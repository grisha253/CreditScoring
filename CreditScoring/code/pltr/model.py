from dataclasses import dataclass
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def _lasso_qp(G, c, theta, lam):
    k = len(c)
    pen = np.full(k, lam)
    pen[0] = 0.0

    def fun(x):
        th = x[:k] - x[k:]
        g = G @ th - c
        f = 0.5 * th @ (G @ th) - c @ th + pen @ (x[:k] + x[k:])
        return f, np.r_[g + pen, -g + pen]

    x0 = np.r_[np.maximum(theta, 0), np.maximum(-theta, 0)]
    x0[0], x0[k] = theta[0], 0.0
    bounds = [(None, None)] + [(0, None)] * (k - 1) + [(0, 0)] + [(0, None)] * (k - 1)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=20000, maxcor=30, ftol=1e-15, gtol=1e-10))
    th = res.x[:k] - res.x[k:]
    th[1:][np.abs(th[1:]) < 1e-12] = 0.0
    return th


def l1_logit_path(Z, y, lambdas, theta0=None, max_irls=50, tol=1e-6):
    n, k = Z.shape
    A = np.column_stack([np.ones(n), Z])
    ybar = y.mean()
    theta = np.zeros(k + 1) if theta0 is None else theta0.copy()
    if theta0 is None:
        theta[0] = np.log(ybar / (1 - ybar))
    out = np.zeros((len(lambdas), k + 1))

    def objective(th, lam):
        eta = A @ th
        return np.mean(np.logaddexp(0, eta) - y * eta) + lam * np.abs(th[1:]).sum()

    for i, lam in enumerate(lambdas):
        f_old = objective(theta, lam)
        for _ in range(max_irls):
            eta = A @ theta
            pr = 1 / (1 + np.exp(-eta))
            w = np.clip(pr * (1 - pr), 1e-5, None)
            z = eta + (y - pr) / w
            Aw = A * w[:, None]
            G = Aw.T @ A / n
            c = Aw.T @ z / n
            new = _lasso_qp(G, c, theta.copy(), lam)
            step, f_new = 1.0, objective(new, lam)
            while f_new > f_old + 1e-12 and step > 1e-4:
                step /= 2
                cand = theta + step * (new - theta)
                f_new = objective(cand, lam)
                new = cand
            converged = abs(f_old - f_new) < tol * max(1.0, abs(f_old))
            theta, f_old = new, f_new
            if converged:
                break
        out[i] = theta
    return out


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AdaptiveLassoLogit:
    def __init__(self, n_lambda=15, lambda_ratio=1e-4, cv=10, ridge_C=1.0, random_state=0, max_iter=2000, n_jobs=1,
                 solver="path"):
        self.solver = solver
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.cv = cv
        self.ridge_C = ridge_C
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    LIBLINEAR_CS = np.logspace(-3, 1, 13)

    def _fit_path(self, Zt, y):
        if self.solver == "path":
            return l1_logit_path(Zt, y, self.lambdas_)
        return np.array([self._liblinear(C, Zt, y) for C in self.LIBLINEAR_CS])

    def _liblinear(self, C, Zt, y):
        m = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", tol=1e-4, max_iter=self.max_iter,
                               random_state=self.random_state).fit(Zt, y)
        return np.r_[m.intercept_, m.coef_.ravel()]

    def _fold_loglik(self, Zt, y, tr, va):
        path = self._fit_path(Zt[tr], y[tr])
        eta = path[:, :1].T + Zt[va] @ path[:, 1:].T
        p = np.clip(_sigmoid(eta), 1e-12, 1 - 1e-12)
        yv = y[va][:, None]
        return list(np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p), axis=0))

    def fit(self, F, y):
        F = np.asarray(F, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = F.mean(0)
        self.sd_ = F.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (F - self.mean_) / self.sd_

        ridge = LogisticRegression(C=self.ridge_C, l1_ratio=0.0, solver="lbfgs", max_iter=self.max_iter)
        ridge.fit(Z, y)
        self.ridge_coef_ = ridge.coef_.ravel()
        self.scale_ = np.abs(self.ridge_coef_)
        Zt = Z * self.scale_

        lam_max = np.max(np.abs(Zt.T @ (y - y.mean()))) / len(y)
        self.lambdas_ = lam_max * np.logspace(0, np.log10(self.lambda_ratio), self.n_lambda)
        if self.solver == "liblinear":
            self.lambdas_ = 1 / (len(y) * self.LIBLINEAR_CS)
        folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        ll = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._fold_loglik)(Zt, y, tr, va) for tr, va in folds.split(Zt, y)))
        self.cv_loglik_ = ll.mean(0)
        self.best_index_ = int(np.argmax(self.cv_loglik_))
        self.lambda_ = float(self.lambdas_[self.best_index_])

        if self.solver == "path":
            self.path_ = self._fit_path(Zt, y)
            theta = self.path_[self.best_index_]
        else:
            theta = self._liblinear(self.LIBLINEAR_CS[self.best_index_], Zt, y)
        self.coef_, self.intercept_ = self._to_original(theta)
        return self

    def _to_original(self, theta):
        theta_z = theta[1:] * self.scale_
        coef = theta_z / self.sd_
        intercept = float(theta[0] - np.sum(theta_z * self.mean_ / self.sd_))
        return coef, intercept

    def path(self, F_test):
        out = []
        for lam, theta in zip(self.lambdas_, self.path_):
            coef, b0 = self._to_original(theta)
            out.append((float(lam), int(np.sum(coef != 0)), _sigmoid(np.asarray(F_test, float) @ coef + b0)))
        return out

    def decision_function(self, F):
        return np.asarray(F, float) @ self.coef_ + self.intercept_

    def predict_proba(self, F):
        return _sigmoid(self.decision_function(F))

    @property
    def C_(self):
        return self.lambda_

    @property
    def n_active(self):
        return int(np.sum(self.coef_ != 0))


@dataclass(frozen=True)
class Cond:
    j: int
    le: bool
    thr: float

    def mask(self, X):
        return X[:, self.j] <= self.thr if self.le else X[:, self.j] > self.thr

    def text(self, names, digits=4):
        op = "<" if self.le else "≥"
        return f"{names[self.j]} {op} {_fmt(self.thr, digits)}"


def _fmt(x, digits):
    return f"{x:.{digits}g}"


@dataclass(frozen=True)
class Rule:
    conds: tuple

    def values(self, X):
        m = np.ones(len(X), dtype=bool)
        for c in self.conds:
            m &= c.mask(X)
        return m.astype(np.int8)

    def text(self, names):
        return " & ".join(c.text(names) for c in self.conds)

    @property
    def n_predicates(self):
        return len(self.conds)


def _univariate_rule(x, y, j, random_state):
    t = DecisionTreeClassifier(max_depth=1, random_state=random_state).fit(x.reshape(-1, 1), y).tree_
    if t.node_count < 3:
        return None
    return Rule((Cond(j, True, float(t.threshold[0])),))


def _bivariate_rule(X2, y, j, k, random_state):
    t = DecisionTreeClassifier(max_leaf_nodes=3, random_state=random_state).fit(X2, y).tree_
    if t.node_count < 5:
        return None
    cols = (j, k)
    left, right = t.children_left[0], t.children_right[0]
    inner = left if t.children_left[left] != -1 else right
    root = Cond(cols[t.feature[0]], inner == left, float(t.threshold[0]))
    second = Cond(cols[t.feature[inner]], True, float(t.threshold[inner]))
    return Rule((root, second))


def extract_rules(X, y, random_state=0):
    p = X.shape[1]
    uni = [r for j in range(p) if (r := _univariate_rule(X[:, j], y, j, random_state)) is not None]
    bi = [r for j, k in combinations(range(p), 2)
          if (r := _bivariate_rule(X[:, [j, k]], y, j, k, random_state)) is not None]
    return uni, bi


def drop_redundant(rules, X):
    kept, seen = [], set()
    for r in rules:
        v = r.values(X)
        if v.min() == v.max():
            continue
        key, comp = np.packbits(v).tobytes(), np.packbits(1 - v).tobytes()
        if key in seen or comp in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


class PLTR:
    def __init__(self, include_linear=True, cv=10, random_state=0, n_jobs=1, solver="path"):
        self.include_linear = include_linear
        self.cv = cv
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _design(self, X):
        V = np.column_stack([r.values(X) for r in self.rules_]).astype(float)
        return np.column_stack([X, V]) if self.include_linear else V

    def fit(self, X, y, names=None):
        X = np.asarray(X, float)
        self.names_ = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
        uni, bi = extract_rules(X, y, self.random_state)
        self.n_candidates_ = {"univariate": len(uni), "bivariate": len(bi)}
        self.rules_ = drop_redundant(uni + bi, X)
        self.lasso_ = AdaptiveLassoLogit(cv=self.cv, random_state=self.random_state, n_jobs=self.n_jobs,
                                         solver=self.solver)
        self.lasso_.fit(self._design(X), y)
        return self

    def predict_proba(self, X):
        return self.lasso_.predict_proba(self._design(np.asarray(X, float)))


    def terms(self):
        out = []
        coefs = self.lasso_.coef_
        offset = 0
        if self.include_linear:
            for j, name in enumerate(self.names_):
                out.append((name, "linear", 1, float(coefs[j])))
            offset = len(self.names_)
        for i, r in enumerate(self.rules_):
            kind = "univariate" if r.n_predicates == 1 else "bivariate"
            out.append((r.text(self.names_), kind, r.n_predicates, float(coefs[offset + i])))
        return out

    def active_terms(self):
        return [t for t in self.terms() if t[3] != 0]

    def interpretability(self):
        act = self.active_terms()
        return {
            "decision_set": len(act),
            "max_predicates": max((t[2] for t in act), default=0),
            "active_linear": sum(t[1] == "linear" for t in act),
            "active_univariate": sum(t[1] == "univariate" for t in act),
            "active_bivariate": sum(t[1] == "bivariate" for t in act),
        }

    def average_marginal_effects(self, X):
        p = self.predict_proba(X)
        scale = float(np.mean(p * (1 - p)))
        return [(label, kind, coef * scale) for label, kind, _, coef in self.active_terms()]
