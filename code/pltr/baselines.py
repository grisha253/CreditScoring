from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .model import AdaptiveLassoLogit


def quad_features(X):
    p = X.shape[1]
    inter = [X[:, j] * X[:, k] for j, k in combinations(range(p), 2)]
    return np.column_stack([X, X ** 2] + inter)


class _Standardised:
    def _std_fit(self, F):
        self.mean_, self.sd_ = F.mean(0), F.std(0)
        self.sd_[self.sd_ == 0] = 1.0

    def _std(self, F):
        return (F - self.mean_) / self.sd_


class LinearLogit(_Standardised):
    def __init__(self, quadratic=False, max_iter=5000):
        self.quadratic = quadratic
        self.max_iter = max_iter

    def _F(self, X):
        X = np.asarray(X, float)
        return quad_features(X) if self.quadratic else X

    def fit(self, X, y):
        F = self._F(X)
        self._std_fit(F)
        self.model_ = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=self.max_iter).fit(self._std(F), y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._std(self._F(X)))[:, 1]

    def interpretability(self):
        return {"decision_set": int(np.sum(self.model_.coef_ != 0)), "max_predicates": 1}


class QuadraticALassoLogit:
    def __init__(self, cv=10, random_state=0, n_jobs=1):
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.lasso_ = AdaptiveLassoLogit(cv=self.cv, random_state=self.random_state, n_jobs=self.n_jobs)
        self.lasso_.fit(quad_features(np.asarray(X, float)), y)
        return self

    def predict_proba(self, X):
        return self.lasso_.predict_proba(quad_features(np.asarray(X, float)))

    def interpretability(self):
        return {"decision_set": self.lasso_.n_active, "max_predicates": 2}


class RandomForest:
    def __init__(self, grid=(100, 200, 300, 500), random_state=0, n_jobs=-1):
        self.grid = grid
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        rf = RandomForestClassifier(n_estimators=self.grid[0], max_features="sqrt", oob_score=True,
                                    warm_start=True, n_jobs=self.n_jobs, random_state=self.random_state)
        best, scores = None, {}
        for n in self.grid:
            rf.set_params(n_estimators=n)
            rf.fit(X, y)
            oob = rf.oob_decision_function_[:, 1]
            ok = ~np.isnan(oob)
            scores[n] = float(np.mean((oob[ok] - y[ok]) ** 2))
            if best is None or scores[n] < scores[best]:
                best = n
        self.oob_brier_ = scores
        self.n_trees_ = best
        if best != self.grid[-1]:
            rf = RandomForestClassifier(n_estimators=best, max_features="sqrt", oob_score=True,
                                        n_jobs=self.n_jobs, random_state=self.random_state).fit(X, y)
        self.model_ = rf
        oob = rf.oob_decision_function_[:, 1]
        self.oob_proba_ = np.where(np.isnan(oob), np.nanmean(oob), oob)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, 1]

    def interpretability(self):
        leaves = [e.get_n_leaves() for e in self.model_.estimators_]
        depth = [e.get_depth() for e in self.model_.estimators_]
        return {"decision_set": float(np.mean(leaves)) * len(leaves), "n_trees": len(leaves),
                "leaves_per_tree": float(np.mean(leaves)), "max_predicates": float(np.mean(depth))}


class Boosting:
    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, scoring="loss",
            random_state=self.random_state).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, 1]

    def interpretability(self):
        n = int(self.model_.n_iter_)
        leaves = [pred.get_n_leaf_nodes() for it in self.model_._predictors for pred in it]
        depth = [pred.get_max_depth() for it in self.model_._predictors for pred in it]
        return {"decision_set": float(np.sum(leaves)), "n_trees": n, "leaves_per_tree": float(np.mean(leaves)),
                "max_predicates": float(np.mean(depth))}
