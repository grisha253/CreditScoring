import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from pltr import PLTR, Boosting, LinearLogit, QuadraticALassoLogit, RandomForest
from pltr.data import FEATURES, describe, impute_mean, load_kaggle
from pltr.metrics import all_metrics, auc, cutoff_matching_defaults, misclassification_cost

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUT = RESULTS / "kaggle.json"

MODELS = ["Linear logit", "Non-linear logit", "Non-linear logit + ALasso", "Random forest", "PLTR", "Boosting"]
COST_FN = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50]


def make(name, seed, jobs):
    return {
        "Linear logit": lambda: LinearLogit(),
        "Non-linear logit": lambda: LinearLogit(quadratic=True),
        "Non-linear logit + ALasso": lambda: QuadraticALassoLogit(random_state=seed, n_jobs=jobs),
        "Random forest": lambda: RandomForest(grid=(100, 200, 300), random_state=seed, n_jobs=jobs),
        "PLTR": lambda: PLTR(random_state=seed, n_jobs=jobs),
        "Boosting": lambda: Boosting(random_state=seed),
    }[name]()


def run_split(X, y, rep, half, tr, te, jobs):
    Xtr, Xte = impute_mean(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    out = {"rep": rep, "half": half, "metrics": {}, "interpretability": {}, "cost": {}, "seconds": {}}
    probs = {}
    for name in MODELS:
        t0 = time.time()
        m = make(name, seed=10 * rep + half, jobs=jobs).fit(Xtr, ytr) if name != "PLTR" else \
            make(name, seed=10 * rep + half, jobs=jobs).fit(Xtr, ytr, names=FEATURES)
        p_tr = m.oob_proba_ if name == "Random forest" else m.predict_proba(Xtr)
        p_te = m.predict_proba(Xte)
        cut = cutoff_matching_defaults(ytr, p_tr)
        out["metrics"][name] = all_metrics(yte, p_te, cut)
        out["interpretability"][name] = m.interpretability()
        out["cost"][name] = {c: misclassification_cost(yte, p_te, cut, c) for c in COST_FN}
        out["seconds"][name] = time.time() - t0
        probs[name] = p_te
        if name == "PLTR":
            out["pltr_candidates"] = m.n_candidates_
            out["pltr_rules_after_dedup"] = len(m.rules_)
            out["pltr_lambda"] = m.lasso_.lambda_
            out["pltr_lambda_index"] = m.lasso_.best_index_
            out["pltr_path"] = [{"step": i, "lambda": lam, "n_active": k, "AUC": auc(yte, pp)}
                                for i, (lam, k, pp) in enumerate(m.lasso_.path(m._design(Xte)))]
        print(f"  rep {rep} half {half} {name:28s} AUC {out['metrics'][name]['AUC']:.4f} "
              f"({out['seconds'][name]:.0f}s)", flush=True)
    return out


def five_by_two_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
    t = d[0, 0] / np.sqrt(s2.mean())
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), df=5))


def summarise(splits):
    S = {"n_splits": len(splits)}
    S["table1"] = {m: {k: float(np.nanmean([s["metrics"][m][k] for s in splits])) for k in
                       ["AUC", "PGI", "PCC", "KS", "BS"]} for m in MODELS}
    S["table1_std"] = {m: {k: float(np.nanstd([s["metrics"][m][k] for s in splits])) for k in
                           ["AUC", "PGI", "PCC", "KS", "BS"]} for m in MODELS}
    S["table2"] = {}
    for m in ["Linear logit", "Random forest", "PLTR", "Boosting", "Non-linear logit + ALasso"]:
        keys = splits[0]["interpretability"][m].keys()
        S["table2"][m] = {k: float(np.mean([s["interpretability"][m][k] for s in splits])) for k in keys}
    base = {c: np.mean([s["cost"]["Linear logit"][c] for s in splits]) for c in COST_FN}
    S["cost_reduction_vs_linear"] = {
        m: {c: float(1 - np.mean([s["cost"][m][c] for s in splits]) / base[c]) for c in COST_FN} for m in MODELS}
    S["seconds"] = {m: float(np.mean([s["seconds"][m] for s in splits])) for m in MODELS}

    if len(splits) == 10:
        tests = {}
        grid = lambda m, k: np.array([[s["metrics"][m][k] for s in splits if s["rep"] == r] for r in range(5)])
        for a, b in [("PLTR", "Linear logit"), ("PLTR", "Random forest"), ("Boosting", "PLTR"),
                     ("Random forest", "Linear logit"), ("PLTR", "Non-linear logit + ALasso")]:
            tests[f"{a} vs {b}"] = {k: dict(zip(["diff", "t", "p"], five_by_two_t(grid(a, k), grid(b, k))))
                                    for k in ["AUC", "BS", "KS"]}
        S["tests_5x2cv"] = tests

    path = {}
    for s in splits:
        for pt in s["pltr_path"]:
            path.setdefault(pt["step"], []).append((pt["n_active"], pt["AUC"]))
    S["pltr_path"] = [{"step": k, "n_active": float(np.mean([a for a, _ in v])), "AUC": float(np.mean([b for _, b in v]))}
                      for k, v in sorted(path.items())]
    S["pltr_selected_step"] = [s["pltr_lambda_index"] for s in splits]
    S["pltr_candidates"] = splits[0]["pltr_candidates"]
    return S


def full_sample_rules(X, y, top=10):
    Xf, _ = impute_mean(X, X[:1])
    m = PLTR(random_state=0, n_jobs=2).fit(Xf, y, names=FEATURES)
    ame = m.average_marginal_effects(Xf)
    df = pd.DataFrame(ame, columns=["term", "kind", "ame"])
    df["abs"] = df["ame"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs")
    share = {r.text(FEATURES): float(r.values(Xf).mean()) for r in m.rules_}
    df["share_on"] = df["term"].map(share)
    df.to_csv(RESULTS / "kaggle_rules.csv", index=False)
    rules_only = df[df["kind"] != "linear"].head(top)
    return {"rules_top": rules_only.to_dict("records"), "all_terms": df.to_dict("records"),
            "interpretability": m.interpretability()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT / "data" / "cs-training.csv"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    X, y, df = load_kaggle(args.data)
    RESULTS.mkdir(exist_ok=True)
    state = json.loads(OUT.read_text()) if OUT.exists() else {}
    state["data"] = describe(df)
    splits = state.get("splits", [])
    done = {(s["rep"], s["half"]) for s in splits}

    for rep in range(args.reps):
        folds = list(StratifiedKFold(2, shuffle=True, random_state=rep).split(X, y))
        for half, (tr, te) in enumerate(folds):
            if (rep, half) in done:
                continue
            print(f"rep {rep} half {half}", flush=True)
            splits.append(run_split(X, y, rep, half, tr, te, args.jobs))
            state["splits"] = splits
            OUT.write_text(json.dumps(state, indent=1, default=float))

    state["summary"] = summarise(splits)
    print("full-sample PLTR for the rules table", flush=True)
    state["rules"] = full_sample_rules(X, y)
    OUT.write_text(json.dumps(state, indent=1, default=float))
    print(json.dumps(state["summary"]["table1"], indent=1))


if __name__ == "__main__":
    main()
