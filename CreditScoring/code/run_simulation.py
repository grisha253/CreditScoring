import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from pltr import PLTR, Boosting, LinearLogit, RandomForest
from pltr.metrics import auc, brier, cutoff_matching_defaults, pcc
from pltr.simulation import simulate

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "simulation.json"
PS = [4, 8, 12, 16, 20]
REGIMES = ["paper", "bernoulli"]


def one_replication(regime, p, rep):
    seed = 1000 * p + rep + (0 if regime == "paper" else 500_000)
    X, y, P = simulate(p, n=5000, labels=regime, rng=seed)
    tr, te = np.arange(2500), np.arange(2500, 5000)
    models = {
        "Linear logit": LinearLogit(),
        "Non-linear logit": LinearLogit(quadratic=True),
        "Random forest": RandomForest(grid=(100, 200, 300), random_state=rep, n_jobs=1),
        "PLTR": PLTR(random_state=rep, solver="liblinear"),
        "Boosting": Boosting(random_state=rep),
    }
    rows = []
    for name, m in models.items():
        t0 = time.time()
        m.fit(X[tr], y[tr])
        p_tr = m.oob_proba_ if name == "Random forest" else m.predict_proba(X[tr])
        cut = cutoff_matching_defaults(y[tr], p_tr)
        p_te = m.predict_proba(X[te])
        row = dict(regime=regime, p=p, rep=rep, model=name, PCC=pcc(y[te], p_te, cut), AUC=auc(y[te], p_te),
                   BS=brier(y[te], p_te), seconds=time.time() - t0)
        if name in ("PLTR", "Random forest"):
            row.update(m.interpretability())
        rows.append(row)
    oracle_cut = cutoff_matching_defaults(y[tr], P[tr])
    rows.append(dict(regime=regime, p=p, rep=rep, model="True probabilities", PCC=pcc(y[te], P[te], oracle_cut),
                     AUC=auc(y[te], P[te]), BS=brier(y[te], P[te]), seconds=0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    OUT.parent.mkdir(exist_ok=True)
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["regime"], r["p"], r["rep"]) for r in rows}
    todo = [(g, p, k) for k in range(args.reps) for g in REGIMES for p in PS if (g, p, k) not in done]
    print(f"{len(todo)} replications to run")

    t0 = time.time()
    batch = 2 * args.jobs
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = Parallel(n_jobs=args.jobs)(delayed(one_replication)(*t) for t in chunk)
        for r in res:
            rows.extend(r)
        OUT.write_text(json.dumps(rows, indent=1))
        print(f"{i + len(chunk)}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
