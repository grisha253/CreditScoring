import json
import sys
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"

NAVY = "#0D2D6A"
LIME = "#D4F500"
LAVENDER = "#C7B0EA"
GRAY = "#8A8FA3"
LIGHT = "#F3F4F8"
INK = "#3A4260"

COLORS = {
    "Linear logit": GRAY,
    "Non-linear logit": LAVENDER,
    "Non-linear logit + ALasso": "#9C86C9",
    "Random forest": "#6A4FB3",
    "PLTR": NAVY,
    "Boosting": "#A3BA00",
    "True probabilities": "#B8BCCB",
}
SHORT = {
    "Linear logit": "Linear\nlogit",
    "Non-linear logit": "Non-linear\nlogit",
    "Non-linear logit + ALasso": "Non-linear\nlogit + ALasso",
    "Random forest": "Random\nforest",
    "PLTR": "PLTR",
    "Boosting": "Gradient\nboosting",
}

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#B8BCCB",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlecolor": NAVY,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#E6E8EF",
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("figure", name)


def sim_frame():
    return pd.DataFrame(json.loads((RESULTS / "simulation.json").read_text()))


def sim_summary(d):
    g = d.groupby(["regime", "p", "model"])
    out = {}
    for (regime, p, model), x in g:
        r = out.setdefault(regime, {}).setdefault(str(p), {})
        r[model] = {"PCC": float(x.PCC.mean()), "PCC_se": float(x.PCC.std(ddof=1) / np.sqrt(len(x))),
                    "AUC": float(x.AUC.mean()), "n": int(len(x))}
        if "decision_set" in x and x["decision_set"].notna().any():
            r[model]["decision_set"] = float(x["decision_set"].mean())
            r[model]["max_predicates"] = float(x["max_predicates"].mean())
            if "leaves_per_tree" in x and x["leaves_per_tree"].notna().any():
                r[model]["leaves_per_tree"] = float(x["leaves_per_tree"].mean())
                r[model]["n_trees"] = float(x["n_trees"].mean())
    diffs = {}
    for regime in d.regime.unique():
        for p in sorted(d.p.unique()):
            sub = d[(d.regime == regime) & (d.p == p)].pivot_table(index="rep", columns="model", values="PCC")
            for a, b in [("PLTR", "Random forest"), ("PLTR", "Linear logit"), ("Boosting", "PLTR")]:
                x = sub[a] - sub[b]
                diffs.setdefault(regime, {}).setdefault(str(p), {})[f"{a} - {b}"] = {
                    "mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}
    return {"by_p": out, "diffs": diffs, "reps": int(d.rep.nunique())}


def fig_sim(d, regime, name, title, w, h, models, oracle=False):
    fig, ax = plt.subplots(figsize=(w, h))
    sub = d[d.regime == regime]
    ps = sorted(sub.p.unique())
    for m in models + (["True probabilities"] if oracle else []):
        x = sub[sub.model == m].groupby("p").PCC
        mu, se = x.mean() * 100, x.std(ddof=1) / np.sqrt(x.count()) * 100
        style = dict(color=COLORS[m], lw=2.6 if m == "PLTR" else 2, marker="o", ms=5,
                     ls="--" if m in ("True probabilities", "Boosting") else "-")
        label = "True probabilities (ceiling)" if m == "True probabilities" else m
        ax.plot(ps, mu, label=label, zorder=3 if m == "PLTR" else 2, **style)
        ax.fill_between(ps, mu - 1.96 * se, mu + 1.96 * se, color=COLORS[m], alpha=0.15, lw=0)
    ax.set_xticks(ps)
    ax.set_xlabel("Number of predictors p (more thresholds and interactions →)")
    ax.set_ylabel("Correct classification on test, %")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=10, loc="upper right" if regime == "paper" else "lower right", ncol=1)
    fig.tight_layout()
    save(fig, name)


def fig_noise(d, w, h):
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    for ax, regime, title in zip(axes, ["paper", "bernoulli"], ["Paper: y = 1 if PD > median", "Ours: y ~ Bernoulli(PD)"]):
        sub = d[d.regime == regime].pivot_table(index=["p", "rep"], columns="model", values="PCC")
        ps = sorted(sub.index.get_level_values(0).unique())
        for m in ["Random forest", "PLTR", "Boosting"]:
            g = ((sub[m] - sub["Linear logit"]) * 100).groupby(level=0)
            mu, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
            ax.plot(ps, mu, marker="o", ms=4.5, lw=2.6 if m == "PLTR" else 2, color=COLORS[m],
                    ls="--" if m == "Boosting" else "-", label=m.replace("Boosting", "Gradient boosting"),
                    zorder=3 if m == "PLTR" else 2)
            ax.fill_between(ps, mu - 1.96 * se, mu + 1.96 * se, color=COLORS[m], alpha=0.15, lw=0)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks(ps)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.set_xlabel("Number of predictors p")
    axes[0].set_ylabel("Gain over linear logit, pp of PCC")
    axes[1].legend(fontsize=9.5, loc="upper left")
    fig.tight_layout()
    save(fig, "sim_noise")


def kaggle_state():
    f = RESULTS / "kaggle.json"
    return json.loads(f.read_text()) if f.exists() else None


def fig_auc(K, w, h):
    splits = K["splits"]
    models = ["Linear logit", "Non-linear logit", "Non-linear logit + ALasso", "Random forest", "PLTR", "Boosting"]
    fig, ax = plt.subplots(figsize=(w, h))
    mu = [np.mean([s["metrics"][m]["AUC"] for s in splits]) for m in models]
    sd = [np.std([s["metrics"][m]["AUC"] for s in splits]) for m in models]
    paper = {"Linear logit": 0.6983, "Non-linear logit": 0.7660, "Non-linear logit + ALasso": 0.8062,
             "Random forest": 0.8529, "PLTR": 0.8568}
    x = np.arange(len(models))
    bars = ax.bar(x - 0.18, mu, 0.36, yerr=sd, capsize=3, color=[COLORS[m] for m in models],
                  edgecolor=NAVY, linewidth=0.6, label="Ours (mean of 10 test halves)", zorder=3,
                  error_kw=dict(ecolor=INK, lw=1))
    if any(m == "Boosting" for m in models):
        bars[-1].set_hatch("//")
    pv = [paper.get(m, np.nan) for m in models]
    ax.bar(x + 0.18, pv, 0.36, color="white", edgecolor=GRAY, linewidth=1.2, hatch="...", label="Paper (Table 1)",
           zorder=3)
    for i, v in enumerate(mu):
        ax.text(i - 0.18, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=NAVY,
                fontweight="bold")
    for i, v in enumerate(pv):
        if not np.isnan(v):
            ax.text(i + 0.18, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=GRAY)
    ax.set_xticks(x, [SHORT[m] for m in models], fontsize=10)
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("AUC on the test half")
    ax.set_title("Out-of-sample AUC, Kaggle “Give Me Some Credit”, 5×2 cross-validation", loc="left")
    ax.legend(fontsize=10, loc="upper left")
    ax.axhline(0.5, color=GRAY, lw=0.8)
    fig.tight_layout()
    save(fig, "kaggle_auc")


def fig_decision_set(K, w, h):
    T = K["summary"]["table2"]
    rows = [("Linear logit", T["Linear logit"]["decision_set"]),
            ("PLTR", T["PLTR"]["decision_set"]),
            ("Gradient boosting", T["Boosting"]["decision_set"]),
            ("Random forest", T["Random forest"]["decision_set"])]
    fig, ax = plt.subplots(figsize=(w, h))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [COLORS["Linear logit"], COLORS["PLTR"], COLORS["Boosting"], COLORS["Random forest"]][::-1]
    ax.barh(names, vals, color=cols, edgecolor=NAVY, linewidth=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v * 1.25, i, f"{v:,.0f}", va="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1, max(vals) * 40)
    ax.set_xlabel("Size of the decision set (log scale)")
    ax.set_title("How many rules each model uses", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "decision_set")


def fig_cost(K, w, h):
    C = K["summary"]["cost_reduction_vs_linear"]
    fig, ax = plt.subplots(figsize=(w, h))
    for m in ["Non-linear logit + ALasso", "Random forest", "PLTR", "Boosting"]:
        xs = sorted(int(c) for c in C[m])
        ys = [C[m][str(c)] * 100 if str(c) in C[m] else C[m][c] * 100 for c in xs]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=2.6 if m == "PLTR" else 2, color=COLORS[m],
                ls="--" if m == "Boosting" else "-", label=m)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("Cost of granting a loan to a bad borrower, C_FN (C_FP = 1)")
    ax.set_ylabel("Cost reduction vs linear logit, %")
    ax.set_title("Misclassification cost saved relative to logit", loc="left")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "cost")


def fig_boosting(K, w, h):
    splits = K["splits"]
    models = ["Linear logit", "Random forest", "PLTR", "Boosting"]
    metrics = ["AUC", "KS"]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(metrics))
    width = 0.2
    for i, m in enumerate(models):
        mu = [np.mean([s["metrics"][m][k] for s in splits]) for k in metrics]
        sd = [np.std([s["metrics"][m][k] for s in splits]) for k in metrics]
        b = ax.bar(x + (i - 1.5) * width, mu, width, yerr=sd, capsize=3, color=COLORS[m], edgecolor=NAVY,
                   linewidth=0.6, label=m.replace("Boosting", "Gradient boosting"), zorder=3,
                   error_kw=dict(ecolor=INK, lw=1))
        for xi, v, e in zip(x + (i - 1.5) * width, mu, sd):
            ax.text(xi, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=NAVY, va="bottom")
    ax.set_xticks(x, ["AUC", "Kolmogorov–Smirnov"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Boosting vs PLTR vs forest (higher = better)", loc="left")
    ax.legend(fontsize=9.5, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "boosting")


def fig_path(K, w, h):
    P = [p for p in K["summary"]["pltr_path"] if p["n_active"] >= 1]
    T1 = K["summary"]["table1"]
    sel = K["summary"]["table2"]["PLTR"]["decision_set"]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot([p["n_active"] for p in P], [p["AUC"] for p in P], marker="o", color=NAVY, lw=2.4, zorder=3,
            label="PLTR along the lasso path")
    ax.axhline(T1["Random forest"]["AUC"], color=COLORS["Random forest"], ls="-", lw=1.5,
               label=f"Random forest ({T1['Random forest']['AUC']:.3f})")
    ax.axvline(sel, color=LIME, lw=3, alpha=0.9, zorder=1, label=f"chosen by 10-fold CV (~{sel:.0f} terms)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xlabel("Number of active terms in the scorecard (log scale)")
    ax.set_ylabel("AUC on the test half")
    ax.set_title("Accuracy vs number of rules (mean over 10 test halves)", loc="left")
    ax.set_xlim(2.5, 60)
    ax.set_ylim(0.81, 0.865)
    leg = ax.legend(fontsize=9.5, loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    leg.set_zorder(5)
    fig.tight_layout()
    save(fig, "path")


def fig_thresholds(w, h):
    from pltr.data import load_kaggle
    _, _, df = load_kaggle(PROJECT / "data" / "cs-training.csv")
    y = df["SeriousDlqin2yrs"]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ru = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)
    bins = np.r_[np.arange(0, 1.05, 0.1), 1.5]
    cut = pd.cut(ru, bins, include_lowest=True)
    rate = y.groupby(cut, observed=True).mean() * 100
    labels = [f"{max(b.left, 0):.1f}" if b.right <= 1.0001 else ">1" for b in rate.index]
    ax = axes[0]
    ax.bar(range(len(rate)), rate.values, color=[NAVY if b.left >= 0.7 else LAVENDER for b in rate.index],
           edgecolor=NAVY, linewidth=0.5, zorder=3)
    ax.set_xticks(range(len(rate)), labels, fontsize=9)
    ax.set_xlabel("Revolving utilisation of credit lines (bin start)")
    ax.set_ylabel("Default rate, %")
    ax.set_title("Utilisation: risk jumps above ~0.7", loc="left", fontsize=12)
    late = df["NumberOfTimes90DaysLate"].clip(upper=4)
    rate2 = y.groupby(late).mean() * 100
    ax = axes[1]
    ax.bar(range(len(rate2)), rate2.values, color=[LAVENDER] + [NAVY] * (len(rate2) - 1), edgecolor=NAVY,
           linewidth=0.5, zorder=3)
    ax.set_xticks(range(len(rate2)), [f"{v:.0f}" if v < 4 else "4+" for v in rate2.index])
    ax.set_xlabel("Times 90+ days late")
    ax.set_title("Late payments: 0 vs ≥1", loc="left", fontsize=12)
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "data_thresholds")
    ru_raw = df["RevolvingUtilizationOfUnsecuredLines"]
    miss = df["MonthlyIncome"].isna()
    return {"ru_high": float(y[ru_raw >= 0.7].mean()), "ru_low": float(y[ru_raw < 0.7].mean()),
            "late0": float(y[df["NumberOfTimes90DaysLate"] == 0].mean()),
            "late1": float(y[df["NumberOfTimes90DaysLate"] == 1].mean()),
            "miss_rate": float(y[miss].mean()), "nomiss_rate": float(y[~miss].mean())}


def fig_tree(w, h):
    from sklearn.tree import DecisionTreeClassifier

    from pltr.data import FEATURES, impute_mean, load_kaggle
    X, y, _ = load_kaggle(PROJECT / "data" / "cs-training.csv")
    X, _ = impute_mean(X, X[:1])
    j, k = FEATURES.index("NumberOfTimes90DaysLate"), FEATURES.index("RevolvingUtilizationOfUnsecuredLines")
    short = {j: "Times 90+ days late", k: "Utilisation"}
    cols = (j, k)
    t = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0).fit(X[:, [j, k]], y).tree_
    L, R = t.children_left, t.children_right
    inner = L[0] if L[L[0]] != -1 else R[0]
    leaf_other = R[0] if inner == L[0] else L[0]
    n = t.n_node_samples
    rate = lambda node: t.value[node][0][1] / t.value[node][0].sum() * 100 if t.value[node][0].sum() > 1.5 \
        else t.value[node][0][1] * 100
    f0, t0 = cols[t.feature[0]], t.threshold[0]
    f1, t1 = cols[t.feature[inner]], t.threshold[inner]

    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, text, fc, tc, wbox=3.6, hbox=1.05, bold=True):
        ax.add_patch(FancyBboxPatch((x - wbox / 2, y - hbox / 2), wbox, hbox, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=NAVY, lw=1))
        ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=11, fontweight="bold" if bold else None,
                linespacing=1.35)

    def edge(x0, y0, x1, y1, label):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3))
        ax.text((x0 + x1) / 2 + (0.15 if x1 > x0 else -0.15), (y0 + y1) / 2, label, ha="left" if x1 > x0 else "right",
                va="center", fontsize=10, color=INK)

    root_txt = f"All loans: {n[0]:,}\ndefault rate {rate(0):.1f}%"
    box(5, 5.75, root_txt, NAVY, "white")
    inner_left = inner == L[0]
    xi, xo = (2.6, 7.4) if inner_left else (7.4, 2.6)
    op_in = f"< {t0:.1f}" if inner_left else f"≥ {t0:.1f}"
    op_out = f"≥ {t0:.1f}" if inner_left else f"< {t0:.1f}"
    edge(5, 5.2, xi, 4.25, f"{short[f0]} {op_in}")
    edge(5, 5.2, xo, 4.25, f"{short[f0]} {op_out}")
    box(xi, 3.7, f"{n[inner]:,} loans\n{rate(inner):.1f}%", "white", NAVY)
    box(xo, 3.7, f"Leaf: {n[leaf_other]:,} loans\n{rate(leaf_other):.1f}%", LIGHT, NAVY, bold=False)
    ll, lr = L[inner], R[inner]
    xll, xlr = (1.75, 5.25) if inner_left else (4.75, 8.25)
    edge(xi, 3.15, xll, 2.05, f"{short[f1]} < {t1:.2f}")
    edge(xi, 3.15, xlr, 2.05, f"≥ {t1:.2f}")
    box(xll, 1.5, f"Rule V₂ = 1\n{n[ll]:,} loans · {rate(ll):.1f}%", LIME, NAVY, wbox=3.2)
    box(xlr, 1.5, f"Leaf: {n[lr]:,} loans\n{rate(lr):.1f}%", LIGHT, NAVY, wbox=3.2, bold=False)
    ax.text(0.05, 0.25, "Highlighted leaf becomes a 0/1 variable in the logistic regression.", fontsize=10,
            color=INK, style="italic")
    fig.tight_layout()
    save(fig, "tree")
    return {"root": [short[f0], float(t0)], "second": [short[f1], float(t1)], "n": [int(n[0]), int(n[ll])],
            "rate_rule": float(rate(ll)), "rate_all": float(rate(0))}


def main():
    summary = {}
    if (RESULTS / "simulation.json").exists():
        d = sim_frame()
        summary["simulation"] = sim_summary(d)
        fig_sim(d, "paper", "sim_paper", "Simulation with threshold effects (paper’s design)", 6.36, 4.05,
                ["Linear logit", "Non-linear logit", "Random forest", "PLTR"])
        fig_noise(d, 6.39, 4.10)
    K = kaggle_state()
    if K and "summary" in K:
        fig_auc(K, 7.73, 3.80)
        fig_decision_set(K, 4.85, 4.30)
        fig_cost(K, 6.39, 4.10)
        fig_boosting(K, 5.85, 4.10)
        fig_path(K, 7.27, 3.30)
        summary["data_facts"] = fig_thresholds(7.35, 3.40)
        summary["tree"] = fig_tree(6.30, 4.10)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
