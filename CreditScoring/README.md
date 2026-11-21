# Research Seminar, Group 6 — PLTR for credit scoring

Adaptation of Dumitrescu, Hué, Hurlin & Tokpavi (2022), *Machine learning for credit scoring: Improving logistic
regression with non-linear decision-tree effects*, European Journal of Operational Research 297(3), 1178–1192.

PLTR (penalised logistic tree regression): rules from one-split trees (one per predictor) and two-split trees
(one per pair of predictors) enter a logistic regression estimated with the adaptive lasso.

## Structure

```
data/                    put cs-training.csv here (Kaggle "Give Me Some Credit"); not included in the repository
code/pltr/model.py       PLTR: rule extraction, duplicate removal, adaptive-lasso logit, marginal effects
code/pltr/baselines.py   linear logit, logit with squares and interactions (+ adaptive lasso), random forest, boosting
code/pltr/metrics.py     AUC, partial Gini (PGI), PCC, KS, Brier score, misclassification cost
code/pltr/simulation.py  Monte Carlo design of the paper (Section 2.1) + noisy-label version
code/pltr/data.py        loading and mean imputation
code/run_simulation.py   simulation experiments  -> results/simulation.json
code/run_kaggle.py       5x2 cross-validation on Kaggle -> results/kaggle.json, results/kaggle_rules.csv
code/make_figures.py     figures/*.png and results/summary.json
```

## Run

```
pip install -r requirements.txt
python code/run_simulation.py      # 20 runs x 5 values of p x 2 label regimes
python code/run_kaggle.py          # 5 x 2 cross-validation, 6 models, then the full-sample rules table
python code/make_figures.py
```

Both experiment scripts save after every step and resume if interrupted. Run times on a 2-core machine:
simulation about 25 minutes, Kaggle about 30 minutes.

## Main results (Kaggle, mean of the 10 test halves of 5 x 2 cross-validation)

| Model | AUC | Paper |
|---|---|---|
| Linear logit | 0.698 | 0.698 |
| Logit + squares and interactions | 0.816 | 0.766 |
| Same with adaptive lasso | 0.818 | 0.806 |
| Random forest | 0.845 | 0.853 |
| PLTR | 0.856 | 0.857 |
| Gradient boosting (ours) | 0.863 | — |

PLTR keeps 36.8 terms on average (paper: 40) and cuts misclassification costs by 17.9% vs the linear logit
(paper: 18.1%). Full tables: results/kaggle.json, results/summary.json.


## Our choices where the paper is not specific

- Two-split trees: three-leaf trees (`max_leaf_nodes=3`); the first leaf of the branch that is split again is kept.
- Duplicates: a rule identical to, or the complement of, a rule already kept is dropped.
- Adaptive lasso: ridge logit (C = 1, standardised features) for the weights, nu = 1. Kaggle: 15 values of lambda
  from lambda_max down to 1e-4 * lambda_max, solved with a glmnet-style proximal-Newton path (`l1_logit_path`,
  checked against liblinear), lambda chosen by 10-fold CV on the log-likelihood. Simulation: liblinear on a grid
  of 13 values of C from 1e-3 to 10 (`solver="liblinear"`), same CV rule.
- PGI: Gini index (2·AUC − 1) on the loans with predicted PD in [0, 0.4].
- PCC cutoff: predicted defaults = observed defaults on the learning half; out-of-bag PDs for the random forest.
- Random forest: sqrt(p) features per split, number of trees from {100, 200, 300} by out-of-bag Brier score.
- Gradient boosting (our extension): scikit-learn `HistGradientBoostingClassifier`, learning rate 0.05,
  31 leaves, early stopping on 10% of the learning half.
- Simulation: 20 runs per p instead of 100; p in {4, 8, 12, 16, 20}.
