from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SeriousDlqin2yrs"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_kaggle(path):
    df = pd.read_csv(Path(path))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(int)
    return X, y, df


def impute_mean(X_train, X_test):
    mu = np.nanmean(X_train, axis=0)
    X_train, X_test = X_train.copy(), X_test.copy()
    for X in (X_train, X_test):
        r, c = np.where(np.isnan(X))
        X[r, c] = mu[c]
    return X_train, X_test


def describe(df):
    y = df[TARGET]
    return {
        "n_loans": int(len(df)),
        "n_defaults": int(y.sum()),
        "default_rate": float(y.mean()),
        "missing_share": {c: float(df[c].isna().mean()) for c in FEATURES if df[c].isna().any()},
        "quantiles": {c: {q: float(df[c].quantile(q)) for q in (0.5, 0.99, 1.0)} for c in FEATURES},
    }
