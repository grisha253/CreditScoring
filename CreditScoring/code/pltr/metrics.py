import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y, p):
    return float(roc_auc_score(y, p))


def pgi(y, p, a=0.0, b=0.4):
    m = (p >= a) & (p <= b)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(2 * roc_auc_score(y[m], p[m]) - 1)


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def cutoff_matching_defaults(y_train, p_train):
    rate = float(np.mean(y_train))
    return float(np.quantile(p_train, 1 - rate))


def pcc(y, p, cutoff):
    return float(np.mean((p > cutoff).astype(int) == y))


def all_metrics(y, p, cutoff):
    return {"AUC": auc(y, p), "PGI": pgi(y, p), "PCC": pcc(y, p, cutoff), "KS": ks(y, p), "BS": brier(y, p)}


def misclassification_cost(y, p, cutoff, c_fn, c_fp=1.0):
    pred = (p > cutoff).astype(int)
    neg, pos = y == 0, y == 1
    fpr = np.mean(pred[neg] == 1)
    fnr = np.mean(pred[pos] == 0)
    return float(c_fp * fpr + c_fn * fnr)
