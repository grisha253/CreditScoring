import numpy as np


def simulate(p, n=5000, labels="paper", scale=1.0, rng=None):
    rng = np.random.default_rng(rng)
    X = rng.standard_normal((n, p))
    lo, hi = np.quantile(X, 0.1, axis=0), np.quantile(X, 0.9, axis=0)
    g = rng.uniform(lo, hi)
    d = rng.uniform(lo, hi)
    b0 = rng.uniform(-1, 1)
    b = rng.uniform(-1, 1, p)
    eta = b0 + (X <= g).astype(float) @ b
    D = (X <= d).astype(float)
    for j in range(p - 1):
        bjk = rng.uniform(-1, 1, p - j - 1)
        eta += D[:, j] * (D[:, j + 1:] @ bjk)
    eta *= scale
    P = 1 / (1 + np.exp(-eta))
    if labels == "paper":
        y = (P > np.median(P)).astype(int)
    elif labels == "bernoulli":
        y = (rng.uniform(size=n) < P).astype(int)
    else:
        raise ValueError(labels)
    return X, y, P
