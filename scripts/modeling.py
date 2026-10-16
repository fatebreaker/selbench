"""Shared modelling and scoring utilities (kernel ridge with closed-form LOO, partial correlations)."""
import numpy as np

ALPHAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def zscore(df):
    sd = df.std().replace(0, 1).fillna(1)
    return ((df - df.mean()) / sd).fillna(0)


def block_kernel(blocks):
    """Linear kernel with each standardised block contributing equally."""
    K = None
    for b in blocks:
        Xb = zscore(b).values.astype(np.float64) / np.sqrt(b.shape[1])
        Kb = Xb @ Xb.T
        K = Kb if K is None else K + Kb
    return K / len(blocks)


def kernel_ridge_loo(K_tr, y, K_te):
    """Kernel ridge with intercept; alpha (scaled by mean diag) chosen by closed-form LOO error."""
    ym = y.mean()
    yc = y - ym
    w, V = np.linalg.eigh(K_tr)
    w = np.clip(w, 0, None)
    Vy = V.T @ yc
    scale = np.mean(np.diag(K_tr))
    best = None
    for a in ALPHAS * scale:
        shrink = w / (w + a)
        fitted = V @ (shrink * Vy)
        hdiag = (V ** 2) @ shrink
        loo = (yc - fitted) / np.clip(1 - hdiag - 1.0 / len(y), 1e-6, None)
        err = np.mean(loo ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    a = best[1]
    coef = V @ (Vy / (w + a))
    return ym + K_te @ coef, a


def partial_corr(x, y, g):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(g)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    X = np.c_[np.ones(ok.sum()), g[ok]]
    rx = x[ok] - X @ np.linalg.lstsq(X, x[ok], rcond=None)[0]
    ry = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])
