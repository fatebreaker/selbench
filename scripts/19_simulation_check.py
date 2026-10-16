"""Simulation check of the decomposition, triad reliabilities and the c^2 identity.

Three synthetic screens share drug-specific selectivity t_d but have partly screen-specific fitness axes
(each loads rho_f on a shared factor, so the true cross-screen axis correlation is rho_f^2) and screen-specific noise. The study's pipeline (rank-normal z,
leave-drug-out fitness axis, OLS residual) is applied unchanged. Estimated reliabilities and c^2 for a
predictor with known correlation to t_d are compared with their true values.
Output: results/simulation/summary.json.
"""
import numpy as np
import pandas as pd

from common import RESULTS, rank_normal, write_json

N_DRUGS, N_CELLS, SEED = 250, 400, 2026


def screen(t, fit, noise_sd, beta_sd, rng):
    beta = np.abs(rng.normal(0.8, beta_sd, N_DRUGS))           # positive loadings on the fitness axis
    raw = beta[:, None] * fit[None, :] + t + rng.normal(0, noise_sd, t.shape)
    Z = np.vstack([rank_normal(r) for r in raw])
    tot, n = Z.sum(0), Z.shape[0]
    G = (tot[None, :] - Z) / (n - 1)
    R = np.empty_like(Z)
    for i in range(n):
        X = np.c_[np.ones(N_CELLS), G[i]]
        R[i] = Z[i] - X @ np.linalg.lstsq(X, Z[i], rcond=None)[0]
    return Z, R, G


def run(rng, rho_f, noise_sd):
    t = rng.normal(0, 1, (N_DRUGS, N_CELLS)) * rng.uniform(0.2, 1.5, (N_DRUGS, 1))   # drug-specific selectivity
    shared = rng.normal(size=N_CELLS)
    fits = [rho_f * shared + np.sqrt(1 - rho_f ** 2) * rng.normal(size=N_CELLS) for _ in range(3)]
    S = [screen(t, f, ns, 0.3, rng) for f, ns in zip(fits, noise_sd)]
    est_axis = np.corrcoef(S[0][0].mean(0), S[1][0].mean(0))[0, 1]
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    rel_est, rel_true, c2_est, c2_true = [], [], [], []
    for d in range(N_DRUGS):
        r = [S[k][1][d] for k in range(3)]
        rab, rac, rbc = corr(r[0], r[1]), corr(r[0], r[2]), corr(r[1], r[2])
        if rbc > 0.1:
            rel_est.append((rab * rac, rbc))
            rel_true.append(corr(r[0], t[d]) ** 2)                # share of screen-A residual explained by t
        f = 0.5 * t[d] / t[d].std() + rng.normal(0, 1, N_CELLS)   # predictor with known corr to t
        g1, g2 = S[1][2][d], S[2][2][d]
        def partial(x, y, g):
            X = np.c_[np.ones(N_CELLS), g]
            rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            return corr(rx, ry)
        p1, p2 = partial(f, S[1][0][d], g1), partial(f, S[2][0][d], g2)
        if rbc > 0.1:
            c2_est.append((p1 * p2, corr(r[1], r[2])))
            c2_true.append(corr(f, t[d]) ** 2)
    num, den = np.array(rel_est).T
    n2, d2 = np.array(c2_est).T
    return {'rho_f': rho_f, 'noise_sd': noise_sd, 'axis_corr_true': rho_f ** 2, 'axis_corr_est': float(est_axis),
            'reliability_est': float(np.median(num) / np.median(den)), 'reliability_true_median': float(np.median(rel_true)),
            'c2_est': float(np.median(n2) / np.median(d2)), 'c2_true_median': float(np.median(c2_true))}


def main():
    rng = np.random.default_rng(SEED)
    rows = [run(rng, rho_f, ns) for rho_f in (0.2, 0.6) for ns in ([1.0, 1.0, 1.0], [0.7, 1.2, 1.5])]
    res = pd.DataFrame(rows)
    write_json(RESULTS / 'simulation' / 'summary.json', {'runs': rows})
    print(res.round(3).to_string())


if __name__ == '__main__':
    main()
