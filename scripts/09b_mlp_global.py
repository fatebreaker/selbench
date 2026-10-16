"""Global two-tower neural drug-response model (GPU) under the same split designs as 09_global_model.py.

Three stages, because the GPU environment has only numpy and torch:
  export  (.venv):      drug/cell feature matrices, pair tables and fold assignments -> results/global_mlp/inputs
  train   (.venv-gpu):  per screen and split, 5-fold out-of-fold predictions on one GPU
  score   (.venv):      same metrics as 09 (global r, per-drug raw and selective correlations, cross-screen)
Architecture: drug MLP (356->256->128) and cell MLP (186->256->128), concatenated, then 256->128->1;
ReLU, dropout 0.1, Adam (lr 2e-3, weight decay 1e-5), batch 4096, MSE on raw AUC, up to 40 epochs
with early stopping on 10% of training pairs (patience 4).
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results' / 'global_mlp'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
SEED = 13


def export():
    import pandas as pd
    import importlib.util
    spec = importlib.util.spec_from_file_location('gm', Path(__file__).with_name('09_global_model.py'))
    gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
    from common import PROCESSED, RESULTS
    inp = OUT / 'inputs'; inp.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    Z = {s: pd.read_parquet(RESULTS / 'decomposition' / f'z_{s}.parquet') for s in SCREENS}
    part = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    CF = gm.cell_features(sorted(set(long.depmap_id)))
    DF = gm.drug_features(sorted(set(long.compound)))
    np.save(inp / 'cell_X.npy', CF.values.astype(np.float32)); np.save(inp / 'drug_X.npy', DF.values.astype(np.float32))
    pd.Series(CF.index).to_csv(inp / 'cells.csv', index=False); pd.Series(DF.index).to_csv(inp / 'drugs.csv', index=False)
    cidx = {c: i for i, c in enumerate(CF.index)}; didx = {d: i for i, d in enumerate(DF.index)}
    rng = np.random.default_rng(SEED)
    for s in SCREENS:
        d = long[(long.screen == s) & long.compound.isin(Z[s].index) & long.depmap_id.isin(Z[s].columns)
                 & long.depmap_id.isin(CF.index) & long.depmap_id.isin(part.index)].reset_index(drop=True)
        drugs = np.array(sorted(d.compound.unique()))
        dfold = dict(zip(drugs, rng.permutation(np.arange(len(drugs)) % 5)))
        np.savez(inp / f'{s}.npz', drug=d.compound.map(didx).values, cell=d.depmap_id.map(cidx).values,
                 y=d.auc.values.astype(np.float32), random_pairs=rng.integers(0, 5, len(d)),
                 cell_blind=d.depmap_id.map(part).values, drug_blind=d.compound.map(dfold).values)
        d[['compound', 'depmap_id', 'auc']].to_csv(inp / f'{s}_pairs.csv.gz', index=False)
        print(s, len(d))


def train(screen, device, splits):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    inp = OUT / 'inputs'
    Xd = torch.tensor(np.load(inp / 'drug_X.npy'), device=device)
    Xc = np.load(inp / 'cell_X.npy'); Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
    Xc = torch.tensor(Xc.astype(np.float32), device=device)
    z = np.load(inp / f'{screen}.npz')
    drug, cell = torch.tensor(z['drug'], device=device), torch.tensor(z['cell'], device=device)
    y_np = z['y']; mu, sd = float(y_np.mean()), float(y_np.std())
    y = torch.tensor((y_np - mu) / sd, device=device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = nn.Sequential(nn.Linear(Xd.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU())
            self.c = nn.Sequential(nn.Linear(Xc.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU())
            self.h = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

        def forward(self, di, ci):
            return self.h(torch.cat([self.d(Xd[di]), self.c(Xc[ci])], 1)).squeeze(1)

    for split in splits:
        folds = z[split]; pred = np.zeros(len(y_np), np.float32)
        for f in range(5):
            tr_all = np.where(folds != f)[0]; te = np.where(folds == f)[0]
            g = np.random.default_rng(SEED + f); g.shuffle(tr_all)
            nval = len(tr_all) // 10; va, tr = tr_all[:nval], tr_all[nval:]
            net = Net().to(device); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
            best, best_state, bad = np.inf, None, 0
            tr_t = torch.tensor(tr, device=device); va_t = torch.tensor(va, device=device)
            for epoch in range(40):
                net.train()
                perm = tr_t[torch.randperm(len(tr_t), device=device)]
                for k in range(0, len(perm), 4096):
                    b = perm[k:k + 4096]
                    loss = ((net(drug[b], cell[b]) - y[b]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(((net(drug[va_t], cell[va_t]) - y[va_t]) ** 2).mean())
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k_: v.detach().clone() for k_, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 4:
                        break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                te_t = torch.tensor(te, device=device)
                pred[te] = (net(drug[te_t], cell[te_t]).cpu().numpy() * sd + mu)
            print(screen, split, f, 'epochs', epoch + 1, 'val', round(best, 4), flush=True)
        np.save(OUT / f'pred_{screen}_{split}.npy', pred)


def score():
    import pandas as pd
    import importlib.util
    spec = importlib.util.spec_from_file_location('gm', Path(__file__).with_name('09_global_model.py'))
    gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
    from common import PROCESSED, RESULTS
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    Z = {s: pd.read_parquet(RESULTS / 'decomposition' / f'z_{s}.parquet') for s in SCREENS}
    G = {s: gm.loo_fitness(Z[s]) for s in SCREENS}
    rows, cellblind = [], {}
    for s in SCREENS:
        pairs = pd.read_csv(OUT / 'inputs' / f'{s}_pairs.csv.gz')
        for split in ['random_pairs', 'cell_blind', 'drug_blind']:
            d = pairs.copy(); d['pred_global_mlp'] = np.load(OUT / f'pred_{s}_{split}.npy')
            rows += gm.score(d, s, Z[s], G[s], split)
            if split == 'cell_blind':
                cellblind[s] = d
    for T in SCREENS:
        for E in SCREENS:
            if T == E:
                continue
            j = long[long.screen == E][['compound', 'depmap_id', 'auc']].merge(
                cellblind[T][['compound', 'depmap_id', 'pred_global_mlp']], on=['compound', 'depmap_id'])
            if j.compound.nunique() < 10:
                continue
            for r in gm.score(j, E, Z[E], G[E], f'cross_from_{T}'):
                r['train'] = T; rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'scores.csv', index=False)
    print(res.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['export', 'train', 'score'])
    ap.add_argument('--screen'); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--splits', nargs='+', default=['random_pairs', 'cell_blind', 'drug_blind'])
    a = ap.parse_args()
    if a.stage == 'export':
        export()
    elif a.stage == 'train':
        train(a.screen, a.device, a.splits)
    else:
        score()
