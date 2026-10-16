"""Retraining in drevalpy (revision round 1; editor items 4-5, Reviewer 1 M4/M5).

This study's predictors are implemented as drevalpy models (subclasses of drevalpy.models.drp_model.DRPModel) and are
run through drevalpy 1.6.0's own pipeline functions with DrEval's features, splits and hyperparameter tuning, next to
DrEval's models, on GDSC2 (AUC_curvecurator, the uncensored endpoint; and LN_IC50_curvecurator) and on CTRPv2
(LN_IC50_curvecurator, the leaderboard setting; and AUC_curvecurator), with leave-cell-line-out (LCO) cross-validation
and drevalpy's cross-study prediction on the other DrEval datasets.

This study's models (registered in drevalpy's MODEL_FACTORY at run time):
  FitnessOnlyRidge         kernel ridge from expression to each training cell's mean rank-normal response z over
                           all drugs (the drug-agnostic axis), broadcast to every drug; placed on the response scale
                           per drug as mu_d - a_d * f_hat (a_d: training slope of the drug's sensitivity on the
                           leave-drug-out axis).
  PerDrugRidgeRaw          per-drug kernel ridge on the raw target z (rank-normal sensitivity of the training
                           cells), placed as mu_d - sd_d * z_hat.
  PerDrugRidgeSelective    per-drug kernel ridge on the selective target r (OLS residual of z on the training
                           cells' leave-drug-out axis), placed as mu_d - sd_d * r_hat.
  ...DepMap variants       the same with DepMap 24Q4 expression (top-5,000 variance genes) instead of DrEval's
                           landmark-gene expression (feature-source control); 'Dm' suffix.
  GradientBoostingZ / GradientBoostingSelective / SimpleNeuralNetworkSelective / EnsembleMFSelective
                           DrEval's own GradientBoosting / SimpleNeuralNetwork / EnsembleMF (same features,
                           hyperparameter grid, tuning and early stopping) trained on z or on the selective residual r
                           instead of the raw response, the per-pair targets computed from the training (and
                           early-stopping) cells of each split only; predictions placed as mu_d - sd_d * t_hat.
All targets are derived inside train() from the cells drevalpy passes for training, so nothing from validation or test
cells enters a target. Kernel ridge: linear kernel on features standardised on the training cells (DrEval features:
arcsinh first, as drevalpy does), regularisation chosen per drug by closed-form leave-one-out (as in
modeling.kernel_ridge_loo); the hyperparameter grid passed to drevalpy is a single configuration.

Reduced tuning (for cost; approved by the orchestrator): SimpleNeuralNetworkSelective is not re-tuned but reuses, per
split, the configuration DrEval's tuning chose for the raw-target SimpleNeuralNetwork (REUSE_TUNING); RandomForest's
grid is restricted to max_depth = 30, the value DrEval's tuning selected in all 10 leaderboard splits (GRID_FILTER).

The runner reproduces drevalpy.experiment.drug_response_experiment's per-split loop body (hpam_tune ->
train_and_predict on train+validation -> cross_study_prediction -> predictions_split_<k>.csv) so that splits can run in
parallel processes; it calls only drevalpy's own functions, writes drevalpy's output layout and seeds with drevalpy's
seed_everything(42) per task. Splits are created once per setting with drevalpy.experiment.prepare_response_splits
(10-fold LCO, seed 42). The only change to a DrEval model is that SimpleNeuralNetwork's DataLoaders load data
in-process (0 worker processes instead of 8; server process limit; the model, the batches and their order are
unchanged).

Usage:
  58_retrain_drevalpy.py prepare --setting GDSC2_AUC
  58_retrain_drevalpy.py run --setting GDSC2_AUC --model RandomForest --splits 0 1
  58_retrain_drevalpy.py launch --plan plan.txt --max-procs 6 --gpus 2 3 4 6 7   (lines: SETTING MODEL SPLIT)
  58_retrain_drevalpy.py status
Outputs: results/revision_retrain/drevalpy_runs/<setting>/<dataset>/LCO/<model>/{predictions,cross_study,best_hpams},
         results/revision_retrain/drevalpy_runs/<setting>/timings/<model>_split<k>.csv, logs/58_*.log.
For CTRPv2_LNIC50 the splits are DrEval's published leaderboard split files (data/dreval/splits.zip), so these runs are
directly comparable with the leaderboard's released predictions; the other settings use drevalpy's own split
procedure (prepare_response_splits, seed 42), which reproduces the published CTRPv2 splits exactly (57).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402
from modeling import ALPHAS  # noqa: E402

OUT = PROJECT / 'results' / 'revision_retrain'
DATA = OUT / 'drevalpy_data'
RUNS = OUT / 'drevalpy_runs'
LOGS = OUT / 'logs'
SETTINGS = {
    'GDSC2_AUC': ('GDSC2', 'AUC'),
    'GDSC2_LNIC50': ('GDSC2', 'LN_IC50'),
    'CTRPv2_LNIC50': ('CTRPv2', 'LN_IC50'),
    'CTRPv2_AUC': ('CTRPv2', 'AUC'),
    'GDSC1_AUC': ('GDSC1', 'AUC'),
}
ALL_DATASETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
N_SPLITS = 10
TEST_MODE = 'LCO'
GPU_MODELS = {'SimpleNeuralNetwork', 'SimpleNeuralNetworkSelective', 'EnsembleMF', 'EnsembleMFSelective', 'DIPK', 'SRMF'}
MIN_TRAIN_CELLS = 25
# selective-target variants whose hyperparameters are taken from the raw-target model's tuning in the same split
# instead of re-running DrEval's full grid (SimpleNeuralNetwork: 4 configurations at ~0.5 h each per split)
REUSE_TUNING = {'SimpleNeuralNetworkSelective': 'SimpleNeuralNetwork'}
# reduced grids (orchestrator-approved for cost): RandomForest keeps only max_depth = 30, the configuration DrEval's
# tuning selected in 10 of 10 splits of the CTRPv2 leaderboard (data/dreval/leaderboard_Oct26/LCO/RandomForest)
GRID_FILTER = {'RandomForest': {'max_depth': 30}}


# ----------------------------------------------------------------------------------------------------------------
# kernel ridge with closed-form LOO regularisation (identical to modeling.kernel_ridge_loo, but returns coefficients)
def kr_fit(K_tr, y):
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
    return ym, V @ (Vy / (w + a)), a


def decompose_training(drug_ids, cell_ids, response):
    """Training-cell decomposition on the sensitivity scale (-response): z, leave-drug-out axis g, residual r."""
    df = pd.DataFrame({'drug': np.asarray(drug_ids).astype(str), 'cell': np.asarray(cell_ids).astype(str),
                       'y': np.asarray(response, dtype=float)})
    df = df[np.isfinite(df.y)]
    obs = df.groupby(['drug', 'cell'], as_index=False).y.mean()
    Z, G, R, _ = selbench.decompose(obs.assign(response=-obs.y)[['drug', 'cell', 'response']])
    Y = obs.pivot(index='drug', columns='cell', values='y')
    return obs, Y, Z, G, R


def register_models():
    """Define this study's drevalpy models and add them to drevalpy's model factories (idempotent)."""
    from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
    from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY
    from drevalpy.models.baselines.sklearn_models import GradientBoosting
    from drevalpy.models.EnsembleMF import EnsembleMF
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
    from drevalpy.models.utils import load_and_select_gene_features
    from sklearn.preprocessing import StandardScaler

    if 'PerDrugRidgeRaw' in MODEL_FACTORY:
        return

    depmap_cache = {}

    def depmap_expr():
        if 'x' not in depmap_cache:
            depmap_cache['x'] = pd.read_parquet(OUT / 'inputs' / 'feat_depmap_expr.parquet').astype('float64')
        return depmap_cache['x']

    class SelectivityKernelRidge(DRPModel):
        """Base class: this study's kernel-ridge predictors as drevalpy models."""
        cell_line_views = ['gene_expression']
        drug_views = []
        early_stopping = False
        is_single_drug_model = False
        NAME = 'SelectivityKernelRidge'
        FEATURES = 'dreval'
        TARGET = 'raw'

        def __init__(self):
            super().__init__()
            self.scaler = None

        @classmethod
        def get_model_name(cls):
            return cls.NAME

        @classmethod
        def get_hyperparameter_set(cls):
            return [{'feature_source': cls.FEATURES,
                     'gene_list': 'landmark_genes' if cls.FEATURES == 'dreval' else 'depmap24q4_top5000_variance',
                     'target': cls.TARGET, 'alpha_selection': 'closed-form LOO per drug',
                     'alphas_times_mean_kernel_diag': [float(a) for a in ALPHAS], 'min_train_cells': MIN_TRAIN_CELLS}]

        def build_model(self, hyperparameters):
            self.log_hyperparameters(hyperparameters)
            self.hyperparameters = hyperparameters

        def load_cell_line_features(self, data_path, dataset_name):
            if self.FEATURES == 'dreval':
                return load_and_select_gene_features('gene_expression', 'landmark_genes', data_path, dataset_name)
            names = pd.read_csv(os.path.join(data_path, dataset_name, 'cell_line_names.csv'), dtype=str)
            dep = depmap_expr()
            feats = {}
            for cl, cid in zip(names.cell_line_name, names.cellosaurus_id):
                if cid in dep.index and cl not in feats:
                    feats[cl] = {'gene_expression': dep.loc[cid].values}
            return FeatureDataset(features=feats, meta_info={'gene_expression': dep.columns.values})

        def load_drug_features(self, data_path, dataset_name):
            return None

        def _features(self, cell_line_input, cells, fit):
            X = np.asarray(cell_line_input.get_feature_matrix(view='gene_expression', identifiers=cells), float)
            if self.FEATURES == 'dreval':
                X = np.arcsinh(X)
            if fit:
                self.scaler = StandardScaler().fit(X)
            return self.scaler.transform(X) / np.sqrt(X.shape[1])

        def train(self, output, cell_line_input, drug_input=None, output_earlystopping=None,
                  model_checkpoint_dir='checkpoints'):
            obs, Y, Z, G, R = decompose_training(output.drug_ids, output.cell_line_ids, output.response)
            cells = np.array(sorted(obs.cell.unique()))
            X = self._features(cell_line_input, cells, fit=True)
            K = X @ X.T
            self.X_train, self.cells = X, pd.Index(cells)
            self.mu = Y.mean(axis=1)
            self.sd = Y.std(axis=1, ddof=1)
            self.global_mean = float(obs.y.mean())
            self.models = {}
            if self.TARGET == 'fitness':
                f = Z.mean(axis=0)
                f = f[np.isfinite(f.values)]
                idx = self.cells.get_indexer(f.index)
                ym, coef, a = kr_fit(K[np.ix_(idx, idx)], f.values)
                self.models['__fitness__'] = (idx, ym, coef)
                slope = {}
                for d in Z.index:
                    s, g = -Y.loc[d].reindex(Z.columns).values, G.loc[d].values
                    ok = np.isfinite(s) & np.isfinite(g)
                    if ok.sum() > MIN_TRAIN_CELLS:
                        slope[d] = np.cov(s[ok], g[ok])[0, 1] / np.var(g[ok], ddof=1)
                self.slope = pd.Series(slope, dtype=float)
            else:
                M = Z if self.TARGET == 'raw' else R
                for d in M.index:
                    y = M.loc[d].reindex(cells).values
                    ok = np.isfinite(y)
                    if ok.sum() < MIN_TRAIN_CELLS:
                        continue
                    idx = np.where(ok)[0]
                    ym, coef, a = kr_fit(K[np.ix_(idx, idx)], y[ok])
                    self.models[d] = (idx, ym, coef)

        def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
            cell_line_ids, drug_ids = np.asarray(cell_line_ids).astype(str), np.asarray(drug_ids).astype(str)
            pred = np.full(len(cell_line_ids), self.global_mean)
            if len(cell_line_ids) == 0:
                return pred
            uc = np.unique(cell_line_ids)
            Kq = self._features(cell_line_input, uc, fit=False) @ self.X_train.T
            row = pd.Index(uc).get_indexer(cell_line_ids)
            if self.TARGET == 'fitness':
                idx, ym, coef = self.models['__fitness__']
                fh = (ym + Kq[:, idx] @ coef)[row]
                mu = pd.Series(drug_ids).map(self.mu).fillna(self.global_mean).values
                a = pd.Series(drug_ids).map(self.slope).fillna(float(self.slope.mean())).values
                return mu - a * fh
            for d in np.unique(drug_ids):
                rows = np.where(drug_ids == d)[0]
                if d in self.models:
                    idx, ym, coef = self.models[d]
                    pred[rows] = self.mu[d] - self.sd[d] * (ym + Kq[np.ix_(row[rows], idx)] @ coef)
                elif d in self.mu.index:
                    pred[rows] = self.mu[d]
            return pred

    def make(name, features, target):
        return type(name, (SelectivityKernelRidge,), {'NAME': name, 'FEATURES': features, 'TARGET': target})

    ours = [make('FitnessOnlyRidge', 'dreval', 'fitness'), make('PerDrugRidgeRaw', 'dreval', 'raw'),
            make('PerDrugRidgeSelective', 'dreval', 'selective'), make('FitnessOnlyRidgeDm', 'depmap', 'fitness'),
            make('PerDrugRidgeRawDm', 'depmap', 'raw'), make('PerDrugRidgeSelectiveDm', 'depmap', 'selective')]

    class TransformedTargetMixin:
        """Train a DrEval global model on z or on the selective residual r of the training cells."""
        TARGET = 'selective'
        BASE = None

        @classmethod
        def get_model_name(cls):
            return cls.NAME

        @classmethod
        def get_hyperparameter_set(cls):
            return cls.BASE.get_hyperparameter_set()      # DrEval's own grid and tuning

        def _targets(self, ds, Z, R):
            M = Z if self.TARGET == 'z' else R
            t = M.stack()
            key = pd.MultiIndex.from_arrays([ds.drug_ids.astype(str), ds.cell_line_ids.astype(str)])
            v = t.reindex(key).values
            ok = np.isfinite(v)
            return DrugResponseDataset(response=v[ok], cell_line_ids=ds.cell_line_ids[ok], drug_ids=ds.drug_ids[ok],
                                       tissues=ds.tissue[ok] if ds.tissue is not None else None,
                                       dataset_name=ds.dataset_name)

        def train(self, output, cell_line_input, drug_input=None, output_earlystopping=None,
                  model_checkpoint_dir='checkpoints'):
            parts = [output] + ([output_earlystopping] if output_earlystopping is not None and len(output_earlystopping)
                                else [])
            obs, Y, Z, G, R = decompose_training(np.concatenate([p.drug_ids for p in parts]),
                                                 np.concatenate([p.cell_line_ids for p in parts]),
                                                 np.concatenate([p.response for p in parts]))
            Ytr = decompose_training(output.drug_ids, output.cell_line_ids, output.response)[1]
            self.mu, self.sd = Ytr.mean(axis=1), Ytr.std(axis=1, ddof=1)
            self.global_mean = float(np.nanmean(output.response))
            es = self._targets(output_earlystopping, Z, R) if len(parts) > 1 else output_earlystopping
            super().train(self._targets(output, Z, R), cell_line_input, drug_input, es, model_checkpoint_dir)

        def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
            t = np.asarray(super().predict(cell_line_ids, drug_ids, cell_line_input, drug_input), float)
            d = pd.Series(np.asarray(drug_ids).astype(str))
            mu, sd = d.map(self.mu).values, d.map(self.sd).values
            return np.where(np.isfinite(mu) & np.isfinite(sd), mu - sd * t, self.global_mean)

    def make_tt(name, base, target):
        return type(name, (TransformedTargetMixin, base), {'NAME': name, 'BASE': base, 'TARGET': target})

    ours += [make_tt('GradientBoostingZ', GradientBoosting, 'z'),
             make_tt('GradientBoostingSelective', GradientBoosting, 'selective'),
             make_tt('SimpleNeuralNetworkSelective', SimpleNeuralNetwork, 'selective'),
             make_tt('EnsembleMFSelective', EnsembleMF, 'selective')]
    for cls in ours:
        MODEL_FACTORY[cls.NAME] = cls
        MULTI_DRUG_MODEL_FACTORY[cls.NAME] = cls


def patch_snn_workers(n=0):
    """drevalpy's SimpleNeuralNetwork builds its DataLoaders with 8 worker processes; load data in-process instead.

    Only the loading mechanism changes (the shuffling sampler runs in the main process either way), so the model,
    the batches and their order are unchanged; this keeps every job a single process (server process limit).
    """
    from drevalpy.models.SimpleNeuralNetwork import utils as snn_utils
    if getattr(snn_utils.DataLoader, '_patched', False):
        return
    orig = snn_utils.DataLoader

    def loader(*a, **kw):
        kw['num_workers'] = n
        if n == 0:
            kw['persistent_workers'] = False
        return orig(*a, **kw)
    loader._patched = True
    snn_utils.DataLoader = loader


# ----------------------------------------------------------------------------------------------------------------
def load_setting(setting, with_cross_study=True):
    from drevalpy.datasets.loader import load_dataset
    ds, measure = SETTINGS[setting]
    rd = load_dataset(dataset_name=ds, path_data=str(DATA), measure=measure, curve_curator=True)
    cs = []
    if with_cross_study:
        cs = [load_dataset(dataset_name=t, path_data=str(DATA), measure=measure, curve_curator=True)
              for t in ALL_DATASETS if t != ds]
    return rd, cs


def result_path(setting):
    ds, _ = SETTINGS[setting]
    return RUNS / setting / ds / TEST_MODE


def prepare(setting):
    """Create (or, for the leaderboard setting, copy DrEval's published) 10-fold LCO splits."""
    import zipfile
    from drevalpy.experiment import prepare_response_splits, seed_everything
    seed_everything(42)
    rd, _ = load_setting(setting, with_cross_study=False)
    rp = result_path(setting)
    if setting == 'CTRPv2_LNIC50' and not (rp / 'splits').exists():
        # the leaderboard's own split files (train/validation/validation_es/early_stopping/test per split)
        (rp / 'splits').mkdir(parents=True)
        with zipfile.ZipFile(PROJECT / 'data' / 'dreval' / 'splits.zip') as z:
            for n in z.namelist():
                if n.startswith('splits/cv_split_') and n.endswith('.csv'):
                    (rp / 'splits' / Path(n).name).write_bytes(z.read(n))
    n = prepare_response_splits(rd, split_path=str(rp / 'splits'), result_path=str(rp), split_label=TEST_MODE,
                                test_mode=TEST_MODE, n_cv_splits=N_SPLITS, overwrite=False,
                                result_folder_exists=rp.exists())
    rd.load_splits(path=str(rp / 'splits'))
    test_cells = [set(s['test'].cell_line_ids) for s in rd.cv_splits]
    assert sum(len(c) for c in test_cells) == len(set.union(*test_cells)), 'test folds overlap'
    print(setting, 'splits:', n, 'rows:', len(rd), 'test cells per split:', [len(c) for c in test_cells], flush=True)


def run(setting, model_name, splits, metric='RMSE'):
    """drevalpy.experiment.drug_response_experiment's loop body for one model and the given splits."""
    from drevalpy.experiment import (cross_study_prediction, generate_data_saving_path, get_datasets_from_cv_split,
                                     hpam_tune, seed_everything, train_and_predict)
    from drevalpy.models import MODEL_FACTORY
    register_models()
    patch_snn_workers(0)
    rd, cross = load_setting(setting)
    rp = result_path(setting)
    assert (rp / 'splits').exists(), 'run prepare first'
    rd.load_splits(path=str(rp / 'splits'))
    model_class = MODEL_FACTORY[model_name]
    pred_dir = Path(generate_data_saving_path(model_name=model_name, drug_id=None, result_path=str(rp),
                                              suffix='predictions'))
    hp_dir = Path(generate_data_saving_path(model_name=model_name, drug_id=None, result_path=str(rp),
                                            suffix='best_hpams'))
    hpam_set = model_class.get_hyperparameter_set()
    if model_name in GRID_FILTER:
        hpam_set = [h for h in hpam_set if all(h.get(a) == v for a, v in GRID_FILTER[model_name].items())]
        assert hpam_set, f'grid filter leaves no configuration for {model_name}'
    timing = []
    for k in splits:
        pf = pred_dir / f'predictions_split_{k}.csv'
        if pf.exists():
            print(f'{model_name} split {k} exists, skipping', flush=True)
            continue
        t0 = time.time()
        hpam_set_k = hpam_set
        if model_name in REUSE_TUNING:
            # reduced tuning grid (orchestrator-approved for cost): reuse the configuration that DrEval's tuning
            # selected for the raw-target model in the same split; wait until that model has finished tuning
            src = result_path(setting) / REUSE_TUNING[model_name] / 'best_hpams' / f'best_hpams_split_{k}.json'
            while not src.exists():
                time.sleep(60)
            hpam_set_k = [json.loads(src.read_text())]
            t0 = time.time()
        seed_everything(42)
        train_ds, val_ds, es_ds, test_ds = get_datasets_from_cv_split(rd.cv_splits[k], model_class, model_name, None)
        model = model_class()
        best = hpam_tune(model=model, train_dataset=train_ds, validation_dataset=val_ds, early_stopping_dataset=es_ds,
                         hpam_set=hpam_set_k, response_transformation=None, metric=metric, path_data=str(DATA),
                         model_checkpoint_dir='TEMPORARY')
        with open(hp_dir / f'best_hpams_split_{k}.json', 'w', encoding='utf-8') as f:
            json.dump(best, f)
        t1 = time.time()
        train_ds.add_rows(val_ds)
        train_ds.shuffle(random_state=42)
        test_ds = train_and_predict(model=model, hpams=best, path_data=str(DATA), train_dataset=train_ds,
                                    prediction_dataset=test_ds,
                                    early_stopping_dataset=es_ds if model.early_stopping else None,
                                    response_transformation=None, model_checkpoint_dir='TEMPORARY')
        t2 = time.time()
        for cs in cross:
            cs.remove_nan_responses()
            cross_study_prediction(dataset=cs, model=model, test_mode=TEST_MODE, train_dataset=train_ds,
                                   path_data=str(DATA), early_stopping_dataset=es_ds if model.early_stopping else None,
                                   response_transformation=None, path_out=str(pred_dir.parent), split_index=k,
                                   single_drug_id=None)
        test_ds.to_csv(str(pf))
        t3 = time.time()
        timing.append({'setting': setting, 'model': model_name, 'split': k, 'tuning_s': t1 - t0, 'final_fit_s': t2 - t1,
                       'cross_study_s': t3 - t2, 'total_s': t3 - t0, 'n_hpam_configs': len(hpam_set_k),
                       'host_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '')})
        print(f'{model_name} split {k} done in {t3 - t0:.0f}s (tuning {t1 - t0:.0f}s)', flush=True)
        tf = RUNS / setting / 'timings' / f'{model_name}_split{k}.csv'
        tf.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(timing[-1:]).to_csv(tf, index=False)


def running_tasks():
    """(setting, model, split) of every '58_retrain_drevalpy.py run' process on this machine (any launcher)."""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        args = [x.decode(errors='ignore') for x in args if x]
        if len(args) > 2 and args[1].endswith('58_retrain_drevalpy.py') and args[2] == 'run':
            try:
                out.append((args[args.index('--setting') + 1], args[args.index('--model') + 1],
                            args[args.index('--splits') + 1]))
            except (ValueError, IndexError):
                out.append(('?', '?', '?'))
    return out


def launch(plan, max_procs, gpus, cores, pin=False):
    """Run the (setting, model, split) tasks of a plan file as separate single-threaded processes, in file order.

    max_procs caps the number of running '58_retrain_drevalpy.py run' processes that belong to this plan (found by
    scanning /proc, so a launcher can be replaced without killing its running tasks, and launchers with disjoint plans
    and caps adding up to 6 respect the machine limit); a task already running elsewhere is retried later and dropped
    once its predictions exist. GPU models get the least-loaded GPU.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    tasks = [tuple(line.split()) for line in Path(plan).read_text().splitlines() if line.strip()]
    plan_set = set(tasks)
    gpu_load = {g: 0 for g in gpus}
    running = []
    slot_ids = list(cores) if pin else list(range(max_procs))
    while tasks or running:
        for item in list(running):
            p, setting, m, k, slot, gpu, fh = item
            if p.poll() is not None:
                running.remove(item)
                fh.close()
                if gpu is not None:
                    gpu_load[gpu] -= 1
                print(time.strftime('%H:%M:%S'), f'finished {setting} {m} split {k} rc={p.returncode}', flush=True)
        busy = running_tasks()
        n_busy = len([t for t in busy if t in plan_set])      # cap applies to this plan's tasks
        deferred = []
        while tasks and n_busy < max_procs:
            setting, m, k = tasks.pop(0)
            if (result_path(setting) / m / 'predictions' / f'predictions_split_{k}.csv').exists():
                continue
            if (setting, m, str(k)) in busy:
                deferred.append((setting, m, k))
                continue
            used = {it[4] for it in running}
            slot = next((c for c in slot_ids if c not in used), slot_ids[0])
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       LOKY_MAX_CPU_COUNT='1', PYTHONHASHSEED='42', PYTHONDONTWRITEBYTECODE='1')
            gpu = None
            if m in GPU_MODELS and gpus:
                gpu = min(gpu_load, key=gpu_load.get)
                gpu_load[gpu] += 1
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
            fh = open(LOGS / f'58_{setting}_{m}_split{k}.log', 'a')
            cmd = [sys.executable, str(Path(__file__).resolve()), 'run', '--setting', setting, '--model', m,
                   '--splits', str(k)]
            if pin:
                cmd = ['taskset', '-c', str(slot)] + cmd
            p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p, setting, m, k, slot, gpu, fh))
            n_busy += 1
            print(time.strftime('%H:%M:%S'), f'started {setting} {m} split {k} gpu {gpu} (busy {n_busy})', flush=True)
        tasks = tasks + deferred          # tasks running elsewhere are re-checked later
        if deferred and not running:
            time.sleep(50)
        time.sleep(10)


def status():
    rows = []
    for setting in SETTINGS:
        rp = result_path(setting)
        if not rp.exists():
            continue
        for md in sorted(p for p in rp.iterdir() if p.is_dir() and p.name != 'splits'):
            n = len(list((md / 'predictions').glob('predictions_split_*.csv')))
            rows.append({'setting': setting, 'model': md.name, 'splits_done': n})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'run', 'launch', 'status'])
    ap.add_argument('--setting', choices=list(SETTINGS))
    ap.add_argument('--model')
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--splits', nargs='*', type=int, default=list(range(N_SPLITS)))
    ap.add_argument('--max-procs', type=int, default=6)
    ap.add_argument('--gpus', nargs='*', type=int, default=[2, 3, 4, 6, 7])
    ap.add_argument('--cores', nargs='*', type=int, default=[40, 41, 42, 43, 44, 45])
    ap.add_argument('--pin', action='store_true')
    ap.add_argument('--plan', help='task file: one "SETTING MODEL SPLIT" per line')
    args = ap.parse_args()
    if args.cmd == 'prepare':
        prepare(args.setting)
    elif args.cmd == 'run':
        run(args.setting, args.model, args.splits)
    elif args.cmd == 'launch':
        launch(args.plan, args.max_procs, args.gpus, args.cores, args.pin)
    else:
        status()


if __name__ == '__main__':
    main()
