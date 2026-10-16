"""Shared paths and helpers for the selectivity-ceiling study."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RAW = REPO / 'data' / 'raw'
OWN_RAW = PROJECT / 'data' / 'raw'
PROCESSED = PROJECT / 'data' / 'processed'
RESULTS = PROJECT / 'results'
FIGURES = PROJECT / 'figures'

DISCOVERY_SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
LAB = {'PRISM': 'Broad', 'CTRP': 'Broad', 'GDSC1': 'Sanger', 'GDSC2': 'Sanger',
       'gCSI': 'Genentech', 'NCI60': 'NCI'}

SALT_WORDS = ('hydrochloride', 'dihydrochloride', 'trihydrochloride', 'hcl', 'mesylate', 'dimesylate',
              'maleate', 'sodium', 'disodium', 'potassium', 'tosylate', 'ditosylate', 'citrate', 'phosphate',
              'diphosphate', 'sulfate', 'bisulfate', 'acetate', 'fumarate', 'hemifumarate', 'tartrate', 'bitartrate',
              'succinate', 'malate', 'lactate', 'besylate', 'bromide', 'hydrobromide', 'chloride', 'iodide',
              'monohydrate', 'dihydrate', 'trihydrate', 'hydrate', 'salt', 'calcium', 'magnesium', 'free base',
              'freebase', 'esylate', 'isethionate', 'napsylate', 'oxalate', 'nitrate', 'trifluoroacetate', 'tfa')


def norm_name(name):
    """Lower-case alphanumeric drug name with common salt/hydrate words removed."""
    if not isinstance(name, str):
        return ''
    s = name.lower()
    s = re.sub(r'\(.*?\)', ' ', s)
    for w in sorted(SALT_WORDS, key=len, reverse=True):
        s = re.sub(r'\b' + re.escape(w) + r'\b', ' ', s)
    return re.sub(r'[^a-z0-9]', '', s)


def norm_cell(name):
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def inchikey25(smiles):
    """Connectivity and stereo blocks of the InChIKey of the largest neutralised fragment."""
    return _inchikey(smiles, 25)


def inchikey14(smiles):
    """Connectivity block of the InChIKey of the largest neutralised fragment (None if unparsable)."""
    return _inchikey(smiles, 14)


def _inchikey(smiles, n):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    smi = smiles.split(',')[0].strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key[:n] if key else None


def rank_normal(x):
    """Rank-based inverse normal transform of a 1-D array (NaN preserved, ties averaged)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    n = ok.sum()
    if n:
        out[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / n)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {('|'.join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path, obj):
    obj = _jsonable(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')


def load_models():
    m = pd.read_csv(RAW / 'depmap24q4_metadata' / 'Model.csv')
    return m
