"""Bit-for-bit regeneration of the frozen predictions from the hashed pre-retrieval inputs (revision; Reviewer 3, M1
request 3).

For each frozen prediction script this harness

1. checks that the script, and the sibling modules it imports, still have the SHA-256 recorded in
   protocol/freeze_manifest_amendment2.json (the frozen scripts themselves are never edited);
2. records the SHA-256 and modification time of every input file the script reads, and whether the input is
   covered by a freeze manifest;
3. writes a COPY of the script to results/revision_repro/regenerated/_patched_scripts/ in which only the output
   path is changed (plus, for script 22, the hard-coded worker count 32 -> 6, and, for script 30, the directory from
   which the hashed sibling module 06_load_confirmation.py is loaded, which would otherwise resolve to the copy's
   directory). Every patch must match exactly once; the unified diff is saved next to the copy;
4. runs the copy with the analysis interpreter (.venv, Python 3.8; OPENBLAS/OMP/MKL threads = 1,
   PYTHONDONTWRITEBYTECODE=1, cwd = scripts/), writing only to results/revision_repro/regenerated/<target>/.
   If strace is available, the run is traced (open/openat, following forked workers) and the files opened under
   the project are summarised, which shows that no drug-response (outcome) file was opened;
5. compares every regenerated file with the frozen file: SHA-256 equality and numeric equality (identical
   index/columns/dtypes/NaN pattern; maximum absolute difference; number of differing cells).

Targets: pdx (30_pdx_transfer_predictions.py -> results/pdx_transfer), beataml (22_beataml_transfer_predictions.py
-> results/beataml_transfer), consensus (05_consensus_predictions.py -> results/frozen_predictions) and features
(03_cell_features.py -> data/processed/features: the DepMap feature blocks that the three prediction scripts read and
that no manifest covers; regenerated from the public DepMap 24Q4 files to close the provenance chain),
decomposition (02_decompose.py -> results/decomposition, hashed; from the hashed data/processed/discovery_long.csv.gz)
and harmonize (01_harmonize_discovery.py -> the hashed data/processed tables, from the public discovery screens; its
output directory is rebound by one line inserted after the imports). harmonize_tiebreak is a DIAGNOSTIC copy of 01
that also fixes 7 hash-seed-dependent InChIKey tie-breaks (see TARGETS); it is not an output-path-only patch.

Usage (from scripts/; the default runs every target, upstream first, in about 15 minutes):
    ../../.venv/bin/python 61_regenerate_frozen.py
    ../../.venv/bin/python 61_regenerate_frozen.py --targets pdx beataml consensus
    ../../.venv/bin/python 61_regenerate_frozen.py --targets features decomposition harmonize harmonize_tiebreak
    ../../.venv/bin/python 61_regenerate_frozen.py --compare-only --targets pdx beataml consensus features
Outputs: results/revision_repro/regenerated/{pdx_transfer,beataml_transfer,frozen_predictions,features,decomposition,
processed,processed_tiebreak}/,
_patched_scripts/<target>/ (copies and diffs), _provenance/ (stdout, compressed strace logs), comparison.csv,
comparison.json.
"""
import argparse
import difflib
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
RESULTS = PROJECT / 'results'
REGEN = RESULTS / 'revision_repro' / 'regenerated'
PATCHED = REGEN / '_patched_scripts'
PROV = REGEN / '_provenance'
MANIFESTS = [PROJECT / 'protocol' / f for f in
             ['freeze_manifest.json', 'freeze_manifest_amendment1.json', 'freeze_manifest_amendment2.json']]
ANALYSIS_PY = PROJECT.parent / '.venv' / 'bin' / 'python'
MAX_WORKERS = 6

TARGETS = {
    'pdx': {
        'script': '30_pdx_transfer_predictions.py',
        'frozen_dir': RESULTS / 'pdx_transfer',
        'out_dir': REGEN / 'pdx_transfer',
        'args': [],
        'siblings': ['common.py', 'modeling.py', '06_load_confirmation.py'],
        'patches': [
            ("HERE = Path(__file__).resolve().parent",
             f"HERE = Path({str(SCRIPTS)!r})"),
            ("OUT = RESULTS / 'pdx_transfer'",
             "OUT = RESULTS / 'revision_repro' / 'regenerated' / 'pdx_transfer'"),
        ],
    },
    'beataml': {
        'script': '22_beataml_transfer_predictions.py',
        'frozen_dir': RESULTS / 'beataml_transfer',
        'out_dir': REGEN / 'beataml_transfer',
        'args': [],
        'siblings': ['common.py', 'modeling.py'],
        'patches': [
            ("OUT = RESULTS / 'beataml_transfer'",
             "OUT = RESULTS / 'revision_repro' / 'regenerated' / 'beataml_transfer'"),
            ("with ProcessPoolExecutor(32, initializer=init, initargs=(payload,)) as ex:",
             f"with ProcessPoolExecutor({MAX_WORKERS}, initializer=init, initargs=(payload,)) as ex:"),
        ],
    },
    # upstream, unhashed inputs of the three scripts above: DepMap 24Q4 feature blocks, regenerated from the public
    # DepMap files by the hashed script 03 (not a frozen prediction; closes the provenance chain)
    'features': {
        'script': '03_cell_features.py',
        'frozen_dir': PROJECT / 'data' / 'processed' / 'features',
        'out_dir': REGEN / 'features',
        'args': [],
        'siblings': ['common.py'],
        'patches': [
            ("out = PROCESSED / 'features'",
             "out = PROCESSED.parents[1] / 'results' / 'revision_repro' / 'regenerated' / 'features'"),
        ],
    },
    # hashed processed inputs (discovery_long.csv.gz, compounds.csv, compound_merges.csv, cells.csv), from the public
    # discovery screens opened before the study (PRISM, CTRP, GDSC1/2) and DepMap metadata
    'harmonize': {
        'script': '01_harmonize_discovery.py',
        'frozen_dir': PROJECT / 'data' / 'processed',
        'out_dir': REGEN / 'processed',
        'files': ['compound_merges.csv', 'discovery_long.csv.gz', 'compounds.csv', 'cells.csv',
                  'gdsc_compound_mapping.csv', 'harmonize_summary.json'],
        'args': [],
        'siblings': ['common.py'],
        'patches': [
            ("                    sha256, write_json)\n",
             "                    sha256, write_json)\n"
             "PROCESSED = PROCESSED.parents[1] / 'results' / 'revision_repro' / 'regenerated' / 'processed'\n"),
        ],
    },
    # DIAGNOSTIC (not an output-path-only patch). Script 01 is not deterministic: the structure-merge step picks a
    # compound's InChIKey with max(set(keys), key=keys.count), so ties between equally frequent keys (33 compounds) are
    # broken by set iteration order, which depends on Python's per-process string-hash seed. Only 7 of the ties can
    # change a merge. This copy fixes those 7 to the keys implied by the frozen compound_merges.csv (each compound's
    # merge partner in the frozen file shares exactly one of its candidate keys); everything else is unchanged.
    'harmonize_tiebreak': {
        'script': '01_harmonize_discovery.py',
        'frozen_dir': PROJECT / 'data' / 'processed',
        'out_dir': REGEN / 'processed_tiebreak',
        'files': ['compound_merges.csv', 'discovery_long.csv.gz', 'compounds.csv', 'cells.csv',
                  'gdsc_compound_mapping.csv', 'harmonize_summary.json'],
        'args': [],
        'siblings': ['common.py'],
        'patches': [
            ("                    sha256, write_json)\n",
             "                    sha256, write_json)\n"
             "PROCESSED = PROCESSED.parents[1] / 'results' / 'revision_repro' / 'regenerated' / 'processed_tiebreak'\n"),
            ("    merge = {}\n    for key, group in pd.Series(ik).groupby(lambda d: ik[d]):",
             "    ik.update(TIEBREAK)  # diagnostic: tie-breaks implied by the frozen compound_merges.csv\n"
             "    merge = {}\n    for key, group in pd.Series(ik).groupby(lambda d: ik[d]):"),
            ("def split_ids(s):",
             "TIEBREAK = " + repr({
                 'DPC-002021': 'UHDGCWIWMRVCDJ-CCXZUQQUSA', 'DPC-000903': 'QYZOGCMHVIGURT-UHFFFAOYSA',
                 'DPC-000474': 'BIIVYFLTOXDAOV-SJKOYZFVSA', 'DPC-004226': 'NWIBSHFKIJFRCO-WUDYKRTCSA',
                 'DPC-000898': 'WRLVHADVOGFZOZ-UHFFFAOYSA', 'DPC-001793': 'DQLATGHUWYMOKM-UHFFFAOYSA',
                 'DPC-001134': 'OYVAGSVQBOHSSS-UAPAGMARSA'}) + "\n\n\ndef split_ids(s):"),
        ],
    },
    # hashed upstream product read by script 05: the discovery decomposition, from the hashed discovery_long.csv.gz
    'decomposition': {
        'script': '02_decompose.py',
        'frozen_dir': RESULTS / 'decomposition',
        'out_dir': REGEN / 'decomposition',
        'args': [],
        'siblings': ['common.py'],
        'patches': [
            ("out = RESULTS / args.out",
             "out = RESULTS / 'revision_repro' / 'regenerated' / args.out"),
        ],
    },
    'consensus': {
        'script': '05_consensus_predictions.py',
        'frozen_dir': RESULTS / 'frozen_predictions',
        'out_dir': REGEN / 'frozen_predictions',
        # the original pipeline call (run_discovery_pipeline.sh) with 30 -> 6 workers
        'args': ['--out', 'frozen_predictions', '--workers', str(MAX_WORKERS)],
        'siblings': ['common.py', 'modeling.py'],
        'patches': [
            ("out = RESULTS / args.out",
             "out = RESULTS / 'revision_repro' / 'regenerated' / args.out"),
        ],
    },
}
# directories that hold outcome (drug-response) data that the frozen predictions must not have used
OUTCOME_DIRS = ['data/raw/pdx_bruna_responses', 'data/raw/beataml_responses', 'data/raw/confirmation',
                'data/raw/pharmacodb_responses', 'data/processed/confirmation', 'data/processed/pdx_bruna',
                'data/processed/beataml']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes():
    """path (relative to the project) -> {manifest file name: sha256}."""
    out = {}
    for mf in MANIFESTS:
        m = json.loads(mf.read_text())
        for files in m['files'].values():
            for p, h in files.items():
                out.setdefault(p, {})[mf.name] = h
    return out


def utc_mtime(path):
    return datetime.fromtimestamp(os.stat(path).st_mtime, timezone.utc).isoformat(timespec='seconds')


def check_scripts(spec, hashes):
    rows = []
    for name in [spec['script']] + spec['siblings']:
        rel = f'scripts/{name}'
        h = sha256(SCRIPTS / name)
        rec = hashes.get(rel, {})
        rows.append({'file': rel, 'sha256': h, 'manifests': sorted(rec),
                     'matches_all_manifests': bool(rec) and all(v == h for v in rec.values())})
        if not rows[-1]['matches_all_manifests']:
            raise SystemExit(f'{rel} does not match its freeze-manifest hash; refusing to regenerate from it')
    return rows


def write_patched_copy(name, spec):
    src = (SCRIPTS / spec['script']).read_text()
    new = src
    for old, rep in spec['patches']:
        n = new.count(old)
        if n != 1:
            raise SystemExit(f"patch target {old!r} found {n} times in {spec['script']}")
        new = new.replace(old, rep)
    (PATCHED / name).mkdir(parents=True, exist_ok=True)
    dst = PATCHED / name / spec['script']
    dst.write_text(new)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        fromfile=f"scripts/{spec['script']} (frozen)",
                                        tofile=f"_patched_scripts/{name}/{spec['script']} (copy)"))
    (PATCHED / name / (spec['script'] + '.diff')).write_text(diff)
    return dst, diff


def run_copy(name, spec, use_strace):
    out_dir = spec['out_dir']
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dst, _ = write_patched_copy(name, spec)
    env = dict(os.environ)
    # PYTHONPYCACHEPREFIX points at an empty, never-written directory, so that the hashed .py sources (not cached
    # bytecode in scripts/__pycache__) are compiled, and the trace shows them being read
    env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': str(REGEN / '_no_pycache'),
                'PYTHONPATH': str(SCRIPTS)})
    cmd = [str(ANALYSIS_PY), str(dst)] + spec['args']
    PROV.mkdir(parents=True, exist_ok=True)
    trace = PROV / f'{name}.strace'
    for old in PROV.glob(f'{name}.strace*'):
        old.unlink()
    if use_strace:      # one trace file per process (-ff), so that worker syscalls are not split across lines
        cmd = ['strace', '-ff', '--seccomp-bpf', '-qq', '-e', 'trace=open,openat', '-e', 'signal=none',
               '-o', str(trace)] + cmd
    t0 = time.time()
    log = PROV / f'{name}.stdout.txt'
    with open(log, 'w') as fh:
        rc = subprocess.call(cmd, cwd=str(SCRIPTS), env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    info = {'target': name, 'command': ' '.join(cmd), 'returncode': rc, 'seconds': round(dt, 1),
            'started_utc': datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec='seconds')}
    parts = sorted(PROV.glob(f'{name}.strace.*'))
    if use_strace and parts:
        info['opened'] = summarise_trace(parts)
        info['traced_processes'] = len(parts)
        with gzip.open(str(trace) + '.txt.gz', 'wt') as g:
            for part in parts:
                g.write(f'### {part.name}\n')
                g.write(part.read_text(errors='replace'))
                part.unlink()
    if rc != 0:
        raise SystemExit(f'{name}: regeneration failed (see {log})')
    return info


_OPEN = re.compile(r'open(?:at)?\((?:AT_FDCWD, |\d+, )?"([^"]+)", ([A-Z_|]+)[^)]*\)\s+=\s+(-?\d+)')


def summarise_trace(parts):
    """Files under the project (or the parent repository's data) opened successfully, split into read and written."""
    read, written = set(), set()
    roots = [str(PROJECT), str(PROJECT.parent / 'data')]
    lines = (line for part in parts for line in open(part, errors='replace'))
    for line in lines:
        m = _OPEN.search(line)
        if not m or int(m.group(3)) < 0:
            continue
        path, flags = os.path.normpath(os.path.join(str(SCRIPTS), m.group(1))), m.group(2)
        if not any(path.startswith(r) for r in roots) or os.path.isdir(path):
            continue
        rel = os.path.relpath(path, str(PROJECT))
        if '__pycache__' in rel:
            continue
        (written if ('O_WRONLY' in flags or 'O_RDWR' in flags or 'O_CREAT' in flags) else read).add(rel)
    outcome = sorted(p for p in read | written if any(p.startswith(d) for d in OUTCOME_DIRS))
    outside = sorted(p for p in written if not p.startswith('results/revision_repro/regenerated/'))
    return {'read': sorted(read), 'written': sorted(written), 'outcome_files_opened': outcome,
            'written_outside_regenerated': outside}


def input_records(opened, hashes):
    rows = []
    for rel in opened.get('read', []):
        if rel.startswith('results/revision_repro/'):
            continue
        p = PROJECT / rel
        if not p.is_file():
            continue
        h = sha256(p)
        rec = hashes.get(rel, {})
        rows.append({'input': rel, 'sha256': h, 'mtime_utc': utc_mtime(p), 'in_manifests': sorted(rec),
                     'matches_manifest': (all(v == h for v in rec.values()) if rec else None)})
    return rows


def _num(x):
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None


def compare_frames(a, b):
    res = {'shape_frozen': list(a.shape), 'shape_regenerated': list(b.shape),
           'index_equal': bool(a.index.equals(b.index)), 'columns_equal': bool(a.columns.equals(b.columns)),
           'dtypes_equal': bool(a.shape[1] == b.shape[1] and (a.dtypes.values == b.dtypes.values).all())}
    if not (res['index_equal'] and res['columns_equal']):
        return {**res, 'values_identical': False, 'max_abs_diff': None, 'n_diff': None}
    num_a, num_b, n_diff, mad, nan_ok = [], [], 0, 0.0, True
    for c in a.columns:
        x, y = a[c], b[c]
        xa, ya = _num(x), _num(y)
        if xa is None or ya is None or x.dtype == object:
            same = x.astype(str).values == y.astype(str).values
            n_diff += int((~same).sum())
            continue
        na, nb = np.isnan(xa), np.isnan(ya)
        nan_ok &= bool((na == nb).all())
        both = ~na & ~nb
        d = np.abs(xa[both] - ya[both])
        if d.size:
            mad = max(mad, float(d.max()))
        n_diff += int((xa[both] != ya[both]).sum() + (na != nb).sum())
        num_a.append(xa); num_b.append(ya)
    return {**res, 'nan_pattern_equal': nan_ok, 'values_identical': n_diff == 0, 'max_abs_diff': mad,
            'n_diff': n_diff, 'n_values': int(a.size)}


def compare_json(a, b, path=''):
    """List of (path, frozen, regenerated) for differing leaves; numeric leaves compared exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += compare_json(a.get(k), b.get(k), f'{path}/{k}')
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum((compare_json(x, y, f'{path}[{i}]') for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [(path, a, b)]


def compare_file(frozen, regen):
    row = {'file': str(frozen.relative_to(PROJECT)), 'regenerated': str(regen.relative_to(PROJECT)),
           'sha256_frozen': sha256(frozen) if frozen.exists() else None,
           'sha256_regenerated': sha256(regen) if regen.exists() else None}
    row['sha256_equal'] = row['sha256_frozen'] is not None and row['sha256_frozen'] == row['sha256_regenerated']
    if frozen.suffix == '.gz' and frozen.exists() and regen.exists():
        # gzip headers carry the write time, so compressed bytes can differ while the content is identical
        dec = [hashlib.sha256(gzip.open(str(f), 'rb').read()).hexdigest() for f in (frozen, regen)]
        row['sha256_decompressed_equal'] = dec[0] == dec[1]
    if not (frozen.exists() and regen.exists()):
        row.update({'values_identical': False, 'note': 'missing file'})
        return row
    suf = ''.join(frozen.suffixes[-2:]) if frozen.name.endswith('.csv.gz') else frozen.suffix
    if suf == '.parquet':
        row.update(compare_frames(pd.read_parquet(frozen), pd.read_parquet(regen)))
    elif suf in ('.csv', '.csv.gz'):
        row.update(compare_frames(pd.read_csv(frozen), pd.read_csv(regen)))
    elif suf == '.json':
        diffs = compare_json(json.loads(frozen.read_text()), json.loads(regen.read_text()))
        row.update({'values_identical': not diffs, 'n_diff': len(diffs), 'max_abs_diff': 0.0 if not diffs else None,
                    'note': '; '.join(f'{p}: {x} vs {y}' for p, x, y in diffs[:5])})
    else:
        row.update({'values_identical': row['sha256_equal'], 'note': 'byte comparison only'})
    return row


def environment():
    import pyarrow
    import scipy
    out = {'python': sys.version.split()[0], 'executable': str(ANALYSIS_PY), 'platform': platform.platform(),
           'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
           'pyarrow': pyarrow.__version__, 'cpu_count': os.cpu_count()}
    try:
        cfg = np.__config__.get_info('blas_ilp64_opt_info') or np.__config__.get_info('blas_opt_info')
        out['blas'] = cfg.get('libraries')
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', choices=list(TARGETS),
                    default=['harmonize', 'harmonize_tiebreak', 'decomposition', 'features', 'consensus', 'beataml', 'pdx'])
    ap.add_argument('--compare-only', action='store_true')
    ap.add_argument('--no-strace', action='store_true')
    args = ap.parse_args()
    if Path(sys.executable).resolve() != ANALYSIS_PY.resolve():
        print(f'note: harness interpreter {sys.executable}; frozen scripts are run with {ANALYSIS_PY}')
    REGEN.mkdir(parents=True, exist_ok=True)
    hashes = manifest_hashes()
    use_strace = (not args.no_strace) and shutil.which('strace') is not None
    report_path = REGEN / 'comparison.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'targets': {}}
    report['environment'] = environment()
    for name in args.targets:
        spec = TARGETS[name]
        entry = report['targets'].get(name, {})
        entry['script_checks'] = check_scripts(spec, hashes)
        if not args.compare_only:
            print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] regenerating {name}', flush=True)
            info = run_copy(name, spec, use_strace)
            entry['run'] = {k: v for k, v in info.items() if k != 'opened'}
            if 'opened' in info:
                entry['provenance'] = info['opened']
                entry['inputs'] = input_records(info['opened'], hashes)
            print(f"  done in {info['seconds']} s", flush=True)
        _, diff = (write_patched_copy(name, spec) if args.compare_only
                   else (None, (PATCHED / name / (spec['script'] + '.diff')).read_text()))
        entry['patch_diff'] = diff
        rows = []
        frozen_files = (sorted(spec['frozen_dir'] / f for f in spec['files']) if 'files' in spec
                        else sorted(p for p in spec['frozen_dir'].iterdir() if p.is_file()))
        regen_files = sorted(p.name for p in spec['out_dir'].iterdir() if p.is_file()) if spec['out_dir'].exists() else []
        for f in frozen_files:
            r = compare_file(f, spec['out_dir'] / f.name)
            rec = hashes.get(r['file'], {})
            r['frozen_matches_manifest'] = (all(v == r['sha256_frozen'] for v in rec.values()) if rec else None)
            rows.append(r)
        entry['extra_regenerated_files'] = sorted(set(regen_files) - {f.name for f in frozen_files})
        entry['files'] = rows
        report['targets'][name] = entry
    report['generated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report_path.write_text(json.dumps(report, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)) + '\n')
    flat = [{'target': t, **{k: v for k, v in r.items() if k not in ('shape_frozen', 'shape_regenerated')}}
            for t, e in report['targets'].items() for r in e.get('files', [])]
    cols = ['target', 'file', 'sha256_frozen', 'sha256_regenerated', 'sha256_equal', 'values_identical',
            'max_abs_diff', 'n_diff', 'n_values', 'index_equal', 'columns_equal', 'dtypes_equal', 'nan_pattern_equal',
            'frozen_matches_manifest', 'sha256_decompressed_equal', 'note']
    T = pd.DataFrame(flat).reindex(columns=cols)
    T.to_csv(REGEN / 'comparison.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['target', 'file', 'sha256_equal', 'values_identical', 'max_abs_diff', 'n_diff']].to_string(index=False))


if __name__ == '__main__':
    main()
