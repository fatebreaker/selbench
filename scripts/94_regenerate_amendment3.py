"""Bit-for-bit regeneration of the Amendment 3 frozen predictions (editor round 2, C4), following
61_regenerate_frozen.py exactly.

The harness
1. checks that scripts/70_pdm_transfer_predictions.py and the sibling modules it imports (common.py, modeling.py,
   06_load_confirmation.py) have the SHA-256 recorded in protocol/freeze_manifest_amendment3.json (and in every
   earlier manifest that lists them); the frozen script is never edited;
2. writes a COPY of the script to results/revision_repro/regenerated_amendment3/_patched_scripts/ in which only the
   output path is changed (plus, as for script 30 in 61, the directory from which the hashed sibling module
   06_load_confirmation.py is loaded, which would otherwise resolve to the copy's directory); every patch must match
   exactly once and the unified diff is saved next to the copy;
3. runs the copy under strace (open/openat, following forked processes) with the analysis interpreter that wrote the
   frozen files (.venv, Python 3.8, pyarrow 9.0.0: the frozen parquet files record 'parquet-cpp-arrow version 9.0.0'),
   OPENBLAS/OMP/MKL threads = 1, PYTHONDONTWRITEBYTECODE=1, cwd = scripts/, writing only to
   results/revision_repro/regenerated_amendment3/pdm_transfer/;
4. records every project file the run opened, its SHA-256, and whether it matches the Amendment 3 manifest; confirms
   that no response file (data/raw/pdm_responses, data/processed/pdm*, results/pdm_confirm*, and the earlier
   confirmation outcome directories) was opened and nothing was written outside the regeneration directory;
5. compares each of the 21 frozen files in results/pdm_transfer/ (as listed in the manifest) with the regenerated file:
   SHA-256 equality, then numeric equality (61's compare_file).
The frozen outputs are only read. Outputs: results/revision_repro/regenerated_amendment3/{pdm_transfer/,
_patched_scripts/, _provenance/, comparison.csv, comparison.json}.
Usage (from scripts/): ../../.venv-dreval/bin/python 94_regenerate_amendment3.py [--python PY --tag TAG]
  --python/--tag run the same check with another interpreter (e.g. an environment rebuilt from env/ lockfiles) into
  results/revision_repro/regenerated_amendment3_<TAG>/, leaving the original run untouched.
"""
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import gzip
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
RESULTS = PROJECT / 'results'
_spec = importlib.util.spec_from_file_location('rg61', SCRIPTS / '61_regenerate_frozen.py')
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)          # read-only import of 61's helpers (sha256, compare_file, environment)

import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument('--python', default=None, help='interpreter to run the frozen script with (default: the study .venv)')
_ap.add_argument('--tag', default=None, help='write to regenerated_amendment3_<tag>/ instead (a second, independent run)')
ARGS = _ap.parse_args()
SUB = 'regenerated_amendment3' + (f'_{ARGS.tag}' if ARGS.tag else '')
REGEN = RESULTS / 'revision_repro' / SUB
PATCHED = REGEN / '_patched_scripts'
PROV = REGEN / '_provenance'
MANIFEST3 = PROJECT / 'protocol' / 'freeze_manifest_amendment3.json'
MANIFESTS = [PROJECT / 'protocol' / f for f in ['freeze_manifest.json', 'freeze_manifest_amendment1.json',
                                                 'freeze_manifest_amendment2.json', 'freeze_manifest_amendment3.json']]
ANALYSIS_PY = Path(ARGS.python) if ARGS.python else PROJECT.parent / '.venv' / 'bin' / 'python'
FROZEN_DIR = RESULTS / 'pdm_transfer'
OUT_DIR = REGEN / 'pdm_transfer'
SPEC = {
    'script': '70_pdm_transfer_predictions.py',
    'siblings': ['common.py', 'modeling.py', '06_load_confirmation.py'],
    'patches': [
        ("HERE = Path(__file__).resolve().parent",
         f"HERE = Path({str(SCRIPTS)!r})"),
        ("OUT = RESULTS / 'pdm_transfer'",
         f"OUT = RESULTS / 'revision_repro' / {SUB!r} / 'pdm_transfer'"),
    ],
}
OUTCOME_DIRS = ['data/raw/pdm_responses', 'data/processed/pdm', 'data/processed/pdm_corrected', 'results/pdm_confirm',
                'results/pdm_estimates', 'results/confirmation_pdm', 'data/raw/pdx_bruna_responses',
                'data/raw/beataml_responses', 'data/raw/confirmation', 'data/raw/pharmacodb_responses',
                'data/processed/confirmation', 'data/processed/pdx_bruna', 'data/processed/beataml']


def manifest_hashes():
    out = {}
    for mf in MANIFESTS:
        m = json.loads(mf.read_text())
        for files in m['files'].values():
            for p, h in files.items():
                out.setdefault(p, {})[mf.name] = h
    return out


def check_scripts(hashes):
    rows = []
    for name in [SPEC['script']] + SPEC['siblings']:
        rel = f'scripts/{name}'
        h = rg.sha256(SCRIPTS / name)
        rec = hashes.get(rel, {})
        ok3 = rec.get(MANIFEST3.name) == h
        rows.append({'file': rel, 'sha256': h, 'manifests': sorted(rec), 'in_amendment3_manifest': MANIFEST3.name in rec,
                     'matches_amendment3': ok3, 'matches_all_manifests': bool(rec) and all(v == h for v in rec.values())})
        if not rows[-1]['matches_all_manifests'] or (MANIFEST3.name in rec and not ok3):
            raise SystemExit(f'{rel} does not match its freeze-manifest hash; refusing to regenerate from it')
    return rows


def write_patched_copy():
    import difflib
    src = (SCRIPTS / SPEC['script']).read_text()
    new = src
    for old, rep in SPEC['patches']:
        n = new.count(old)
        if n != 1:
            raise SystemExit(f'patch target {old!r} found {n} times')
        new = new.replace(old, rep)
    PATCHED.mkdir(parents=True, exist_ok=True)
    dst = PATCHED / SPEC['script']
    dst.write_text(new)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        fromfile=f"scripts/{SPEC['script']} (frozen)",
                                        tofile=f"_patched_scripts/{SPEC['script']} (copy)"))
    (PATCHED / (SPEC['script'] + '.diff')).write_text(diff)
    return dst, diff


_OPEN = re.compile(r'open(?:at)?\((?:AT_FDCWD, |\d+, )?"([^"]+)", ([A-Z_|]+)[^)]*\)\s+=\s+(-?\d+)')


def summarise_trace(parts):
    read, written, attempted = set(), set(), set()
    roots = [str(PROJECT), str(PROJECT.parent / 'data')]
    for part in parts:
        for line in open(part, errors='replace'):
            m = _OPEN.search(line)
            if not m:
                continue
            path, flags = os.path.normpath(os.path.join(str(SCRIPTS), m.group(1))), m.group(2)
            if not any(path.startswith(r) for r in roots) or os.path.isdir(path):
                continue
            rel = os.path.relpath(path, str(PROJECT))
            if '__pycache__' in rel:
                continue
            if int(m.group(3)) < 0:
                attempted.add(rel)
                continue
            (written if ('O_WRONLY' in flags or 'O_RDWR' in flags or 'O_CREAT' in flags) else read).add(rel)
    every = read | written | attempted
    outcome = sorted(p for p in every if any(p.startswith(d) for d in OUTCOME_DIRS))
    outside = sorted(p for p in written if not p.startswith(f'results/revision_repro/{SUB}/'))
    return {'read': sorted(read), 'written': sorted(written), 'failed_open_attempts': sorted(attempted),
            'outcome_files_opened_or_attempted': outcome, 'written_outside_regenerated': outside}


def run_copy():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    dst, diff = write_patched_copy()
    env = dict(os.environ)
    env.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': str(REGEN / '_no_pycache'),
                'PYTHONPATH': str(SCRIPTS)})
    if ARGS.python:                      # a rebuilt environment: no user site-packages either
        env['PYTHONNOUSERSITE'] = '1'
    PROV.mkdir(parents=True, exist_ok=True)
    trace = PROV / 'pdm.strace'
    for old in PROV.glob('pdm.strace*'):
        old.unlink()
    cmd = ['strace', '-ff', '--seccomp-bpf', '-qq', '-e', 'trace=open,openat', '-e', 'signal=none', '-o', str(trace),
           str(ANALYSIS_PY), str(dst)]
    t0 = time.time()
    with open(PROV / 'pdm.stdout.txt', 'w') as fh:
        rc = subprocess.call(cmd, cwd=str(SCRIPTS), env=env, stdout=fh, stderr=subprocess.STDOUT)
    info = {'command': ' '.join(cmd), 'returncode': rc, 'seconds': round(time.time() - t0, 1),
            'started_utc': datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec='seconds')}
    parts = sorted(PROV.glob('pdm.strace.*'))
    info['traced_processes'] = len(parts)
    opened = summarise_trace(parts)
    with gzip.open(str(trace) + '.txt.gz', 'wt') as g:
        for part in parts:
            g.write(f'### {part.name}\n')
            g.write(part.read_text(errors='replace'))
            part.unlink()
    if rc != 0:
        raise SystemExit(f'regeneration failed (see {PROV / "pdm.stdout.txt"})')
    return info, opened, diff


def input_records(opened, hashes):
    rows = []
    for rel in opened['read']:
        if rel.startswith('results/revision_repro/'):
            continue
        p = PROJECT / rel
        if not p.is_file():
            continue
        h = rg.sha256(p)
        rec = hashes.get(rel, {})
        rows.append({'input': rel, 'sha256': h, 'mtime_utc': rg.utc_mtime(p), 'in_manifests': sorted(rec),
                     'in_amendment3_manifest': MANIFEST3.name in rec,
                     'matches_manifest': (all(v == h for v in rec.values()) if rec else None)})
    return rows


def main():
    hashes = manifest_hashes()
    m3 = json.loads(MANIFEST3.read_text())
    frozen = sorted(PROJECT / p for p in m3['files']['pdm_transfer_predictions'])
    report = {'manifest': str(MANIFEST3.relative_to(PROJECT)), 'manifest_sha256': rg.sha256(MANIFEST3),
              'frozen_utc': m3['frozen_utc'], 'script_checks': check_scripts(hashes)}
    print('scripts verified against the Amendment 3 manifest', flush=True)
    info, opened, diff = run_copy()
    report['run'] = info
    report['provenance'] = opened
    report['inputs'] = input_records(opened, hashes)
    report['patch_diff'] = diff
    rows = []
    for f in frozen:
        r = rg.compare_file(f, OUT_DIR / f.relative_to(FROZEN_DIR))
        r['frozen_matches_manifest'] = (r['sha256_frozen'] == m3['files']['pdm_transfer_predictions'][str(f.relative_to(PROJECT))])
        rows.append(r)
    report['files'] = rows
    regen_files = sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob('*') if p.is_file())
    report['extra_regenerated_files'] = sorted(set(regen_files) - {str(f.relative_to(FROZEN_DIR)) for f in frozen})
    T = pd.DataFrame(rows)
    report['summary'] = {
        'files_compared': int(len(T)), 'sha256_identical': int(T.sha256_equal.sum()),
        'values_identical': int(T.values_identical.fillna(False).astype(bool).sum()),
        'all_frozen_match_manifest': bool(T.frozen_matches_manifest.all()),
        'inputs_read': len(report['inputs']),
        'inputs_in_amendment3_manifest': int(sum(r['in_amendment3_manifest'] for r in report['inputs'])),
        'inputs_matching_manifest': int(sum(bool(r['matches_manifest']) for r in report['inputs'])),
        'inputs_not_in_any_manifest': [r['input'] for r in report['inputs'] if not r['in_manifests']],
        'outcome_files_opened_or_attempted': opened['outcome_files_opened_or_attempted'],
        'written_outside_regenerated': opened['written_outside_regenerated']}
    env_cmd = ("import sys, platform, numpy, pandas, scipy, pyarrow, json; print(json.dumps({'python': sys.version.split()[0], "
               "'platform': platform.platform(), 'numpy': numpy.__version__, 'pandas': pandas.__version__, "
               "'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__}))")
    # unhashed inputs: cross-reference 61's earlier bit-for-bit regeneration of the DepMap feature blocks
    prev = RESULTS / 'revision_repro' / 'regenerated' / 'comparison.csv'
    notes = {}
    if prev.exists():
        pc = pd.read_csv(prev)
        for rel in report['summary']['inputs_not_in_any_manifest']:
            hit = pc[pc.file == rel]
            cur = next(r['sha256'] for r in report['inputs'] if r['input'] == rel)
            if len(hit):
                notes[rel] = {'regenerated_by_61_sha256_equal': bool(hit.sha256_equal.iloc[0]),
                              'sha256_in_61_comparison': hit.sha256_frozen.iloc[0],
                              'same_file_now': hit.sha256_frozen.iloc[0] == cur}
            else:
                notes[rel] = 'not hashed and not regenerated (public DepMap file or fold table)'
    report['summary']['unhashed_input_notes'] = notes
    report['environment'] = {**json.loads(subprocess.check_output([str(ANALYSIS_PY), '-c', env_cmd]).decode()),
                             'executable': str(ANALYSIS_PY), 'cpu_count': os.cpu_count()}
    report['generated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    (REGEN / 'comparison.json').write_text(json.dumps(
        report, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)) + '\n')
    cols = ['file', 'sha256_frozen', 'sha256_regenerated', 'sha256_equal', 'values_identical', 'max_abs_diff', 'n_diff',
            'n_values', 'frozen_matches_manifest', 'note']
    T.reindex(columns=cols).to_csv(REGEN / 'comparison.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['file', 'sha256_equal', 'values_identical', 'max_abs_diff']].to_string(index=False))
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
