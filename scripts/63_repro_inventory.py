"""Environment pins and timeline evidence for the revision (Reviewer 3, M1 request 4 and M8 request 2).

1. env/ (project root): for each interpreter used by the study
     venv         ../.venv         Python 3.8, main analysis (a venv over miniconda's base Python with system and user
                                   site-packages enabled: three package layers)
     venv-dreval  ../.venv-dreval  Python 3.11, DrEval re-scoring (a conda environment plus pip packages)
     venv-gpu     ../.venv-gpu     Python 3.10, GPU neural model (a venv over the system Python, isolated)
   writes `pip freeze` (requirements-<name>.lock.txt), an inventory of every importable distribution with its
   location and whether a copy earlier on sys.path shadows it (inventory-<name>.csv), `conda list --explicit` where conda
   manages the packages (conda-<name>.explicit.txt; conda-miniconda3-base.*.txt for the base layer of .venv), and
   env/python-versions.txt (interpreters, site layers, PYTHONPATH, key package versions, BLAS) and env/toolchain.txt
   (tectonic, OpenTimestamps client, OS).
2. results/revision_repro/file_times.csv: modification and birth times (UTC; ext4 birth times via GNU stat; a
   directory's birth time is its creation time, a file's is its last whole-file rewrite) and sizes of every file and
   directory under data/raw,
   data/processed, protocol, results (except the revision_* directories of the revision workstreams), logs, scripts,
   selbench, figures and manuscript;
   results/revision_repro/timeline_evidence.json: internal timestamps (gzip header MTIME of .gz files written by the
   pipeline, PDF /CreationDate, manifest frozen_utc, retrieved_utc.txt files, fetch logs, pipeline.log) and the
   attestations in the OpenTimestamps proof and its pre-upgrade backup.
Read-only except for env/ and results/revision_repro/. Run from scripts/ with ../../.venv/bin/python.
"""
import csv
import gzip
import json
import os
import platform
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
REPO = PROJECT.parent
ENV = PROJECT / 'env'
OUT = PROJECT / 'results' / 'revision_repro'
INTERPRETERS = {'venv': REPO / '.venv' / 'bin' / 'python', 'venv-dreval': REPO / '.venv-dreval' / 'bin' / 'python',
                'venv-gpu': REPO / '.venv-gpu' / 'bin' / 'python'}
CONDA = Path('/home/htran/miniconda3/bin/conda')
TECTONIC = Path('/home/htran/.local/bin/tectonic')
KEY = ['numpy', 'scipy', 'pandas', 'pyarrow', 'sklearn', 'rdkit', 'openpyxl', 'xlrd', 'h5py', 'matplotlib',
       'statsmodels', 'torch', 'drevalpy', 'lightgbm', 'xgboost', 'requests']

PROBE = r'''
import json, os, site, sys
try:
    import importlib.metadata as md
except ImportError:
    md = None
rows, seen = [], set()
for p in sys.path:
    if not p or not os.path.isdir(p) or md is None:
        continue
    for d in md.distributions(path=[p]):
        name = (d.metadata['Name'] or '?')
        key = name.lower().replace('_', '-')
        rows.append({'name': name, 'version': d.version, 'location': p, 'effective': key not in seen})
        seen.add(key)
key = {}
for m in KEYS:
    try:
        mod = __import__(m)
        key[m] = {'version': getattr(mod, '__version__', '?'), 'file': getattr(mod, '__file__', '?')}
        if m == 'torch':
            key[m]['cuda'] = getattr(mod.version, 'cuda', None)
    except Exception as e:
        key[m] = None
blas = None
try:
    import numpy as np
    try:
        cfg = np.show_config(mode='dicts')
        blas = {k: cfg['Build Dependencies'][k].get('name') for k in ('blas', 'lapack')}
    except TypeError:
        blas = {k: np.__config__.get_info(k).get('libraries') for k in ('blas_ilp64_opt_info', 'blas_opt_info')}
except Exception:
    pass
pools = None
try:
    import numpy, scipy.linalg, threadpoolctl
    pools = [{k: d.get(k) for k in ('internal_api', 'version', 'architecture', 'filepath')}
             for d in threadpoolctl.threadpool_info()]
except Exception:
    pass
print(json.dumps({'pools': pools, 'version': sys.version, 'executable': sys.executable, 'realpath': os.path.realpath(sys.executable),
                  'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'enable_user_site': site.ENABLE_USER_SITE,
                  'user_site': site.getusersitepackages() if site.ENABLE_USER_SITE else None, 'sys_path': sys.path,
                  'rows': rows, 'key': key, 'blas': blas}))
'''


# packages imported by the study scripts run with .venv (third-party only); their runtime closure at the versions the
# .venv actually imports is written to env/requirements-venv.rebuild.txt (the full freeze spans three layers and
# contains conda-built `@ file://` entries that pip cannot install)
ANALYSIS_TOP = ['numpy', 'scipy', 'pandas', 'pyarrow', 'scikit-learn', 'rdkit', 'matplotlib', 'openpyxl']
CLOSURE = r'''
import json, re
import importlib.metadata as md
try:
    from packaging.markers import Marker
except Exception:
    Marker = None
seen, stack = {}, list(TOP)
while stack:
    n = stack.pop()
    key = n.lower().replace('_', '-')
    if key in seen:
        continue
    try:
        d = md.distribution(n)
    except md.PackageNotFoundError:
        continue
    seen[key] = (d.metadata['Name'], d.version)
    for r in d.requires or []:
        req, _, marker = r.partition(';')
        if 'extra' in marker:
            continue
        if marker.strip() and Marker is not None:
            try:
                if not Marker(marker).evaluate():
                    continue
            except Exception:
                pass
        stack.append(re.split(r'[ <>=!~\[(]', req.strip())[0])
print(json.dumps(sorted(seen.values())))
'''


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    return r.stdout if r.returncode == 0 else f'# command failed ({r.returncode}): {" ".join(map(str, cmd))}\n{r.stderr}'


def layer(loc, info):
    loc = os.path.realpath(loc)
    if loc.startswith(os.path.realpath(info['prefix'])):
        return 'environment'
    if info.get('user_site') and loc.startswith(os.path.realpath(info['user_site'])):
        return 'user-site (~/.local)'
    if loc.startswith(os.path.realpath(info['base_prefix'])):
        return 'base interpreter'
    return 'other (PYTHONPATH or cwd)'


def environments():
    ENV.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = [f'# Interpreters of the selectivity_ceiling study; recorded {now} by scripts/63_repro_inventory.py', '',
             f"PYTHONPATH of the analysis shell (~/.bashrc): {os.environ.get('PYTHONPATH', '')!r}",
             '  (an unrelated project directory plus a trailing empty entry, i.e. the working directory; the scripts '
             'do not import from it; unset it in a clean rebuild)', '']
    for name, py in INTERPRETERS.items():
        info = json.loads(run([py, '-c', PROBE.replace('KEYS', repr(KEY))]))
        freeze = run([py, '-m', 'pip', 'freeze'])
        pipv = run([py, '-m', 'pip', '--version']).strip()
        header = (f'# pip freeze of {REPO.name}/.{name} ({info["version"].split()[0]}); {pipv}\n'
                  f'# interpreter: {info["realpath"]}; prefix {info["prefix"]}; base {info["base_prefix"]}\n'
                  f'# recorded {now}. See inventory-{name}.csv for the location (layer) of each distribution.\n')
        (ENV / f'requirements-{name}.lock.txt').write_text(header + freeze)
        with open(ENV / f'inventory-{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'version', 'layer', 'effective', 'location'])
            w.writeheader()
            for r in info['rows']:
                w.writerow({**r, 'layer': layer(r['location'], info)})
        cfg = Path(info['prefix']) / 'pyvenv.cfg'
        lines += [f'== .{name}', f"python {info['version'].split()[0]}  ({info['version']})",
                  f"executable {info['executable']} -> {info['realpath']}",
                  f"prefix {info['prefix']}; base_prefix {info['base_prefix']}",
                  'pyvenv.cfg: ' + (' | '.join(cfg.read_text().split('\n')).strip(' |') if cfg.exists() else
                                    'none (conda environment)' if (Path(info['prefix']) / 'conda-meta').exists() else 'none'),
                  f"user site enabled: {info['enable_user_site']} ({info['user_site']})", 'sys.path:']
        lines += [f'  {p}' for p in info['sys_path']]
        n_eff = sum(r['effective'] for r in info['rows'])
        lines += [f"distributions: {len(info['rows'])} found, {n_eff} effective, "
                  f"{len(info['rows']) - n_eff} shadowed by an earlier copy on sys.path", 'key packages:']
        for m, v in info['key'].items():
            if v:
                lines.append(f"  {m:12s} {v['version']:14s} {layer(os.path.dirname(v['file']), info) if v['file'] != '?' else ''}"
                             f"  {v['file']}" + (f"  cuda {v['cuda']}" if v.get('cuda') else ''))
        lines += [f"BLAS/LAPACK: {info['blas']}"]
        for d in info.get('pools') or []:
            lines.append(f"  loaded: {d['internal_api']} {d['version']} kernels {d['architecture']} ({d['filepath']})")
        lines.append('')
        if (Path(info['prefix']) / 'conda-meta').exists() and CONDA.exists():
            (ENV / f'conda-{name}.explicit.txt').write_text(run([CONDA, 'list', '-p', info['prefix'], '--explicit']))
            (ENV / f'conda-{name}.list.txt').write_text(run([CONDA, 'list', '-p', info['prefix']]))
        if name == 'venv':
            pins = json.loads(run([py, '-c', CLOSURE.replace('TOP', repr(ANALYSIS_TOP))]))
            (ENV / 'requirements-venv.rebuild.txt').write_text(
                f'# Minimal pinned set to rebuild .venv in a clean CPython {info["version"].split()[0]}: the third-party '
                f'packages imported by the\n# study scripts that run with .venv ({", ".join(ANALYSIS_TOP)}) and their '
                f'runtime dependencies, at the\n# versions the .venv actually imports across its three layers '
                f'(see inventory-venv.csv). Recorded {now}.\n'
                + ''.join(f'{n}=={v}\n' for n, v in pins))
        if name == 'venv' and CONDA.exists():
            base = info['base_prefix']
            (ENV / 'conda-miniconda3-base.explicit.txt').write_text(run([CONDA, 'list', '-p', base, '--explicit']))
            (ENV / 'conda-miniconda3-base.list.txt').write_text(run([CONDA, 'list', '-p', base]))
    (ENV / 'python-versions.txt').write_text('\n'.join(lines) + '\n')
    tool = [f'# Toolchain; recorded {now}', '',
            f"tectonic: {run([TECTONIC, '--version']).strip()!r} ({TECTONIC}; called by scripts/16_build_documents.py "
            f"as `tectonic --keep-logs main.tex` and `supplement.tex`)",
            f'tectonic bundle: {_tectonic_bundle()}',
            'OpenTimestamps client: opentimestamps-client 0.7.2 (pip), installed in a session-scratch virtualenv that '
            'no longer exists (Reviewer 3, M8); reinstall with `pip install opentimestamps-client==0.7.2`',
            f"ots on PATH now: {shutil.which('ots')}",
            f"conda: {run([CONDA, '--version']).strip() if CONDA.exists() else 'absent'}",
            f'OS: {platform.platform()}', f"glibc: {' '.join(platform.libc_ver())}",
            f"CPU: {_cpu()}", f'cores: {os.cpu_count()}']
    (ENV / 'toolchain.txt').write_text('\n'.join(tool) + '\n')


def _tectonic_bundle():
    cache = Path.home() / '.cache' / 'Tectonic'
    out = []
    for u in sorted((cache / 'urls').glob('*')) if (cache / 'urls').exists() else []:
        url = u.name.replace(',58,', ':').replace(',47,', '/')
        digest = u.read_text().strip()
        red = cache / 'redirects' / f'{digest}.txt'
        out.append(f"{url} (digest {digest}" + (f'; resolves to {red.read_text().strip()}' if red.exists() else '') + ')')
    return '; '.join(out) or 'no cached bundle found'


def _cpu():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')


def _varuint(b, i):
    v, s = 0, 0
    while True:
        x = b[i]; i += 1
        v |= (x & 0x7f) << s
        s += 7
        if not x & 0x80:
            return v, i


def ots_attestations(path):
    """Attestations in an OpenTimestamps proof: pending calendar URLs and Bitcoin block heights."""
    b = Path(path).read_bytes()
    out = {'file': str(Path(path).relative_to(PROJECT)), 'bytes': len(b), 'mtime_utc': utc(os.stat(path).st_mtime),
           'pending': [], 'bitcoin_block_heights': []}
    for m in re.finditer(re.escape(bytes.fromhex('83dfe30d2ef90c8e')), b):
        n, i = _varuint(b, m.end())
        ln, j = _varuint(b, i)
        out['pending'].append(b[j:j + ln].decode(errors='replace'))
    for m in re.finditer(re.escape(bytes.fromhex('0588960d73d71901')), b):
        n, i = _varuint(b, m.end())
        h, _ = _varuint(b, i)
        out['bitcoin_block_heights'].append(h)
    return out


def _stat_batch(paths):
    """birth and modification times (epoch seconds; birth 0 if unknown) from GNU stat, which exposes ext4 birth times."""
    out = {}
    for i in range(0, len(paths), 400):
        chunk = [str(p) for p in paths[i:i + 400]]
        r = subprocess.run(['stat', '-c', '%W|%.9Y|%n'] + chunk, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            w, y, n = line.split('|', 2)
            out[n] = (float(w), float(y))
    return out


def file_times():
    roots = ['data/raw', 'data/processed', 'protocol', 'results', 'logs', 'scripts', 'selbench', 'figures', 'manuscript']
    paths = [PROJECT, PROJECT / 'data', PROJECT / 'env'] if (PROJECT / 'env').exists() else [PROJECT, PROJECT / 'data']
    for r in roots:
        paths += [PROJECT / r] + sorted((PROJECT / r).rglob('*'))
    keep = []
    for p in paths:
        rel = p.relative_to(PROJECT)
        # results/revision_*: outputs of the revision workstreams (this one and others), not of the study
        if str(rel).startswith('results/revision_') or '__pycache__' in rel.parts:
            continue
        keep.append(p)
    st = _stat_batch(keep)
    rows = []
    for p in keep:
        rel = p.relative_to(PROJECT)
        w, y = st.get(str(p), (0.0, p.stat().st_mtime))
        row = {'path': str(rel) if str(rel) != '.' else '(project directory)', 'kind': 'dir' if p.is_dir() else 'file',
               'top': rel.parts[0] if rel.parts and rel.parts[0] != 'data' else '/'.join(rel.parts[:2]),
               'size': p.stat().st_size if p.is_file() else '', 'mtime_utc': utc(y),
               'birth_utc': utc(w) if w > 0 else '', 'internal_utc': '', 'internal_source': ''}
        if p.is_file() and p.suffix == '.gz' and not str(rel).startswith('data/raw'):
            with open(p, 'rb') as f:
                h = f.read(10)
            if h[:2] == b'\x1f\x8b':
                mt = struct.unpack('<I', h[4:8])[0]
                if mt:
                    row.update(internal_utc=utc(mt), internal_source='gzip header MTIME')
        if p.is_file() and p.suffix == '.pdf':
            m = re.search(rb'/CreationDate\s*\(D:(\d{14})([^)]*)\)', p.read_bytes())
            if m:
                row.update(internal_utc=_pdfdate(m.group(1).decode(), m.group(2).decode()), internal_source='PDF /CreationDate')
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'file_times.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def _pdfdate(d, tz):
    t = datetime.strptime(d, '%Y%m%d%H%M%S')
    m = re.match(r"([+-Z])(\d\d)?'?(\d\d)?", tz or 'Z')
    if m and m.group(1) in '+-' and m.group(2):
        off = (int(m.group(2)) * 60 + int(m.group(3) or 0)) * (1 if m.group(1) == '+' else -1)
        t = t.replace(tzinfo=timezone.utc) - __import__('datetime').timedelta(minutes=off)
    else:
        t = t.replace(tzinfo=timezone.utc)
    return t.isoformat(timespec='seconds')


def evidence(rows):
    ev = {'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'manifests': {}, 'retrieved_utc': {},
          'fetch_logs': {}, 'pipeline_log': [], 'ots': [], 'internal_timestamps': [r for r in rows if r['internal_utc']]}
    for mf in sorted((PROJECT / 'protocol').glob('freeze_manifest*.json')):
        m = json.loads(mf.read_text())
        ev['manifests'][mf.name] = {'frozen_utc': m.get('frozen_utc'), 'mtime_utc': utc(mf.stat().st_mtime),
                                    'files': sum(len(v) for v in m['files'].values())}
    for p in sorted((PROJECT / 'data' / 'raw').rglob('retrieved_utc.txt')):
        ev['retrieved_utc'][str(p.relative_to(PROJECT))] = p.read_text().strip()
    for p in sorted((PROJECT / 'data' / 'raw').rglob('fetch_log.json')):
        try:
            ev['fetch_logs'][str(p.relative_to(PROJECT))] = json.loads(p.read_text())
        except ValueError:
            ev['fetch_logs'][str(p.relative_to(PROJECT))] = p.read_text()[:2000]
    for name in ['pipeline.log', 'pipeline_run2.out', 'secondary_done.txt']:
        p = PROJECT / 'logs' / name
        if p.exists():
            ev['pipeline_log'].append({'file': name, 'text': p.read_text()})
    for p in sorted((PROJECT / 'protocol').glob('*.ots*')):
        ev['ots'].append(ots_attestations(p))
    (OUT / 'timeline_evidence.json').write_text(json.dumps(ev, indent=1, default=str) + '\n')
    return ev


def main():
    environments()
    rows = file_times()
    ev = evidence(rows)
    print('env files:', sorted(p.name for p in ENV.iterdir()))
    print('file times:', len(rows), '; internal timestamps:', len(ev['internal_timestamps']))
    print(json.dumps(ev['ots'], indent=1))


if __name__ == '__main__':
    main()
