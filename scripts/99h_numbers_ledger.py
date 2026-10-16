"""Round 3, editor item 9 (R1-M4): the authoritative numbers ledger of the manuscript.

Every number in the abstract, Introduction, Results and Discussion (main text), in the figure captions
(sections/figures.tex), the Extended Data figure captions (sections/extended_data.tex), Box 1 (sections/box1.tex) and
the Methods (sections/methods.tex) is extracted from the LaTeX sources on every run (digits, powers of ten,
scientific notation, ordinals, spelled-out counts, multiples and fractions, dates and clock times; citation keys, SHA-256,
figure/panel/note/table labels, the \st... table macros, LaTeX lengths, subscripts such as IC$_{50}$ and formula
constants are skipped) and matched to a hand-checked map, results/ledger/ledger_map.csv, which gives for each number
occurrence

  section, anchor, number    where it is: the section, a short string of the surrounding LaTeX (whitespace collapsed)
                             with the number written as [[number]], and the number as written
  quantity                   what it is, in plain language
  kind                       'source' (re-read from a deposited file and compared), 'derived' (arithmetic on deposited
                             values, given in `expr`), 'design' (a design constant or count fixed by the study design;
                             its source is named, the value is not re-read), 'wording' (a spelled-out word that is not
                             a quantity) or 'unverifiable' (no deposited source; stated in `notes`)
  source_file + locator      the primary source (path relative to the project root) and where in it: json_path (a JSON
                             list of keys) or a CSV row_filter (pandas query) + column (+ agg: min, max, median, count,
                             nunique, ...)
  transform, compare, tol    how the source value becomes the written number (x100 for percentages, abs, neg); the
                             comparison is rounding to the decimals written unless compare is 'le'/'ge' (bounds such
                             as P <= 1e-4), 'approx' with an absolute tolerance (words such as 'about a third') or
                             'text' (dates and clock times, compared as written; derived expressions can use UTC() to
                             format a timestamp, OTS() for the block heights of an OpenTimestamps proof and MANIFEST()
                             for the files of a freeze manifest whose SHA-256 has changed since)
  script                     the script that produced the source file
  alt_*                      a second source, usually results/paper_numbers.json (or key_numbers*.json), re-read and
                             compared as well
  headline, recipe           about 50 numbers marked for the independent human verification (condition C3), each with
                             a one-line recipe for recomputing it from the deposited files
  topic                      groups reconciled together (IMPROVE floor, Fig. 4b, tissue baseline, Holm, capture, 2x2)

On each run the script re-extracts the numbers, re-reads every source value and writes (results/ledger/)
  numbers_ledger.csv         one row per number occurrence, with status
                               match | rounding mismatch | value mismatch | unmapped | anchor not found (text changed)
                               | design constant | wording (not a quantity) | derived in text | unverifiable
                               | source error
                             and the second source's status (alt_status)
  ledger_check.csv           problem rows only: rounding / value mismatches, unmapped numbers, anchors not found,
                             source errors, unverifiable numbers and numbers whose second source disagrees
  summary.json               counts by status and by section, the reconciliation by topic, coverage of
                             paper_numbers.json, sha256 of every source file used, of results/paper_numbers.json and of
                             the manuscript sections, the frozen hashes and a timestamp
  change_log.csv             one row appended whenever a hashed file (source, paper_numbers.json, manuscript section)
                             differs from its frozen hash (file, old hash, new hash, date)
The freeze: on the first run results/paper_numbers.json is copied to results/ledger/paper_numbers_frozen_<date>.json
and the sha256 of every hashed file is recorded in results/ledger/frozen_hashes.json (copied into summary.json); later
runs never overwrite a frozen hash (files used for the first time are added with their first-seen hash).

When the text changes, the numbers of the edited sentences become 'anchor not found (text changed)' (their map rows)
and 'unmapped' (the new occurrences); update the map rows (anchor, number, locator) and rerun.
--dump-occurrences PATH writes every extracted occurrence with a proposed unique anchor, to start new map rows.

Usage (from scripts/):  ../../.venv/bin/python 99h_numbers_ledger.py [--dump-occurrences PATH]
                         [--sections-dir DIR --out-dir DIR]   (testing on an edited copy of the sections)
Reads only; writes only results/ledger/ (or --out-dir). No network access; single process.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings('ignore', message='This pattern is interpreted as a regular expression')

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / 'manuscript' / 'sections'
OUT = ROOT / 'results' / 'ledger'
MAP = OUT / 'ledger_map.csv'
PAPER_NUMBERS = 'results/paper_numbers.json'

SECTIONS = [('abstract', 'abstract.tex', 'main'), ('introduction', 'introduction.tex', 'main'),
            ('results', 'results.tex', 'main'), ('discussion', 'discussion.tex', 'main'),
            ('figures', 'figures.tex', 'captions'), ('extended_data', 'extended_data.tex', 'captions'),
            ('box1', 'box1.tex', 'box'), ('methods', 'methods.tex', 'methods')]

# ----------------------------------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------------------------------

# spans whose digits are not numbers of the paper (labels, citation keys, LaTeX lengths and macros, formula constants)
SKIP_PATTERNS = [
    r'(?<!\\)%[^\n]*',                                              # comments (not escaped \%)
    r'\\(?:cite|ref|label|input|includegraphics|begin|end|newcommand)\*?(?:\[[^\]]*\])?\{[^}]*\}(?:\[[^\]]*\])?',
    r'\\st[A-Za-z]+',                                             # supplementary-table macros
    r'[\d.]+\\(?:textwidth|linewidth|columnwidth)',               # LaTeX lengths
    r'Figs?\.~?\s*\d+[a-g]?(?:\s*(?:,|--|–)\s*[a-g](?![a-z]))*',  # figure and panel labels
    r'Figure~?\s*\d+',
    r'Supplementary\s+Notes?~?\s*\d+(?:\s*(?:,|and~?)\s*\d+)*',   # supplementary note labels
    r'Supplementary\s+(?:Fig\.|Figure)s?~?\s*\d+',
    r'Box~?\s*\d+',
    r'Amendments?~?\s*\d+',
    r'[Ss]teps?~?\s*\d+',                                         # Box 1 step references
    r'\\textbf\{\d+\.',                                           # Box 1 step numbers
    r'NCI-60',
    r'SHA-\d+',                                                     # hash algorithm names
    r'_\{[^}]*\}', r'_[A-Za-z0-9]',                               # subscripts (IC_{50}, log_{10})
    r'(?<!10)(?<!10\$)\^\{[^}]*\}', r'\^[0-9]',                            # superscripts other than powers of ten
    r'\(1-(?=[A-Za-z\\])',                                        # (1 - r^2), (1-\rho), (1-m) in formulas
]
SKIP_RE = re.compile('|'.join(f'(?:{p})' for p in SKIP_PATTERNS))

NUM_RE = re.compile(
    r'(?P<pow>(?<![\d.])10\$?\^\{(?P<pexp>[-−]?\d+)\})'
    r'|(?P<sci>(?<![\d.])(?P<mant>\d+(?:\.\d+)?)\s*\\times\s*10\^\{(?P<exp>[-−]?\d+)\})'
    r'|(?P<num>(?<![A-Za-z\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:st|nd|rd|th)?(?![A-Za-z\d]))')

WORDS = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
         'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
         'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'twice': 2, 'half': 0.5, 'third': 1 / 3, 'two-thirds': 2 / 3,
         'quarter': 0.25, 'fifth': 0.2, 'four-fifths': 0.8}
WORD_RE = re.compile(r'(?<![A-Za-z\-])(' + '|'.join(sorted((re.escape(w) for w in WORDS), key=len, reverse=True))
                     + r')(?![A-Za-z\-])', re.IGNORECASE)
# spelled-out multiples: 'four-fold', 'ten-fold' and the first word of 'two- to four-fold' (the word is the token)
_NUMW = 'two|three|four|five|six|seven|eight|nine|ten'
FOLD_RE = re.compile(rf'(?<![A-Za-z\-])({_NUMW})(?=-(?:fold\b|\s+to\s+(?:{_NUMW})-fold\b))', re.IGNORECASE)
ORDINAL_WORD_RE = re.compile(r'(?<=ranked )(second|third|first)\b|\b(fourth)\b|(?<=[Oo]nly )(one)\b|\b([Nn]one)\b')
ORDINAL_VALUES = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'one': 1, 'none': 0}
# dates and clock times are single tokens (compared as text: compare = 'text')
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'
DATETIME_RE = re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})|(?P<time>(?<![\d:])\d{1,2}:\d{2}:\d{2}(?![\d:]))'
                         rf'|(?P<dmy>(?<!\d)\d{{1,2}}~?\s?(?:{MONTHS})(?:\s\d{{4}})?)')


def norm_ws(s):
    return re.sub(r'\s+', ' ', s)


def extract(section, path):
    """Number occurrences of one .tex file: list of dicts (in text order)."""
    raw = path.read_text()
    # mask: True where a character belongs to a skipped span
    mask = np.zeros(len(raw), bool)
    for m in SKIP_RE.finditer(raw):
        mask[m.start():m.end()] = True
    occ = []

    seen = set()

    def add(start, end, text, value, kind):
        if mask[start:end].any() or (start, end) in seen:
            return
        seen.add((start, end))
        line = raw.count('\n', 0, start) + 1
        # sign: an explicit minus written just before the number ($-$, −, -) that is not a range dash (--)
        pre = raw[max(0, start - 4):start]
        sign = ''
        if re.search(r'(\$-\$|−)~?$', pre) or (re.search(r'(?<![-\w])-$', pre)):
            sign = '-'
        elif re.search(r'(\$\+\$|\+)~?$', pre):
            sign = '+'
        post = raw[end:end + 2]
        unit = '%' if post.startswith('\\%') else ''
        occ.append(dict(section=section, line=line, start=start, end=end, number=text, sign=sign, unit=unit,
                        value=value, kind_extracted=kind))

    for m in DATETIME_RE.finditer(raw):
        kind = 'date' if (m.group('date') or m.group('dmy')) else 'time'
        add(m.start(), m.end(), m.group(0), float('nan'), kind)
        mask[m.start():m.end()] = True                 # digits inside a date or time are not separate numbers
    for m in NUM_RE.finditer(raw):
        if m.group('pow'):
            add(m.start(), m.end(), m.group('pow'), 10.0 ** int(m.group('pexp').replace('−', '-')), 'sci')
        elif m.group('sci'):
            mant = float(m.group('mant'))
            ex = int(m.group('exp').replace('−', '-'))
            add(m.start(), m.end(), m.group('sci'), mant * 10 ** ex, 'sci')
        else:
            t = m.group('num')
            ordn = re.search(r'(st|nd|rd|th)$', t)
            core = t[:ordn.start()] if ordn else t
            add(m.start(), m.end(), t, float(core.replace(',', '')), 'ordinal' if ordn else 'digits')
    for m in FOLD_RE.finditer(raw):
        add(m.start(), m.end(), m.group(1), WORDS[m.group(1).lower()], 'word')
    for m in WORD_RE.finditer(raw):
        add(m.start(), m.end(), m.group(1), WORDS[m.group(1).lower()], 'word')
    for m in ORDINAL_WORD_RE.finditer(raw):
        w = next(g for g in m.groups() if g)
        add(m.start(), m.end(), w, ORDINAL_VALUES[w.lower()], 'count word' if w.lower() in ('one', 'none') else 'ordinal word')
    occ.sort(key=lambda o: o['start'])
    for o in occ:
        o['context'] = norm_ws(raw[max(0, o['start'] - 160):o['start']]) + '[[' + o['number'] + ']]' + \
            norm_ws(raw[o['end']:o['end'] + 100])
        o['before'] = norm_ws(raw[max(0, o['start'] - 160):o['start']])
        o['after'] = norm_ws(raw[o['end']:o['end'] + 100])
    return occ, raw


def all_occurrences(ms=None):
    out = []
    for sec, fn, part in SECTIONS:
        occ, _ = extract(sec, (ms or MS) / fn)
        for i, o in enumerate(occ):
            o['part'] = part
            o['occ_index'] = i
        out.extend(occ)
    return out


def make_anchor(o, occs_same_section):
    """Shortest anchor (LaTeX before + [[number]] + LaTeX after) that matches only this occurrence in its section."""
    for b, a in [(40, 12), (60, 20), (90, 30), (130, 60), (160, 100)]:
        anc = o['before'][-b:] + '[[' + o['number'] + ']]' + o['after'][:a]
        if sum(anc in p['context'] for p in occs_same_section) == 1:
            return anc
    return anc


# ----------------------------------------------------------------------------------------------------------------------
# Source values
# ----------------------------------------------------------------------------------------------------------------------

MAP_COLUMNS = ['id', 'section', 'anchor', 'number', 'quantity', 'kind', 'source_file', 'json_path', 'row_filter',
               'column', 'agg', 'expr', 'transform', 'compare', 'tol', 'script', 'alt_source', 'alt_json_path',
               'alt_row_filter', 'alt_column', 'alt_agg', 'headline', 'recipe', 'topic', 'notes']

_cache = {}
USED_FILES = set()


def _path(f):
    p = ROOT / f
    if not p.exists():
        raise FileNotFoundError(f)
    USED_FILES.add(f)
    return p


def load_json(f):
    if ('j', f) not in _cache:
        _cache[('j', f)] = json.loads(_path(f).read_text())
    return _cache[('j', f)]


def load_csv(f):
    if ('c', f) not in _cache:
        _cache[('c', f)] = pd.read_csv(_path(f), low_memory=False)
    return _cache[('c', f)]


def J(f, *keys):
    x = load_json(f)
    for k in keys:
        if isinstance(x, list):
            k = int(k)
        x = x[k]
    return x


AGG = {'min': lambda s: s.min(), 'max': lambda s: s.max(), 'median': lambda s: s.median(), 'mean': lambda s: s.mean(),
       'sum': lambda s: s.sum(), 'count': lambda s: float(len(s)), 'nunique': lambda s: float(s.nunique()),
       'first': lambda s: s.iloc[0], 'absmin': lambda s: s.abs().min(), 'absmax': lambda s: s.abs().max()}


def C(f, row_filter='', column='', agg=''):
    """CSV value: rows matching a pandas query, one column, optional aggregation (exactly one row without agg)."""
    d = load_csv(f)
    if row_filter:
        d = d.query(row_filter, engine='python')
    if agg == 'count' and not column:
        return float(len(d))
    s = d[column]
    if agg and s.dtype == object and s.astype(str).str.startswith('[').any():
        s = pd.Series([v for x in s for v in first_number(x)])   # intervals stored as '[lo, hi]' strings
    if agg:
        return AGG[agg](s)
    if len(s) != 1:
        raise ValueError(f'{len(s)} rows match {row_filter!r} in {f}')
    return s.iloc[0]


def first_number(x):
    """Numbers stored as strings such as '[0.0262, 0.1691]' -> list of floats."""
    return [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e-?\d+)?', str(x))]


def read_value(src, json_path, row_filter, column, agg):
    if not src:
        raise ValueError('no source')
    if src.endswith('.json'):
        keys = json.loads(json_path) if json_path else []
        v = J(src, *keys)
        if agg == 'count' and isinstance(v, (list, tuple, dict)):
            return float(len(v))
        if agg:
            if isinstance(v, (list, tuple)):
                v = AGG[agg](pd.Series([float(t) for t in v]))
            elif isinstance(v, str):
                v = AGG[agg](pd.Series(first_number(v)))
        return v
    return C(src, row_filter, column, agg)


def apply_transform(v, transform):
    if isinstance(v, str):
        nums = first_number(v)
        if len(nums) != 1:
            raise ValueError(f'non-numeric source value {v!r}')
        v = nums[0]
    if isinstance(v, (bool, np.bool_)):
        v = float(v)
    v = float(v)
    for t in [t.strip() for t in (transform or '').split(',') if t.strip()]:
        if t == 'x100':
            v *= 100
        elif t == 'abs':
            v = abs(v)
        elif t == 'neg':
            v = -v
        elif t == 'x-1':
            v = v - 1
        elif t.startswith('x'):
            v *= float(t[1:])
        else:
            raise ValueError(f'unknown transform {t}')
    return v


def written_value(row_number, sign, kind):
    """Value as written (signed) and its number of decimals."""
    t = row_number
    if '\\times' not in t and '^{' in t:
        ex = int(re.search(r'\^\{([-−]?\d+)\}', t).group(1).replace('−', '-'))
        return 10.0 ** ex, -ex
    if '\\times' in t:
        m = re.match(r'(\d+(?:\.\d+)?)\s*\\times\s*10\^\{([-−]?\d+)\}', t)
        mant, ex = m.group(1), int(m.group(2).replace('−', '-'))
        dec = (len(mant.split('.')[1]) if '.' in mant else 0) - ex
        return float(mant) * 10 ** ex, dec
    if kind in ('word', 'ordinal word', 'count word', 'date', 'time'):
        return None, None
    core = re.sub(r'(st|nd|rd|th)$', '', t).replace(',', '')
    dec = len(core.split('.')[1]) if '.' in core else 0
    v = float(core)
    return (-v if sign == '-' else v), dec


def compare(x, w, dec, how, tol):
    """Status of source value x (transformed) against the written value w with dec decimals."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'source error', 'source value is NaN'
    if how in ('le', 'ge'):
        ok = x <= w * (1 + 1e-9) if how == 'le' else x >= w * (1 - 1e-9)
        if ok:
            return 'match', 'bound holds'
        near = abs(x - w) <= 0.5 * 10.0 ** (-dec) + 1e-15
        return ('rounding mismatch' if near else 'value mismatch'), \
            f'bound {"<=" if how == "le" else ">="} {w:.4g} violated by source {x:.4g}' + (' (within rounding)' if near else '')
    if how == 'approx' or (tol not in ('', None) and not (isinstance(tol, float) and math.isnan(tol))):
        t = float(tol)
        return ('match', f'approximate (tolerance {t})') if abs(x - w) <= t + 1e-12 else \
            ('value mismatch', f'|{x:.4g} - {w:.4g}| > {t}')
    unit = 10.0 ** (-dec)
    d = abs(x - w)
    if d <= 0.5 * unit + 1e-9 * max(1, abs(w)):
        return 'match', ''
    if d <= 1.5 * unit + 1e-9:
        return 'rounding mismatch', f'source {x:.6g} rounds to {round(x, dec):.{max(dec, 0)}f}'
    return 'value mismatch', f'source {x:.6g}'


def T(f, row_filter=''):
    """CSV rows matching a pandas query, as a DataFrame (for derived expressions)."""
    d = load_csv(f)
    return d.query(row_filter, engine='python') if row_filter else d


def LS(d):
    """Names in a directory of the project (read-only listing; e.g. DrEval's released model directories)."""
    p = ROOT / d
    if not p.is_dir():
        raise FileNotFoundError(d)
    return sorted(x.name for x in p.iterdir())


def UTC(s, fmt):
    """An ISO-8601 timestamp (or date) from a source file, written as the text writes it: 'date' (2026-10-14), 'time'
    (18:29:40), 'dmy' (14 October 2026) or 'dm' (14 October); a value without a zone is taken as UTC."""
    t = dt.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    t = t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)
    return {'date': t.strftime('%Y-%m-%d'), 'time': t.strftime('%H:%M:%S'),
            'dmy': f'{t.day} {t.strftime("%B")} {t.year}', 'dm': f'{t.day} {t.strftime("%B")}'}[fmt]


def OTS(f):
    """Bitcoin block heights attested in an OpenTimestamps proof (.ots, read-only): the varuint payload after each
    BitcoinBlockHeaderAttestation tag."""
    b = _path(f).read_bytes()
    tag = bytes.fromhex('0588960d73d71901')

    def varuint(i):
        v = s = 0
        while True:
            x = b[i]
            i += 1
            v |= (x & 0x7f) << s
            s += 7
            if not x & 0x80:
                return v, i
    out, i = [], b.find(tag)
    while i >= 0:
        _, j = varuint(i + len(tag))      # payload length
        out.append(varuint(j)[0])         # block height
        i = b.find(tag, i + 1)
    return sorted(out)


def MANIFEST(f):
    """Files hashed in a freeze manifest and those whose current SHA-256 differs (read-only)."""
    m = load_json(f)
    files = {p: h for grp in m['files'].values() for p, h in grp.items()}
    changed = sorted(p for p, h in files.items() if not (ROOT / p).exists() or sha256(ROOT / p) != h)
    return {'files': len(files), 'changed': changed}    # (the members are not added to the hashed source files)


def eval_expr(expr, values):
    env = {'J': J, 'C': C, 'T': T, 'LS': LS, 'UTC': UTC, 'OTS': OTS, 'MANIFEST': MANIFEST, 're': re, 'np': np, 'pd': pd,
           'V': lambda i: values[i], 'abs': abs, 'min': min, 'max': max,
           'round': round, 'len': len, 'sum': sum, 'float': float, 'math': math}
    return eval(expr, env)  # noqa: S307 (expressions come from the hand-checked map)


# ----------------------------------------------------------------------------------------------------------------------
# Hashes, freeze and change log
# ----------------------------------------------------------------------------------------------------------------------

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze_and_log(used, today, now, ms):
    """Frozen copy of paper_numbers.json and frozen hashes (written once); change log of source and section files.

    The frozen hashes live in results/ledger/frozen_hashes.json (copied into summary.json on every run); if that file
    is missing, the 'frozen' block of an existing summary.json is used. A file first used after the freeze is added
    with its hash at first use. Every hash that differs from the frozen one is appended once to change_log.csv.
    """
    fh_path = OUT / 'frozen_hashes.json'
    frozen = None
    if fh_path.exists():
        frozen = json.loads(fh_path.read_text())
    elif (OUT / 'summary.json').exists():
        prev = json.loads((OUT / 'summary.json').read_text())
        if 'frozen' in prev and 'paper_numbers' in prev:
            frozen = dict(frozen_utc=prev['frozen']['frozen_utc'], files=prev['frozen']['files'],
                          paper_numbers_frozen_copy=prev['paper_numbers']['frozen_copy'],
                          paper_numbers_sha256=prev['paper_numbers']['frozen_sha256'])
    pn = ROOT / PAPER_NUMBERS
    if frozen is None:
        dst = OUT / f'paper_numbers_frozen_{today}.json'
        if not dst.exists():
            shutil.copy2(pn, dst)
        frozen = {'frozen_utc': now, 'paper_numbers_frozen_copy': os.path.relpath(dst, ROOT),
                  'paper_numbers_sha256': sha256(dst),
                  'paper_numbers_mtime_utc': dt.datetime.fromtimestamp(pn.stat().st_mtime, dt.timezone.utc)
                  .isoformat(timespec='seconds'),
                  'review_round3_mtimes_utc': {
                      p.name: dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(timespec='seconds')
                      for p in sorted((ROOT / 'manuscript' / 'review_round3').glob('*.md'))},
                  'note': 'results/paper_numbers.json as it was at the freeze (compare its file time with the round-3 '
                          'review files); later changes of any hashed file are appended to change_log.csv',
                  'files': {}}
    files = {f: ROOT / f for f in sorted(set(used) | {PAPER_NUMBERS})}
    files.update({os.path.relpath(ms / fn, ROOT): ms / fn for _, fn, _ in SECTIONS})
    current = {f: sha256(p) for f, p in sorted(files.items()) if p.exists()}
    changes, added = [], False
    for f, h in current.items():
        old = frozen['files'].get(f)
        if old is None:
            frozen['files'][f] = h          # first seen: its hash at first use is the frozen one
            added = True
        elif old != h:
            changes.append((f, old, h))
    if added or not fh_path.exists():
        fh_path.write_text(json.dumps(frozen, indent=1))
    log = OUT / 'change_log.csv'
    prev = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=['file', 'old_sha256', 'new_sha256', 'date'])
    new = [dict(file=f, old_sha256=o, new_sha256=n, date=now) for f, o, n in changes
           if not ((prev.file == f) & (prev.new_sha256 == n)).any()]
    if new or not log.exists():
        pd.concat([prev, pd.DataFrame(new, columns=prev.columns)], ignore_index=True).to_csv(log, index=False)
    return frozen, current, changes


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

PROBLEMS = {'rounding mismatch', 'value mismatch', 'unmapped', 'anchor not found (text changed)', 'source error',
            'unverifiable'}


class Row(dict):
    """A map row with attribute access (pandas Series attributes such as .agg or .transform would shadow columns)."""
    __getattr__ = dict.__getitem__


def blank(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-occurrences', help='write every extracted occurrence with a proposed anchor to this CSV')
    ap.add_argument('--sections-dir', help='read the .tex sections from this directory instead of manuscript/sections '
                                           '(testing only; outputs then go to --out-dir)')
    ap.add_argument('--out-dir', help='write outputs here instead of results/ledger (the map is still read from '
                                      'results/ledger/ledger_map.csv)')
    args = ap.parse_args()
    global OUT
    ms = Path(args.sections_dir).resolve() if args.sections_dir else MS
    if args.sections_dir and not args.out_dir:
        raise SystemExit('--sections-dir requires --out-dir (the frozen ledger state must not be touched by tests)')
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    today = dt.date.today().isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    occ = all_occurrences(ms)
    by_sec = {}
    for o in occ:
        by_sec.setdefault(o['section'], []).append(o)
    if args.dump_occurrences:
        rows = [dict(section=o['section'], line=o['line'], number=o['number'], sign=o['sign'], unit=o['unit'],
                     kind_extracted=o['kind_extracted'], anchor=make_anchor(o, by_sec[o['section']]))
                for o in occ]
        pd.DataFrame(rows).to_csv(args.dump_occurrences, index=False)
        print(f'{len(rows)} occurrences -> {args.dump_occurrences}')
        return

    M = pd.read_csv(MAP, dtype=str, keep_default_na=False)
    missing_cols = [c for c in MAP_COLUMNS if c not in M.columns]
    if missing_cols:
        raise SystemExit(f'ledger_map.csv lacks columns {missing_cols}')

    MR = [Row(x) for x in M.to_dict('records')]
    # match map rows to occurrences
    assigned = {}
    map_status = {}
    for i, r in enumerate(MR):
        cands = [o for o in by_sec.get(r.section, []) if r.anchor in o['context'] and o['number'] == r.number]
        if len(cands) == 1 and id(cands[0]) not in assigned:
            assigned[id(cands[0])] = i
        elif len(cands) > 1:
            map_status[i] = ('anchor not found (text changed)', f'anchor matches {len(cands)} occurrences')
        else:
            map_status[i] = ('anchor not found (text changed)', 'anchor not in the current text')

    # evaluate map rows (sources first, then derived rows that may reference them)
    values, alt_values, results = {}, {}, {}
    order = sorted(range(len(MR)), key=lambda i: MR[i]['kind'] == 'derived')
    for i in order:
        r = MR[i]
        res = dict(source_value='', alt_value='', status='', detail='')
        try:
            if r.kind == 'source':
                v = read_value(r.source_file, r.json_path, r.row_filter, r.column, r.agg)
                res['source_value'] = v
                values[r.id] = v if r.compare == 'text' else apply_transform(v, r.transform)
            elif r.kind == 'derived':
                v = eval_expr(r.expr, values)
                res['source_value'] = v
                values[r.id] = v if r.compare == 'text' else apply_transform(v, r.transform)
            if r.kind in ('source', 'derived') and not blank(r.alt_source):
                av = read_value(r.alt_source, r.alt_json_path, r.alt_row_filter, r.alt_column, r.alt_agg)
                res['alt_value'] = av
                alt_values[r.id] = apply_transform(av, r.transform)
        except Exception as e:  # noqa: BLE001
            res['status'] = 'source error'
            res['detail'] = f'{type(e).__name__}: {e}'
        results[i] = res

    rows = []
    matched_occ = set()
    for i, r in enumerate(MR):
        o = next((p for p in occ if id(p) in assigned and assigned[id(p)] == i), None)
        res = results[i]
        base = dict(id=r.id, section=r.section, part=o['part'] if o else '', line=o['line'] if o else '',
                    number=r.number, written=(o['sign'] + o['number'] + o['unit']) if o else r.number,
                    anchor=r.anchor, quantity=r.quantity, kind=r.kind, source_file=r.source_file,
                    locator=(r.json_path or (f'{r.row_filter} :: {r.column}' + (f' :: {r.agg}' if r.agg else '')))
                    if r.kind == 'source' else r.expr, transform=r.transform, script=r.script,
                    source_value=res['source_value'], value_for_text='', alt_source=r.alt_source,
                    alt_locator=r.alt_json_path or (f'{r.alt_row_filter} :: {r.alt_column}' if r.alt_column else ''),
                    alt_value=res['alt_value'], alt_status='', headline=r.headline, recipe=r.recipe, topic=r.topic,
                    notes=r.notes)
        if o is not None:
            matched_occ.add(id(o))
        if i in map_status:
            base['status'], base['detail'] = map_status[i]
        elif res['status']:
            base['status'], base['detail'] = res['status'], res['detail']
        elif r.kind in ('source', 'derived') and r.compare == 'text':
            x = res['source_value']
            base['value_for_text'] = x
            ok = norm_ws(str(x)).replace('~', ' ').strip() == norm_ws(o['number']).replace('~', ' ').strip()
            base['status'] = ('derived in text' if r.kind == 'derived' else 'match') if ok else 'value mismatch'
            base['detail'] = 'text comparison' if ok else f'source gives {x!r}'
        elif r.kind in ('source', 'derived'):
            w, dec = written_value(r.number, o['sign'], o['kind_extracted'])
            if w is None:  # spelled-out number
                w = o['value']
                dec = 0 if float(w).is_integer() else 2
            x = values[r.id]
            base['value_for_text'] = x
            st, det = compare(x, w, dec, r.compare or 'round', r.tol)
            if st == 'match' and w == 0 and o['sign'] == '-' and x > 0:      # '-0.00' written for a positive value
                st, det = 'rounding mismatch', f'written with a minus sign; source {x:.4g} is positive'
            if r.kind == 'derived' and st == 'match':
                st = 'derived in text'
                det = (det + '; ' if det else '') + 'arithmetic checked'
            base['status'], base['detail'] = st, det
            if r.id in alt_values:
                ast, adet = compare(alt_values[r.id], w, dec, r.compare or 'round', r.tol)
                base['alt_status'] = ast + (f' ({adet})' if adet else '')
        elif r.kind == 'design':
            base['status'], base['detail'] = 'design constant', ''
        elif r.kind == 'wording':
            base['status'], base['detail'] = 'wording (not a quantity)', ''
        elif r.kind == 'unverifiable':
            base['status'], base['detail'] = 'unverifiable', 'no deposited source; see notes'
        else:
            base['status'], base['detail'] = 'source error', f'unknown kind {r.kind!r}'
        rows.append(base)
    for o in occ:
        if id(o) not in matched_occ:
            rows.append(dict(id='', section=o['section'], part=o['part'], line=o['line'], number=o['number'],
                             written=o['sign'] + o['number'] + o['unit'],
                             anchor=make_anchor(o, by_sec[o['section']]), status='unmapped',
                             detail='number in the text without a map row'))
    L = pd.DataFrame(rows)
    sec_order = {s: k for k, (s, _, _) in enumerate(SECTIONS)}
    L['_s'] = L.section.map(sec_order)
    L['_l'] = pd.to_numeric(L.line, errors='coerce')
    L = L.sort_values(['_s', '_l', 'id']).drop(columns=['_s', '_l'])
    L.to_csv(OUT / 'numbers_ledger.csv', index=False)
    altbad = L.alt_status.fillna('').str.contains('mismatch')
    chk = L[L.status.isin(PROBLEMS) | altbad].copy()
    chk.insert(0, 'check_reason', np.where(chk.status.isin(PROBLEMS), chk.status, 'second source disagrees'))
    chk.to_csv(OUT / 'ledger_check.csv', index=False)
    recon = {}
    for t, g in L[L.topic.fillna('') != ''].groupby('topic'):
        recon[t] = dict(numbers=int(len(g)), status=g.status.value_counts().to_dict(),
                        ids=';'.join(g.id.astype(str)),
                        problems=g[g.status.isin(PROBLEMS) | g.alt_status.fillna('').str.contains('mismatch')][
                            ['id', 'written', 'status', 'detail', 'alt_status']].fillna('').to_dict('records'))
    pn_rows = L[L.alt_source.fillna('') == PAPER_NUMBERS]

    frozen, current, changes = freeze_and_log(USED_FILES, today, now, ms)
    summary = dict(
        generated_utc=now, script='scripts/99h_numbers_ledger.py', map='results/ledger/ledger_map.csv',
        occurrences=int(len(occ)), map_rows=int(len(M)),
        counts_by_status=L.status.value_counts().to_dict(),
        counts_by_section_and_status={s: g.status.value_counts().to_dict() for s, g in L.groupby('section')},
        headline_numbers=int((L.headline.astype(str).isin(['1', '1.0'])).sum()),
        paper_numbers_coverage=dict(
            mapped_source_rows=int(L.kind.isin(['source']).sum()),
            rows_with_paper_numbers_counterpart=int(len(pn_rows)),
            rows_without_counterpart=int(L.notes.fillna('').str.contains('no counterpart').sum()),
            counterpart_disagreements=int(pn_rows.alt_status.fillna('').str.contains('mismatch').sum())),
        alt_source_disagreements=int(L.alt_status.fillna('').str.contains('mismatch').sum()),
        paper_numbers=dict(path=PAPER_NUMBERS, sha256=current.get(PAPER_NUMBERS),
                           frozen_copy=frozen['paper_numbers_frozen_copy'],
                           frozen_sha256=frozen['paper_numbers_sha256'],
                           changed_since_freeze=current.get(PAPER_NUMBERS) != frozen['paper_numbers_sha256']),
        frozen=dict(frozen_utc=frozen['frozen_utc'], note=frozen.get('note', ''),
                    paper_numbers_mtime_utc=frozen.get('paper_numbers_mtime_utc', ''),
                    review_round3_mtimes_utc=frozen.get('review_round3_mtimes_utc', {}), files=frozen['files']),
        source_files_sha256=current,
        changed_since_freeze=[dict(file=f, old_sha256=a, new_sha256=b) for f, a, b in changes],
        manuscript_sha256={fn: sha256(ms / fn) for _, fn, _ in SECTIONS},
        reconciliation=recon, problems=L[L.status.isin(PROBLEMS) | L.alt_status.fillna('').str.contains('mismatch')][
            ['id', 'section', 'line', 'written', 'status', 'detail', 'alt_status']].fillna('').to_dict('records'))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary['counts_by_status'], indent=1))
    print('by section:', json.dumps(summary['counts_by_section_and_status']))


if __name__ == '__main__':
    sys.exit(main())
