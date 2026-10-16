"""Submission inventory (round-4 editor, B2 and B4): project-tree hash, change inventory and deposit plan.

Read-only on everything except submission/ (written here). For the frozen state of the text:
  tree_manifest.csv        every file of the project tree (path, bytes, SHA-256), sorted by path; __pycache__ and *.pyc
                           are skipped. The project-tree hash is the SHA-256 of the lines "path<TAB>sha256<LF>" in
                           that order (summary.json), so anyone can recompute it from the manifest or the tree.
  change_inventory.csv     every file hashed in the four freeze manifests (original, amendments 1-3): unchanged,
                           changed or missing now; for the two append-only logs, whether the frozen version is an exact
                           byte prefix of the current file (checked by hashing every prefix that ends at a newline).
  unhashed_inputs.csv      the four Amendment 3 regeneration inputs that no manifest hashed: their SHA-256 now against
                           the SHA-256 the regeneration recorded (results/revision_repro/regenerated_amendment3/
                           comparison.json), with what can and cannot be shown about them.
  deposit_plan.csv         each file assigned to a deposit group (B2): code and lockfiles; frozen predictions,
                           manifests and timestamp proofs; derived tables; source data and ledger; manuscript; and
                           what is not deposited (public raw data, retrievable by the fetch scripts), with sizes.
  summary.json             counts, sizes, the tree hash and the time of the run.
Nothing is uploaded; deposit and DOIs are for the authors.
Usage: 104_submission_inventory.py
"""
import csv
import hashlib
import json
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'submission'
MANIFESTS = ['freeze_manifest.json', 'freeze_manifest_amendment1.json', 'freeze_manifest_amendment2.json',
             'freeze_manifest_amendment3.json']
APPEND_ONLY = {'protocol/deviations.md', 'protocol/post_freeze_notes.md'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tree_files():
    for p in sorted(PROJECT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        rel = p.relative_to(PROJECT).as_posix()
        if rel.startswith('submission/'):
            continue
        yield rel, p


def manifest_entries():
    for m in MANIFESTS:
        d = json.loads((PROJECT / 'protocol' / m).read_text())
        for group, files in d['files'].items():
            for rel, h in files.items():
                yield m, group, rel, h


def is_prefix(p, frozen_hash):
    data = (PROJECT / p).read_bytes()
    ends = [i + 1 for i, c in enumerate(data) if c == 0x0A] + [len(data)]
    return any(hashlib.sha256(data[:e]).hexdigest() == frozen_hash for e in ends)


def deposit_group(rel):
    top = rel.split('/')[0]
    if rel.startswith('protocol/') or rel.endswith('.ots') or rel.startswith('results/frozen') or '/frozen_' in rel:
        return 'B2-2 frozen predictions, manifests, registrations and timestamp proofs'
    if top in ('scripts', 'selbench', 'env') or rel in ('README.md', 'HANDOFF.md'):
        return 'B2-1 code, tests and lockfiles'
    if rel.startswith('results/ledger/') or rel.startswith('manuscript/source_data/'):
        return 'B2-4 source data and numbers ledger'
    if rel.startswith('results/revision_retrain/drevalpy_runs/'):
        return 'B2-3b retraining run directories (models, predictions; large)'
    if top in ('results', 'logs'):
        return 'B2-3 derived tables and logs'
    if top in ('manuscript', 'figures'):
        return 'manuscript, figures, supplementary data'
    if rel.startswith('data/raw/'):
        return 'not deposited: public raw data (fetch scripts and logs record source and hashes)'
    if top == 'data':
        return 'B2-3 derived tables and logs'
    return 'other'


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    rows, lines = [], []
    for rel, p in tree_files():
        h = sha256(p)
        rows.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': h, 'deposit_group': deposit_group(rel)})
        lines.append(f'{rel}\t{h}\n')
    tree_hash = hashlib.sha256(''.join(lines).encode()).hexdigest()
    with open(OUT / 'tree_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['path', 'bytes', 'sha256', 'deposit_group'])
        w.writeheader(); w.writerows(rows)
    now = {r['path']: r['sha256'] for r in rows}

    inv = []
    for m, group, rel, h in manifest_entries():
        cur = now.get(rel)
        status = 'missing' if cur is None else ('unchanged' if cur == h else 'changed')
        prefix = ''
        if status == 'changed' and rel in APPEND_ONLY:
            prefix = 'frozen version is a byte prefix' if is_prefix(rel, h) else 'NOT a prefix'
        inv.append({'manifest': m, 'group': group, 'path': rel, 'frozen_sha256': h, 'current_sha256': cur or '',
                    'status': status, 'append_only_check': prefix})
    with open(OUT / 'change_inventory.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(inv[0]))
        w.writeheader(); w.writerows(inv)

    comp = json.loads((PROJECT / 'results/revision_repro/regenerated_amendment3/comparison.json').read_text())
    notes = comp['summary'].get('unhashed_input_notes', {})
    recorded = {i['input']: i['sha256'] for i in comp['inputs'] if isinstance(i, dict)}
    explain = {
        'data/processed/features/expression_top.parquet':
            'derived from the DepMap expression file by the feature script; regenerated bit-for-bit by script 61 '
            '(same SHA-256), so its content is fixed by its inputs and code',
        'data/raw/depmap24q4/OmicsExpressionProteinCodingGenesTPMLogp1.csv':
            'public DepMap 24Q4 release file, downloaded before the original freeze; not hashed at the time. Its '
            'identity can be checked against the DepMap release download, not against a pre-freeze record',
        'data/raw/depmap24q4/PortalCompounds.csv':
            'public DepMap 24Q4 release file; as above',
        'results/models/cell_folds.csv':
            'cell-line fold table of the discovery models, written before the original freeze; not hashed at the '
            'time. Unchanged since the regeneration (same SHA-256); no pre-freeze record exists'}
    un = []
    for rel in notes:
        cur = now.get(rel, '')
        un.append({'path': rel, 'sha256_now': cur, 'sha256_at_regeneration': recorded.get(rel, ''),
                   'same_as_regeneration': cur == recorded.get(rel, ''), 'what_can_be_shown': explain.get(rel, '')})
    with open(OUT / 'unhashed_inputs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(un[0]))
        w.writeheader(); w.writerows(un)

    plan = {}
    for r in rows:
        g = plan.setdefault(r['deposit_group'], {'files': 0, 'bytes': 0})
        g['files'] += 1; g['bytes'] += r['bytes']
    with open(OUT / 'deposit_plan.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['deposit_group', 'files', 'bytes', 'GB'])
        for g, v in sorted(plan.items()):
            w.writerow([g, v['files'], v['bytes'], f"{v['bytes'] / 1e9:.3f}"])

    cnt = {}
    for r in inv:
        cnt[r['status']] = cnt.get(r['status'], 0) + 1
    changed = sorted({r['path'] for r in inv if r['status'] == 'changed'})
    summary = {'run_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': round(time.time() - t0),
               'tree_files': len(rows), 'tree_bytes': sum(r['bytes'] for r in rows), 'tree_sha256': tree_hash,
               'tree_hash_definition': 'SHA-256 of "path<TAB>sha256<LF>" lines of tree_manifest.csv in path order',
               'manifest_entries': len(inv), 'manifest_status_counts': cnt,
               'changed_files': changed,
               'append_only_ok': all(r['append_only_check'] == 'frozen version is a byte prefix'
                                     for r in inv if r['status'] == 'changed' and r['path'] in APPEND_ONLY),
               'unhashed_inputs_same_as_regeneration': all(u['same_as_regeneration'] for u in un),
               'deposit_plan': plan}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    verification_kit(tree_hash)
    print(json.dumps({k: v for k, v in summary.items() if k != 'deposit_plan'}, indent=1))


KIT_HEAD = """# Verification kit for the named analyst (round-4 editor, condition B3)

Generated by `scripts/104_submission_inventory.py` from the numbers ledger (`results/ledger/ledger_map.csv`, headline
rows) for the frozen text with project-tree hash `{tree}`. Work from the deposited files only, without the study's
scripts, and report every discrepancy (value, file, your code) to the corresponding author; discrepancies are disclosed
in the paper, not fixed silently.

## 1. Re-implement the core estimators

Per drug d, with profile z, prediction p and the evaluation dataset's general-sensitivity axis g (rank-normalized
profiles; g = leave-drug-out mean of the other active drugs' profiles; Methods and Box 1):

1. **Exact decomposition.** r_pz = r_pg r_zg + sqrt((1 - r_pg^2)(1 - r_zg^2)) r_pz.g, with r_pz.g the partial
   correlation of p and z given g. Check it holds to machine precision on any model and drug.
2. **Axis and selective capture.** A = sum_d r_pg r_zg / sum_d r_zg^2; S = sum_d sqrt((1 - r_pg^2)(1 - r_zg^2)) r_pz.g /
   sum_d (1 - r_zg^2); perfect-predictor axis share = mean_d r_zg^2. Pool over models by summing over models and drugs.
3. **Two-way bootstrap.** Resample drugs and cell lines independently (B = 1,000), recompute pooled ratios, take
   percentile intervals.
4. **Reliability-adjusted attenuation test.** Excess = (share of the loss carried by the axis term) - (axis term's share
   within the training dataset), with each dataset's axis term divided by its axis split-half reliability (100 random
   drug halves); a clear loss is one whose raw-loss interval excludes 0; a reading rejects proportional attenuation when
   the excess interval excludes 0. Source: `results/revision_round4/simulation/realdata_attenuation_reliability_adjusted.csv`.
5. **Transfer correlations.** Across models, Spearman correlation between a within-dataset statistic (median raw
   per-drug r, DrEval's normalized Pearson, S) and the same predictions' mean selective-both in each other dataset;
   pooled over targets as equal-weight Fisher-z means; bootstrap resamples models, drugs and cells. Source:
   `results/revision_round4/transfer_prediction_r4/`.
6. **Dataset x endpoint contrasts.** Differences of pooled axis shares between the four cells, the interaction = endpoint
   effect in CTRP v2 minus endpoint effect in GDSC2. Source: `results/revision_round2/two_by_two/`.

## 2. Reproduce the headline numbers

Each row: where the number is, what it is, the deposited file it comes from and a one-line recipe. Tick each, or record
the value you obtain.

| # | Section | Written | Quantity | Source file | Recipe | Your value |
|---|---|---|---|---|---|---|
"""

KIT_TAIL = """
## 3. Check each abstract claim (a human author)

Read the abstract sentence by sentence against the Results paragraph and the files above; record agreement or the
change needed. The ledger (`results/ledger/numbers_ledger.csv`) lists every other number in the text with its source.

Analyst: ____________________  Date: __________  Discrepancies found: ____
"""


def verification_kit(tree_hash):
    import pandas as pd
    m = pd.read_csv(PROJECT / 'results/ledger/ledger_map.csv')
    h = m[m.headline.fillna(0).astype(str).isin(['1', '1.0', 'True', 'true'])]
    esc = lambda x: '' if pd.isna(x) else str(x).replace('|', '/').replace('\n', ' ')  # noqa: E731
    lines = [KIT_HEAD.format(tree=tree_hash)]
    for i, r in enumerate(h.itertuples(), 1):
        lines.append(f'| {i} | {r.section} | {esc(r.number)} | {esc(r.quantity)} | `{esc(r.source_file)}` | '
                     f'{esc(r.recipe)} | |\n')
    lines.append(KIT_TAIL)
    (OUT / 'verification_kit.md').write_text(''.join(lines))


if __name__ == '__main__':
    main()
