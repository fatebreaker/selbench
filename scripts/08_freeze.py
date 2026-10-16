"""Write protocol/freeze_manifest.json: SHA-256 of the pre-registration, every script, the discovery
decomposition, the frozen predictions and the rehearsal outcome, plus a check that no confirmation
outcome file exists anywhere in the repository. Refuses to overwrite an existing manifest.
"""
import datetime
import json
import sys

from common import PROJECT, REPO, sha256

CONFIRMATION_PATTERNS = ('gCSI_GRdata', 'gCSI_GRvalues', 'DTP_NCI60', 'NCI60_RAW', 'NCI60_ZSCORE')


def main():
    manifest = PROJECT / 'protocol' / 'freeze_manifest.json'
    if manifest.exists():
        sys.exit(f'{manifest} exists; the freeze is one-time')
    groups = {
        'protocol': sorted((PROJECT / 'protocol').glob('*.md')),
        'scripts': sorted(p for p in (PROJECT / 'scripts').glob('*') if p.suffix in ('.py', '.sh')),
        'decomposition': sorted((PROJECT / 'results' / 'decomposition').glob('*')),
        'frozen_predictions': sorted((PROJECT / 'results' / 'frozen_predictions').glob('*')),
        'rehearsal': sorted((PROJECT / 'results' / 'confirmation_rehearsal_GDSC2').glob('*')) +
                     sorted((PROJECT / 'results' / 'rehearsal_predictions_noGDSC2').glob('*.json')),
        'processed_inputs': [PROJECT / 'data' / 'processed' / f for f in
                             ('compounds.csv', 'compound_merges.csv', 'discovery_long.csv.gz', 'cells.csv')],
        'confirmation_metadata': sorted((PROJECT / 'data' / 'raw' / 'gcsi_metadata').glob('*')),
    }
    found = [str(p.relative_to(REPO)) for p in REPO.rglob('*')
             if p.is_file() and any(k in p.name for k in CONFIRMATION_PATTERNS) and '.venv' not in p.parts]
    if found:
        sys.exit(f'confirmation outcome files already present: {found}')
    out = {'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'confirmation_outcome_files_present': found,
           'files': {g: {str(p.relative_to(PROJECT)): sha256(p) for p in ps if p.is_file()} for g, ps in groups.items()}}
    manifest.write_text(json.dumps(out, indent=1) + '\n')
    print(manifest, sum(len(v) for v in out['files'].values()), 'files hashed at', out['frozen_utc'])


if __name__ == '__main__':
    main()
