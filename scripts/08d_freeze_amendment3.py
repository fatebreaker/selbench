"""Freeze Amendment 3 before any response retrieval for the patient-derived datasets (one-time; refuses to overwrite).

After this script, stamp the manifest publicly: `ots stamp protocol/freeze_manifest_amendment3.json`
(OpenTimestamps; only the SHA-256 digest leaves the machine). 73_fetch_pdm_responses.py refuses to run without both
files.
"""
import datetime
import json
import sys

from common import PROJECT, REPO, sha256

DATASETS = ['bladder', 'pancreatic', 'liver', 'sarcoma', 'mpnst']


def main():
    manifest = PROJECT / 'protocol' / 'freeze_manifest_amendment3.json'
    if manifest.exists():
        sys.exit(f'{manifest} exists; the freeze is one-time')
    present = [str(p.relative_to(REPO)) for ds in DATASETS for p in REPO.rglob(f'{ds}_experiments*')
               if '.venv' not in str(p)]
    responses = PROJECT / 'data' / 'raw' / 'pdm_responses'
    processed = PROJECT / 'data' / 'processed' / 'pdm'
    if present or (responses.exists() and any(responses.iterdir())) or processed.exists() or \
            (PROJECT / 'results' / 'confirmation_PDM').exists():
        sys.exit(f'response files or outputs already present: {present}')
    groups = {
        'protocol': sorted((PROJECT / 'protocol').glob('*.md')),
        'scripts': sorted(p for p in (PROJECT / 'scripts').glob('*') if p.suffix in ('.py', '.sh')),
        'selbench': sorted((PROJECT / 'selbench').glob('*.py')),
        'frozen_predictions': sorted((PROJECT / 'results' / 'frozen_predictions').glob('*')),
        'pdm_transfer_predictions': sorted((PROJECT / 'results' / 'pdm_transfer').rglob('*')),
        'pdm_inputs': sorted((PROJECT / 'data' / 'raw' / 'pdm_inputs').glob('*')),
        'pdm_simulation_checks': sorted((PROJECT / 'results' / 'pdm_simulation').glob('*')),
        'discovery_classes': [PROJECT / 'results' / 'decomposition' / 'cross_screen_pairs.csv'],
        'previous_manifests': [PROJECT / 'protocol' / 'freeze_manifest.json',
                               PROJECT / 'protocol' / 'freeze_manifest_amendment1.json',
                               PROJECT / 'protocol' / 'freeze_manifest_amendment2.json',
                               PROJECT / 'protocol' / 'freeze_manifest_amendment2.json.ots'],
    }
    out = {'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'response_files_present': present,
           'files': {g: {str(p.relative_to(PROJECT)): sha256(p) for p in ps if p.is_file()} for g, ps in groups.items()}}
    manifest.write_text(json.dumps(out, indent=1) + '\n')
    print(manifest, sum(len(v) for v in out['files'].values()), 'files hashed at', out['frozen_utc'])


if __name__ == '__main__':
    main()
