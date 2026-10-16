"""Freeze Amendment 2 before any PDX_Bruna response retrieval (one-time; refuses to overwrite).

After this script, stamp the manifest publicly: `ots stamp protocol/freeze_manifest_amendment2.json`
(OpenTimestamps; only the SHA-256 digest leaves the machine). 33_fetch_pdx_responses.py refuses to run without
both files.
"""
import datetime
import json
import sys

from common import PROJECT, REPO, sha256


def main():
    manifest = PROJECT / 'protocol' / 'freeze_manifest_amendment2.json'
    if manifest.exists():
        sys.exit(f'{manifest} exists; the freeze is one-time')
    present = [str(p.relative_to(REPO)) for p in REPO.rglob('PDX_Bruna*.csv') if '.venv' not in str(p)]
    responses = PROJECT / 'data' / 'raw' / 'pdx_bruna_responses'
    processed = PROJECT / 'data' / 'processed' / 'pdx_bruna'
    if present or (responses.exists() and any(responses.iterdir())) or processed.exists():
        sys.exit(f'PDX_Bruna response files already present: {present}')
    groups = {
        'protocol': sorted((PROJECT / 'protocol').glob('*.md')),
        'scripts': sorted(p for p in (PROJECT / 'scripts').glob('*') if p.suffix in ('.py', '.sh')),
        'frozen_predictions': sorted((PROJECT / 'results' / 'frozen_predictions').glob('*')),
        'pdx_transfer_predictions': sorted((PROJECT / 'results' / 'pdx_transfer').glob('*')),
        'pdx_inputs': sorted((PROJECT / 'data' / 'raw' / 'pdx_bruna_inputs').glob('*')),
        'discovery_classes': [PROJECT / 'results' / 'decomposition' / 'cross_screen_pairs.csv'],
        'previous_manifests': [PROJECT / 'protocol' / 'freeze_manifest.json',
                               PROJECT / 'protocol' / 'freeze_manifest_amendment1.json'],
    }
    out = {'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'response_files_present': present,
           'files': {g: {str(p.relative_to(PROJECT)): sha256(p) for p in ps if p.is_file()} for g, ps in groups.items()}}
    manifest.write_text(json.dumps(out, indent=1) + '\n')
    print(manifest, sum(len(v) for v in out['files'].values()), 'files hashed at', out['frozen_utc'])


if __name__ == '__main__':
    main()
