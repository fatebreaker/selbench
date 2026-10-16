"""Freeze Amendment 1 before any CCLE/FIMM/GRAY/BeatAML response retrieval (one-time; refuses to overwrite)."""
import datetime
import json
import sys

from common import PROJECT, REPO, sha256

OUTCOME_PATTERNS = ('probit_curve_fits', 'raw_inhibitor', 'CCLE_long', 'FIMM_long', 'GRAY_long', 'beataml_long')


def main():
    manifest = PROJECT / 'protocol' / 'freeze_manifest_amendment1.json'
    if manifest.exists():
        sys.exit(f'{manifest} exists; the freeze is one-time')
    present = [str(p.relative_to(REPO)) for p in REPO.rglob('*') if p.is_file() and '.venv' not in p.parts
               and any(k in p.name for k in OUTCOME_PATTERNS)]
    responses = PROJECT / 'data' / 'raw' / 'pharmacodb_responses'
    if present or (responses.exists() and any(responses.iterdir())):
        sys.exit(f'response files already present: {present}')
    groups = {
        'protocol': sorted((PROJECT / 'protocol').glob('*.md')),
        'scripts': sorted(p for p in (PROJECT / 'scripts').glob('*') if p.suffix in ('.py', '.sh')),
        'frozen_predictions': sorted((PROJECT / 'results' / 'frozen_predictions').glob('*')),
        'transfer_predictions': sorted((PROJECT / 'results' / 'beataml_transfer').glob('*')),
        'metadata_inputs': sorted((PROJECT / 'data' / 'raw' / 'pharmacodb_metadata').glob('*')) +
                           sorted((PROJECT / 'data' / 'raw' / 'beataml_inputs').glob('*')),
        'original_manifest': [PROJECT / 'protocol' / 'freeze_manifest.json'],
    }
    out = {'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'response_files_present': present,
           'files': {g: {str(p.relative_to(PROJECT)): sha256(p) for p in ps if p.is_file()} for g, ps in groups.items()}}
    manifest.write_text(json.dumps(out, indent=1) + '\n')
    print(manifest, sum(len(v) for v in out['files'].values()), 'files hashed at', out['frozen_utc'])


if __name__ == '__main__':
    main()
