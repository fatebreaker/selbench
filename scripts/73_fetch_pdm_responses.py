"""Fetch the response files of the five Amendment 3 datasets AFTER the freeze and its public timestamp.

Refuses to run unless protocol/freeze_manifest_amendment3.json and its OpenTimestamps proof (.ots) both exist.
Downloads <dataset>_experiments.tsv.gz for bladder, pancreatic, liver, sarcoma and MPNST from CoderData v2.4 (figshare
article 33990850, version 4, the version of the frozen input files). The figshare file ids, MD5 checksums and sizes of
version 4 are pinned below (taken from figshare's file listing before the freeze; a listing holds no response value).
Logs UTC time, size and SHA-256 in data/raw/pdm_responses/fetch_log.json.
"""
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import PROJECT

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fi', HERE / '69_fetch_pdm_inputs.py')
fi = importlib.util.module_from_spec(spec); spec.loader.exec_module(fi)
MANIFEST = PROJECT / 'protocol' / 'freeze_manifest_amendment3.json'
OUT = PROJECT / 'data' / 'raw' / 'pdm_responses'
VERSION = 4
PINNED = {
    'bladder_experiments.tsv.gz': (69973449, 'a2b1259d32d9fa15fc30bccb8c4918cf', 112869),
    'pancreatic_experiments.tsv.gz': (69973407, '80531ff88a1f33eae347e334aac6369f', 11338),
    'liver_experiments.tsv.gz': (69973485, '7317f84b2afdde98ab564a03db1a0308', 251168),
    'sarcoma_experiments.tsv.gz': (69973398, '9349c57a2e47dac64bf868d741fee994', 1998),
    'mpnst_experiments.tsv.gz': (69973446, '86cc16dff62ee607b16f9b7d5d3226a0', 103375),
}


def main():
    if not MANIFEST.exists() or not MANIFEST.with_suffix('.json.ots').exists():
        sys.exit('Amendment 3 freeze manifest and its OpenTimestamps proof must exist before retrieval')
    OUT.mkdir(parents=True, exist_ok=True)
    log = {'article': fi.ARTICLE, 'version': VERSION,
           'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
           'started_utc': datetime.now(timezone.utc).isoformat(), 'files': []}
    for name, (fid, md5_expected, size) in PINNED.items():
        t = datetime.now(timezone.utc).isoformat()
        blob = fi.get(f'https://api.figshare.com/v2/file/download/{fid}')
        md5 = hashlib.md5(blob).hexdigest()
        assert md5 == md5_expected and len(blob) == size, (name, 'differs from the pinned version-4 file')
        (OUT / name).write_bytes(blob)
        log['files'].append({'name': name, 'figshare_file_id': fid, 'size': len(blob), 'md5': md5,
                             'sha256': hashlib.sha256(blob).hexdigest(), 'requested_utc': t})
        print(name, len(blob), flush=True)
        time.sleep(1)
    log['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'fetch_log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
