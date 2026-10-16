"""Amendment 3 (in preparation): fetch only the INPUT files of CoderData's patient-derived-model datasets.

CoderData v2.4 (figshare article 33990850; PNNL-CompBio/coderdata) stores each modality of each dataset as a separate
file. This script downloads, for the patient-derived datasets, only the sample tables, transcriptomics, mutations
and drug tables. It refuses any file whose name contains 'experiments' or 'combinations' (drug responses), so no
outcome reaches this project before the freeze. Every request is logged with its UTC time, figshare file id, size
and SHA-256 in data/raw/pdm_inputs/fetch_log.json.
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone

from common import PROJECT

ARTICLE = 33990850
DATASETS = ['bladder', 'pancreatic', 'colorectal', 'liver', 'sarcoma', 'mpnst', 'novartis']
KINDS = ['samples.csv', 'transcriptomics.csv.gz', 'mutations.csv.gz', 'drugs.tsv.gz']
FORBIDDEN = ('experiments', 'combinations')
OUT = PROJECT / 'data' / 'raw' / 'pdm_inputs'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = json.loads(get(f'https://api.figshare.com/v2/articles/{ARTICLE}/files?page=1&page_size=500'))
    by_name = {f['name']: f for f in files}
    log = {'article': ARTICLE, 'started_utc': datetime.now(timezone.utc).isoformat(), 'files': []}
    for ds in DATASETS:
        for kind in KINDS:
            name = f'{ds}_{kind}'
            assert not any(x in name for x in FORBIDDEN), name
            f = by_name.get(name)
            if f is None:
                log['files'].append({'name': name, 'status': 'absent'})
                continue
            t = datetime.now(timezone.utc).isoformat()
            blob = get(f"https://api.figshare.com/v2/file/download/{f['id']}")
            (OUT / name).write_bytes(blob)
            log['files'].append({'name': name, 'figshare_file_id': f['id'], 'size': len(blob),
                                 'sha256': hashlib.sha256(blob).hexdigest(), 'requested_utc': t})
            print(name, len(blob), flush=True)
            time.sleep(1)
    log['completed_utc'] = datetime.now(timezone.utc).isoformat()
    log['not_requested'] = sorted(n for n in by_name if any(x in n for x in FORBIDDEN))
    (OUT / 'fetch_log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
