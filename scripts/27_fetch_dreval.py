"""Parallel ranged download of DrEval's Zenodo files (curated datasets and published model results)."""
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import PROJECT

RECORD = 'https://zenodo.org/api/records/23159774'
OUT = PROJECT / 'data' / 'dreval'
CHUNK = 16 * 1024 * 1024


def get_range(url, start, end, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except Exception:
            pass
    raise RuntimeError(f'failed range {start}-{end}')


def fetch(name, size, checksum):
    path = OUT / name
    if path.exists() and path.stat().st_size == size:
        return
    url = f'{RECORD}/files/{name}/content'
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    with open(path, 'wb') as f:
        f.truncate(size)
    with ThreadPoolExecutor(16) as ex:
        for (s, e), data in zip(ranges, ex.map(lambda r: get_range(url, *r), ranges)):
            with open(path, 'r+b') as f:
                f.seek(s); f.write(data)
    algo, _, digest = checksum.partition(':')
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    assert h.hexdigest() == digest, f'checksum mismatch for {name}'
    print(name, size, 'ok', flush=True)


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(f'{RECORD}', timeout=120).read())
    files = {f['key']: (f['size'], f['checksum']) for f in meta['files']}
    (OUT / 'zenodo_record.json').write_text(json.dumps(meta, indent=1))
    for n in names:
        fetch(n, *files[n])


if __name__ == '__main__':
    main(sys.argv[1:])
