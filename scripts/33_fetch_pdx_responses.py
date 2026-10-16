"""Fetch the PDX_Bruna response member (PDX_Bruna/PDX_Bruna.csv) AFTER the Amendment 2 freeze and its public timestamp.

Refuses to run unless protocol/freeze_manifest_amendment2.json and its OpenTimestamps proof (.ots) both exist.
Downloads only that member by HTTP range, checks its CRC-32 against the zip central directory and writes it to
data/raw/pdx_bruna_responses/ with a UTC log.
"""
import datetime
import importlib.util
import json
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

from common import PROJECT

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fi', HERE / '29_fetch_pdx_inputs.py')
fi = importlib.util.module_from_spec(spec); spec.loader.exec_module(fi)
MANIFEST = PROJECT / 'protocol' / 'freeze_manifest_amendment2.json'
OUT = PROJECT / 'data' / 'raw' / 'pdx_bruna_responses'


def main():
    if not MANIFEST.exists() or not MANIFEST.with_suffix('.json.ots').exists():
        sys.exit('Amendment 2 freeze manifest and its OpenTimestamps proof must exist before retrieval')
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(fi.RECORD, timeout=120).read())
    f = [x for x in meta['files'] if x['key'] == fi.NAME][0]
    size, url = f['size'], f'{fi.RECORD}/files/{fi.NAME}/content'
    z = zipfile.ZipFile(fi.Tail(size, fi.get(url, size - 2 * 1024 * 1024, size - 1)))
    info = z.getinfo(fi.RESPONSE_MEMBER)
    hdr = fi.get(url, info.header_offset, info.header_offset + 29)
    n_name, n_extra = struct.unpack('<HH', hdr[26:30])
    start = info.header_offset + 30 + n_name + n_extra
    raw = fi.get(url, start, start + info.compress_size - 1)
    data = zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw
    assert zlib.crc32(data) & 0xffffffff == info.CRC, 'CRC mismatch'
    (OUT / 'PDX_Bruna.csv').write_bytes(data)
    log = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'member': fi.RESPONSE_MEMBER,
           'bytes': len(data), 'crc32': f'{info.CRC:08x}', 'zip_checksum': f['checksum'],
           'record_version_date': meta['metadata'].get('publication_date')}
    (OUT / 'fetch_log.json').write_text(json.dumps(log, indent=1))
    print(log)


if __name__ == '__main__':
    main()
