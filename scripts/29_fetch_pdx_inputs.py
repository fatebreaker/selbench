"""Fetch only the INPUT members of DrEval's PDX_Bruna.zip (Zenodo 23159774) by HTTP range requests.

The response member (PDX_Bruna/PDX_Bruna.csv) is never requested: this script reads the zip central directory,
then downloads the local header and compressed bytes of the listed input members only, checks each CRC-32 and
writes them to data/raw/pdx_bruna_inputs/. A UTC log of every request is written next to them, so that the
amendment can document that no response bytes were on disk before its freeze. Run once, before Amendment 2.
"""
import datetime
import json
import struct
import sys
import urllib.request
import zipfile
import zlib

from common import PROJECT

RECORD = 'https://zenodo.org/api/records/23159774'
NAME = 'PDX_Bruna.zip'
RESPONSE_MEMBER = 'PDX_Bruna/PDX_Bruna.csv'
INPUTS = ['PDX_Bruna/cell_line_names.csv', 'PDX_Bruna/drug_names.csv', 'PDX_Bruna/drug_smiles.csv',
          'PDX_Bruna/gene_expression.csv', 'PDX_Bruna/copy_number_variation_gistic.csv']
OUT = PROJECT / 'data' / 'raw' / 'pdx_bruna_inputs'


def get(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    assert len(data) == end - start + 1, 'short read'
    return data


class Tail:
    """File-like object exposing only the zip tail (central directory) at its true offsets."""

    def __init__(self, size, tail_bytes):
        self.size, self.tail, self.pos = size, tail_bytes, 0
        self.base = size - len(tail_bytes)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if self.pos < self.base:
            raise IOError('read outside the central directory')
        end = self.size if n < 0 else min(self.size, self.pos + n)
        out = self.tail[self.pos - self.base:end - self.base]
        self.pos = end
        return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(urllib.request.urlopen(RECORD, timeout=120).read())
    f = [x for x in meta['files'] if x['key'] == NAME][0]
    size, url = f['size'], f'{RECORD}/files/{NAME}/content'
    log = [{'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'record': RECORD,
            'record_version_date': meta['metadata'].get('publication_date'), 'zip': NAME, 'zip_size': size,
            'zip_checksum': f['checksum'], 'never_requested': RESPONSE_MEMBER}]
    tail = get(url, size - 2 * 1024 * 1024, size - 1)
    log.append({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'range': [size - 2 * 1024 * 1024, size - 1],
                'what': 'central directory'})
    z = zipfile.ZipFile(Tail(size, tail))
    resp = z.getinfo(RESPONSE_MEMBER)
    # exact span of the response member's compressed bytes, from its 30-byte local header (sizes only, no data)
    rh = get(url, resp.header_offset, resp.header_offset + 29)
    r_name, r_extra = struct.unpack('<HH', rh[26:30])
    r_data0 = resp.header_offset + 30 + r_name + r_extra
    r_data1 = r_data0 + resp.compress_size - 1
    log.append({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'range': [resp.header_offset, resp.header_offset + 29], 'what': 'response member local header only',
                'response_data_span_never_requested': [r_data0, r_data1]})
    for name in INPUTS:
        assert name != RESPONSE_MEMBER
        info = z.getinfo(name)
        hdr = get(url, info.header_offset, info.header_offset + 29)
        n_name, n_extra = struct.unpack('<HH', hdr[26:30])
        start = info.header_offset + 30 + n_name + n_extra
        end = start + info.compress_size - 1
        # no requested range may touch the response member's data bytes
        assert end < r_data0 or info.header_offset > r_data1, f'{name} overlaps the response member'
        raw = get(url, start, end)
        data = zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw
        assert zlib.crc32(data) & 0xffffffff == info.CRC, f'CRC mismatch for {name}'
        (OUT / name.split('/')[-1]).write_bytes(data)
        log.append({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'member': name,
                    'range': [info.header_offset, end], 'bytes': len(data), 'crc32': f'{info.CRC:08x}'})
        print(name, len(data), 'ok', flush=True)
    (OUT / 'fetch_log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    sys.exit(main())
