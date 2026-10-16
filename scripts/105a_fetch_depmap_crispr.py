"""Round 5 (second domain): fetch the public DepMap 24Q4 CRISPR gene-effect inputs for the gene-dependency benchmark.

What: downloads, from the public DepMap 24Q4 release (figshare article 27993248), the four files that the CRISPR
gene-dependency analysis (scripts/105b_crispr_dependency_benchmark.py) needs and that are not yet in
data/raw/depmap24q4/:
  CRISPRGeneEffect.csv                 Chronos gene effects (cell line x gene; more negative = more dependent)
  CRISPRInferredCommonEssentials.csv   DepMap's inferred common-essential genes (excluded from the benchmark)
  AchillesNonessentialControls.csv     nonessential control genes (excluded)
  Model.csv                            cell-line annotation (OncotreeLineage)
and, for the axis analysis of 105b (what the general-dependency axis is), three small screen-quality files:
  CRISPRInferredModelEfficacy.csv      Chronos-inferred screen efficacy per model and library
  AchillesScreenQCReport.csv           per-screen QC metrics (e.g. separation of common essentials from nonessentials)
  CRISPRScreenMap.csv                  ScreenID -> ModelID
File ids are looked up in the article's public file listing (api.figshare.com/v2/articles/27993248/files); each file
is streamed from api.figshare.com/v2/file/download/<id> to disk while its SHA-256 is computed, and checked against
the listing's size (and MD5 when the listing gives one). No credentials, personal information or custom headers
beyond a generic User-Agent are sent. Files already present are not re-downloaded or overwritten: they are hashed
and logged as 'already_present'.
Inputs: network access to api.figshare.com.
Outputs: data/raw/depmap24q4/{CRISPRGeneEffect.csv, CRISPRInferredCommonEssentials.csv,
AchillesNonessentialControls.csv, Model.csv, CRISPRInferredModelEfficacy.csv, AchillesScreenQCReport.csv,
CRISPRScreenMap.csv}; data/raw/depmap24q4/fetch_log_105a_crispr.json: one entry per run under 'runs' (earlier runs
are kept), each with the article id, start/end UTC times and per file: name, figshare file id, size, sha256, md5
check, request and completion UTC times (or 'already_present' with the hash time).
Modeled on scripts/69_fetch_pdm_inputs.py.
Usage: python scripts/105a_fetch_depmap_crispr.py
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone

from common import OWN_RAW

ARTICLE = 27993248
NAMES = ['CRISPRGeneEffect.csv', 'CRISPRInferredCommonEssentials.csv', 'AchillesNonessentialControls.csv',
         'Model.csv', 'CRISPRInferredModelEfficacy.csv', 'AchillesScreenQCReport.csv', 'CRISPRScreenMap.csv']
OUT = OWN_RAW / 'depmap24q4'
LOG = OUT / 'fetch_log_105a_crispr.json'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def stream(url, path):
    """Stream url to path; returns (size, sha256, md5)."""
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    tmp = path.with_name(path.name + '.part')
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, 'wb') as f:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            f.write(chunk)
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    tmp.rename(path)
    return n, sha.hexdigest(), md5.hexdigest()


def hash_file(path):
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return n, sha.hexdigest(), md5.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    files = json.loads(get(f'https://api.figshare.com/v2/articles/{ARTICLE}/files?page=1&page_size=500'))
    by_name = {f['name']: f for f in files}
    log = {'article': ARTICLE, 'listing_url': f'https://api.figshare.com/v2/articles/{ARTICLE}/files?page=1&page_size=500',
           'listing_files': len(files), 'started_utc': started, 'files': []}
    for name in NAMES:
        f = by_name.get(name)
        if f is None:
            log['files'].append({'name': name, 'status': 'absent_from_listing'})
            continue
        path = OUT / name
        entry = {'name': name, 'figshare_file_id': f['id'], 'listing_size': f.get('size'),
                 'listing_md5': f.get('computed_md5') or f.get('supplied_md5'),
                 'url': f"https://api.figshare.com/v2/file/download/{f['id']}"}
        if path.exists():
            n, sha, md5 = hash_file(path)
            entry.update({'status': 'already_present', 'size': n, 'sha256': sha, 'md5': md5,
                          'hashed_utc': datetime.now(timezone.utc).isoformat()})
        else:
            entry['requested_utc'] = datetime.now(timezone.utc).isoformat()
            n, sha, md5 = stream(entry['url'], path)
            entry.update({'status': 'downloaded', 'size': n, 'sha256': sha, 'md5': md5,
                          'completed_utc': datetime.now(timezone.utc).isoformat()})
            time.sleep(1)
        entry['size_matches_listing'] = (entry['listing_size'] is None) or (int(entry['listing_size']) == entry['size'])
        entry['md5_matches_listing'] = (entry['listing_md5'] is None) or (entry['listing_md5'] == entry['md5'])
        log['files'].append(entry)
        print(name, entry['status'], entry['size'], entry['sha256'][:16], entry['size_matches_listing'],
              entry['md5_matches_listing'], flush=True)
    log['completed_utc'] = datetime.now(timezone.utc).isoformat()
    runs = []
    if LOG.exists():
        old = json.loads(LOG.read_text())
        runs = old['runs'] if 'runs' in old else [old]
    LOG.write_text(json.dumps({'runs': runs + [log]}, indent=1) + '\n')


if __name__ == '__main__':
    main()
