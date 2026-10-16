"""Extract DrEval's full published results (Zenodo 23159774, all_results.zip) into compact parquet files.

all_results/true_vs_pred.csv (12 GB) holds the out-of-fold predictions of every model in the DrEval paper:
CTRPv2-trained models evaluated within CTRPv2 ('predictions') and on CCLE, CTRPv1, GDSC1 and GDSC2 without
retraining ('cross-study-*'), for the LPO, LCO, LDO and LTO test modes (plus feature-randomisation runs, skipped).
Output: data/dreval/all_results_parquet/<rand_setting>__<test_mode>.parquet with columns algorithm, pubchem_id,
cell_line_name, cellosaurus_id, CV_split, y_true, y_pred.
"""
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import PROJECT

SRC = PROJECT / 'data' / 'dreval' / 'all_results.zip'
OUT = PROJECT / 'data' / 'dreval' / 'all_results_parquet'
KEEP = ['algorithm', 'pubchem_id', 'cell_line_name', 'cellosaurus_id', 'CV_split', 'y_true', 'y_pred']
SCHEMA = pa.schema([('algorithm', pa.string()), ('pubchem_id', pa.string()), ('cell_line_name', pa.string()),
                    ('cellosaurus_id', pa.string()), ('CV_split', pa.int16()), ('y_true', pa.float32()),
                    ('y_pred', pa.float32())])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['unzip', '-p', str(SRC), 'all_results/true_vs_pred.csv'], stdout=subprocess.PIPE)
    writers, counts = {}, {}
    reader = pd.read_csv(proc.stdout, chunksize=2_000_000, dtype={'pubchem_id': str, 'cell_line_name': str,
                                                                    'cellosaurus_id': str, 'algorithm': str,
                                                                    'rand_setting': str, 'test_mode': str})
    for i, chunk in enumerate(reader):
        chunk = chunk[(chunk.rand_setting == 'predictions') | chunk.rand_setting.str.startswith('cross-study')]
        for (rs, tm), g in chunk.groupby(['rand_setting', 'test_mode']):
            key = f'{rs}__{tm}'
            t = pa.Table.from_pandas(g[KEEP].astype({'CV_split': 'int16', 'y_true': 'float32', 'y_pred': 'float32'}),
                                     schema=SCHEMA, preserve_index=False)
            if key not in writers:
                writers[key] = pq.ParquetWriter(OUT / f'{key}.parquet', SCHEMA, compression='zstd')
            writers[key].write_table(t)
            counts[key] = counts.get(key, 0) + len(g)
        print('chunk', i, flush=True)
    for w in writers.values():
        w.close()
    proc.wait()
    pd.Series(counts).sort_index().to_csv(OUT / 'row_counts.csv', header=['rows'])
    print(pd.Series(counts).sort_index())
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
