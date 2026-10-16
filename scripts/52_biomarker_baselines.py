"""Revision analysis (R2 Major 7, R1 M8): canonical single-feature biomarker baselines. EXPLORATORY, not confirmatory.

The drug -> biomarker list below was fixed from the established pharmacogenomic literature before any biomarker
score was computed in this project. Every listed pair that maps to a discovery compound is reported; none is dropped
on its result. Each biomarker is a training-free, cell-blind predictor f = sign * feature (expression in log TPM,
mutation status, copy number or fusion status from DepMap 24Q4 basal data).

1. c^2 with the same triad estimator as the paper (49_ceiling_estimands: per-drug sum of bias-corrected numerators
   over sum of denominators) on the evaluation pairs used for the selective ridge (matched terms: same drug, training
   screen T, evaluation pair E1,E2; for a training-free predictor T only fixes the pairs and weights), compared drug
   by drug with ridge_sel (cell-bootstrap intervals, paired), and on all evaluation pairs (drugs in >= 2 screens).
   Within-screen selective associations pc(f, z_E | g_E) are reported for each discovery screen.
2. Selective transfer to PDX_Bruna (breast PDX-derived models; expression and GISTIC copy number inputs) and BeatAML
   (primary AML; RNA-seq and WES inputs): partial correlation of the biomarker with the drug's profile given the
   cohort's leave-drug-out fitness axis (frozen 07_confirm.build; MIN_CELLS = 20 for PDX as in 31_pdx_confirm, 30 for
   BeatAML), next to the frozen cell-line model's transfer for the same drug (results/confirmation_*/scores.csv,
   family 'sel'). PDX intervals resample tumours (37 models from 20 tumours); a tumour-level estimate averages each
   tumour's models. BeatAML intervals resample patients.
Outputs: results/revision_ceiling/52_*.csv and 52_summary.json.
"""
import importlib.util
import itertools
import os
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OWN_RAW, PROCESSED, RESULTS, norm_name, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ce49', HERE / '49_ceiling_estimands.py')
ce = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ce)
_spec = importlib.util.spec_from_file_location('confirm', HERE / '07_confirm.py')
confirm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(confirm)

OUT = Path(os.environ.get('REVISION_OUT', RESULTS / 'revision_ceiling'))   # override only for test runs
FEAT = PROCESSED / 'features'
DEC = RESULTS / 'decomposition'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
N_CELL_BOOT = int(os.environ.get('REVISION_N_CELL_BOOT', 400))
N_BOOT = int(os.environ.get('REVISION_N_BOOT', 2000))
SEED = 5201
MIN_PAIRS_TRANSFER = 15
RAS = ['KRAS', 'NRAS', 'HRAS']

# ---------------------------------------------------------------------------------------------------------------
# Curated a priori: (biomarker, kind, genes, sign, drug names, tier, rationale/reference). sign = +1: high feature
# (or mutation / wild type as named) -> more sensitive. kind: expr | wt (no hotspot or damaging mutation) |
# hotspot_any (hotspot mutation in any listed gene) | cn | fusion.
# ---------------------------------------------------------------------------------------------------------------
CATALOG = [
    ('SLFN11 expression', 'expr', ['SLFN11'], +1,
     ['topotecan', 'irinotecan', 'camptothecin', 'SN-38', 'exatecan'], 'core',
     'TOP1 poisons; Zoppoli et al. 2012 PNAS; Barretina et al. 2012 Nature'),
    ('SLFN11 expression', 'expr', ['SLFN11'], +1,
     ['talazoparib', 'olaparib', 'niraparib', 'rucaparib', 'veliparib'], 'core',
     'PARP inhibitors; Murai et al. 2016 Oncotarget; Lok et al. 2017 Clin Cancer Res'),
    ('NQO1 expression', 'expr', ['NQO1'], +1, ['17-AAG', 'tanespimycin', 'alvespimycin'], 'core',
     'NQO1 reduces benzoquinone ansamycins; Kelland et al. 1999 JNCI; Guo et al. 2005 Cancer Res'),
    ('TP53 wild type', 'wt', ['TP53'], +1,
     ['nutlin-3', 'nutlin-3a', 'idasanutlin', 'AMG-232', 'CGM097', 'HDM201', 'RG7112', 'SAR405838'], 'core',
     'MDM2 antagonists need wild-type p53; Vassilev et al. 2004 Science'),
    ('BRAF hotspot', 'hotspot_any', ['BRAF'], +1,
     ['PLX4720', 'vemurafenib', 'dabrafenib', 'SB590885', 'encorafenib', 'AZ628', 'GDC-0879', 'RAF265'], 'core',
     'BRAF V600 inhibitors; Tsai et al. 2008 PNAS; Garnett et al. 2012 Nature'),
    ('BRAF or RAS hotspot', 'hotspot_any', ['BRAF'] + RAS, +1,
     ['trametinib', 'selumetinib', 'PD-0325901', 'cobimetinib', 'refametinib', 'binimetinib', 'CI-1040', 'pimasertib'],
     'core', 'MEK inhibitors in MAPK-activated lines; Solit et al. 2006 Nature; Barretina et al. 2012 Nature'),
    ('ERBB2 expression', 'expr', ['ERBB2'], +1,
     ['lapatinib', 'afatinib', 'neratinib', 'CP-724714', 'canertinib', 'tucatinib', 'sapitinib'], 'core',
     'HER2-amplified lines; Konecny et al. 2006 Cancer Res'),
    ('EGFR expression', 'expr', ['EGFR'], +1, ['erlotinib', 'gefitinib', 'afatinib', 'osimertinib'], 'core',
     'EGFR-dependent lines; Lynch et al. 2004 NEJM; Paez et al. 2004 Science (mutation); expression as continuous marker'),
    ('EGFR hotspot', 'hotspot_any', ['EGFR'], +1, ['erlotinib', 'gefitinib', 'afatinib', 'osimertinib'], 'core',
     'Activating EGFR mutations; Lynch et al. 2004 NEJM; Paez et al. 2004 Science'),
    ('BCL2 expression', 'expr', ['BCL2'], +1, ['venetoclax', 'navitoclax', 'ABT-737', 'obatoclax'], 'core',
     'BH3 mimetics; Oltersdorf et al. 2005 Nature; Souers et al. 2013 Nat Med'),
    ('MCL1 expression (resistance)', 'expr', ['MCL1'], -1, ['venetoclax', 'navitoclax', 'ABT-737'], 'core',
     'MCL1 confers resistance to BCL2/BCL-XL inhibitors; van Delft et al. 2006 Cancer Cell'),
    ('BCL2L1 expression (resistance)', 'expr', ['BCL2L1'], -1, ['venetoclax'], 'core',
     'BCL-XL confers resistance to BCL2-selective inhibition; Souers et al. 2013 Nat Med'),
    ('ABCB1 expression (resistance)', 'expr', ['ABCB1'], -1,
     ['paclitaxel', 'docetaxel', 'vinblastine', 'vincristine', 'vinorelbine', 'doxorubicin', 'daunorubicin',
      'epirubicin', 'etoposide', 'dactinomycin', 'colchicine'], 'core',
     'P-glycoprotein substrates; Gottesman et al. 2002 Nat Rev Cancer; Szakacs et al. 2004 Cancer Cell'),
    ('SLC35F2 expression', 'expr', ['SLC35F2'], +1, ['YM-155', 'sepantronium'], 'extended',
     'SLC35F2 imports YM155; Winter et al. 2014 Nat Chem Biol'),
    ('RB1 expression', 'expr', ['RB1'], +1, ['palbociclib', 'ribociclib', 'abemaciclib'], 'extended',
     'RB1 loss confers resistance to CDK4/6 inhibition; Fry et al. 2004 Mol Cancer Ther'),
    ('BCR-ABL1 fusion', 'fusion', ['BCR--ABL1'], +1, ['imatinib', 'nilotinib', 'dasatinib', 'bosutinib', 'ponatinib'],
     'extended', 'Druker et al. 1996 Nat Med'),
    ('ALK expression', 'expr', ['ALK'], +1, ['crizotinib', 'alectinib', 'NVP-TAE684', 'ceritinib'], 'extended',
     'ALK fusions and mutations; McDermott et al. 2008 Cancer Res'),
    ('MET expression', 'expr', ['MET'], +1, ['crizotinib', 'PHA-665752', 'SU11274', 'capmatinib', 'tepotinib'], 'extended',
     'MET-amplified lines; Smolen et al. 2006 PNAS'),
    ('DCK expression', 'expr', ['DCK'], +1, ['cytarabine', 'gemcitabine', 'decitabine'], 'extended',
     'Deoxycytidine kinase activates nucleoside analogues'),
    ('MGMT expression (resistance)', 'expr', ['MGMT'], -1, ['temozolomide'], 'extended',
     'MGMT repairs O6-methylguanine; Hegi et al. 2005 NEJM'),
    ('PIK3CA hotspot', 'hotspot_any', ['PIK3CA'], +1, ['alpelisib', 'BYL719', 'taselisib'], 'extended',
     'PI3K-alpha selective inhibitors; Fritsch et al. 2014 Mol Cancer Ther'),
    ('ESR1 expression', 'expr', ['ESR1'], +1, ['tamoxifen', 'fulvestrant'], 'extended', 'ER-positive breast lines'),
    ('FLT3 expression', 'expr', ['FLT3'], +1, ['quizartinib', 'gilteritinib', 'crenolanib', 'midostaurin'], 'extended',
     'FLT3-driven AML; Smith et al. 2012 Nature (quizartinib)'),
]


def catalog_frame():
    comp = pd.read_csv(PROCESSED / 'compounds.csv')
    idx = {}
    for r in comp.itertuples():
        for n in str(r.all_names).split('|') + [str(r.name)]:
            idx.setdefault(norm_name(n), set()).add(r.compound)
    rows = []
    for bm, kind, genes, sign, drugs, tier, ref in CATALOG:
        for dn in drugs:
            for c in sorted(idx.get(norm_name(dn), set())):
                rows.append({'biomarker': bm, 'kind': kind, 'genes': '|'.join(genes), 'sign': sign, 'drug_name': dn,
                             'compound': c, 'tier': tier, 'reference': ref})
    cat = pd.DataFrame(rows).drop_duplicates(['biomarker', 'compound'])
    # the first-listed biomarker of a compound is its pre-specified primary biomarker
    cat['primary_for_drug'] = ~cat.duplicated('compound')
    return cat


def cell_features(cat):
    expr = pd.read_parquet(FEAT / 'expression_all.parquet')
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    hot = pd.read_csv(OWN_RAW / 'depmap24q4' / 'OmicsSomaticMutationsMatrixHotspot.csv', index_col=0)
    hot.columns = [c.split(' (')[0] for c in hot.columns]
    hot = hot.loc[:, ~hot.columns.duplicated()]
    fus = pd.read_csv(OWN_RAW / 'depmap24q4' / 'OmicsFusionFiltered.csv', usecols=['ModelID', 'FusionName'])
    profiled_fusion = set(expr.index)          # fusion calls come from RNA-seq: lines with expression profiles
    feats = {}
    for r in cat.drop_duplicates('biomarker').itertuples():
        genes = r.genes.split('|')
        if r.kind == 'expr':
            v = expr[genes[0]]
        elif r.kind == 'wt':
            v = 1.0 - mut[genes[0]]
        elif r.kind == 'hotspot_any':
            v = (hot.reindex(columns=genes).fillna(0) > 0).any(axis=1).astype(float)
        elif r.kind == 'fusion':
            pos = set(fus.loc[fus.FusionName.isin(genes), 'ModelID'])
            v = pd.Series([1.0 if m in pos else 0.0 for m in sorted(profiled_fusion)], index=sorted(profiled_fusion))
        else:
            raise ValueError(r.kind)
        feats[r.biomarker] = r.sign * v.astype(float)
    return feats


# ---------------------------------------------------------------------------------------------------------------
def c2_terms(f_all, d, A, member, T_list=None):
    """Terms for predictor vector f (global cell order) and drug d: (T, E1, E2) as for the ridge if T_list is given
    (matched), otherwise every evaluation pair once."""
    present = [s for s in SCREENS if d in A[s][0]]
    if T_list is None:
        configs = [('none', E1, E2) for E1, E2 in itertools.combinations(present, 2)]
    else:
        configs = [(T, E1, E2) for T in T_list if T in present
                   for E1, E2 in itertools.combinations([s for s in present if s != T], 2)]
    return configs


def eval_term(f_all, d, E1, E2, A, member, W):
    z1, g1, r1 = (A[E1][k][d] for k in range(3))
    z2, g2, r2 = (A[E2][k][d] for k in range(3))
    ok = member[E1] & member[E2] & np.isfinite(f_all) & np.isfinite(z1) & np.isfinite(z2) & np.isfinite(g1) & np.isfinite(g2)
    if ok.sum() < ce.MIN_COMMON:
        return None
    X = np.c_[f_all[ok], z1[ok], g1[ok], z2[ok], g2[ok], r1[ok], r2[ok]]
    if np.std(X[:, 0]) < 1e-12:
        return None
    p1, p2, r12, n = ce.dis_term(X)
    bp1, bp2, br12, bn = ce.dis_term(X, W[:, ok])
    return (p1[0], p2[0], r12[0], n[0]), (bp1, bp2, br12, bn)


def drug_c2(term_list):
    """Per-drug bias-corrected ratio of sums and per-drug RoM, with cell-bootstrap draws."""
    if not term_list:
        return None
    pt = np.array([t[0] for t in term_list])          # k x 4
    bt = np.array([t[1] for t in term_list])          # k x 4 x B
    num = ce.bias_corrected_product(pt[:, 0], pt[:, 1], pt[:, 2], pt[:, 3])
    est = num.sum() / pt[:, 2].sum()
    rom = np.median(pt[:, 0] * pt[:, 1]) / np.median(pt[:, 2])
    bnum = ce.bias_corrected_product(bt[:, 0], bt[:, 1], bt[:, 2], bt[:, 3])
    with np.errstate(invalid='ignore', divide='ignore'):
        best = np.nansum(bnum, 0) / np.nansum(bt[:, 2], 0)
    return est, rom, best, len(term_list), num.sum(), pt[:, 2].sum()


def within_screen(f_all, d, A, member):
    out = {}
    for s in SCREENS:
        if d not in A[s][0]:
            continue
        z, g = A[s][0][d], A[s][1][d]
        ok = member[s] & np.isfinite(f_all) & np.isfinite(z) & np.isfinite(g)
        if ok.sum() < ce.MIN_COMMON or np.std(f_all[ok]) < 1e-12:
            continue
        C, _ = ce.wcov(np.c_[f_all[ok], z[ok], g[ok]], [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
        out[s] = float(ce.pcorr_from(C, 0, 1, 2)[0])
    return out


# ---------------------------------------------------------------------------------------------------------------
def partial(x, y, g, w=None):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g)
    if m.sum() < MIN_PAIRS_TRANSFER or np.std(x[m]) < 1e-12:
        return np.nan, int(m.sum())
    C, _ = ce.wcov(np.c_[x[m], y[m], g[m]], [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)], None if w is None else w[:, m])
    return ce.pcorr_from(C, 0, 1, 2), int(m.sum())


def cluster_weights(clusters, n_boot, rng):
    """Bootstrap weights over samples that resample clusters (tumours or patients)."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    W = np.zeros((n_boot, len(codes)))
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        W[b] = cnt[codes]
    return W


def transfer_rows(cohort, Z, Gl, feats, cat, frozen, clusters, rng, cluster_level=None, frozen_pred=None):
    """frozen: compound -> stored frozen-model transfer; frozen_pred: frozen prediction matrix (recomputed under the
    same cluster-bootstrap draws for a paired difference)."""
    rows = []
    W = cluster_weights(clusters, N_BOOT, rng)
    for r in cat.itertuples():
        if r.compound not in Z.index or r.biomarker not in feats:
            continue
        f = feats[r.biomarker].reindex(Z.columns).values.astype(float)
        z, g = Z.loc[r.compound].values, Gl.loc[r.compound].values
        est, n = partial(f, z, g)
        if not np.isfinite(np.atleast_1d(est)[0]):
            continue
        b, _ = partial(f, z, g, W)
        row = {'cohort': cohort, 'biomarker': r.biomarker, 'compound': r.compound, 'drug_name': r.drug_name,
               'tier': r.tier, 'primary_for_drug': r.primary_for_drug, 'n': n,
               'n_positive': int(np.nansum(f[np.isfinite(z)] > 0)) if r.kind != 'expr' else np.nan,
               'biomarker_sel': float(np.atleast_1d(est)[0]), 'ci_lo': ce.ci(b)[0], 'ci_hi': ce.ci(b)[1],
               'frozen_model_sel': frozen.get(r.compound, np.nan)}
        if frozen_pred is not None and r.compound in frozen_pred.index:
            p = frozen_pred.loc[r.compound].reindex(Z.columns).values.astype(float)
            pe, _ = partial(p, z, g)
            pb, _ = partial(p, z, g, W)
            if np.isfinite(np.atleast_1d(pe)[0]):
                row['frozen_model_sel_recomputed'] = float(np.atleast_1d(pe)[0])
                row['frozen_ci_lo'], row['frozen_ci_hi'] = ce.ci(pb)
                row['diff_biomarker_minus_model'] = row['biomarker_sel'] - row['frozen_model_sel_recomputed']
                row['diff_ci_lo'], row['diff_ci_hi'] = ce.ci(b - pb)
        if cluster_level is not None:
            row['biomarker_sel_cluster_level'] = cluster_level(f, r.compound)
        rows.append(row)
    return pd.DataFrame(rows)


def pdx_transfer(cat, rng):
    confirm.MIN_CELLS = 20
    long = pd.read_csv(PROCESSED / 'pdx_bruna' / 'pdx_long.csv.gz').rename(columns={'sample': 'depmap_id'})
    M, Z, Gl, R = confirm.build(long)
    expr = pd.read_csv(OWN_RAW / 'pdx_bruna_inputs' / 'gene_expression.csv', index_col=0)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    cn = pd.read_csv(OWN_RAW / 'pdx_bruna_inputs' / 'copy_number_variation_gistic.csv', index_col=0)
    feats = {}
    for r in cat.drop_duplicates('biomarker').itertuples():
        g = r.genes.split('|')[0]
        if r.kind == 'expr' and g in expr.columns:
            feats[r.biomarker] = r.sign * expr[g].astype(float)
    if 'ERBB2' in cn.columns:          # secondary: GISTIC copy number of ERBB2 (amplification)
        feats['ERBB2 copy number (GISTIC)'] = cn['ERBB2'].astype(float)
    cat2 = pd.concat([cat, cat[cat.biomarker == 'ERBB2 expression'].assign(biomarker='ERBB2 copy number (GISTIC)', kind='cn',
                                                                          primary_for_drug=False)])
    sc = pd.read_csv(RESULTS / 'confirmation_PDX_Bruna' / 'scores.csv')
    frozen = sc[sc.family == 'sel'].set_index('compound').sel.to_dict()
    tumour = pd.Series([c.split('-')[0] for c in Z.columns], index=Z.columns)

    def tumour_level(f, d):
        df = pd.DataFrame({'f': f, 'z': Z.loc[d].values, 'g': Gl.loc[d].values, 't': tumour.values}).dropna()
        a = df.groupby('t').mean()
        if len(a) < 8 or a.f.std() < 1e-12:
            return np.nan
        C, _ = ce.wcov(a[['f', 'z', 'g']].values, [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
        return float(ce.pcorr_from(C, 0, 1, 2)[0])
    fp = pd.read_parquet(RESULTS / 'pdx_transfer' / 'pred_sel.parquet')
    out = transfer_rows('PDX_Bruna', Z, Gl, feats, cat2, frozen, tumour.values, rng, tumour_level, fp)
    out['n_tumours'] = [len(set(tumour[Z.loc[d].dropna().index])) for d in out.compound]
    return out


def beataml_transfer(cat, rng):
    long = pd.read_csv(PROCESSED / 'beataml' / 'beataml_long.csv.gz').rename(columns={'sample': 'depmap_id'})
    confirm.MIN_CELLS = 30
    M, Z, Gl, R = confirm.build(long)
    ex = pd.read_csv(OWN_RAW / 'beataml_inputs' / 'beataml_waves1to4_norm_exp_dbgap.txt', sep='\t')
    ex = ex.drop(columns=['stable_id', 'description', 'biotype']).drop_duplicates('display_label').set_index('display_label').T
    samples = pd.read_csv(RESULTS / 'beataml_transfer' / 'samples.csv').drop_duplicates('dbgap_rnaseq_sample')
    rna2dna = samples.dropna(subset=['dbgap_dnaseq_sample']).set_index('dbgap_rnaseq_sample').dbgap_dnaseq_sample
    rna2patient = samples.set_index('dbgap_rnaseq_sample').patientId
    mut = pd.read_csv(OWN_RAW / 'beataml_inputs' / 'beataml_wes_wv1to4_mutations_dbgap.txt', sep='\t',
                      usecols=['dbgap_sample_id', 'variant_classification', 'hgvsp_short', 'symbol'])
    prot = {'missense_variant', 'frameshift_variant', 'stop_gained', 'inframe_deletion', 'inframe_insertion',
            'splice_acceptor_variant', 'splice_donor_variant', 'protein_altering_variant', 'start_lost', 'stop_lost'}
    mut = mut[mut.variant_classification.str.split('&').apply(lambda v: any(x in prot for x in v))]
    dna_samples = set(rna2dna.values)

    def status(pred):
        hit = set(mut.loc[pred(mut), 'dbgap_sample_id'])
        return pd.Series({r: (1.0 if d in hit else 0.0) for r, d in rna2dna.items() if d in dna_samples})
    hotspot = {
        'BRAF': lambda m: (m.symbol == 'BRAF') & m.hgvsp_short.astype(str).str.startswith('p.V600'),
        'KRAS': lambda m: (m.symbol == 'KRAS') & m.hgvsp_short.astype(str).str.match(r'p\.(G12|G13|Q61)'),
        'NRAS': lambda m: (m.symbol == 'NRAS') & m.hgvsp_short.astype(str).str.match(r'p\.(G12|G13|Q61)'),
        'HRAS': lambda m: (m.symbol == 'HRAS') & m.hgvsp_short.astype(str).str.match(r'p\.(G12|G13|Q61)'),
        'EGFR': lambda m: (m.symbol == 'EGFR') & m.hgvsp_short.astype(str).str.match(r'p\.(L858|G719|T790|E746|L747|S768|L861)'),
        'PIK3CA': lambda m: (m.symbol == 'PIK3CA') & m.hgvsp_short.astype(str).str.match(r'p\.(E542|E545|H1047)'),
    }
    feats = {}
    for r in cat.drop_duplicates('biomarker').itertuples():
        genes = r.genes.split('|')
        if r.kind == 'expr' and genes[0] in ex.columns:
            feats[r.biomarker] = r.sign * ex[genes[0]].astype(float)
        elif r.kind == 'wt':
            feats[r.biomarker] = r.sign * (1.0 - status(lambda m, g=genes[0]: m.symbol == g))
        elif r.kind == 'hotspot_any' and all(g in hotspot for g in genes):
            st = None
            for g in genes:
                s = status(hotspot[g])
                st = s if st is None else np.maximum(st, s)
            feats[r.biomarker] = r.sign * st
    feats['FLT3 mutation (WES)'] = status(lambda m: m.symbol == 'FLT3')
    cat2 = pd.concat([cat, cat[cat.biomarker == 'FLT3 expression'].assign(biomarker='FLT3 mutation (WES)', kind='mut',
                                                                         primary_for_drug=False)])
    sc = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    frozen = sc[sc.family == 'sel'].set_index('compound').sel.to_dict()
    patients = rna2patient.reindex(Z.columns).fillna(pd.Series(Z.columns, index=Z.columns)).values
    fp = pd.read_parquet(RESULTS / 'beataml_transfer' / 'pred_sel.parquet')
    return transfer_rows('BeatAML', Z, Gl, feats, cat2, frozen, patients, rng, None, fp)


# ---------------------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cat = catalog_frame()
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    G = {s: ce.loo_fitness(Z[s]) for s in SCREENS}
    cells = np.array(sorted(set().union(*[set(Z[s].columns) for s in SCREENS])))
    member = {s: np.isin(cells, np.asarray(Z[s].columns)) for s in SCREENS}
    A = {s: (ce.Aligned(Z[s], cells), ce.Aligned(G[s], cells), ce.Aligned(R[s], cells)) for s in SCREENS}
    W = ce.cell_weights(len(cells), N_CELL_BOOT, np.random.default_rng(SEED + 1))
    pairs = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    rep_sel = pairs.groupby('compound').rho_sel.mean()
    cat['n_screens'] = [sum(c in Z[s].index for s in SCREENS) for c in cat.compound]
    cat['rep_class'] = pd.cut(cat.compound.map(rep_sel), ce.CLASS_BINS, labels=ce.CLASS_LABELS).astype(str)
    cat.to_csv(OUT / '52_biomarker_catalog.csv', index=False)
    feats = cell_features(cat)
    ridge = {T: ce.Aligned(pd.read_parquet(RESULTS / 'models' / f'pred_ridge_sel_{T}.parquet'), cells) for T in SCREENS}
    rep = {T: ce.Aligned(pd.read_parquet(RESULTS / 'models' / f'pred_replicate_{T}.parquet'), cells) for T in SCREENS}

    rows = []
    for r in cat[cat.n_screens >= 2].itertuples():
        d = r.compound
        f = feats[r.biomarker].reindex(cells).values.astype(float)
        row = {'biomarker': r.biomarker, 'compound': d, 'drug_name': r.drug_name, 'tier': r.tier,
               'primary_for_drug': r.primary_for_drug, 'n_screens': r.n_screens, 'rep_class': r.rep_class,
               'n_cells_feature': int(np.isfinite(f).sum())}
        if r.kind != 'expr':
            row['n_positive_cells'] = int(np.nansum(f[np.isfinite(f)] > 0)) if r.sign > 0 else int(np.nansum(f[np.isfinite(f)] < 0))
        for s, v in within_screen(f, d, A, member).items():
            row[f'pc_{s}'] = v
        # all evaluation pairs (training-free predictor)
        tl = [x for x in (eval_term(f, d, E1, E2, A, member, W) for _, E1, E2 in c2_terms(f, d, A, member)) if x is not None]
        res = drug_c2(tl)
        if res:
            row.update({'c2_biomarker_allpairs': res[0], 'c2_biomarker_allpairs_ci_lo': ce.ci(res[2])[0],
                        'c2_biomarker_allpairs_ci_hi': ce.ci(res[2])[1], 'n_pairs': res[3]})
        # matched to the ridge's terms (drugs in >= 3 screens)
        if r.n_screens >= 3:
            tb, trd, trp = [], [], []
            for T, E1, E2 in c2_terms(f, d, A, member, T_list=SCREENS):
                x = eval_term(f, d, E1, E2, A, member, W)
                y = eval_term(ridge[T][d], d, E1, E2, A, member, W) if d in ridge[T] else None
                zr = eval_term(rep[T][d], d, E1, E2, A, member, W) if d in rep[T] else None
                if x is not None and y is not None:
                    tb.append(x); trd.append(y)
                    if zr is not None:
                        trp.append(zr)
            rb, rr, rp = drug_c2(tb), drug_c2(trd), drug_c2(trp)
            if rb and rr:
                diff = rb[2] - rr[2]
                row.update({'c2_biomarker': rb[0], 'c2_biomarker_ci_lo': ce.ci(rb[2])[0], 'c2_biomarker_ci_hi': ce.ci(rb[2])[1],
                            'c2_biomarker_rom': rb[1], 'c2_ridge_sel': rr[0], 'c2_ridge_ci_lo': ce.ci(rr[2])[0],
                            'c2_ridge_ci_hi': ce.ci(rr[2])[1], 'c2_ridge_rom': rr[1], 'diff_biomarker_minus_ridge': rb[0] - rr[0],
                            'diff_ci_lo': ce.ci(diff)[0], 'diff_ci_hi': ce.ci(diff)[1], 'n_terms': rb[3],
                            'num_biomarker': rb[4], 'num_ridge': rr[4], 'den_sum': rb[5],
                            'mean_r12': rb[5] / rb[3], 'c2_replicate': rp[0] if rp else np.nan,
                            'num_replicate': rp[4] if rp else np.nan, 'den_replicate': rp[5] if rp else np.nan,
                            'n_terms_replicate': rp[3] if rp else np.nan,
                            # per-drug ratios are unstable when the evaluation screens share little selectivity
                            'estimable': bool(rb[5] / rb[3] > 0.1 and (ce.ci(diff)[1] - ce.ci(diff)[0]) < 1.0)})
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / '52_biomarker_c2_perdrug.csv', index=False)
    print(per[['biomarker', 'drug_name', 'n_screens', 'rep_class', 'c2_biomarker_allpairs', 'c2_biomarker', 'c2_ridge_sel',
               'diff_ci_lo', 'diff_ci_hi', 'c2_biomarker_rom', 'c2_ridge_rom']].round(3).to_string(), flush=True)

    # pooled comparison over drugs (primary biomarker per drug; per-drug ratio of means; drug-cluster bootstrap)
    pool_rows = []
    for label, sub in [('core, primary biomarker per drug', per[(per.tier == 'core') & per.primary_for_drug]),
                       ('all tiers, primary biomarker per drug', per[per.primary_for_drug]),
                       ('core, reproducible drugs', per[(per.tier == 'core') & per.primary_for_drug & (per.rep_class == 'reproducible')])]:
        sub = sub.dropna(subset=['c2_biomarker', 'c2_ridge_sel'])
        if len(sub) < 3:
            continue
        nterm = sub.n_terms.values
        nb, nr, dd = sub.num_biomarker.values / nterm, sub.num_ridge.values / nterm, sub.den_sum.values / nterm
        C = ce.drug_counts(len(sub), N_BOOT, rng)
        eb, er = nb.sum() / dd.sum(), nr.sum() / dd.sum()
        bb, br = (C @ nb) / (C @ dd), (C @ nr) / (C @ dd)
        rs = sub.dropna(subset=['num_replicate'])
        rep_pooled = float((rs.num_replicate / rs.n_terms_replicate).sum() / (rs.den_replicate / rs.n_terms_replicate).sum())
        pool_rows.append({'set': label, 'n_drugs': len(sub), 'c2_replicate_pooled_same_drugs': rep_pooled,
                          'c2_biomarker_pooled': eb, 'ci_lo': ce.ci(bb)[0], 'ci_hi': ce.ci(bb)[1],
                          'c2_ridge_pooled': er, 'ridge_ci_lo': ce.ci(br)[0], 'ridge_ci_hi': ce.ci(br)[1],
                          'diff': eb - er, 'diff_ci_lo': ce.ci(bb - br)[0], 'diff_ci_hi': ce.ci(bb - br)[1],
                          'n_biomarker_higher': int((sub.c2_biomarker > sub.c2_ridge_sel).sum()),
                          'n_biomarker_higher_ci_excludes_0': int((sub.diff_ci_lo > 0).sum()),
                          'n_ridge_higher_ci_excludes_0': int((sub.diff_ci_hi < 0).sum()),
                          'median_c2_biomarker': float(sub.c2_biomarker.median()), 'median_c2_ridge': float(sub.c2_ridge_sel.median())})
    pool = pd.DataFrame(pool_rows)
    pool.to_csv(OUT / '52_biomarker_c2_pooled.csv', index=False)
    print(pool.round(3).to_string(), flush=True)

    # transfer to patient-derived cells
    pdx = pdx_transfer(cat, np.random.default_rng(SEED + 2))
    pdx.to_csv(OUT / '52_biomarker_transfer_pdx.csv', index=False)
    ba = beataml_transfer(cat, np.random.default_rng(SEED + 3))
    ba.to_csv(OUT / '52_biomarker_transfer_beataml.csv', index=False)
    tr_sum = []
    for name, t in [('PDX_Bruna', pdx), ('BeatAML', ba)]:
        for label, sub in [('all listed pairs', t), ('primary biomarker per drug', t[t.primary_for_drug])]:
            sub = sub.dropna(subset=['biomarker_sel'])
            both = sub.dropna(subset=['frozen_model_sel'])
            tr_sum.append({'cohort': name, 'set': label, 'n_pairs': len(sub), 'n_drugs': sub.compound.nunique(),
                           'median_biomarker_sel': float(sub.biomarker_sel.median()) if len(sub) else np.nan,
                           'frac_positive': float((sub.biomarker_sel > 0).mean()) if len(sub) else np.nan,
                           'n_ci_excludes_0_positive': int((sub.ci_lo > 0).sum()),
                           'n_with_frozen': len(both), 'median_frozen_model_sel_same_drugs': float(both.frozen_model_sel.median()) if len(both) else np.nan,
                           'n_biomarker_above_frozen': int((both.biomarker_sel > both.frozen_model_sel).sum()),
                           'n_biomarker_above_frozen_ci_excludes_0': int((both.get('diff_ci_lo', pd.Series(dtype=float)) > 0).sum()),
                           'n_frozen_above_biomarker_ci_excludes_0': int((both.get('diff_ci_hi', pd.Series(dtype=float)) < 0).sum()),
                           'max_abs_frozen_recompute_diff': float((both.frozen_model_sel - both.get('frozen_model_sel_recomputed', both.frozen_model_sel)).abs().max()) if len(both) else np.nan})
    trs = pd.DataFrame(tr_sum)
    trs.to_csv(OUT / '52_biomarker_transfer_summary.csv', index=False)
    print(pdx.round(3).to_string(), '\n', ba.round(3).to_string(), '\n', trs.round(3).to_string(), flush=True)
    write_json(OUT / '52_summary.json', {'seed': SEED, 'n_catalog_pairs': len(cat), 'n_catalog_compounds': int(cat.compound.nunique()),
                                         'pooled': pool.to_dict(orient='records'), 'transfer': trs.to_dict(orient='records'),
                                         'status': 'exploratory; biomarker list fixed a priori; not confirmatory'})


if __name__ == '__main__':
    main()
