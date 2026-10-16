"""Revision analyses of the discovery reproducibility classes (reviewer 2 Major 8 and minor 5; reviewer 3 minor 10).

1. Mechanism-class curation (R2 minor 5). The regex classifier of 12_drug_classes.py assigns the first class whose
   pattern matches ANY target or MOA token, so promiscuous compounds land in the class of a secondary target
   (docetaxel -> BCL2 family because BCL2 is listed; BMS-536924 -> MAPK because MAPK1 is listed). Every one of the
   353 multi-screen drugs is reviewed manually below (CURATION); each change carries a reason and a change type:
     secondary_target_match  regex matched a secondary/off-target token; moved to the class of the primary target
     missed_by_regex         target not covered by the regex (JNK, p38, TGFBR1, NTRK, ROCK, G9a, SIRT, MRE11 ...)
     new_class               moved into a class added by the curation (DNA-damaging agent, Nuclear receptor /
                             hormone, Other oncology (non-kinase))
     split_other             stays in the remainder of the old 'Other / non-oncology' bin, now 'Non-oncology / tool'
     mechanism_revised       the annotated mechanism has been revised in the literature (e.g. RITA, tivantinib)
     judgement               a defensible call on a multi-target compound (e.g. sorafenib -> Other kinase)
   Outputs the curated table, the class summary under the old, curated and conservative (mechanism_revised and
   judgement changes reverted) schemes, and n per class.
2. Activity confounding (R2 Major 8): Spearman of mean selective agreement with the minimum activity fraction
   across screens (fraction of lines with AUC < 0.8) and with the minimum AUC s.d.; class composition by activity
   bin; class medians within activity strata; regression of reproducibility on class plus activity covariates.
3. Cross-laboratory-only classification and per-drug standard errors (R3 minor 10): a joint cell-line bootstrap
   (Poisson weights shared by all screens) of every drug's mean selective cross-screen correlation.

Inputs are read-only: results/drug_classes/drug_table.csv, results/decomposition/{cross_screen_pairs.csv,
drug_screen_stats.csv, resid_*.parquet, z_*.parquet}, data/processed/compounds.csv. Outputs: results/revision_transfer/.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from common import PROCESSED, RESULTS, write_json

OUT = RESULTS / 'revision_transfer'
DEC = RESULTS / 'decomposition'
SEED = 20261015
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15
N_BOOT_CELL = 1000
N_BOOT_DRUG = 5000
LAB = {'PRISM': 'Broad', 'CTRP': 'Broad', 'GDSC1': 'Sanger', 'GDSC2': 'Sanger'}

MAPK, EGFR, MDM2, PI3K = 'MAPK (RAF/MEK/ERK)', 'EGFR/ERBB', 'MDM2–p53', 'PI3K/AKT/mTOR'
BCL2, DDR, TOPO, ANTIMET = 'BCL2 family / IAP', 'DNA damage response', 'Topoisomerase', 'Antimetabolite'
MITO, CDK, EPI, PROT = 'Mitosis / microtubule', 'CDK', 'Epigenetic (HDAC/BET/DNMT)', 'Proteasome / HSP90'
OKIN = 'Other kinase'
DNADMG, NUCREC = 'DNA-damaging agent', 'Nuclear receptor / hormone'
OTHONC, NONONC = 'Other oncology (non-kinase)', 'Non-oncology / tool'
OLD_OTHER = 'Other / non-oncology'

# compound -> (expected name, new class, change type, reason). Drugs not listed keep their class (reason 'confirmed').
CURATION = {
    # ---- Antimetabolite
    'DPC-004226': ('MITOMYCIN-C', DNADMG, 'new_class', 'DNA crosslinker (alkylating); matched "DNA SYNTHESIS" in the MOA text, not an antimetabolite'),
    'DPC-001735': ('CICLOPIROX', NONONC, 'secondary_target_match', 'antifungal iron chelator; RRM1 is an indirect target'),
    # ---- BCL2 family / IAP
    'DPC-005918': ('SEPANTRONIUM BROMIDE', DNADMG, 'mechanism_revised', 'YM155: cytotoxicity is DNA damage after SLC35F2-mediated uptake (Winter 2014); survivin suppression is secondary'),
    'DPC-002313': ('DOCETAXEL', MITO, 'secondary_target_match', 'taxane microtubule stabiliser; BCL2 phosphorylation listed as a target'),
    'DPC-004880': ('PACLITAXEL', MITO, 'secondary_target_match', 'taxane microtubule stabiliser; BCL2 listed as a target'),
    'DPC-006008': ('SNS-032', CDK, 'secondary_target_match', 'CDK2/7/9 inhibitor; MCL1 loss is downstream of CDK9 inhibition'),
    'DPC-001793': ('CIS-DDP', DNADMG, 'secondary_target_match', 'cisplatin, platinum DNA crosslinker; XIAP listed as a target'),
    # ---- CDK
    'DPC-003413': ('INDISULAM', OTHONC, 'mechanism_revised', 'DCAF15-dependent RBM39 degrader (Han 2017; Uehara 2017); the CDK annotation is historical'),
    'DPC-005783': ('SB 216763', OKIN, 'secondary_target_match', 'GSK3A/B inhibitor; CDK2 listed as an off-target'),
    'DPC-005080': ('PF-573228', OKIN, 'secondary_target_match', 'FAK (PTK2) inhibitor; CDKs listed as off-targets'),
    # ---- DNA damage response
    'DPC-001409': ('BX-795', OKIN, 'secondary_target_match', 'PDK1/TBK1/IKKe multi-kinase inhibitor; CHEK1 listed as an off-target'),
    'DPC-001410': ('BX-912', PI3K, 'secondary_target_match', 'PDK1 (PDPK1) inhibitor, the AKT-activating kinase of the PI3K pathway; CHEK1 listed as an off-target'),
    # ---- EGFR/ERBB
    'DPC-002683': ('EXEL-2880', OKIN, 'secondary_target_match', 'foretinib, MET/VEGFR2 inhibitor; EGFR listed among many targets'),
    'DPC-000474': ('ALVOCIDIB', CDK, 'secondary_target_match', 'flavopiridol, pan-CDK (CDK9) inhibitor; EGFR listed as a target'),
    # ---- Epigenetic
    'DPC-003889': ('LOVASTATIN', NONONC, 'secondary_target_match', 'statin (HMGCR inhibitor); HDAC2 listed as a target'),
    'DPC-001977': ('CURCUMIN', NONONC, 'secondary_target_match', 'pleiotropic natural product; DNMT3B listed among many targets'),
    'DPC-006431': ('FEDRATINIB', OKIN, 'secondary_target_match', 'JAK2 inhibitor; BRD4 binding is secondary'),
    'DPC-004929': ('PARTHENOLIDE', NONONC, 'secondary_target_match', 'NF-kB/IKK-inhibiting sesquiterpene lactone; HDAC1 depletion is secondary'),
    # ---- MAPK
    'DPC-001140': ('BMS 536924', OKIN, 'secondary_target_match', 'IGF1R/INSR inhibitor; MAPK1 listed among off-targets'),
    'DPC-001674': ('LADUVIGLUSIB', OKIN, 'secondary_target_match', 'CHIR-99021, GSK3A/B inhibitor; MAPK1 listed as an off-target'),
    'DPC-001198': ('BOSULIF', OKIN, 'secondary_target_match', 'bosutinib, SRC/ABL inhibitor; MAP2K1/2 listed among off-targets'),
    'DPC-001056': ('BHG712', OKIN, 'secondary_target_match', 'EPHB4 inhibitor; RAF listed as an off-target'),
    'DPC-000083': ('284461-73-0', OKIN, 'judgement', 'sorafenib: multikinase VEGFR/PDGFR/KIT/FLT3/RET inhibitor; weak cellular RAF/MEK/ERK inhibitor, not BRAF-mutant selective'),
    'DPC-005513': ('REGORAFENIB', OKIN, 'judgement', 'multikinase VEGFR/TIE2/KIT/RET inhibitor; RAF is one of many targets'),
    # ---- MDM2-p53
    'DPC-005596': ('RITA', DNADMG, 'mechanism_revised', 'p53-independent; bioactivated by SULT1A1 to a DNA crosslinker (Wanzel 2016)'),
    # ---- Mitosis / microtubule
    'DPC-000366': ('AG-13736', OKIN, 'secondary_target_match', 'axitinib, VEGFR1-3 inhibitor; PLK4 listed as an off-target'),
    'DPC-005992': ('SL 0101-1', OKIN, 'secondary_target_match', 'RSK inhibitor; AURKB listed as an off-target'),
    'DPC-003673': ('KW-2449', OKIN, 'judgement', 'developed as a FLT3 inhibitor (FLT3/ABL/Aurora multikinase)'),
    # ---- Other kinase
    'DPC-006521': ('TIVANTINIB', MITO, 'mechanism_revised', 'cytotoxicity is MET-independent microtubule disruption (Katayama 2013; Basilico 2013)'),
    'DPC-000798': ('AZ-3146', MITO, 'judgement', 'MPS1/TTK spindle-checkpoint kinase inhibitor, grouped with Aurora/PLK mitotic kinases'),
    'DPC-006940': ('WP1130', OTHONC, 'secondary_target_match', 'partially selective deubiquitinase inhibitor (USP9X/USP5/USP14/UCHL5); JAK2 listed'),
    # ---- PI3K/AKT/mTOR
    'DPC-001018': ('BERZOSERTIB', DDR, 'secondary_target_match', 'VX-970, ATR inhibitor; MTOR/PIK3CG listed as off-targets'),
    'DPC-001165': ('BMS-754807', OKIN, 'secondary_target_match', 'IGF1R/INSR inhibitor; AKT1 listed as a target'),
    'DPC-005574': ('RIGOSERTIB', MITO, 'secondary_target_match', 'annotated PLK1 inhibitor; acts as a microtubule destabiliser (Jost 2020)'),
    'DPC-000802': ('AZ20', DDR, 'secondary_target_match', 'ATR inhibitor; MTOR listed as an off-target'),
    'DPC-002657': ('ETP-46464', DDR, 'secondary_target_match', 'ATR inhibitor (with mTOR activity)'),
    'DPC-006997': ('YM 201636', OKIN, 'secondary_target_match', 'PIKfyve (class III phosphoinositide kinase, endolysosomal), not PI3K/AKT/mTOR signalling'),
    # ---- Proteasome / HSP90
    'DPC-002456': ('ELESCLOMOL', OTHONC, 'mechanism_revised', 'copper ionophore; cuproptosis via FDX1 (Tsvetkov 2019, 2022); not an HSP90 inhibitor'),
    # ---- Other / non-oncology: kinase, epigenetic, DDR and antimetabolite agents missed by the regex
    'GDSC1:1463': ('AZD1332', OKIN, 'missed_by_regex', 'TRK (NTRK1-3) inhibitor'),
    'DPC-004216': ('MIRIN', DDR, 'missed_by_regex', 'MRE11 nuclease inhibitor (DNA repair)'),
    'DPC-003080': ('GSK 269962A', OKIN, 'missed_by_regex', 'ROCK1/2 inhibitor'),
    'DPC-003570': ('JNK INHIBITOR VIII', OKIN, 'missed_by_regex', 'JNK inhibitor'),
    'DPC-003168': ('GW-441756', OKIN, 'missed_by_regex', 'TrkA (NTRK1) inhibitor'),
    'DPC-001106': ('BIRB 796', OKIN, 'missed_by_regex', 'doramapimod, p38 MAPK inhibitor (not RAF/MEK/ERK)'),
    'DPC-005817': ('SB-525334', OKIN, 'missed_by_regex', 'ALK5 (TGFBR1) kinase inhibitor'),
    'DPC-005981': ('SKEPINONE-L', OKIN, 'missed_by_regex', 'p38 MAPK inhibitor'),
    'DPC-000154': ('5XE', OKIN, 'missed_by_regex', 'TGFBR1 kinase inhibitor'),
    'DPC-006903': ('VX 702', OKIN, 'missed_by_regex', 'p38 MAPK inhibitor'),
    'DPC-005814': ('SB-431542', OKIN, 'missed_by_regex', 'ALK4/5/7 (TGFBR1) kinase inhibitor'),
    'DPC-003641': ('KIN001-201', OKIN, 'missed_by_regex', 'p38 MAPK inhibitor'),
    'DPC-006744': ('UNC 0683', EPI, 'missed_by_regex', 'G9a/GLP (EHMT1/2) histone methyltransferase inhibitor'),
    'DPC-001123': ('BIX-01294', EPI, 'missed_by_regex', 'G9a/GLP (EHMT1/2) histone methyltransferase inhibitor'),
    'DPC-000374': ('AGK-2', EPI, 'missed_by_regex', 'SIRT2 (class III HDAC) inhibitor'),
    'DPC-002678': ('EX 527', EPI, 'missed_by_regex', 'SIRT1 (class III HDAC) inhibitor'),
    'DPC-006383': ('TENOVIN-6', EPI, 'missed_by_regex', 'SIRT1/2 (class III HDAC) inhibitor'),
    'DPC-004401': ('MYCOPHENOLIC-ACID', ANTIMET, 'missed_by_regex', 'IMPDH inhibitor (guanine-nucleotide antimetabolite)'),
    'DPC-000898': ('BAM7', BCL2, 'missed_by_regex', 'direct BAX activator (BCL2 family)'),
    # ---- Other / non-oncology: DNA-damaging chemotherapy
    'DPC-001134': ('BLEOMYCIN A2', DNADMG, 'new_class', 'radiomimetic DNA strand-break inducer'),
    'DPC-002466': ('ELOXATIN', DNADMG, 'new_class', 'oxaliplatin, platinum DNA crosslinker'),
    'DPC-003981': ('M-39831', DNADMG, 'new_class', 'temozolomide (M&B 39831), DNA methylating agent'),
    'DPC-001676': ('CHLORAMBUCIL', DNADMG, 'new_class', 'nitrogen mustard alkylating agent'),
    'DPC-002004': ('CYCLOPHOSPHAMIDE', DNADMG, 'new_class', 'nitrogen mustard prodrug (needs hepatic CYP activation; little activity in vitro)'),
    'DPC-003361': ('IFOSFAMIDE', DNADMG, 'new_class', 'nitrogen mustard prodrug (needs hepatic CYP activation)'),
    'DPC-000976': ('BENDAMUSTINE', DNADMG, 'new_class', 'nitrogen mustard alkylating agent'),
    'DPC-002046': ('DACARBAZINE', DNADMG, 'new_class', 'methylating prodrug (needs hepatic activation)'),
    'DPC-001479': ('CARBOPLATIN', DNADMG, 'new_class', 'platinum DNA crosslinker'),
    # ---- Other / non-oncology: nuclear receptor and hormone agents
    'DPC-001552': ('CD-437', NUCREC, 'new_class', 'RARgamma agonist retinoid (also reported POLA1 inhibitor)'),
    'DPC-002900': ('FULVESTRANT', NUCREC, 'new_class', 'ER antagonist/degrader'),
    'DPC-000232': ('ABIRATERONE', NUCREC, 'new_class', 'CYP17A1 inhibitor (androgen synthesis)'),
    'DPC-000757': ('ATRA', NUCREC, 'new_class', 'RAR agonist (all-trans retinoic acid)'),
    'DPC-003334': ('ICI-46474', NUCREC, 'new_class', 'tamoxifen, selective ER modulator'),
    'DPC-000482': ('AM-580', NUCREC, 'new_class', 'RARalpha agonist'),
    'DPC-003765': ('LE-135', NUCREC, 'new_class', 'RARbeta-selective retinoid ligand'),
    'DPC-001049': ('BEXAROTENE', NUCREC, 'new_class', 'RXR agonist'),
    'DPC-001080': ('BICALUTAMIDE', NUCREC, 'new_class', 'androgen receptor antagonist'),
    # ---- Other / non-oncology: oncology agents with non-kinase targets
    'DPC-005016': ('PEVONEDISTAT', OTHONC, 'new_class', 'NEDD8-activating enzyme (NAE) inhibitor'),
    'DPC-006994': ('YK 4-279', OTHONC, 'new_class', 'EWS-FLI1/RNA helicase A inhibitor'),
    'DPC-002562': ('ERASTIN', OTHONC, 'new_class', 'system xc- (SLC7A11) inhibitor, ferroptosis inducer'),
    'DPC-004391': ('MST 312', OTHONC, 'new_class', 'telomerase inhibitor'),
    'DPC-004766': ('OMACETAXINE MEPESUCCINATE', OTHONC, 'new_class', 'protein synthesis inhibitor (approved for CML)'),
    'DPC-001544': ('CCT007093', OTHONC, 'new_class', 'PPM1D (WIP1) phosphatase inhibitor'),
    'DPC-004707': ('XAV-939', OTHONC, 'new_class', 'tankyrase (WNT) inhibitor'),
    'DPC-000635': ('APO866', OTHONC, 'new_class', 'NAMPT inhibitor'),
    'DPC-003660': ('KPT185', OTHONC, 'new_class', 'XPO1 (exportin) inhibitor'),
    'DPC-002438': ('EHT 1864', OTHONC, 'new_class', 'RAC GTPase inhibitor'),
    'DPC-006572': ('TOSEDOSTAT', OTHONC, 'new_class', 'aminopeptidase inhibitor (AML trials)'),
    'DPC-004679': ('NSC319726', OTHONC, 'new_class', 'mutant-p53 (R175H) reactivator'),
    'DPC-004874': ('P 22077', OTHONC, 'new_class', 'USP7/USP47 deubiquitinase inhibitor'),
    'DPC-003142': ('GTPL5238', OTHONC, 'new_class', 'procaspase-3 activator'),
    'DPC-006507': ('TIPIFARNIB', OTHONC, 'new_class', 'farnesyltransferase inhibitor'),
    'DPC-003811': ('LGK-974', OTHONC, 'new_class', 'porcupine (WNT secretion) inhibitor'),
    'DPC-003118': ('GSK2830371', OTHONC, 'new_class', 'allosteric WIP1 (PPM1D) inhibitor'),
    'DPC-006937': ('WNT-C59', OTHONC, 'new_class', 'porcupine (WNT secretion) inhibitor'),
    'DPC-003769': ('LENALIDOMIDE', OTHONC, 'new_class', 'cereblon molecular glue (IKZF1/3 degrader)'),
    'DPC-004213': ('MIRA 1', OTHONC, 'new_class', 'mutant-p53 reactivator'),
    'DPC-006658': ('TRIPTOLIDE', OTHONC, 'new_class', 'XPB/TFIIH inhibitor (transcription)'),
    'DPC-002570': ('ERISMODEGIB', OTHONC, 'new_class', 'sonidegib, SMO antagonist (hedgehog)'),
    'DPC-000373': ('AGI-6780', OTHONC, 'new_class', 'mutant-IDH2 inhibitor'),
    'DPC-001077': ('BIBR-1532', OTHONC, 'new_class', 'telomerase inhibitor'),
    'DPC-006858': ('VISMODEGIB', OTHONC, 'new_class', 'SMO antagonist (hedgehog)'),
    'DPC-004677': ('NSC23766', OTHONC, 'new_class', 'RAC1-GEF interaction inhibitor'),
    'DPC-001983': ('PIDNARULEX', OTHONC, 'new_class', 'CX-5461, RNA polymerase I inhibitor / G-quadruplex stabiliser'),
    # ---- Other / non-oncology: remainder (label change only)
    'DPC-002848': ('FLUVASTATIN', NONONC, 'split_other', 'statin (HMGCR)'),
    'DPC-004751': ('OLIGOMYCIN A', NONONC, 'split_other', 'ATP synthase inhibitor (mitochondrial tool)'),
    'DPC-000773': ('AVAGACESTAT', NONONC, 'split_other', 'gamma-secretase inhibitor developed for Alzheimer disease'),
    'DPC-003485': ('ISOEVODIAMINE', NONONC, 'split_other', 'natural product (TRPV1)'),
    'DPC-005204': ('PIPERLONGUMINE', NONONC, 'split_other', 'natural product ROS inducer'),
    'DPC-005959': ('SIMVASTATIN', NONONC, 'split_other', 'statin (HMGCR)'),
    'DPC-003924': ('LY-2183240', NONONC, 'split_other', 'FAAH inhibitor'),
    'DPC-005797': ('SB-225002', NONONC, 'split_other', 'CXCR2 antagonist'),
    'DPC-004833': ('OUABAIN', NONONC, 'split_other', 'cardiac glycoside (Na/K-ATPase)'),
    'DPC-002773': ('FINGOLIMOD HYDROCHLORIDE; GILENYA; TDI-132; IMUSERA; GILENIA', NONONC, 'split_other', 'S1P receptor modulator'),
    'DPC-003516': ('ISTRADEFYLLINE', NONONC, 'split_other', 'adenosine A2A antagonist'),
    'DPC-002006': ('CICLOSPORIN', NONONC, 'split_other', 'calcineurin inhibitor'),
    'DPC-000296': ('ACETYLCYSTEINE', NONONC, 'split_other', 'mucolytic / antioxidant'),
    'DPC-004559': ('NICLOSAMIDE', NONONC, 'split_other', 'anthelmintic, mitochondrial uncoupler'),
    'DPC-003165': ('GW-405833', NONONC, 'split_other', 'cannabinoid CB2 ligand'),
    'DPC-006223': ('TACROLIMUS', NONONC, 'split_other', 'calcineurin inhibitor'),
    'DPC-001760': ('CIMETIDINE', NONONC, 'split_other', 'histamine H2 antagonist'),
    'DPC-001043': ('BETULINIC ACID', NONONC, 'split_other', 'natural product'),
    'DPC-006119': ('STEMREGENIN 1', NONONC, 'split_other', 'AHR antagonist (stem-cell expansion)'),
    'DPC-002649': ('ETOMOXIR', NONONC, 'split_other', 'CPT1 inhibitor'),
    'DPC-003524': ('ITRACONAZOLE', NONONC, 'split_other', 'azole antifungal'),
}
NOTES = {  # kept in their class, with a caveat
    'DPC-003550': 'serdemetan: annotated MDM2 antagonist; p53-independent activity reported',
    'DPC-006799': 'vandetanib: VEGFR2/EGFR/RET multikinase; EGFR is the relevant target in cancer-cell screens',
    'DPC-001973': 'CUDC-101: dual EGFR/HDAC inhibitor',
    'DPC-000393': 'AK176060: 2-oxoglutarate oxygenase (KDM/EGLN) annotation; mechanism uncertain',
    'DPC-001682': 'pyrimethamine: antiparasitic antifolate (DHFR), kept by mechanism',
    'DPC-003676': 'tirbanibulin: SRC/tubulin dual inhibitor; cytotoxicity is mainly tubulin',
    'DPC-000655': 'AR-12 (OSU-03012): weak PDK1 inhibitor; mechanism uncertain',
}


def load_table():
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    dt['targets'] = comp.targets.reindex(dt.index)
    return dt


def curate(dt):
    missing = [c for c in CURATION if c not in dt.index]
    assert not missing, f'curation keys not in drug table: {missing}'
    rows = []
    for d, r in dt.iterrows():
        if d in CURATION:
            name, new, ctype, reason = CURATION[d]
            assert name == r['name'], (d, name, r['name'])
        else:
            new, ctype = r['class'], 'unchanged'
            reason = 'confirmed: primary target/mechanism consistent with class'
            if r['class'] == OLD_OTHER:
                raise ValueError(f'{d} {r["name"]} in the old Other bin was not reviewed')
        if d in NOTES:
            reason += f'; note: {NOTES[d]}'
        conservative = r['class'] if ctype in ('mechanism_revised', 'judgement') else new
        rows.append({'compound': d, 'name': r['name'], 'old_class': r['class'], 'new_class': new,
                     'conservative_class': conservative, 'changed': ctype not in ('unchanged', 'split_other'),
                     'change_type': ctype, 'reason': reason, 'rho_sel': r.rho_sel, 'n_pairs': int(r.n_pairs),
                     'targets': r.targets})
    cur = pd.DataFrame(rows)
    assert cur.compound.is_unique and len(cur) == len(dt)
    return cur


def class_summary(dt, col, rng):
    out = []
    for cls, g in dt.groupby(col):
        v = g.rho_sel.values
        boots = np.median(rng.choice(v, size=(N_BOOT_DRUG, len(v)), replace=True), axis=1)
        out.append({'class': cls, 'drugs': len(g), 'n_reproducible': int((v >= REPRODUCIBLE).sum()),
                    'median_rho_sel': float(np.median(v)), 'median_ci95_lo': float(np.percentile(boots, 2.5)),
                    'median_ci95_hi': float(np.percentile(boots, 97.5)),
                    'frac_reproducible': float((v >= REPRODUCIBLE).mean()),
                    'median_fitness_r2': float(g.fitness_r2.median()),
                    'gcsi_n': int(g.confirm_gCSI.notna().sum()), 'gcsi_median': g.confirm_gCSI.median(),
                    'nci_n': int(g.confirm_NCI60.notna().sum()), 'nci_median': g.confirm_NCI60.median(),
                    'small_n_flag': len(g) < 5})
    s = pd.DataFrame(out).sort_values('median_rho_sel', ascending=False).reset_index(drop=True)
    s['rank'] = np.arange(1, len(s) + 1)
    return s


# ------------------------------------------------------------------ activity
def activity_table(dt):
    st = pd.read_csv(DEC / 'drug_screen_stats.csv')
    st = st[st.compound.isin(dt.index)]
    agg = st.groupby('compound').agg(min_frac_active=('frac_auc_below_0p8', 'min'),
                                     max_frac_active=('frac_auc_below_0p8', 'max'),
                                     mean_frac_active=('frac_auc_below_0p8', 'mean'),
                                     min_auc_sd=('auc_sd', 'min'), mean_auc_sd=('auc_sd', 'mean'),
                                     n_screens=('screen', 'size'))
    return agg


def activity_bins(x):
    return pd.cut(x, [-0.001, 0.05, 0.20, 0.50, 1.0], labels=['<5%', '5-20%', '20-50%', '>=50%'])


def activity_strata(x):
    return pd.cut(x, [-0.001, 0.10, 0.50, 1.0], labels=['<10% (inactive in >=1 screen)', '10-50%', '>=50%'])


ACTIVITY_SPECS = {  # activity covariates; the first is primary
    'logit_quadratic': 'lf + I(lf**2) + lsd',
    'spline': 'bs(min_frac_active, df=4) + bs(lsd, df=3)',
    'rank_spline': 'bs(pf, df=5) + bs(psd, df=4)',
    'bins': 'C(activity_bin) + lsd',
}


def class_regression(df, col, spec):
    """OLS of mean selective agreement on class + activity covariates (HC3). Adjusted class means are marginal
    (standardised) means: the average prediction over all 353 drugs' observed activity with the class set to c."""
    from patsy import build_design_matrices
    d = df.copy()
    d['lf'] = np.log(np.clip(d.min_frac_active, 0.01, 0.99) / (1 - np.clip(d.min_frac_active, 0.01, 0.99)))
    d['lsd'] = np.log(d.min_auc_sd)
    d['pf'] = d.min_frac_active.rank(pct=True)
    d['psd'] = d.min_auc_sd.rank(pct=True)
    d['activity_bin'] = d.activity_bin.astype(str)
    d['cls'] = d[col]
    rhs = ACTIVITY_SPECS[spec]
    base = smf.ols('rho_sel ~ C(cls)', data=d).fit()
    act = smf.ols(f'rho_sel ~ {rhs}', data=d).fit()
    full = smf.ols(f'rho_sel ~ C(cls) + {rhs}', data=d).fit()
    f_test = full.compare_f_test(act)  # nested F-test for class given activity (classical SEs)
    cov = full.get_robustcov_results(cov_type='HC3').cov_params()
    info = full.model.data.design_info
    rows = []
    for cls in sorted(d.cls.unique()):
        dd = d.copy(); dd['cls'] = cls
        xbar = np.asarray(build_design_matrices([info], dd)[0]).mean(axis=0)
        est = float(xbar @ full.params.values); se = float(np.sqrt(xbar @ cov @ xbar))
        g = d[d.cls == cls].rho_sel
        rows.append({'class': cls, 'drugs': len(g), 'unadjusted_mean': g.mean(), 'unadjusted_median': g.median(),
                     'adjusted_mean': est, 'adj_ci95_lo': est - 1.96 * se, 'adj_ci95_hi': est + 1.96 * se})
    grid = pd.DataFrame(rows)
    grid['spec'] = spec
    big = grid[grid.drugs >= 5]
    rho_rank = spearmanr(big.unadjusted_median, big.adjusted_mean).correlation
    top5_unadj = set(big.sort_values('unadjusted_median', ascending=False)['class'].head(5))
    top5_adj = set(big.sort_values('adjusted_mean', ascending=False)['class'].head(5))
    stats = {'activity_terms': rhs, 'r2_class_only': base.rsquared, 'r2_activity_only': act.rsquared,
             'r2_class_plus_activity': full.rsquared,
             'class_given_activity_F': float(f_test[0]), 'class_given_activity_p': float(f_test[1]),
             'class_given_activity_df': int(f_test[2]),
             'spearman_unadjusted_median_vs_adjusted_mean_classes_n5': float(rho_rank),
             'n_classes_n5': int(len(big)), 'top5_unadjusted': sorted(top5_unadj), 'top5_adjusted': sorted(top5_adj),
             'top5_overlap': len(top5_unadj & top5_adj)}
    return grid, stats


def standardized_class_means(df, col, strata_col):
    """Direct standardisation: class mean within each activity stratum, weighted by the pooled stratum shares;
    classes absent from a stratum are given the regression-free within-stratum overall mean (reported as coverage)."""
    w = df[strata_col].value_counts(normalize=True)
    overall = df.groupby(strata_col).rho_sel.mean()
    rows = []
    for cls, g in df.groupby(col):
        m = g.groupby(strata_col).rho_sel.mean()
        cov = [s for s in w.index if s in m.index and np.isfinite(m[s])]
        val = sum(w[s] * (m[s] if s in cov else overall[s]) for s in w.index)
        rows.append({'class': cls, 'standardized_mean': val, 'strata_covered': len(cov), 'drugs': len(g)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cell bootstrap of reproducibility
def pair_blocks(pairs):
    """Per screen pair: residual matrices on the common cells (zero-filled) plus masks, as in 02_decompose."""
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in LAB}
    Zc = {s: pd.read_parquet(DEC / f'z_{s}.parquet').columns for s in LAB}
    blocks = []
    for (a, b), g in pairs.groupby(['screen_a', 'screen_b']):
        g = g[g.rho_sel.notna()]
        cells = Zc[a].intersection(Zc[b])
        drugs = list(g.compound)
        X = R[a].loc[drugs, cells].values.T
        Y = R[b].loc[drugs, cells].values.T
        M = (np.isfinite(X) & np.isfinite(Y)).astype(float)
        blocks.append({'a': a, 'b': b, 'cells': cells, 'drugs': drugs, 'X': np.where(M > 0, X, 0.0),
                       'Y': np.where(M > 0, Y, 0.0), 'M': M, 'rho': g.rho_sel.values})
    return blocks


def weighted_corr(W, X, Y, M):
    S0 = W @ M
    Sx, Sy = W @ (M * X), W @ (M * Y)
    Sxx, Syy, Sxy = W @ (M * X * X), W @ (M * Y * Y), W @ (M * X * Y)
    mx, my = Sx / S0, Sy / S0
    cov = Sxy / S0 - mx * my
    vx, vy = Sxx / S0 - mx ** 2, Syy / S0 - my ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return cov / np.sqrt(vx * vy)


def bootstrap_reproducibility(pairs, drugs, rng):
    blocks = pair_blocks(pairs)
    union = sorted(set().union(*[set(b['cells']) for b in blocks]))
    pos = {c: i for i, c in enumerate(union)}
    Wall = rng.poisson(1.0, size=(N_BOOT_CELL, len(union))).astype(float)
    didx = {d: i for i, d in enumerate(drugs)}
    sums = np.zeros((N_BOOT_CELL, len(drugs))); cnt = np.zeros(len(drugs))
    sums_x = np.zeros((N_BOOT_CELL, len(drugs))); cnt_x = np.zeros(len(drugs))
    check = []
    for b in blocks:
        idx = np.array([pos[c] for c in b['cells']])
        r0 = weighted_corr(np.ones((1, len(idx))), b['X'], b['Y'], b['M'])[0]
        check.append(float(np.nanmax(np.abs(r0 - b['rho']))))
        rb = weighted_corr(Wall[:, idx], b['X'], b['Y'], b['M'])
        cross = LAB[b['a']] != LAB[b['b']]
        for j, d in enumerate(b['drugs']):
            k = didx[d]
            sums[:, k] += rb[:, j]; cnt[k] += 1
            if cross:
                sums_x[:, k] += rb[:, j]; cnt_x[k] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_all = sums / cnt
        mean_x = np.where(cnt_x > 0, sums_x / np.where(cnt_x > 0, cnt_x, 1), np.nan)
    return mean_all, mean_x, max(check)


def classify(v):
    return np.where(v >= REPRODUCIBLE, 'reproducible', np.where(v < NONREPRODUCIBLE, 'non_reproducible', 'intermediate'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dt = load_table()
    summary = {}

    # ---------------- 1. curation
    cur = curate(dt)
    cur.drop(columns=['targets']).to_csv(OUT / 'curated_classes.csv', index=False)
    dt = dt.join(cur.set_index('compound')[['new_class', 'conservative_class', 'change_type']])
    sums = []
    for col, label in [('class', 'old'), ('new_class', 'curated'), ('conservative_class', 'curated_conservative')]:
        s = class_summary(dt, col, rng)
        s.insert(0, 'scheme', label)
        sums.append(s)
    cs = pd.concat(sums, ignore_index=True)
    cs.to_csv(OUT / 'class_summary_curated.csv', index=False)
    old, new = sums[0].set_index('class'), sums[1].set_index('class')
    changes = cur[cur.changed]
    summary['curation'] = {
        'drugs_reviewed': int(len(cur)), 'reassigned': int(len(changes)),
        'by_change_type': cur.change_type.value_counts().to_dict(),
        'reassigned_from_named_classes': int((changes.old_class != OLD_OTHER).sum()),
        'n_per_class_old': old.drugs.to_dict(), 'n_per_class_curated': new.drugs.to_dict(),
        'median_rho_sel_curated': new.median_rho_sel.round(3).to_dict(),
        'frac_reproducible_curated': new.frac_reproducible.round(3).to_dict(),
        'rank_old': old['rank'].to_dict(), 'rank_curated': new['rank'].to_dict(),
        'top5_old': list(sums[0]['class'].head(5)), 'top5_curated': list(sums[1]['class'].head(5)),
        'top5_curated_conservative': list(sums[2]['class'].head(5)),
        'reviewer_examples': cur[cur.compound.isin(['DPC-001140', 'DPC-001409', 'GDSC1:1463', 'DPC-001134',
                                                    'DPC-001165'])][['name', 'old_class', 'new_class']].to_dict(orient='records')}

    # ---------------- 2. activity confounding
    act = activity_table(dt)
    df = dt.join(act)
    df['activity_bin'] = activity_bins(df.min_frac_active)
    df['activity_stratum'] = activity_strata(df.min_frac_active)
    df['repro_class'] = classify(df.rho_sel.values)
    sp = {}
    for c in ['min_frac_active', 'max_frac_active', 'mean_frac_active', 'min_auc_sd', 'mean_auc_sd']:
        r = spearmanr(df.rho_sel, df[c])
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(df), len(df))
            boots.append(spearmanr(df.rho_sel.values[i], df[c].values[i]).correlation)
        sp[c] = {'spearman': float(r.correlation), 'p': float(r.pvalue),
                 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    nonrep = df[df.repro_class == 'non_reproducible']
    summary['activity'] = {
        'spearman_mean_selective_agreement_vs': sp,
        'n_nonreproducible': int(len(nonrep)),
        'nonreproducible_inactive_in_any_screen_lt10pct': int((nonrep.min_frac_active < 0.10).sum()),
        'nonreproducible_inactive_in_any_screen_lt5pct': int((nonrep.min_frac_active < 0.05).sum()),
        'reproducible_inactive_in_any_screen_lt10pct': int(((df.repro_class == 'reproducible') & (df.min_frac_active < 0.10)).sum())}
    bins = df.groupby('activity_bin').agg(drugs=('rho_sel', 'size'), median_rho_sel=('rho_sel', 'median'),
                                          frac_reproducible=('rho_sel', lambda v: float((v >= REPRODUCIBLE).mean())),
                                          frac_nonreproducible=('rho_sel', lambda v: float((v < NONREPRODUCIBLE).mean())))
    comp_new = pd.crosstab(df.activity_bin, df.new_class)
    comp_old = pd.crosstab(df.activity_bin, df['class'])
    comp_new.columns = [f'curated: {c}' for c in comp_new.columns]
    comp_old.columns = [f'old: {c}' for c in comp_old.columns]
    bins.join(comp_new).join(comp_old).reset_index().to_csv(OUT / 'activity_bins.csv', index=False)
    summary['activity']['bins'] = bins.round(3).reset_index().astype({'activity_bin': str}).to_dict(orient='records')

    strata_rows = []
    for col, label in [('new_class', 'curated'), ('class', 'old')]:
        t = df.groupby([col, 'activity_stratum']).rho_sel.agg(['size', 'median', 'mean']).reset_index()
        t.columns = ['class', 'activity_stratum', 'drugs', 'median_rho_sel', 'mean_rho_sel']
        t.insert(0, 'scheme', label)
        strata_rows.append(t[t.drugs > 0])
    pd.concat(strata_rows, ignore_index=True).to_csv(OUT / 'activity_class_strata.csv', index=False)

    regs, regstats = [], {}
    for col, label in [('new_class', 'curated'), ('class', 'old')]:
        stdz = standardized_class_means(df, col, 'activity_stratum').set_index('class')
        for spec in ACTIVITY_SPECS:
            grid, st = class_regression(df, col, spec)
            grid = grid.join(stdz[['standardized_mean', 'strata_covered']], on='class')
            grid['scheme'] = label
            grid['rank_unadjusted'] = grid.unadjusted_median.rank(ascending=False, method='min')
            grid['rank_adjusted'] = grid.adjusted_mean.rank(ascending=False, method='min')
            regs.append(grid)
            regstats[f'{label}|{spec}'] = st
    pd.concat(regs, ignore_index=True).to_csv(OUT / 'activity_regression.csv', index=False)
    summary['activity']['regression'] = regstats
    summary['activity']['strata_n'] = df.activity_stratum.value_counts().astype(int).to_dict()

    # ---------------- 3. cross-lab classification and per-drug SEs
    pairs = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    pairs = pairs[pairs.compound.isin(df.index)].copy()
    pairs['cross_lab'] = [LAB[a] != LAB[b] for a, b in zip(pairs.screen_a, pairs.screen_b)]
    g = pairs.groupby('compound')
    df['n_pairs_cross_lab'] = g.cross_lab.sum().reindex(df.index).astype(int)
    df['rho_sel_cross_lab'] = pairs[pairs.cross_lab].groupby('compound').rho_sel.mean().reindex(df.index)
    df['rho_sel_same_lab'] = pairs[~pairs.cross_lab].groupby('compound').rho_sel.mean().reindex(df.index)
    drugs = list(df.index)
    boot_all, boot_x, maxdiff = bootstrap_reproducibility(pairs, drugs, rng)
    df['se_boot'] = np.nanstd(boot_all, axis=0, ddof=1)
    df['ci95_lo'] = np.nanpercentile(boot_all, 2.5, axis=0)
    df['ci95_hi'] = np.nanpercentile(boot_all, 97.5, axis=0)
    df['p_class_stable'] = (classify(boot_all) == df.repro_class.values[None, :]).mean(axis=0)
    df['p_reproducible'] = (boot_all >= REPRODUCIBLE).mean(axis=0)
    df['se_boot_cross_lab'] = np.where(df.n_pairs_cross_lab > 0, np.nanstd(boot_x, axis=0, ddof=1), np.nan)
    v = pairs.assign(var=(1 - pairs.rho_sel ** 2) ** 2 / (pairs.n_common - 4)).groupby('compound')['var']
    df['se_analytic_indep'] = np.sqrt(v.sum() / v.size() ** 2).reindex(df.index)
    df['cross_lab_class'] = np.where(df.rho_sel_cross_lab.notna(), classify(df.rho_sel_cross_lab.fillna(0).values), 'no_cross_lab_pair')
    has_x = df.n_pairs_cross_lab > 0
    agree = pd.crosstab(df.loc[has_x, 'repro_class'], df.loc[has_x, 'cross_lab_class'])
    pr = pairs.dropna(subset=['rho_sel'])
    both = df[(df.rho_sel_cross_lab.notna()) & (df.rho_sel_same_lab.notna())]
    summary['crosslab'] = {
        'bootstrap_identity_check_max_abs_diff': maxdiff,
        'drugs_single_pair': int((df.n_pairs == 1).sum()),
        'drugs_same_lab_pairs_only': int((df.n_pairs_cross_lab == 0).sum()),
        'drugs_with_cross_lab_pair': int(has_x.sum()),
        'frac_reproducible_all_pairs': float((df.rho_sel >= REPRODUCIBLE).mean()),
        'frac_reproducible_cross_lab_only': float((df.loc[has_x, 'rho_sel_cross_lab'] >= REPRODUCIBLE).mean()),
        'frac_nonreproducible_cross_lab_only': float((df.loc[has_x, 'rho_sel_cross_lab'] < NONREPRODUCIBLE).mean()),
        'class_agreement_cross_lab_vs_all': float((df.loc[has_x, 'repro_class'] == df.loc[has_x, 'cross_lab_class']).mean()),
        'agreement_table': {f'{i}|{j}': int(agree.loc[i, j]) for i in agree.index for j in agree.columns},
        'median_pair_rho_sel_same_lab': float(pr[~pr.cross_lab].rho_sel.median()),
        'median_pair_rho_sel_cross_lab': float(pr[pr.cross_lab].rho_sel.median()),
        'n_pairs_same_lab': int((~pr.cross_lab).sum()), 'n_pairs_cross_lab': int(pr.cross_lab.sum()),
        'drugs_with_both': int(len(both)),
        'median_within_drug_same_minus_cross': float((both.rho_sel_same_lab - both.rho_sel_cross_lab).median()),
        'median_se_boot': float(df.se_boot.median()), 'iqr_se_boot': [float(df.se_boot.quantile(0.25)), float(df.se_boot.quantile(0.75))],
        'median_se_boot_single_pair': float(df.loc[df.n_pairs == 1, 'se_boot'].median()),
        'median_se_boot_six_pairs': float(df.loc[df.n_pairs == 6, 'se_boot'].median()),
        'frac_class_stable_ge_0p9': float((df.p_class_stable >= 0.9).mean()),
        'reproducible_with_p_reproducible_ge_0p9': int(((df.repro_class == 'reproducible') & (df.p_reproducible >= 0.9)).sum()),
        'n_reproducible': int((df.repro_class == 'reproducible').sum())}
    cols = ['name', 'class', 'new_class', 'conservative_class', 'change_type', 'rho_sel', 'rho_raw', 'n_pairs', 'repro_class',
            'se_boot', 'se_analytic_indep', 'ci95_lo', 'ci95_hi', 'p_reproducible', 'p_class_stable',
            'n_pairs_cross_lab', 'rho_sel_cross_lab', 'rho_sel_same_lab', 'cross_lab_class', 'se_boot_cross_lab',
            'min_frac_active', 'max_frac_active', 'min_auc_sd', 'mean_auc_sd', 'activity_bin', 'activity_stratum',
            'fitness_r2', 'confirm_gCSI', 'confirm_NCI60']
    df[cols].rename(columns={'class': 'old_class'}).to_csv(OUT / 'drug_reproducibility_table.csv', index_label='compound')
    write_json(OUT / '53_summary.json', summary)
    print(json.dumps(summary['curation']['n_per_class_curated'], indent=0))
    print(cs[cs.scheme != 'curated_conservative'][['scheme', 'class', 'drugs', 'median_rho_sel', 'frac_reproducible', 'rank']].to_string())
    print(json.dumps({k: v for k, v in summary['activity'].items() if k != 'bins'}, indent=1, default=str)[:4000])
    print(json.dumps(summary['crosslab'], indent=1))


if __name__ == '__main__':
    main()
