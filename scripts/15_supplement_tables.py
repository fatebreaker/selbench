"""Tables generated from saved results: manuscript/table1.tex (main text), manuscript/supplement_tables.tex and
manuscript/supp_refs.tex (macros \\st<Key> holding each supplementary table's number, used by the main text)."""
import json
import re

import numpy as np
import pandas as pd

from common import PROCESSED, PROJECT, RESULTS

LBL = {'PRISM': 'PRISM', 'CTRP': 'CTRP', 'GDSC1': 'GDSC1', 'GDSC2': 'GDSC2', 'gCSI': 'gCSI', 'NCI60': 'NCI-60',
       'CCLE': 'CCLE', 'FIMM': 'FIMM', 'GRAY': 'GRAY'}
CONFIRM = ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY']
MAN = PROJECT / 'manuscript'


def esc(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('–', '--').replace('#', r'\#')


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    if isinstance(v, (float, np.floating)):
        if v != 0 and abs(v) < 10 ** (-d):
            return f'{v:.1e}'
        return f'{v:.{d}f}'
    v = str(v)
    return v if '\\' in v or '$' in v else esc(v)


def pfmt(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return '--'
    if p < 1e-16:                       # below double precision (underflow or a floor of 1e-300 in the source)
        return r'$<10^{-16}$'
    if p < 1e-3:
        m, e = f'{p:.0e}'.split('e')
        return rf'${m}\times10^{{{int(e)}}}$'
    return f'{p:.2g}'


def sgn(v, d=3):
    return f'{v:.{d}f}'.replace('-', '$-$')


# The study's early names ("fitness axis", "laboratory") were replaced in the paper by "general-sensitivity axis" and
# "dataset"/"institution" (round-2 review, editor C5); captions, headers and cells are written with the current terms.
TERMS = [('Fitness axes', 'General-sensitivity axes'), ('fitness axes', 'general-sensitivity axes'),
         ('Fitness axis', 'General-sensitivity axis'), ('fitness-axis', 'axis'), ('fitness axis', 'general-sensitivity axis'),
         ('Fitness share', 'Axis share'), ('fitness share', 'axis share'), ('Share via fitness', 'Share via axis'),
         ('Fitness retained', 'Axis retained'), ('Fit.\\ retained', 'Axis retained'), ('Fit. retained', 'Axis retained'),
         ('Thr.\\ fitness', 'Thr.\\ axis'), ('Fitness $R^2$', 'Axis $R^2$'), ('fitness only', 'axis only'),
         ('measured fitness', 'measured axis'), ('omics fitness', 'omics-predicted axis'), ('Pred.\\ fit.', 'Pred.\\ axis'),
         ('fitness + selective', 'axis + selective'), ('A-fitness on B', 'A-axis on B'), ('own fitness estimate', 'own axis estimate'),
         ('omics-predicted fitness axis', 'omics-predicted axis'), ('discovery fitness', 'discovery axis'),
         ('fitness-mediated', 'axis-mediated'), ('fitness estimate', 'axis estimate'), ('through fitness', 'through the axis'),
         ('prediction fitness', 'prediction axis'), ('fitness-only', 'axis-only'),
         ('fitness', 'general sensitivity'), ('Fitness', 'General sensitivity'),
         ('Within- versus cross-laboratory', 'Within- versus cross-dataset'), ('within- versus cross-laboratory', 'within- versus cross-dataset'),
         ('Cross-laboratory', 'Cross-dataset'), ('cross-laboratory', 'cross-dataset'), ('other laboratories', 'other datasets'),
         ('single-screen laboratories', 'single-screen institutions'), ('laboratory- and readout-aware', 'institution- and readout-aware'),
         ('three laboratories', 'three institutions'), ('two laboratories', 'two institutions'), ('Laboratories', 'Institutions'),
         ('laboratories', 'institutions'), ('Laboratory', 'Institution'), ('laboratory', 'institution')]


def terms(s):
    s = str(s)
    for old, new in TERMS:
        s = s.replace(old, new)
    return s


TABLE_DATA = {}   # label -> (header, rows), written as Supplementary Data 1 (one CSV per table)


def plain(x):
    """LaTeX cell -> plain text for the CSV export."""
    s = str(x)
    for a_, b_ in [('$\\mu$', 'µ'), ('\\,', ' '), ('\\ ', ' '), ('$-$', '-'), ('\\%', '%'), ('\\_', '_'), ('$\\geq$', '>='), ('$\\leq$', '<='), ('$\\times$', 'x'),
                   ('$\\rightarrow$', '->'), ('$\\Delta$', 'Delta'), ('$\\dagger$', '*'), ('~', ' '), ('--', '-')]:
        s = s.replace(a_, b_)
    s = re.sub(r'\\textit\{([^}]*)\}', r'\1', s)
    s = re.sub(r'[$]', '', s)
    s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
    return s.replace('\\', '').replace('{', '').replace('}', '').strip()


def table(caption, label, header, rows, colspec, small=True):
    TABLE_DATA[label] = ([plain(terms(h)) for h in header], [[plain(terms(fmt(c))) for c in r] for r in rows])
    out = [r'\begin{table}[!htbp]', r'\centering', r'\small' if small else '', rf'\caption{{{terms(caption)}}}', rf'\label{{{label}}}',
           r'\begin{adjustbox}{max width=\textwidth, max totalheight=0.78\textheight}',
           rf'\begin{{tabular}}{{{colspec}}}', r'\toprule', terms(' & '.join(header)) + r' \\', r'\midrule']
    out += [terms(' & '.join(fmt(c) for c in r)) + r' \\' for r in rows]
    out += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}', '']
    return '\n'.join(out)


def display_names():
    c = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    out = {}
    for k, r in c.iterrows():
        names = [n for n in str(r.all_names).split('|') if n and n != 'nan']
        generic = [n for n in names if re.fullmatch(r'[a-z][a-z\-]+', n)]
        if generic:
            out[k] = generic[0]
        elif re.fullmatch(r'\d{2,7}-\d{2}-\d', str(r['name'])) and names:
            out[k] = names[0]
        else:
            out[k] = str(r['name'])
    return out


def jl(p):
    return json.loads((RESULTS / p).read_text())


def h4_pass(h):
    p = h['H4'].get('wilcoxon_p_one_sided')
    return h['H4']['median_difference'] >= 0.15 and p is not None and p < 0.05


def table1():
    """Registered statistics in the five untouched cell-line screens, BeatAML and PDX_Bruna (a supplementary table
    since the revision; the main text shows them as estimates in Fig. 5). NCI-60 P values are not shown: its 39,682
    single-tested compounds on 58 lines are not independent units."""
    H = {k: jl(f'confirmation_{k}/hypotheses.json') for k in CONFIRM}
    B = jl('confirmation_BeatAML/hypotheses.json')

    def h1(h):
        v = [x['disattenuated'] for x in h['H1']['by_discovery'] if x.get('disattenuated') is not None]
        return f'{max(v):.2f}'

    def pf(h, p):
        return 'n.i.' if h is H['NCI60'] else pfmt(p)

    def h3(h):
        x = h['H3']
        if x.get('mwu_p_one_sided') is None:
            return f"n.e.\\ ({x['n_non_reproducible']} non-repr.)"
        return f"{x['median_reproducible']:.2f} vs {x['median_non_reproducible']:.2f}; {pf(h, x['mwu_p_one_sided'])}"

    def h4(h):
        s = f"{h['H4']['median_difference']:.2f}; {pf(h, h['H4']['wilcoxon_p_one_sided'])}"
        if not h4_pass(h):
            return rf'\textit{{{s}}}$^\dagger$'
        return s + (r'$^\ddagger$' if h['H4'].get('pass') is False else '')

    def h5a(h):
        return f"{h['H5']['median_model_sel']:.2f}; {pf(h, h['H5']['model_gt0_p'])}"

    def h5b(h):
        return f"{h['H5']['median_replicate_sel']:.2f} vs {h['H5']['median_model_sel']:.2f}; {pf(h, h['H5']['replicate_gt_model_p'])}"

    rows = [('H1 fitness axis screen-specific (max.\\ disattenuated $r<0.70$)', h1),
            ('H2 agreement is selective (selective/raw $\\geq$0.80)', lambda h: f"{h['H2']['ratio']:.2f}"),
            ('H3 discovery class predicts agreement (MWU $P<0.05$)', h3),
            ('H4 discovery fitness does not transfer (gain $\\geq$0.15, $P<0.05$)', h4),
            ('H5a frozen model $>0$ on reproducible drugs', h5a),
            ('H5b discovery consensus measurement $>$ model', h5b)]
    head = ' & '.join(['Hypothesis (threshold)'] + [LBL[k] for k in CONFIRM]) + r' \\'
    sizes = ' & '.join(['Drugs / cell lines'] + [f"{H[k]['drugs']:,} / {H[k]['cells']}" for k in CONFIRM]) + r' \\'
    out = [r'\begin{table}[h!]', r'\centering', r'\scriptsize',
           r'\caption{\textbf{Registered hypotheses and their outcomes in five untouched cell-line screens, primary '
           r'AML samples and patient-derived breast-cancer cells (Table~1 of the original submission).} '
           r'Statistics are those of the frozen confirmatory scripts; thresholds were fixed before retrieval '
           r'(gCSI and NCI-60: original pre-registration; CCLE, FIMM, GRAY and BeatAML: Amendment~1; PDX\_Bruna: '
           r'Amendment~2, whose manifest was stamped with OpenTimestamps and is anchored in Bitcoin block 971474). '
           r'Entries give the statistic and one-sided $P$; estimates with intervals are in Supplementary '
           r'Table~\stEstimates\ and Extended Data Fig.~6. $^\dagger$Not met. $^\ddagger$Met under the registered rule; the frozen '
           "script's flag was false because $P$ underflowed to zero (Supplementary Note~2). "
           r'n.e., not estimable (fewer than three drugs per class); n.i., NCI-60 $P$ values are not interpretable, '
           r'because its 39,682 single-tested compounds on 58 lines are not independent units. gCSI H3 rests on four '
           r'non-reproducible drugs. H5a/H5b use drugs reproducible in discovery; H5b compares the discovery consensus '
           r'measurement (2--4 screens) with the frozen cell-blind model. H9 is not robust at the tumour level '
           r'(Supplementary Table~\stUnitLevel).}',
           r'\label{tab:registered}',
           r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.6cm} *{5}{>{\raggedright\arraybackslash}X}}',
           r'\toprule', head, r'\midrule', sizes, r'\midrule']
    for lab, f in rows:
        out.append(' & '.join([lab] + [f(H[k]) for k in CONFIRM]) + r' \\')
    h6, h7 = B['H6'], B['H7']
    out += [r'\midrule',
            rf'\multicolumn{{6}}{{l}}{{\textit{{Primary AML samples (BeatAML; {B["drugs"]} inhibitors, {B["samples"]} samples; '
            r'cell-line-trained transfer predictions frozen before retrieval)}} \\',
            rf'H6 transfer of reproducible drugs ($\geq$8 drugs, $P<0.05$) & \multicolumn{{5}}{{l}}{{median selective $r$ = '
            rf'{h6["median_sel"]:.3f} over {h6["drugs"]} drugs; {pfmt(h6["wilcoxon_p_one_sided"])}; met}} \\',
            rf'H7 reproducible $>$ non-reproducible (MWU $P<0.05$) & \multicolumn{{5}}{{l}}{{{sgn(h7["median_reproducible"])} '
            rf'vs {sgn(h7["median_non_reproducible"])} ({h7["n_reproducible"]} and {h7["n_non_reproducible"]} drugs); '
            rf'{pfmt(h7["mwu_p_one_sided"])}; met}} \\']
    P = jl('confirmation_PDX_Bruna/hypotheses.json')
    h8, h9 = P['H8'], P['H9']
    out += [r'\midrule',
            rf'\multicolumn{{6}}{{l}}{{\textit{{Patient-derived breast-cancer cells (PDX\_Bruna; {P["drugs"]} drugs, {P["models"]} '
            r'models from 20 tumours; Amendment~2)}} \\',
            rf'H8 transfer of reproducible drugs ($\geq$8 drugs, $P<0.05$) & \multicolumn{{5}}{{l}}{{median selective $r$ = '
            rf'{h8["median_sel"]:.3f} over {h8["drugs"]} drugs; {pfmt(h8["wilcoxon_p_one_sided"])}; {"met" if h8.get("pass") else "not met"}}} \\',
            rf'H9 reproducible $>$ non-reproducible (MWU $P<0.05$) & \multicolumn{{5}}{{l}}{{{sgn(h9["median_reproducible"])} '
            rf'vs {sgn(h9["median_non_reproducible"])} ({h9["n_reproducible"]} and {h9["n_non_reproducible"]} drugs); '
            rf'{pfmt(h9["mwu_p_one_sided"])}; {"met" if h9.get("pass") else "not met"}}} \\',
            r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    return '\n'.join(out)


# supplementary tables are numbered in order of first citation in the main text; the rest follow in build order
ORDER = ['Screens', 'Protocols', 'AxisPairs', 'Variance', 'SharedActive', 'Lineage', 'Growth', 'AxisContents', 'Variants',
         'LeaderboardMetrics', 'Capture', 'Reliability', 'LineageScores', 'Rankings', 'Normalized', 'CrossTargets', 'Censoring',
         'RangeTail', 'Attenuation', 'EndpointMatched', 'Equivalence', 'Retention', 'RetentionCalibration', 'CrossModes',
         'ImproveWithin', 'ImproveCross', 'Retrain', 'TwoByTwo', 'Dependency', 'DependencyAxis', 'RetrainCross', 'HarmonizationGDSC', 'ClassesCurated', 'MechanismAxis', 'Ceiling', 'CeilingFactor', 'Canonical', 'BiomarkerIncrement', 'Adjusted', 'RetrainTarget',
         'SimulationCeiling',
         'Registered', 'Estimates', 'Holm', 'BeatAML', 'UnitLevel', 'RandomEffects', 'PDX', 'PDM', 'AxisTransfer', 'DrEval', 'Biomarkers',
         'Classes']


def caption_text(tex):
    """Plain-text caption of a typeset table (balanced braces after \\caption)."""
    i = tex.find('\\caption{')
    if i < 0:
        return ''
    j, depth = i + len('\\caption{'), 1
    k = j
    while k < len(tex) and depth:
        depth += {'{': 1, '}': -1}.get(tex[k], 0)
        k += 1
    return plain(terms(tex[j:k - 1]))


def citation_order():
    """Supplementary-table keys in order of first citation: abstract, introduction, results, discussion, figure and
    Extended Data captions, Box 1, Methods, then the Supplementary Information's methods and notes."""
    files = ['sections/abstract.tex', 'sections/introduction.tex', 'sections/results.tex', 'sections/discussion.tex',
             'sections/figures.tex', 'sections/extended_data.tex', 'sections/box1.tex', 'sections/methods.tex',
             'supp_methods.tex', 'supp_notes_revision.tex', 'supplement.tex']
    text = ''
    for f in files:
        if (MAN / f).exists():
            text += re.sub(r'(?<!\\)%.*', '', (MAN / f).read_text()) + '\n'
    return list(dict.fromkeys(re.findall(r'\\st([A-Z][A-Za-z]*)', text)))


# supplementary figures in order of first citation: (macro key, file stem, width, caption)
SUPP_FIGURES = [
    ('Growth', 'supp_figure1_growth_rate', 1.0,
     r'\textbf{General-sensitivity axes are weakly related to growth rate.} Each screen\'s all-drug axis against '
     r'DepMap\'s CRISPR-inferred growth rate for shared cell lines.'),
    ('Normalized', 'supp_figure7_dreval_normalized', 0.62,
     r'\textbf{DrEval\'s own headline metric.} Normalized Pearson (measured and predicted responses minus the '
     r'NaiveMeanEffectsPredictor prediction; mean $\pm$ s.d.\ over ten splits) for the leaderboard models (blue; naive '
     r'baselines grey) and this study\'s predictors (green), on the common drugs and cell lines. The axis-only rows '
     r'place one omics-predicted value per cell line on the ln~IC$_{50}$ scale in three ways (Methods).'),
    ('Modes', 'supp_figure6_crossstudy_modes', 1.0,
     r'\textbf{DrEval\'s full published results scored in CTRP~v2 and in other datasets, by test mode.} For each test '
     r'mode (LCO, unseen cell lines; LTO, unseen tissues; LPO, random pairs; LDO, unseen drugs) and each other dataset, '
     r'the median per-drug (blue) and selective (orange; target axis only) correlation of every CTRP~v2-trained model, '
     r'scored against CTRP~v2 (left, faded) and against the same drug--cell-line pairs in the named dataset (right); '
     r'bars, medians over models.'),
    ('Example', 'supp_figure_worked_example', 1.0,
     r'\textbf{A worked example and what a per-drug correlation means.} \textbf{a}, Trametinib sensitivity in CTRP '
     r'against the general-sensitivity axis, coloured by BRAF or NRAS/KRAS hotspot status. \textbf{b}, Profile (grey) '
     r'and selective residual (coloured) by mutation group; bars, medians. \textbf{c}, Mean selective response of the '
     r'top 10\% or 25\% of cell lines ranked by a prediction with a given per-drug correlation (bivariate normal).'),
    ('DrEval', 'supp_figure3_dreval', 1.0,
     r'\textbf{Re-scoring the DrEval leaderboard.} \textbf{a}, Per-drug selective correlation of this study\'s kernel '
     r'ridge (selective target, landmark genes) against the best published model (EnsembleMF) over the drugs common to '
     r'both. \textbf{b}, Selective correlation of every leaderboard model over all drugs and over the drugs reproducible '
     r'in discovery. \textbf{c}, Selective correlation of the kernel ridge on three input-gene sets, trained on the raw or '
     r'the selective target; dashed line, the best published model.'),
    ('Own', 'supp_figure4_own_models', 1.0,
     r'\textbf{This study\'s own models and the collapse of leaderboard predictions.} \textbf{a}, Median per-drug '
     r'correlations of cell-blind predictors trained in one discovery screen, scored on held-out cell lines in the same '
     r'screen (grey) or other screens (raw, blue; selective, orange). \textbf{b}, Global gradient-boosted trees and a '
     r'two-tower neural network trained on raw AUC under four split designs. \textbf{c}, Selective correlation of each '
     r'DrEval leaderboard model against how collapsed its predictions are onto a single cell-line axis.'),
    ('VOI', 'supp_figure5_value_of_information', 1.0,
     r'\textbf{Value of information of drug-response transcriptomes (Supplementary Note~7).} \textbf{a}, '
     r'Drug-specific correlation with each discovery screen\'s selective residual (drug-blind ridge maps) for the '
     r'measured L1000 response, generated profiles and retrieval baselines, with 95\% drug-bootstrap intervals. '
     r'\textbf{b}, The measured-response map as leading shared principal components are removed. \textbf{c}, A '
     r'basal-omics model combined with the measured response versus the basal model alone.'),
    ('Tahoe', 'supp_figure2_tahoe', 0.75,
     r'\textbf{Tahoe-100M responses carry little transferable selectivity information (Supplementary Note~7).} Pooled '
     r'within-drug correlations with each discovery screen\'s selective residual for the 5\,$\mu$M, 24-h pseudobulk '
     r'response, its strength, 24-h relative cell abundance and basal expression; squares, another screen\'s selective '
     r'residual on the same pairs (median over screens).'),
]


DROP = {'Simulation'}   # superseded by the revision's simulation of the ceiling estimators (SimulationCeiling)



CAMEL = re.compile(r'(?<![\\\w])([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*){2,})(?![\w{])')


def breakable_names(tex):
    """Long CamelCase model names in table rows (e.g. NaiveTissueDrugMeanPredictor) may break before each capital, so
    they do not overrun narrow columns (TeX never hyphenates the first word of a cell)."""
    def fix(m):
        w = m.group(1)
        return w if len(w) < 16 else re.sub(r'(?<=[a-z0-9])(?=[A-Z])', r'\\allowbreak{}', w)
    return '\n'.join(CAMEL.sub(fix, ln) if ' & ' in ln else ln for ln in tex.split('\n'))

def main():
    T = []          # (key, latex) in supplementary order
    names = display_names()

    # ---------------- screens
    load = json.loads((PROCESSED / 'confirmation' / 'load_summary.json').read_text())
    dec = jl('decomposition/summary.json')['screens']
    H = {k: jl(f'confirmation_{k}/hypotheses.json') for k in CONFIRM}
    B = jl('confirmation_BeatAML/hypotheses.json')
    dre = jl('dreval/run.json')
    PX = jl('confirmation_PDX_Bruna/hypotheses.json')
    rows = [['PRISM (secondary, 19Q4)', 'Broad, pooled, barcode mRNA, 5 d', 'AUC (8 doses, $\\leq$10\\,$\\mu$M)', dec['PRISM'][0], dec['PRISM'][1], 'discovery'],
            ['CTRP v2', 'Broad, CellTiter-Glo, 72 h', 'AUC (author re-fit)', dec['CTRP'][0], dec['CTRP'][1], 'discovery'],
            ['GDSC1 (r8.4)', 'Sanger, Syto60/resazurin, 72 h', 'AUC; ln IC$_{50}$', dec['GDSC1'][0], dec['GDSC1'][1], 'discovery'],
            ['GDSC2 (r8.4)', 'Sanger, CellTiter-Glo, 72 h', 'AUC; ln IC$_{50}$', dec['GDSC2'][0], dec['GDSC2'][1], 'discovery'],
            ['gCSI (GR v1.3)', 'Genentech, CellTiter-Glo, 72 h', 'mean relative count; GR\\_AOC', H['gCSI']['drugs'], H['gCSI']['cells'], 'confirmation'],
            ['NCI-60 (CellMiner)', 'NCI DTP, SRB, 48 h', '$-\\log_{10}$GI$_{50}$', f"{H['NCI60']['drugs']:,}", H['NCI60']['cells'], 'confirmation'],
            ['CCLE (2012; PharmacoDB)', 'Novartis/GNF, CellTiter-Glo, 72--84 h', 'AAC (PharmacoDB re-fit)', H['CCLE']['drugs'], H['CCLE']['cells'], 'confirmation (Amendment 1)'],
            ['FIMM (PharmacoDB)', 'FIMM, CellTiter-Glo, 72 h', 'AAC (PharmacoDB re-fit)', H['FIMM']['drugs'], H['FIMM']['cells'], 'confirmation (Amendment 1)'],
            ['GRAY (PharmacoDB)', 'Gray lab breast panel (GR re-release), CellTiter-Glo, 72 h', 'AAC (PharmacoDB re-fit)', H['GRAY']['drugs'], H['GRAY']['cells'], 'confirmation (Amendment 1)'],
            ['BeatAML 2.0', 'primary AML samples, MTS (CellTiter Aqueous), 72 h', 'AUC (probit fit)', B['drugs'], B['samples'], 'patient transfer (Amendment 1)'],
            ['PDX\\_Bruna (DrEval)', 'breast PDX-derived tumour cells, CellTiter-Glo, 6 d', 'AUC (CurveCurator)', PX['drugs'], PX['models'], 'patient-derived transfer (Amendment 2)'],
            ['DrEval CTRPv2 leaderboard', 'CTRP v2 re-fit (CurveCurator)', 'ln IC$_{50}$', dre['drugs_evaluated'], dre['common_cells'], 'benchmark re-analysis'],
            ['DrEval full results', 'CTRPv2-trained; scored in CTRPv2, CTRPv1, CCLE, GDSC1, GDSC2', 'ln IC$_{50}$ (CurveCurator)', '', '', 'cross-dataset re-analysis']]
    out = [r'\begin{table}[h!]', r'\centering', r'\footnotesize', r'\caption{\textbf{Datasets analysed.} Drug (or inhibitor) and cell-line (or sample) counts after the shared filters ($\geq$30 cell lines and $\geq$10 distinct values per drug). Assay descriptions follow the original publications.}',
           r'\label{tab:screens}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{p{2.8cm}p{3.7cm}p{2.7cm}rrp{2.3cm}}', r'\toprule',
           r'Dataset & Institution, assay & Endpoint & Drugs & Lines & Role \\', r'\midrule']
    out += [' & '.join(str(c) for c in r) + r' \\' for r in rows]
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    T.append(('Screens', '\n'.join(out)))

    # ---------------- fitness axis
    fa = jl('fitness_axis/summary.json')
    rows = [[LBL[s], v['drugs'], v['cells'], v['split_half'][0], v['growth_rate_spearman'], v['growth_n'],
             v['suspension_minus_adherent'], v['suspension_n'], v['lineage_r2']] for s, v in fa['per_screen'].items()]
    T.append(('Fitness', table(r'\textbf{Fitness axis by screen.} Split-half reliability (Spearman--Brown, median of 200 drug splits); Spearman correlation with CRISPR-inferred growth rate; mean difference of suspension minus adherent lines (s.d.\ units); variance explained by lineage means.',
                               'tab:fitness', ['Screen', 'Drugs', 'Lines', 'Split-half', 'Growth $\\rho$', 'n', 'Susp.$-$adh.', 'n susp.', 'Lineage $R^2$'],
                               rows, 'lrrrrrrrr')))
    pairs = pd.DataFrame(fa['pairs'])
    rows = [[LBL[r.a], LBL[r.b], r.cells, r.corr, f'{r.ci95[0]:.2f}--{r.ci95[1]:.2f}', r.disattenuated, r.common_drugs, r.corr_common_drugs]
            for r in pairs.itertuples()]
    T.append(('FitnessPairs', table(r'\textbf{Agreement of fitness axes between screens.} Pearson correlation over shared cell lines with 95\% cell-bootstrap interval, disattenuated correlation, and correlation when both axes are computed from shared drugs only.',
                                    'tab:fitnesspairs', ['A', 'B', 'Lines', '$r$', '95\\% CI', 'Disatt.', 'Shared drugs', '$r$ (shared drugs)'], rows, 'llrrrrrr')))

    rob = jl('robustness/summary.json')
    rows = []
    for s, v in rob['shared_fraction_auc'].items():
        rows.append([LBL.get(s, s), 'primary', v['triads'], v['lambda2_median'], f"{v['ci95'][0]:.2f}--{v['ci95'][1]:.2f}",
                     rob['doubling_time'][s]['spearman_vs_doubling_time'], rob['doubling_time'][s]['n']])
    for s, v in rob['shared_fraction_with_ic50'].items():
        if 'lnIC50' in s:
            k = s
            rows.append([LBL.get(s.split('_')[0], s), 'ln IC$_{50}$', v['triads'], v['lambda2_median'],
                         f"{v['ci95'][0]:.2f}--{v['ci95'][1]:.2f}", rob['doubling_time'][k]['spearman_vs_doubling_time'],
                         rob['doubling_time'][k]['n']])
    rows.append(['gCSI', 'GR\\_AOC', None, None, None, rob['doubling_time']['gCSI_GR_AOC']['spearman_vs_doubling_time'],
                 rob['doubling_time']['gCSI_GR_AOC']['n']])
    T.append(('Shared', table(r'\textbf{Shared share of each fitness axis and its relation to proliferation.} $\lambda^2$: share of a screen\'s fitness-axis variance loading on a factor common to the screens, estimated by triad identities over all triads of screens that include it (median and 95\% interval over triads and cell-line bootstrap); ln IC$_{50}$ rows replace the Sanger AUC axes by their ln IC$_{50}$ axes. Spearman correlation with the doubling time measured by gCSI.',
                              'tab:shared', ['Screen', 'Endpoint', 'Triads', '$\\lambda^2$', '95\\% CI', 'Doubling-time $\\rho$', 'Lines'],
                              rows, 'llrrrrr')))

    var = pd.DataFrame(rob['variants'])
    rows = [[esc(r.variant), r.fitness_r2_PRISM, r.fitness_r2_CTRP, r.fitness_r2_GDSC1, r.fitness_r2_GDSC2,
             f'{r.fitness_axis_corr_min:.2f}--{r.fitness_axis_corr_max:.2f}', r.median_raw, r.median_sel] for r in var.itertuples()]
    bl = rob['beyond_lineage']
    rows.append(['selective, lineage means also removed', None, None, None, None, None, None, bl['median_sel_beyond_lineage']])
    T.append(('Variants', table(r'\textbf{Robustness of the decomposition.} Median fitness share per screen, range of cross-screen fitness-axis correlations, and median raw and selective cross-screen agreement of the 353 multi-screen drugs under alternative constructions of the axis. Last row: selective residuals additionally freed of lineage means in each screen (share reproducible ' + f"{100 * bl['frac_reproducible_beyond_lineage']:.0f}\\% versus {100 * bl['frac_reproducible_sel']:.0f}\\%" + ').',
                                'tab:variants', ['Variant', 'PRISM $R^2$', 'CTRP $R^2$', 'GDSC1 $R^2$', 'GDSC2 $R^2$', 'Axis $r$ range', 'Raw', 'Selective'],
                                rows, 'p{4.0cm}rrrrrrr')))

    es = jl('endpoint_sensitivity/summary.json')
    rows = [[k.replace('|', ' ').replace('_', '\\_'), v['drugs'], v['cells'], v['split_half'], v['median_fitness_r2'], v['within_fitness_only_per_drug']]
            for k, v in es['per_endpoint'].items()]
    T.append(('Endpoint', table(r'\textbf{Endpoint sensitivity of the fitness axis.} Within-screen split-half reliability, median fitness share and the drug-agnostic predictor\'s median within-screen per-drug correlation for each screen and dose-response summary.',
                                'tab:endpoint', ['Screen, endpoint', 'Drugs', 'Lines', 'Split-half', 'Fitness $R^2$', 'Fitness-only per-drug'], rows, 'lrrrrr')))
    tab = pd.DataFrame(es['pairs'])
    rows = [[r.a.replace('|', ' ').replace('_', '\\_'), r.b.replace('|', ' ').replace('_', '\\_'), r.cells, r.axis_corr, r.disattenuated, r.drugs,
             r.median_raw, r.median_sel, r.fitness_a_on_b_per_drug] for r in tab.itertuples()]
    T.append(('EndpointPairs', table(r'\textbf{Agreement of fitness axes and drug profiles across endpoints and screens.} Axis correlation over shared cell lines (and disattenuated), median raw and selective cross-screen agreement of shared drugs, and the median per-drug correlation of screen A\'s axis with screen B\'s profiles.',
                                     'tab:endpointpairs', ['A', 'B', 'Lines', 'Axis $r$', 'Disatt.', 'Drugs', 'Raw', 'Selective', 'A-fitness on B'], rows, 'llrrrrrrr')))

    # ---------------- evaluation
    s = pd.read_csv(RESULTS / 'models' / 'score_summary.csv')
    nm = {'fitness_measured': 'measured fitness', 'fitness_omics': 'omics fitness', 'lineage_raw': 'lineage only',
          'ridge_raw': 'ridge (raw)', 'ridge_sel': 'ridge (selective)', 'ridge_crispr': 'ridge + CRISPR',
          'target_prior': 'target dependency', 'replicate': 'replicate'}
    rows = []
    for m in nm:
        w = s[(s.model == m) & (s['train'] == s['eval'])].set_index('train')
        c = s[(s.model == m) & (s['train'] != s['eval'])]
        rows.append([nm[m]] + [w.raw.get(k, np.nan) if m != 'replicate' else None for k in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']]
                    + [c.raw.median(), c.sel.median()])
    T.append(('CellBlind', table(r'\textbf{Cell-blind predictors.} Median per-drug correlation within the training screen (raw), and medians over the 12 train/evaluation screen pairs of the per-pair median raw and selective correlations.',
                                 'tab:cellblind', ['Predictor', 'PRISM', 'CTRP', 'GDSC1', 'GDSC2', 'Other (raw)', 'Other (sel.)'], rows, 'lrrrrrr')))
    rows = []
    for f, lab in [(RESULTS / 'global_model' / 'scores.csv', 'GBT'), (RESULTS / 'global_mlp' / 'scores.csv', 'MLP')]:
        if not f.exists():
            continue
        g = pd.read_csv(f)
        g['split2'] = g.split.where(~g.split.str.startswith('cross_from'), 'cross-screen')
        for (m, sp), gg in g.groupby(['model', 'split2']):
            constant = m == 'drug_mean'     # constant within a drug: per-drug correlations undefined
            rows.append([lab if m.startswith('global') else 'baseline', m, sp, gg.global_r.median(),
                         None if constant else gg.per_drug_raw.median(), None if constant else gg.per_drug_sel.median()])
    T.append(('Global', table(r'\textbf{Global models under different split designs.} Medians over screens (or over train/evaluation pairs for cross-screen).',
                              'tab:global', ['Family', 'Model', 'Split', 'Global $r$', 'Per-drug raw', 'Per-drug selective'], rows, 'lllrrr')))

    D = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv')
    D = D[D.model != "fitness_measured"].rename(columns={"class": "klass"})
    dname = {'ridge_sel': 'ridge, selective (multi-omics)', 'ridge_raw': 'ridge, raw (multi-omics)',
             'ridge_sel_expr': 'ridge, selective (expression)', 'ridge_raw_expr': 'ridge, raw (expression)',
             'ridge_sel_lm': 'ridge, selective (landmark genes)', 'ridge_raw_lm': 'ridge, raw (landmark genes)',
             'fitness_only': 'fitness only (drug-agnostic)'}
    cname = {'neural/factorization': 'neural/fact.'}
    rows = []
    for r in D.itertuples():
        kl = cname.get(r.klass, r.klass)
        if r.drugs == 0:
            rows.append([esc(dname.get(r.model, r.model)), esc(kl), 0, None, None, None, None, None, r.dreval_pearson_norm])
            continue
        rows.append([esc(dname.get(r.model, r.model)), esc(kl), int(r.drugs), r.raw,
                     f'{r.selective:.3f} ({r.selective_lo:.3f}--{r.selective_hi:.3f})', r.mediated_share,
                     r.prediction_fitness_r2, r.reproducible_selective, r.dreval_pearson_norm])
    fm = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv').set_index('model').loc['fitness_measured']
    T.append(('DrEval', table(r'\textbf{The DrEval leaderboard re-scored with the selectivity-aware metric.} CTRPv2 ln IC$_{50}$ (CurveCurator), leave-cell-line-out, 10 splits, ' + f"{dre['common_cells']} cell lines common to all models and {dre['drugs_evaluated']} drugs" + r'. Raw: median per-drug correlation; selective: median partial correlation given the screen\'s leave-drug-out fitness axis (95\% drug-bootstrap interval); through fitness: share of the mean per-drug correlation carried by the fitness axis ($r_{pg}r_{zg}$); prediction fitness $R^2$: median share of each drug\'s predicted profile explained by the leave-drug-out mean of the other predicted profiles (observed screen: ' + f"{dre['median_fitness_r2_observed']:.2f}" + r'); reproducible: selective median over drugs reproducible in discovery; DrEval: the leaderboard\'s own normalized Pearson (mean over splits). The measured fitness axis reaches a raw median of ' + f"{fm.raw:.2f}" + r'. Models of this study were trained on the same splits with DepMap 24Q4 features. NaivePredictor and NaiveDrugMeanPredictor are constant within a drug.',
                              'tab:dreval', ['Model', 'Class', 'Drugs', 'Raw', 'Selective (95\\% CI)', 'Thr.\\ fitness', 'Pred.\\ fit.\\ $R^2$', 'Repro.', 'DrEval'],
                              rows, 'p{3.6cm}p{1.85cm}rr@{~}lrrrr', small=False).replace(r'\begin{table}[h!]', r'\begin{table}[h!]' + '\n' + r'\scriptsize' + '\n' + r'\setlength{\tabcolsep}{3.5pt}')))

    # ---------------- cross-laboratory re-scoring (leaderboard) and DrEval's full results by test mode
    X = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'summary.csv')
    xn = {'ridge_sel_lm': 'ridge, selective (landmark genes)', 'ridge_raw_lm': 'ridge, raw (landmark genes)',
          'ridge_sel_expr': 'ridge, selective (expression)', 'ridge_raw_expr': 'ridge, raw (expression)',
          'ridge_sel': 'ridge, selective (multi-omics)', 'ridge_raw': 'ridge, raw (multi-omics)', 'fitness_only': 'fitness only'}
    W = {t: X[X.target == t].set_index('model') for t in ['GDSC1', 'GDSC2', 'CTRPv1']}
    order = W['GDSC1'].sort_values('sel_cross', ascending=False).index
    rows = []
    for m in order:
        if m == 'SVR':
            continue
        g1, g2, c1 = W['GDSC1'].loc[m], W['GDSC2'].loc[m], W['CTRPv1'].loc[m]
        rows.append([esc(xn.get(m, m)), esc(cname.get(g1['class'], g1['class'])), g1.raw_within, g1.raw_cross, g2.raw_cross,
                     g1.sel_within, g1.sel_cross, g2.sel_cross,
                     g1.mediated_retained, g2.mediated_retained, c1.mediated_retained])
    T.append(('CrossLab', table(r'\textbf{The same leaderboard predictions scored in CTRPv2 and in other laboratories.} Leave-cell-line-out predictions of each model (trained on CTRPv2) scored against CTRPv2 and against GDSC1, GDSC2 and CTRPv1 measurements of the same drugs and cell lines (' + f"{len([m for m in W['GDSC1'].index if m != 'SVR'])} predictors; per drug, the cell lines measured in both screens; published models scored on GDSC1: {int(W['GDSC1'].loc[W['GDSC1']['class'] != 'this study'].drugs.median())} drugs, GDSC2: {int(W['GDSC2'].loc[W['GDSC2']['class'] != 'this study'].drugs.median())}, CTRPv1: {int(W['CTRPv1'].loc[W['CTRPv1']['class'] != 'this study'].drugs.median())}" + r'). Raw and selective: medians over drugs. Fitness retained: mean fitness-mediated term in the other screen divided by that in CTRPv2. CTRPv1 is from the same laboratory as CTRPv2; GDSC2 shares its viability readout; GDSC1 shares neither. Pooled estimates with intervals and paired tests are in Supplementary Table~\stCrossLabSummary.',
                                'tab:crosslab', ['Model', 'Class', 'Raw CTRPv2', 'Raw GDSC1', 'Raw GDSC2', 'Sel. CTRPv2', 'Sel. GDSC1', 'Sel. GDSC2',
                                                 'Fit. ret. GDSC1', 'GDSC2', 'CTRPv1'],
                                rows, 'p{3.4cm}lrrrrrrrrr', small=False).replace(r'\begin{table}[h!]', r'\begin{table}[h!]' + '\n' + r'\scriptsize' + '\n' + r'\setlength{\tabcolsep}{2.6pt}')))
    U = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'uncertainty.csv')
    PC = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'paired_selective_cross.csv')
    rows = []
    for t, lab in [('GDSC1', 'GDSC1 (other lab, other readout)'), ('GDSC2', 'GDSC2 (other lab, same readout)'),
                   ('CTRPv1', 'CTRP v1 (same lab)'), ('CCLE', 'CCLE (17 drugs)')]:
        d = X[(X.target == t) & X['class'].isin(['classical', 'neural/factorization']) & (X.model != 'SVR')]
        q = d[(d.raw_drop > 0.03) & (d.meanraw_within > 0.1)]
        u = U[(U.target == t) & (U.model == 'pooled published models')].iloc[0]
        c = PC[(PC.target == t) & (PC.ours == 'ridge_sel_lm')]
        ci = lambda k: f"{u[k]:.2f} ({u[k + '_lo']:.2f}--{u[k + '_hi']:.2f})"
        rows.append([lab, len(d), int(d.drugs.median()), f"{d.raw_within.median():.2f}/{d.raw_cross.median():.2f}",
                     f"{q.share_of_drop_from_fitness.median():.2f} ({len(q)})" if len(q) >= 3 else '--',
                     ci('share') if t in ('GDSC1', 'GDSC2') else '--', ci('mediated_retained'), ci('selterm_retained'),
                     f"{c.median_diff.min():.2f}; {pfmt(c.wilcoxon_p.max())}; {int((c.wilcoxon_p < 0.05).sum())}/{len(c)}"])
    T.append(('CrossLabSummary', table(r'\textbf{Cross-laboratory decomposition of the published leaderboard models.} Same predictions, drugs and cell lines (per drug, the cell lines measured in both screens). Raw: median over models of the per-drug medians in CTRPv2/other screen. Share (median, models losing $>$0.03): fitness-mediated share of the loss in mean per-drug correlation. Pooled: sums over models and drugs with 95\% drug-bootstrap intervals for the share and for the retained fractions of the fitness-mediated and selective terms. Paired: this study\'s selective-target ridge (landmark genes) versus each published model on cross-laboratory selectivity --- smallest median paired gain; largest Wilcoxon $P$; models beaten at $P<0.05$.',
                                       'tab:crosslabsummary', ['Screen', 'Models', 'Drugs', 'Raw', 'Share', 'Share (pooled)', 'Fitness retained', 'Selective retained', 'Paired'],
                                       rows, 'p{2.6cm}rrlllllp{2.4cm}', small=False).replace(r'\begin{table}[h!]', r'\begin{table}[h!]' + '\n' + r'\scriptsize' + '\n' + r'\setlength{\tabcolsep}{3pt}')))
    M = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    M = M[~M.algorithm.str.startswith('Naive') & ~M.algorithm.str.startswith('ref:')]
    R0 = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    rows = []
    for mode in ['LCO', 'LTO', 'LPO', 'LDO']:
        for t in ['CTRPv1', 'CCLE', 'GDSC2', 'GDSC1']:
            d = M[(M['mode'] == mode) & (M.target == t)]
            if not len(d):
                continue
            q = d[(d.raw_drop > 0.03) & (d.meanraw_within > 0.1)]
            ref = R0[(R0.algorithm == 'ref:CTRPv2_measured') & (R0.target == t) & (R0['mode'] == mode)]
            rows.append([mode, t, len(d), int(d.drugs.median()), d.raw_within.median(), d.raw_cross.median(), d.sel_within.median(),
                         d.sel_cross.median(), q.share_of_drop_from_fitness.median() if len(q) >= 3 else None,
                         d.mediated_retained.median(), ref.sel_cross.iloc[0] if len(ref) else None])
    T.append(('CrossModes', table(r'\textbf{DrEval\'s full published results: within- versus cross-laboratory scores by test mode.} Out-of-fold predictions of the DrEval paper\'s CTRPv2-trained models (LCO, unseen cell lines; LTO, unseen tissues; LPO, random pairs; LDO, unseen drugs) scored against CTRPv2 and against the same drug--cell-line pairs measured in another screen. Medians over models of the per-model drug medians. Share via fitness: fitness-mediated share of the loss in mean per-drug correlation (models losing $>$0.03). Ceiling: selective correlation of CTRPv2\'s own measurement with the other screen.',
                                  'tab:crossmodes', ['Mode', 'Screen', 'Models', 'Drugs', 'Raw within', 'Raw cross', 'Sel. within', 'Sel. cross', 'Share via fitness', 'Fit. retained', 'Ceiling'],
                                  rows, 'llrrrrrrrrr', small=False).replace(r'\begin{table}[h!]', r'\begin{table}[h!]' + '\n' + r'\scriptsize')))
    NM = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv')
    nn = dict(xn, fitness_increment='fitness only (increment over naive)', fitness_increment_tissue_centred='fitness only (within-tissue increment)')
    ND = RESULTS / 'revision_round5' / 'normalized_decomposition' / 'normalized_decomposition.csv'
    ND = pd.read_csv(ND).set_index('model') if ND.exists() else None   # 106: the same metric with the axis removed
    def nd(m, col):
        return ND.at[m, col] if ND is not None and m in ND.index else None
    def frac(m):
        v = nd(m, 'axis_fraction')
        return v if v is not None and nd(m, 'normalized') is not None and nd(m, 'normalized') > 0.1 else None
    rows = [[esc(nn.get(r.model, r.model)), esc(cname.get(r['class'], r['class'])), r.normalized_pearson, r.split_sd,
             nd(r.model, 'axis_adjusted'), frac(r.model)] for _, r in NM.iterrows() if r.model != 'SVR']
    T.append(('Normalized', table(r'\textbf{DrEval\'s own headline metric.} Normalized Pearson as defined by drevalpy (observed and predicted responses minus the NaiveMeanEffectsPredictor prediction; Pearson over the pairs of each split; mean and s.d.\ over the ten splits), on the common drugs and cell lines. The re-implementation reproduces the leaderboard\'s reported values to $<$10$^{-3}$. This study\'s predictors were placed on the ln IC$_{50}$ scale with training-cell statistics only (Methods). Axis-adjusted: the same residuals also regressed, within each split and drug, on the evaluation dataset\'s general-sensitivity axis before pooling (script 106); axis fraction: 1 minus axis-adjusted over normalized on the same pairs, shown for models with a normalized Pearson above 0.1. The fitness-only rows rank cell lines identically for every drug.',
                                  'tab:normalized', ['Model', 'Class', 'Normalized Pearson', 's.d.', 'Axis-adjusted', 'Axis fraction'], rows, 'p{5.0cm}lrrrr')))

    A = pd.read_csv(RESULTS / 'dreval' / 'model_axis_adjusted.csv')
    W = A.pivot_table(index=['model', 'class'], columns='where', values=['selective', 'adjusted']).reset_index()
    W.columns = ['_'.join(c).strip('_') for c in W.columns]
    W = W.sort_values('adjusted_CTRPv2', ascending=False)
    rows = [[esc(xn.get(r.model, r.model)), esc(cname.get(r['class'], r['class'])), r.selective_CTRPv2, r.adjusted_CTRPv2,
             r.selective_GDSC1, r.adjusted_GDSC1, r.selective_GDSC2, r.adjusted_GDSC2] for _, r in W.iterrows() if r.model != 'SVR']
    T.append(('Adjusted', table(r'\textbf{Selectivity hidden by a model\'s own fitness estimate.} Median selective correlation (partial correlation given the evaluation screen\'s fitness axis) and model-axis-adjusted selective correlation (additionally given the model\'s own predicted fitness axis, the leave-drug-out mean of its rank-normal predicted profiles), within CTRPv2 and, with identical predictions and cell lines, in GDSC1 and GDSC2. A raw-trained model\'s imperfect fitness estimate dilutes its selective score; removing its own axis reveals the drug-specific signal it contains.',
                                'tab:adjusted', ['Model', 'Class', 'CTRPv2', 'adj.', 'GDSC1', 'adj.', 'GDSC2', 'adj.'], rows,
                                'p{4.0cm}lrrrrrr', small=False).replace(r'\begin{table}[h!]', r'\begin{table}[h!]' + '\n' + r'\scriptsize')))
    TP = pd.read_csv(RESULTS / 'dreval' / 'two_part_summary.csv')
    tpn = dict(xn, two_part_lm='two-part: fitness + selective (landmark genes)', two_part_expr='two-part: fitness + selective (expression)')
    rows = [[esc(tpn.get(r.model, r.model)), esc(r['where']), None if pd.isna(r.drugs) else int(r.drugs), r.raw,
             None if pd.isna(r.selective) else r.selective, None if pd.isna(r.mediated_share) else r.mediated_share] for _, r in TP.iterrows()]
    T.append(('TwoPart', table(r'\textbf{A two-part model: omics-predicted fitness axis plus per-drug selective model.} The prediction is $b_d\hat f_c+\hat r_{dc}$, with $b_d$ the training cells\' slope of drug $d$ on the fitness axis, $\hat f$ the fitness-only ridge and $\hat r$ the selective-target ridge. Raw and selective: medians over drugs; for the DrEval row, raw is DrEval\'s normalized Pearson. Adding the predicted fitness axis restores raw accuracy but, through its prediction error, lowers the selective score to that of the raw-target ridge.',
                               'tab:twopart', ['Model', 'Scored in', 'Drugs', 'Raw', 'Selective', 'Thr.\\ fitness'], rows, 'p{5.2cm}p{3.4cm}rrrr')))

    # ---------------- selectivity, mechanisms, biomarkers
    cs = pd.read_csv(RESULTS / 'drug_classes' / 'class_summary.csv', index_col=0).sort_values('median_rho_sel', ascending=False)
    rows = [[esc(c), int(r.drugs), r.median_rho_sel, f'{100 * r.frac_reproducible:.0f}\\%', r.median_fitness_r2,
             int(r.gcsi_n), r.gcsi_median if r.gcsi_n else None, int(r.nci_n), r.nci_median if r.nci_n else None] for c, r in cs.iterrows()]
    T.append(('Classes', table(r'\textbf{Selective reproducibility by mechanism class.} Discovery: median mean-selective cross-screen correlation, share reproducible ($\geq$0.3), median fitness share. Confirmation: median selective correlation of the discovery consensus measurement with gCSI and NCI-60.',
                               'tab:classes', ['Class', 'Drugs', 'Median $\\rho_{\\mathrm{sel}}$', 'Repro.', 'Fitness $R^2$', 'gCSI n', 'gCSI', 'NCI-60 n', 'NCI-60'],
                               rows, 'p{4.2cm}rrrrrrrr')))
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0).sort_values('rho_sel', ascending=False).head(30)
    conf = {k: pd.read_csv(RESULTS / f'confirmation_{k}' / 'scores.csv') for k in CONFIRM}
    conf = {k: v[v.family == 'rep_sel_cons'].set_index('compound').sel for k, v in conf.items()}
    ba = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    ba = ba[ba.family == 'sel'].set_index('compound').sel
    rows = []
    for c, r in dt.iterrows():
        vals = [conf[k].get(c, np.nan) for k in CONFIRM]
        rows.append([esc(names.get(c, r['name'])), esc(r['class']), int(r.n_pairs), r.rho_sel] + vals + [ba.get(c, np.nan)])
    T.append(('TopDrugs', table(r'\textbf{The 30 most reproducibly selective discovery drugs.} Mean selective cross-screen correlation over discovery screen pairs; selective agreement of the discovery consensus with each untouched screen; selective correlation of the frozen transfer prediction with BeatAML patient samples.',
                                'tab:topdrugs', ['Drug', 'Class', 'Pairs', '$\\rho_{\\mathrm{sel}}$', 'gCSI', 'NCI-60', 'CCLE', 'FIMM', 'GRAY', 'BeatAML'],
                                rows, 'p{2.6cm}p{3.0cm}rrrrrrrr')))

    bio = jl('biomarkers/summary.json')
    rep = pd.DataFrame(bio['replication'])
    rows = [[r.kind, r.ftype, int(r.discoveries), int(r.tested_pairs), int(r.replicated), f'{100 * r.replication_rate:.0f}\\%']
            for r in rep.sort_values(['ftype', 'kind']).itertuples()]
    rows += [[r['kind'], f"top {r['K']}", None, r['tested_pairs'], None, f"{100 * r['replication_rate']:.0f}\\%"] for r in bio['topk']]
    T.append(('Biomarkers', table(r'\textbf{Replication of single-feature biomarker associations between discovery screens.} Spearman associations of expression (1,000 most variable genes), mutations (genes mutated in $\geq$3\% of screened lines) and lineage (lineages with $\geq$10 lines) with raw or selective profiles of the 353 multi-screen drugs, discovered in one screen (Benjamini--Hochberg $q<0.05$ within screen, profile and feature type) and tested in every other screen measuring the drug (replication: same sign and one-sided $P<0.05$). Top-$K$ rows take each screen\'s $K$ strongest associations per profile type.',
                                  'tab:biomarkers', ['Profile', 'Features', 'Discoveries', 'Tested pairs', 'Replicated', 'Rate'], rows, 'llrrrr')))

    # ---------------- value of information
    v = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_two_way.csv')
    rows = []
    for _, r in v.iterrows():
        lo, hi = [float(x) for x in str(r.map_ci95).strip('()[]').split(',')]
        rows.append([LBL[r['eval']], esc(r.features), int(r.pairs), int(r.drugs), r.strength_r if np.isfinite(r.strength_r) else None,
                     r.map_r, f'{lo:.2f}--{hi:.2f}'])
    T.append(('Lthousand', table(r'\textbf{Value of information of L1000 responses (two-way-centred design).} Response strength and drug-blind ridge-map correlations with the selective residual; 95\% drug-bootstrap intervals for the map.',
                                 'tab:lthousand', ['Screen', 'Features', 'Pairs', 'Drugs', 'Strength $r$', 'Map $r$', '95\\% CI'], rows, 'llrrrrr')))
    inc = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_incremental.csv')
    rows = [[LBL[r['eval']], LBL[r.basal_train], int(r.pairs), r.basal_r, r.measured_r, r.measured_given_basal, r.combined_r] for _, r in inc.iterrows()]
    T.append(('Incremental', table(r'\textbf{Incremental value of measured L1000 responses over cell-blind basal-omics models.}',
                                   'tab:incremental', ['Screen', 'Basal model from', 'Pairs', 'Basal $r$', 'Measured $r$', 'Measured $\\mid$ basal', 'Combined $r$'], rows, 'llrrrrr')))
    rb = pd.read_csv(RESULTS / 'value_of_information' / 'retrieval_baselines.csv')
    fl = {'measured': 'measured response', 'nearest_chemistry': 'nearest training chemistry', 'same_target': 'shared-target training drugs',
          'measured_minus_1pc': 'measured, top PC removed', 'measured_minus_3pc': 'measured, top 3 PCs removed',
          'measured_minus_10pc': 'measured, top 10 PCs removed', 'measured_on_same_target_pairs': 'measured (shared-target pairs)'}
    rows = []
    for _, r in rb[rb.features.isin(fl)].iterrows():
        lo, hi = [float(x) for x in str(r.map_ci95).strip('()[]').split(',')]
        rows.append([LBL[r['eval']], fl[r.features], int(r.pairs), int(r.drugs), r.map_r, f'{lo:.2f}--{hi:.2f}'])
    T.append(('Retrieval', table(r'\textbf{Retrieval baselines and removal of global response programs.} Drug-blind ridge maps scored as in Supplementary Table~\stLthousand. Profiles available for unseen chemistry were built only from measured L1000 responses of the same cell line in the other chemical-scaffold folds: the response to the most similar training compound (Morgan fingerprint Tanimoto; nearest training chemistry) or the mean response to training compounds sharing an annotated target (shared-target). PC removal projects the measured responses off their leading principal components before mapping.',
                                 'tab:retrieval', ['Screen', 'Features', 'Pairs', 'Drugs', 'Map $r$', '95\\% CI'], rows, 'llrrrr')))

    # ---------------- confirmation
    rows = []
    for name, lab in [('gCSI', 'gCSI (primary)'), ('NCI60', 'NCI-60 (primary)'), ('CCLE', 'CCLE (Amendment 1)'),
                      ('FIMM', 'FIMM (Amendment 1)'), ('GRAY', 'GRAY (Amendment 1)'), ('gCSI_GRAOC', 'gCSI GR_AOC'),
                      ('NCI60_cellminerQC', 'NCI-60 CellMiner QC'), ('rehearsal_GDSC2', 'GDSC2 rehearsal')]:
        h = jl(f'confirmation_{name}/hypotheses.json')
        dis = max(x['disattenuated'] for x in h['H1']['by_discovery'] if x.get('disattenuated') is not None)
        rows.append([lab, f"{h['drugs']:,}", h['cells'], dis, h['H2']['ratio'],
                     f"{h['H3']['median_reproducible']:.2f}/{h['H3']['median_non_reproducible']:.2f}", pfmt(h['H3'].get('mwu_p_one_sided')),
                     h['H4']['median_difference'], pfmt(h['H4']['wilcoxon_p_one_sided']),
                     f"{h['H5']['median_model_sel']:.2f}/{h['H5']['median_replicate_sel']:.2f}"])
    T.append(('Confirm', table(r'\textbf{Confirmatory statistics for all untouched screens, secondary endpoints and the GDSC2 rehearsal.} H1: largest disattenuated fitness-axis correlation (threshold $<$0.70). H2: selective/raw ratio ($\geq$0.80). H3: reproducible/non-reproducible medians and one-sided Mann--Whitney $P$. H4: median own-minus-discovery fitness correlation ($\geq$0.15) and one-sided Wilcoxon $P$. H5: model/measurement medians for reproducible drugs.',
                               'tab:confirm_all', ['Analysis', 'Drugs', 'Lines', 'H1', 'H2', 'H3', 'H3 $P$', 'H4', 'H4 $P$', 'H5'], rows, 'p{2.8cm}rrrrrrrrr')))

    b = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    rows = []
    for (fam, cls), g in b.groupby(['family', 'class']):
        rows.append([f'{fam}-trained', cls.replace('_', '-'), len(g), g.raw.median(), g.sel.median()])
    top = b[b.family == 'sel'].sort_values('sel', ascending=False).head(15)
    rows += [[esc(names.get(r.compound, r.compound)), r['class'].replace('_', '-'), int(r.n), r.raw, r.sel] for _, r in top.iterrows()]
    T.append(('BeatAML', table(r'\textbf{Transfer of cell-line-trained predictions to primary AML samples (BeatAML).} Upper block: medians by discovery reproducibility class for selective- and raw-trained frozen transfer predictions (raw: correlation with the inhibitor\'s within-inhibitor rank-normal profile; selective: partial correlation given BeatAML\'s leave-inhibitor-out fitness axis). Lower block: the 15 inhibitors with the highest selective transfer (selective-trained predictions; $n$, samples).',
                               'tab:beataml', ['Predictions / inhibitor', 'Class', 'n', 'Raw', 'Selective'], rows, 'p{4.0cm}lrrr')))

    px = pd.read_csv(RESULTS / 'confirmation_PDX_Bruna' / 'scores.csv')
    rows = []
    for (fam, cls), g in px.groupby(['family', 'class']):
        rows.append([f'{fam}-trained', cls.replace('_', '-'), len(g), g.raw.median(), g.sel.median()])
    top = px[px.family == 'sel'].sort_values('sel', ascending=False).head(15)
    rows += [[esc(names.get(r.compound, r.compound)), r['class'].replace('_', '-'), int(r.n), r.raw, r.sel] for _, r in top.iterrows()]
    PXi = jl('confirmation_PDX_Bruna_lnIC50/hypotheses.json')
    T.append(('PDX', table(r'\textbf{Transfer of cell-line-trained predictions to patient-derived breast-cancer models (PDX\_Bruna; Amendment 2).} As Supplementary Table~\stBeatAML, for ' + f"{PX['drugs']} drugs measured in $\\geq$20 of {PX['models']} models (primary endpoint CurveCurator AUC; fitness-axis split-half reliability {PX['split_half_reliability']:.2f}); predictions exist for the 25 models with expression ($n$, models). Families with \_cn add GISTIC copy number. Secondary endpoint (ln IC$_{{50}}$; {PXi['drugs']} drugs): H8 median {PXi['H8']['median_sel']:.2f}, $P$ = {PXi['H8']['wilcoxon_p_one_sided']:.2g}; H9 {PXi['H9']['median_reproducible']:.2f} vs {PXi['H9']['median_non_reproducible']:.2f}, $P$ = {PXi['H9']['mwu_p_one_sided']:.2g}.",
                           'tab:pdx', ['Predictions / drug', 'Class', 'n', 'Raw', 'Selective'], rows, 'p{4.0cm}lrrr')))

    prot = PROCESSED / 'screen_protocols.csv'
    if prot.exists():
        rows = [['PRISM', 'Broad (PRISM)', 'barcode mRNA abundance', '120 h', '384, pooled', 'single (RPMI)', 'no'],
                ['CTRP v2', 'Broad (CTRP)', 'CellTiter-Glo', '72 h', '1536', 'per line', 'no'],
                ['CTRP v1', 'Broad (CTRP)', 'CellTiter-Glo', '72 h', '384', 'per line', 'no'],
                ['GDSC1', 'Sanger + MGH', 'Syto60 / resazurin', '72 h', '96/384', 'two standard media', 'no'],
                ['GDSC2', 'Sanger', 'CellTiter-Glo', '72 h', '1536', 'two standard media', 'growth plates only'],
                ['gCSI', 'Genentech', 'CellTiter-Glo', '72 h', '384', '5\% FBS (base n.v.)', 'no (nominal doubling)'],
                ['NCI-60', 'NCI DTP', 'SRB (protein)', '48 h', '96', 'single (RPMI)', 'yes'],
                ['CCLE', 'Novartis/GNF', 'CellTiter-Glo', '72--84 h', '1536', 'two standard media', 'no'],
                ['FIMM', 'FIMM', 'CellTiter-Glo', '72 h', '384', 'n.v.', 'no'],
                ['GRAY', 'Gray lab (LBNL/OHSU)', 'CellTiter-Glo', '72 h', '96', 'per line', 'yes (original assay)'],
                ['BeatAML', 'OHSU (ex vivo AML)', 'MTS', '72 h', '384', 'single (RPMI)', 'no'],
                ['PDX\_Bruna', 'Caldas lab (CRUK Cambridge)', 'CellTiter-Glo', '144 h', '384', 'single (serum-free MEGM)', 'no']]
        T.append(('Protocols', table(r'\textbf{Screening protocols.} Laboratory, viability readout, drug exposure, plate format, culture-medium policy and use of a time-zero (pre-treatment) measurement, compiled from the original publications, supplements and portals. Sources, quotations and confidence for every field are in \texttt{data/processed/screen\_protocols\_sources.md}; n.v., not verified (methods not accessible). PharmacoDB\'s GRAY values correspond to the GR re-release of the Gray-laboratory screen.',
                                     'tab:protocols', ['Screen', 'Laboratory', 'Readout', 'Exposure', 'Plate', 'Medium', 'Time zero'], rows,
                                     'lllllll')))

    sim = jl('simulation/summary.json')['runs']
    rows = [[r['rho_f'] ** 2, '/'.join(f'{x:.1f}' for x in r['noise_sd']), r['reliability_true_median'], r['reliability_est'],
             r['c2_true_median'], r['c2_est']] for r in sim]
    T.append(('Simulation', table(r'\textbf{Simulation check of the ceiling estimators.} Three synthetic screens (250 drugs, 400 cell lines) with shared drug-specific selectivity, partly shared fitness axes (true axis correlation shown) and screen-specific noise (s.d.\ per screen), processed with the study pipeline.',
                                  'tab:simulation', ['Axis $r$ (true)', 'Noise s.d.', 'Reliability (true)', 'Reliability (est.)', '$c^2$ (true)', '$c^2$ (est.)'],
                                  rows, 'rlrrrr')))

    T = [kv for kv in T if kv[0] not in DROP]
    T.append(('Registered', table1()))
    import importlib
    rev = importlib.import_module('65_revision_tables')
    T += rev.tables({'table': table, 'fmt': fmt, 'pfmt': pfmt, 'esc': esc})
    keys = [k for k, _ in T]
    first = [k for k in citation_order() if k in keys] + [k for k in ORDER if k in keys]
    first = list(dict.fromkeys(first))      # tables numbered by first citation in the manuscript, then ORDER
    T = sorted(T, key=lambda kv: (first.index(kv[0]) if kv[0] in first else len(first) + keys.index(kv[0])))
    refs = [rf'\newcommand{{\st{k}}}{{{i + 1}}}' for i, (k, _) in enumerate(T)]
    refs.append(rf'\newcommand{{\stCount}}{{{len(T)}}}')
    # Supplementary Data 1: machine-readable versions of every supplementary table
    sd = MAN / 'supplementary_data'
    sd.mkdir(exist_ok=True)
    for old in sd.glob('Supplementary_Table_*.csv'):
        old.unlink()
    book, index = {}, []
    for i, (k, tex) in enumerate(T):
        m = re.search(r'\\label\{([^}]*)\}', tex)
        if m and m.group(1) in TABLE_DATA:
            hdr, rws = TABLE_DATA[m.group(1)]
            width = max([len(r) for r in rws] + [len(hdr)])
            cols = hdr + [f'col{j + 1}' for j in range(len(hdr), width)]
            df = pd.DataFrame([r + [''] * (width - len(r)) for r in rws], columns=cols)
            df.to_csv(sd / f'Supplementary_Table_{i + 1:02d}_{k}.csv', index=False)
            book[f'Table {i + 1}'] = df
        elif '\\begin{tabular' in tex:           # hand-built tables (tabular, tabularx): read the typeset rows
            lines = tex.split('\n')
            body, hdr, in_body = [], None, False
            for j, ln in enumerate(lines):
                if ln.strip() == '\\midrule' and hdr is None:
                    hdr = [plain(terms(c)) for c in lines[j - 1].rstrip('\\ ').split('&')]
                    in_body = True
                    continue
                if ln.strip().startswith('\\bottomrule'):
                    break
                if in_body and '&' in ln and not ln.strip().startswith('\\midrule'):
                    body.append([plain(terms(c)) for c in ln.strip().rstrip('\\').rstrip().rstrip('\\').split('&')])
            if hdr and body:
                width = max([len(r) for r in body] + [len(hdr)])
                cols = hdr + [f'col{j + 1}' for j in range(len(hdr), width)]
                df = pd.DataFrame([r + [''] * (width - len(r)) for r in body], columns=cols)
                df.to_csv(sd / f'Supplementary_Table_{i + 1:02d}_{k}.csv', index=False)
                book[f'Table {i + 1}'] = df
        index.append({'Supplementary Table': i + 1, 'Key': k, 'Title and legend': caption_text(tex)})
    # the submitted Supplementary Tables: one Excel workbook (an index sheet, then one sheet per table)
    try:
        with pd.ExcelWriter(sd / 'Supplementary_Tables.xlsx') as xw:
            pd.DataFrame(index).to_excel(xw, sheet_name='Index', index=False)
            for name, df in book.items():
                df.to_excel(xw, sheet_name=name, index=False)
    except ImportError:
        print('openpyxl not available: Supplementary_Tables.xlsx not written')
    figs = [f for f in SUPP_FIGURES if (PROJECT / 'figures' / f'{f[1]}.pdf').exists()]
    refs += [rf'\newcommand{{\sf{k}}}{{{i + 1}}}' for i, (k, _, _, _) in enumerate(figs)]
    out = []
    for k, name, width, cap in figs:
        out += [r'\begin{figure}[H]', r'\centering', rf'\includegraphics[width={width}\textwidth]{{../figures/{name}.pdf}}',
                rf'\caption{{{cap}}}', rf'\label{{sfig:{k}}}', r'\end{figure}', '']
    (MAN / 'supplement_figures.tex').write_text('\n'.join(out))
    (MAN / 'supp_refs.tex').write_text('\n'.join(refs) + '\n')
    # captions are raw strings: an escaped apostrophe would become a LaTeX acute accent
    (MAN / 'supplement_tables.tex').write_text(breakable_names('\n'.join(t for _, t in T).replace("\\'", "'")))
    print('written', len(T), 'supplementary tables')


if __name__ == '__main__':
    main()
