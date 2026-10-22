"""Assemble manuscript/overleaf_package.zip from the current manuscript sources (figure paths rewritten), with the
passages revised for the colleague's comments (round 8) highlighted in yellow (switch: \\highlighttrue in main.tex).
Usage: python build_overleaf_package.py STAGE_DIR VERSION "DATE" MAIN/SUPP/TABLES_PAGES   (run 15 and 16 first)."""
import re
import shutil
import sys
import zipfile
from pathlib import Path

MS = Path(__file__).resolve().parent.parent / 'manuscript'
FIG = MS.parent / 'figures'
STAGE = Path(sys.argv[1])
VERSION, DATE, PAGES = sys.argv[2], sys.argv[3], sys.argv[4].split('/')

TOP = ['main.tex', 'supplement.tex', 'supplementary_tables.tex', 'references.tex', 'supp_refs.tex', 'wordcount.tex',
       'supplement_tables.tex', 'supplement_figures.tex', 'supp_methods.tex', 'supp_notes_revision.tex',
       'supp_prereg.tex', 'supp_deviations.tex', 'supp_freeze.tex', 'supp_amendment.tex', 'supp_amendment2.tex',
       'supp_amendment3.tex', 'cover_letter_nature_methods.md']
SECTIONS = ['abstract', 'introduction', 'results', 'discussion', 'box1', 'methods', 'figures', 'extended_data']

# Passages revised for a colleague's comments (round 8, 18 October 2026; yellow, \colleague) and for the experiment
# a colleague suggested (round 9, 22 October 2026; green, \colleaguesuggestion): (file, first words, last words[, macro]).
SPANS = [
    ('sections/results.tex', "We use ``selectivity'' operationally,", 'about sparing healthy tissue.'),
    ('sections/results.tex', 'but not for every mechanism:', r'Table~\stMechanismAxis).'),
    ('sections/results.tex', 'a comparison that favours the biomarker,', r'Table~\stBiomarkerIncrement).'),
    ('sections/discussion.tex', 'and for antiproliferative drugs', 'lies on it;'),
    ('sections/discussion.tex', 'Being shared across drugs,', 'which drug suits which tumour.'),
    ('sections/discussion.tex', 'but the two were complementary', 'approached one more screen.'),
    ('sections/discussion.tex', 'they are preliminary.', None),
    ('sections/methods.tex', r'\textbf{Agreement carried by the axis.}', r'Table~\stMechanismAxis).'),
    ('sections/methods.tex', 'What the ridge adds was scored', r'Table~\stBiomarkerIncrement).'),
    ('supplement_tables.tex', 'Cross-screen agreement that rests on the general-sensitivity axis, by mechanism.', None),
    ('supplement_tables.tex', 'What the cell-blind model adds to a canonical biomarker.', None),
    # round 9 (a colleague's suggestion, 22 October 2026): choosing cell lines for follow-up
    ('sections/results.tex', 'The same held when models chose cell lines for follow-up', r'Table~\stFollowUp).', 'colleaguesuggestion'),
    ('sections/discussion.tex', 'When models chose cell lines for follow-up,', 'picks were drug-specific.', 'colleaguesuggestion'),
    ('sections/methods.tex', 'This analysis was added late in the study.', r'Table~\stFollowUp).', 'colleaguesuggestion'),
    ('sections/extended_data.tex', 'Choosing cell lines for follow-up.', None, 'colleaguesuggestion'),
    ('supplement_tables.tex', 'Choosing cell lines for follow-up.', None, 'colleaguesuggestion'),
]
PREAMBLE = r"""\usepackage{xcolor}
% soul v3.2 is bundled with the project: older copies (soul 3.1, as on Overleaf's TeX Live 2025) hide highlighted text
% under the newer LaTeX kernel. The fallback below keeps v3.2 working with kernels that lack \RawParEnd.
\providecommand{\RawParEnd}{}
\usepackage{soul}
% Marked revisions: yellow, in response to a colleague's comments (18 October 2026); green, the follow-up experiment a
% colleague suggested (22 October 2026). For a clean copy, change \highlighttrue to \highlightfalse.
\newif\ifhighlight \highlighttrue
\definecolor{hlsuggestion}{RGB}{160,232,160}
\DeclareRobustCommand{\colleague}[1]{\ifhighlight{\sethlcolor{yellow}\hl{#1}}\else#1\fi}
\DeclareRobustCommand{\colleaguesuggestion}[1]{\ifhighlight{\sethlcolor{hlsuggestion}\hl{#1}}\else#1\fi}
\soulregister{\textbf}{1}
"""
COLOUR_KEY = (r"\ifhighlight\noindent{\small Marked revisions: \colleague{yellow, in response to a colleague's comments "
              r"(18 October 2026)}; \colleaguesuggestion{green, the follow-up experiment a colleague suggested (22 October "
              r"2026)}.}\par\medskip\fi" + '\n')


def words(t):
    return r'\s+'.join(re.escape(w) for w in t.split(' '))


def apply_highlights(stage):
    for rel, a, b, *mac in SPANS:
        macro = mac[0] if mac else 'colleague'
        f = stage / rel
        txt = f.read_text()
        pat = words(a) if b is None else words(a) + r'.*?' + words(b)
        m = list(re.finditer(pat, txt, flags=re.S))
        assert len(m) == 1, (rel, a, len(m))
        m = m[0]
        assert not re.search(r'\n\s*\n', m.group(0)), (rel, a, 'crosses a paragraph')
        span = re.sub(r'(\\st[A-Za-z]+)', r'{\1}', m.group(0))   # braced: soul highlights the number with the rest
        f.write_text(txt[:m.start()] + '\\' + macro + '{' + span + '}' + txt[m.end():])
    for rel in ['main.tex', 'supplementary_tables.tex']:
        f = stage / rel
        txt = f.read_text()
        anchor = '\\input{supp_refs.tex}\n'
        assert txt.count(anchor) == 1 and txt.count('\\maketitle\n') == 1, rel
        f.write_text(txt.replace(anchor, anchor + PREAMBLE).replace('\\maketitle\n', '\\maketitle\n' + COLOUR_KEY))


if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / 'sections').mkdir(parents=True)
(STAGE / 'figures').mkdir()
figs = set()
for rel in TOP + [f'sections/{s}.tex' for s in SECTIONS]:
    txt = (MS / rel).read_text()
    figs |= set(re.findall(r'\.\./figures/([A-Za-z0-9_.-]+\.pdf)', txt))
    (STAGE / rel).write_text(txt.replace('../figures/', 'figures/'))
apply_highlights(STAGE)
for f in sorted(figs):
    shutil.copy2(FIG / f, STAGE / 'figures' / f)
for f in sorted((MS / 'overleaf_extra').glob('*.sty')):   # soul v3.2 (LPPL), see PREAMBLE
    shutil.copy2(f, STAGE / f.name)
leftover = [p for p in STAGE.rglob('*.tex') if '../' in p.read_text()]
assert not leftover, leftover

readme = f"""Drug-response models capture general drug sensitivity far better than drug-specific selectivity
Overleaf package (manuscript {VERSION}, {DATE})

Documents (each compiles on its own; switch with Menu > Main document):
  main.tex                  manuscript: abstract, main text, Box 1, Methods, references, Extended Data ({PAGES[0]} pages)
  supplement.tex            Supplementary Information ({PAGES[1]} pages)
  supplementary_tables.tex  Supplementary Tables, typeset review copy ({PAGES[2]} pages)
  cover_letter_nature_methods.md   draft cover letter (not compiled)

Compiler: pdfLaTeX (Overleaf's default) or XeLaTeX. Sources are plain ASCII and use standard packages.

Where to edit:
  sections/*.tex      the manuscript text (abstract, introduction, results, discussion, box1, methods), figure
                      legends (figures.tex) and Extended Data legends (extended_data.tex)
  supp_methods.tex, supp_notes_revision.tex   supplementary methods and notes
The other files are generated by the analysis scripts (references.tex, supp_refs.tex, wordcount.tex,
supplement_tables.tex, supplement_figures.tex, supp_prereg.tex, supp_deviations.tex, supp_freeze.tex,
supp_amendment*.tex) and are overwritten when the documents are rebuilt; comment on them rather than editing.
Numbers in the text are checked against the results by a numbers ledger, so please flag changed numbers.

Marked revisions, in the Results, Discussion and Methods and the titles of the new Extended Data figure and
Supplementary Tables (a colour key is printed under each title):
  yellow  in response to a colleague's comments (18 October 2026)
  green   the follow-up experiment a colleague suggested (22 October 2026: choosing cell lines for follow-up)
To remove them, change \\highlighttrue to \\highlightfalse near the top of main.tex and supplementary_tables.tex.
The highlighting uses the soul package, v3.2, bundled here (soul.sty, soul-ori.sty, soulutf8.sty): older versions,
such as the one in Overleaf's TeX Live 2025, hide the highlighted words under the current LaTeX kernel.

Placeholders in square brackets [ ... ] (authors, affiliations, statements, AI-use accountability, verification,
access terms, DOIs) are for the authors to complete.

Changes made in Overleaf are not synced back automatically: download the project (Menu > Download > Source) to
return them.
"""
(STAGE / 'README.txt').write_text(readme)

out = MS / 'overleaf_package.zip'
tmp = out.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(STAGE.rglob('*')):
        z.write(p, p.relative_to(STAGE).as_posix())
tmp.replace(out)
n = sum(1 for p in STAGE.rglob('*') if p.is_file())
print(f'{out}: {n} files, {len(figs)} figures, {out.stat().st_size / 1024:.0f} KB')
