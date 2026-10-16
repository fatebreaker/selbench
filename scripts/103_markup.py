"""Marked-up manuscript: word-level differences between the version the referees reviewed (the frozen snapshot in
manuscript/review_round4/snapshot/) and the current manuscript.

For each section file the two versions are split into paragraphs and words and compared with difflib. Inserted text is
set in blue and deleted text in red, struck through where it is plain text. A changed run is marked only if it is safe to
wrap in a LaTeX group (balanced braces, an even number of $, no environment or sectioning commands); otherwise the new
text is shown unmarked and a margin note flags the change. The marked-up sections are compiled with the current
preamble into manuscript/markup/main_markup.pdf.
Usage: 103_markup.py [--base manuscript/review_round4/snapshot]
"""
import argparse
import difflib
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MAN = PROJECT / 'manuscript'
OUT = MAN / 'markup'
SECTIONS = ['abstract', 'introduction', 'results', 'discussion', 'methods', 'figures', 'extended_data', 'box1']
UNSAFE = re.compile(r'\\(begin|end|subsection|section|caption|includegraphics|item|label|input|centering)\b|%')


def safe(tokens):
    t = ' '.join(tokens)
    return (t.count('{') == t.count('}') and t.count('$') % 2 == 0 and not UNSAFE.search(t)
            and t.count('\\left') == t.count('\\right'))


def plain_text(tokens):
    t = ' '.join(tokens)
    return not re.search(r'[\\$^_{}]', t)


def mark(old, new):
    """Word-level diff of two paragraphs -> LaTeX with insertions and deletions marked."""
    a, b = old.split(), new.split()
    out, flagged = [], False
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == 'equal':
            out += b[j1:j2]
            continue
        if op in ('delete', 'replace') and i2 > i1:
            dele = a[i1:i2]
            if plain_text(dele):
                out.append(r'\textcolor{red}{\sout{' + ' '.join(dele) + '}}')
            elif safe(dele):
                out.append(r'{\color{red}[' + ' '.join(dele) + ']}')
            else:
                flagged = True
        if op in ('insert', 'replace') and j2 > j1:
            ins = b[j1:j2]
            if safe(ins):
                out.append(r'{\color{blue}' + ' '.join(ins) + '}')
            else:
                out += ins
                flagged = True
    s = ' '.join(out)
    if flagged:
        s = FLAG[0] + s if not UNSAFE.search(s[:200]) else s
    return s


# change flag for paragraphs that cannot be colour-marked: a margin note in running text; inline inside floats (Box 1,
# figure legends), where \marginpar is not allowed
MARGIN_FLAG, INLINE_FLAG = r'\marginpar{\color{blue}\tiny changed}', r'{\color{blue}\tiny[changed]} '
FLOATS = {'box1', 'figures', 'extended_data'}
FLAG = [MARGIN_FLAG]


def paragraphs(text):
    return re.split(r'\n\s*\n', text)


def diff_file(old_text, new_text):
    po, pn = paragraphs(old_text), paragraphs(new_text)
    res = []
    sm = difflib.SequenceMatcher(a=[p.strip() for p in po], b=[p.strip() for p in pn], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            res += pn[j1:j2]
        elif op == 'insert':
            for p in pn[j1:j2]:
                res.append(mark('', p))
        elif op == 'delete':
            for p in po[i1:i2]:
                if safe(p.split()) and plain_text(p.split()[:5]):
                    res.append(r'{\color{red}[deleted paragraph: ' + ' '.join(p.split()[:25]) + r' \dots]}')
        else:                                   # replace: pair paragraphs in order, extra ones inserted/deleted
            olds, news = po[i1:i2], pn[j1:j2]
            for k in range(max(len(olds), len(news))):
                o = olds[k] if k < len(olds) else ''
                n = news[k] if k < len(news) else ''
                if n:
                    res.append(mark(o, n))
    return '\n\n'.join(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=str(MAN / 'review_round4' / 'snapshot'))
    base = Path(ap.parse_args().base)
    OUT.mkdir(exist_ok=True)
    (OUT / 'sections').mkdir(exist_ok=True)
    for s in SECTIONS:
        new = (MAN / 'sections' / f'{s}.tex')
        old = base / 'sections' / f'{s}.tex'
        if not new.exists():
            continue
        FLAG[0] = INLINE_FLAG if s in FLOATS else MARGIN_FLAG
        text = diff_file(old.read_text() if old.exists() else '', new.read_text())
        (OUT / 'sections' / f'{s}.tex').write_text(text)
    main_tex = (MAN / 'main.tex').read_text()
    main_tex = main_tex.replace(r'\begin{document}', r'\usepackage[normalem]{ulem}' + '\n' + r'\begin{document}')
    main_tex = main_tex.replace(r'\maketitle', r'\maketitle' + '\n' + r'\noindent\textit{Marked-up version: changes since '
                                r'the version reviewed in round 4 (insertions blue, deletions red).}' + '\n')
    main_tex = main_tex.replace('../figures/', '../../figures/')
    for f in ['references.tex', 'supp_refs.tex', 'wordcount.tex']:
        if (MAN / f).exists():
            shutil.copy(MAN / f, OUT / f)
    (OUT / 'main_markup.tex').write_text(main_tex)
    for s in SECTIONS:
        p = OUT / 'sections' / f'{s}.tex'
        if p.exists():
            p.write_text(p.read_text().replace('../figures/', '../../figures/'))
    tectonic = shutil.which('tectonic') or str(PROJECT.parent / 'bin' / 'tectonic')
    r = subprocess.run([tectonic, 'main_markup.tex'], cwd=OUT, capture_output=True, text=True)
    print('tectonic exit', r.returncode)
    if r.returncode:
        print(r.stderr[-3000:])


if __name__ == '__main__':
    main()
