"""Build the supplementary document and compile both PDFs with tectonic.

The pre-registration and deviation records are converted from their frozen Markdown sources (the
pre-registration's SHA-256 is checked against the freeze manifest first) so the supplement shows
exactly the registered text.
"""
import hashlib
import json
import re
import shutil
import subprocess

from common import PROJECT

MAN = PROJECT / 'manuscript'


def breakable(h):
    """Insert \\allowbreak into a long hex digest so LaTeX can wrap it inside \\texttt."""
    return r'\allowbreak{}'.join(h[i:i + 8] for i in range(0, len(h), 8))
UNICODE = {'≥': r'$\geq$', '≤': r'$\leq$', '→': r'$\rightarrow$', '−': '-', '–': '--', '—': '---', '×': r'$\times$',
           'µ': r'$\mu$', '“': '``', '”': "''", '’': "'", '‘': '`', '…': '...', '·': r'$\cdot$'}


SUPERSCRIPT = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
SUPERSCRIPT_CHARS = '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺'


def tex_escape(s):
    out = []
    for part in re.split(r'(`[^`]*`)', s):
        if part.startswith('`') and part.endswith('`') and len(part) > 1:
            t = part[1:-1].replace('\\', r'\textbackslash{}')
            for a, b in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('$', r'\$'), ('{', r'\{'), ('}', r'\}')]:
                t = t.replace(a, b)
            out.append(r'\texttt{' + t.replace('/', r'/\allowbreak{}') + '}')
            continue
        t = part
        for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('$', r'\$'),
                     ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
                     ('<', r'$<$'), ('>', r'$>$')]:
            t = t.replace(a, b)
        for a, b in UNICODE.items():
            t = t.replace(a, b)
        t = re.sub(f'[{SUPERSCRIPT_CHARS}]+', lambda m: '$^{' + m.group(0).translate(SUPERSCRIPT) + '}$', t)
        t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
        t = re.sub(r'(?<![\\w])\*(?!\s)(.+?)(?<!\s)\*', r'\\emph{\1}', t)
        out.append(t)
    return ''.join(out)


def md_to_tex(md):
    lines, out, in_list = md.splitlines(), [], False
    para = []

    def flush():
        if para:
            out.append(tex_escape(' '.join(para)) + '\n')
            para.clear()
    table = []

    def flush_table():
        if table:
            rows = [[tex_escape(c.strip()) for c in r.strip().strip('|').split('|')] for r in table
                    if not re.match(r'^\|?\s*:?-{3,}', r.strip())]
            out.append(r'\begin{center}\begin{tabular}{l' + 'r' * (len(rows[0]) - 1) + r'}\toprule')
            out.append(' & '.join(rows[0]) + r' \\ \midrule')
            out.extend(' & '.join(r) + r' \\' for r in rows[1:])
            out.append(r'\bottomrule\end{tabular}\end{center}')
            table.clear()
    for ln in lines:
        if ln.strip().startswith('|'):
            flush()
            table.append(ln)
            continue
        flush_table()
        if ln.startswith('#'):
            flush()
            if in_list:
                out.append(r'\end{itemize}'); in_list = False
            level = len(ln) - len(ln.lstrip('#'))
            cmd = {1: r'\subsection*', 2: r'\subsubsection*'}.get(level, r'\paragraph*')
            out.append(f'{cmd}{{{tex_escape(ln.lstrip("#").strip())}}}')
        elif re.match(r'^\s*[\*\-] ', ln) or re.match(r'^\d+\. ', ln):
            flush()
            if not in_list:
                out.append(r'\begin{itemize}'); in_list = True
            item = re.sub(r'^\s*([\*\-]|\d+\.) ', '', ln)
            out.append(r'\item ' + tex_escape(item))
        elif ln.strip() == '':
            flush()
            if in_list:
                out.append(r'\end{itemize}'); in_list = False
        elif in_list and ln.startswith('  '):
            out[-1] += ' ' + tex_escape(ln.strip())
        else:
            para.append(ln.strip())
    flush()
    flush_table()
    if in_list:
        out.append(r'\end{itemize}')
    return '\n'.join(out)


MAIN_PARTS = ['abstract', 'introduction', 'results', 'discussion', 'methods', 'figures', 'extended_data']


def order_references():
    """Renumber references.tex in order of first citation (Nature style); drop uncited items, fail on missing keys."""
    text = '\n'.join((MAN / 'sections' / f'{p}.tex').read_text() for p in MAIN_PARTS)
    order = []
    for group in re.findall(r'\\cite\{([^}]*)\}', text):
        for k in (x.strip() for x in group.split(',')):
            if k not in order:
                order.append(k)
    ref = (MAN / 'references.tex').read_text()
    spare = MAN / 'references_uncited.tex'
    if spare.exists():
        ref = ref.replace('\\end{thebibliography}', spare.read_text() + '\n\\end{thebibliography}')
    items = dict(re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\n\\bibitem|\n\n\\end\{thebibliography\}|\n\\end\{thebibliography\})',
                            ref, flags=re.S))
    missing = [k for k in order if k not in items]
    assert not missing, f'cited but not in references.tex: {missing}'
    unused = [k for k in items if k not in order]
    out = ['\\begin{thebibliography}{99}', '\\setlength{\\itemsep}{0pt}', '']
    out += [f'\\bibitem{{{k}}} {items[k].strip()}' for k in order]
    out += ['', '\\end{thebibliography}', '']
    (MAN / 'references.tex').write_text('\n'.join(out))
    # keep the uncited entries out of the list but on disk for later use
    if unused:
        spare.write_text('\n'.join(f'\\bibitem{{{k}}} {items[k].strip()}' for k in unused) + '\n')
    elif spare.exists():
        spare.unlink()
    print(f'references: {len(order)} cited, {len(unused)} uncited moved to references_uncited.tex')


def word_counts():
    """Word counts for the title page (editor: main text <= 4,500 words excluding Methods, Box 1 and legends): words in
    the LaTeX sources after removing comments, citations, environments and commands."""
    def words(t):
        t = re.sub(r'(?<!\\)%.*', '', t)   # comments, not \\%
        t = re.sub(r'\\cite\{[^}]*\}', '', t)
        t = re.sub(r'\\(begin|end)\{[^}]*\}', '', t)
        t = re.sub(r'\\(subsection|section)\*?\{([^}]*)\}', r'\2', t)
        t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', '', t)
        t = re.sub(r'[{}$\\^_~]', ' ', t)
        return len([w for w in t.split() if re.search(r'[A-Za-z0-9]', w)])
    sec = MAN / 'sections'
    main_text = sum(words((sec / f'{x}.tex').read_text()) for x in ['introduction', 'results', 'discussion'])
    out = {'abstract': words((sec / 'abstract.tex').read_text()), 'main': main_text,
           'methods': words((sec / 'methods.tex').read_text()), 'box': words((sec / 'box1.tex').read_text())}
    n_fig = len(re.findall(r'\\includegraphics', (sec / 'figures.tex').read_text()))
    edf = sec / 'extended_data.tex'
    n_ed = len(re.findall(r'\\includegraphics', edf.read_text())) if edf.exists() else 0
    (MAN / 'wordcount.tex').write_text(
        f"Abstract {out['abstract']} words; main text (Introduction, Results, Discussion) {out['main']:,} words; "
        f"Methods {out['methods']:,} words; Box~1, {out['box']} words; {n_fig} figures, 1 box and {n_ed} Extended Data "
        f"figures.\n")
    return out


def main():
    order_references()
    manifest = json.loads((PROJECT / 'protocol' / 'freeze_manifest.json').read_text())
    prereg = PROJECT / 'protocol' / 'preregistration.md'
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert digest == manifest['files']['protocol']['protocol/preregistration.md'], 'pre-registration changed after freeze'
    (MAN / 'supp_prereg.tex').write_text(md_to_tex(prereg.read_text()))
    notes = PROJECT / 'protocol' / 'post_freeze_notes.md'
    (MAN / 'supp_deviations.tex').write_text(md_to_tex((PROJECT / 'protocol' / 'deviations.md').read_text())
                                             + ('\n\n' + md_to_tex(notes.read_text()) if notes.exists() else ''))
    amend = PROJECT / 'protocol' / 'amendment1_extended_confirmation.md'
    adigest = hashlib.sha256(amend.read_bytes()).hexdigest()
    amanifest = json.loads((PROJECT / 'protocol' / 'freeze_manifest_amendment1.json').read_text())
    (MAN / 'supp_amendment.tex').write_text(
        f"Registered and frozen {amanifest['frozen_utc'].replace('T', ' ')[:19]} UTC "
        f"(\\texttt{{protocol/freeze\\_manifest\\_amendment1.json}}, {sum(len(v) for v in amanifest['files'].values())} "
        f"files); this file's SHA-256 \\texttt{{{breakable(adigest)}}}. The text below is reproduced verbatim.\n\n"
        + md_to_tex(amend.read_text()))
    amend2 = PROJECT / 'protocol' / 'amendment2_pdx_confirmation.md'
    m2 = PROJECT / 'protocol' / 'freeze_manifest_amendment2.json'
    a2digest = hashlib.sha256(amend2.read_bytes()).hexdigest()
    a2manifest = json.loads(m2.read_text())
    assert a2digest == a2manifest['files']['protocol']['protocol/amendment2_pdx_confirmation.md'], 'Amendment 2 changed after freeze'
    m2digest = hashlib.sha256(m2.read_bytes()).hexdigest()
    pdx_log = json.loads((PROJECT / 'data' / 'raw' / 'pdx_bruna_responses' / 'fetch_log.json').read_text())
    (MAN / 'supp_amendment2.tex').write_text(
        f"Registered and frozen {a2manifest['frozen_utc'].replace('T', ' ')[:19]} UTC "
        f"(\\texttt{{protocol/freeze\\_manifest\\_amendment2.json}}, {sum(len(v) for v in a2manifest['files'].values())} "
        f"files; manifest SHA-256 \\texttt{{{breakable(m2digest)}}}). The manifest digest was submitted to four public "
        f"OpenTimestamps calendar servers immediately after the freeze (proof \\texttt{{freeze\\_manifest\\_amendment2.json.ots}}). "
        f"The proof was later upgraded to Bitcoin block-header attestations at heights 971474 and 971477 (header time of "
        f"block 971474: 2026-10-14 22:37:15 UTC). It establishes that the manifest existed when that block was mined; that "
        f"the freeze preceded retrieval rests on the local retrieval log, according to which the PDX\\_Bruna response file "
        f"was retrieved at {pdx_log['utc'].replace('T', ' ')[:19]} UTC. This file's SHA-256 "
        f"\\texttt{{{breakable(a2digest)}}}. The text below is reproduced verbatim.\n\n" + md_to_tex(amend2.read_text()))
    amend3 = PROJECT / 'protocol' / 'amendment3_pdm_confirmation.md'
    m3 = PROJECT / 'protocol' / 'freeze_manifest_amendment3.json'
    a3digest = hashlib.sha256(amend3.read_bytes()).hexdigest()
    a3manifest = json.loads(m3.read_text())
    assert a3digest == a3manifest['files']['protocol']['protocol/amendment3_pdm_confirmation.md'], 'Amendment 3 changed after freeze'
    m3digest = hashlib.sha256(m3.read_bytes()).hexdigest()
    pdm_log = json.loads((PROJECT / 'data' / 'raw' / 'pdm_responses' / 'fetch_log.json').read_text())
    (MAN / 'supp_amendment3.tex').write_text(
        f"Registered and frozen {a3manifest['frozen_utc'].replace('T', ' ')[:19]} UTC "
        f"(\\texttt{{protocol/freeze\\_manifest\\_amendment3.json}}, {sum(len(v) for v in a3manifest['files'].values())} "
        f"files; manifest SHA-256 \\texttt{{{breakable(m3digest)}}}). After a human approved the public stamp, the manifest "
        f"digest was submitted to four OpenTimestamps calendar servers at 2026-10-15 05:07:25 UTC (proof "
        f"\\texttt{{freeze\\_manifest\\_amendment3.json.ots}}). The five response files were retrieved from "
        f"{pdm_log['started_utc'].replace('T', ' ')[:19]} UTC, pinned to figshare version {pdm_log['version']} by file "
        f"identifier and MD5. The registered orientation rule misfired on CoderData's \\texttt{{fit\\_ic50}} scale; the "
        f"correction was recorded before the confirmatory script was run (Supplementary Note~2, item 4). This file's "
        f"SHA-256 \\texttt{{{breakable(a3digest)}}}. The text below is reproduced verbatim.\n\n" + md_to_tex(amend3.read_text()))
    (MAN / 'supp_freeze.tex').write_text(
        f"Frozen {manifest['frozen_utc'].replace('T', ' ')[:19]} UTC; pre-registration SHA-256 \\texttt{{{breakable(digest)}}}; "
        f"{sum(len(v) for v in manifest['files'].values())} files hashed in \\texttt{{protocol/freeze\\_manifest.json}}.\n")
    print('word counts', word_counts())
    tectonic = shutil.which('tectonic') or '/home/htran/.local/bin/tectonic'
    for name in ['main', 'supplement', 'supplementary_tables']:
        r = subprocess.run([tectonic, '--keep-logs', f'{name}.tex'], cwd=MAN, capture_output=True, text=True)
        print(name, 'exit', r.returncode)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
        warn = [l for l in (r.stdout + r.stderr).splitlines() if 'warning' in l.lower() and 'Overfull' not in l]
        for w in warn[:15]:
            print('  ', w)


if __name__ == '__main__':
    main()
