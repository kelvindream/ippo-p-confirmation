#!/usr/bin/env python3
"""Build main.tex from ../draft.md (build wiring only; prose untouched).

Pipeline: preprocess unicode/HTML in a temp copy -> pandoc to LaTeX fragment
-> post-process (title page, abstract, figures, references) -> main.tex.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # paper/latex
PANDOC = str(Path.home() / ".local/bin/pandoc")
MD = ROOT.parent / "draft.md"

# ---------------------------------------------------------------- preprocess
SUP_MAP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
           "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
CHAR_MAP = {
    "α": r"\ensuremath{\alpha}", "γ": r"\ensuremath{\gamma}",
    "λ": r"\ensuremath{\lambda}", "π": r"\ensuremath{\pi}",
    "→": r"\ensuremath{\rightarrow}", "−": r"\ensuremath{-}",
    "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}",
    "≈": r"\ensuremath{\approx}", "±": r"\ensuremath{\pm}",
    "×": r"\ensuremath{\times}", "·": r"\ensuremath{\cdot}",
    "–": r"\textendash{}", "—": r"\textemdash{}",
    "★": r"\ensuremath{\star}",
}


def preprocess(text: str) -> str:
    text = text.replace("<sup>", r"\textsuperscript{").replace("</sup>", "}")
    text = text.replace("p\u0303", r"\~{p}")  # combining tilde on p
    # superscript digit runs -> \textsuperscript{...}
    text = re.sub("[" + "".join(SUP_MAP) + "]+",
                  lambda m: r"\textsuperscript{" +
                            "".join(SUP_MAP[c] for c in m.group(0)) + "}",
                  text)
    for k, v in CHAR_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"^---+\s*$", "", text, flags=re.M)  # decorative hrules
    return text


def pandoc(md_text: str) -> str:
    p = subprocess.run(
        [PANDOC, "-f", "markdown+superscript+subscript", "-t", "latex",
         "--wrap=none"],
        input=md_text, capture_output=True, text=True, check=True)
    return p.stdout


# ---------------------------------------------------------------- postprocess
HEAD = r"\\(?:sub){0,2}section\{"


def find_block(frag: str, start_pat: str) -> tuple[int, int, str]:
    m = re.compile(start_pat).search(frag)
    if not m:
        return (-1, -1, "")
    nxt = re.search(HEAD, frag[m.end():])
    end = m.end() + (nxt.start() if nxt else len(frag) - m.end())
    return m.start(), end, frag[m.start():end]


def strip_num(t: str) -> str:
    return re.sub(r"^(\d+\.)*\d+\.\s+", "", t)


def main() -> None:
    raw = MD.read_text(encoding="utf-8")
    pre = preprocess(raw)
    (ROOT / "_preproc.md").write_text(pre, encoding="utf-8")
    frag = pandoc(pre)

    # ---- title (H1) --------------------------------------------------------
    m = re.match(r"\\section\{(.+?)\}(?:\\label\{[^}]*\})?\s*\n", frag, re.S)
    assert m, "title not found"
    title = m.group(1)
    frag = frag[m.end():]

    # ---- Title Page section: discard (rebuilt by hand) ---------------------
    s, e, _ = find_block(frag, r"\\subsection\{Title Page\}")
    assert s >= 0, "Title Page section not found"
    frag = frag[:s] + frag[e:]

    # ---- Abstract + Keywords ------------------------------------------------
    s, e, abst = find_block(frag, r"\\subsection\{Abstract\}")
    assert s >= 0, "Abstract not found"
    frag = frag[:s] + frag[e:]
    abst = re.sub(r"\\subsection\{Abstract\}(?:\\label\{[^}]*\})?\s*",
                  "", abst)
    kw = ""
    mk = re.search(r"(\\textbf\{Keywords:\}.*?)(?:\n\n|\Z)", abst, re.S)
    if mk:
        kw = mk.group(1)
        abst = (abst[:mk.start()] + abst[mk.end():]).strip()
    kw = re.sub(r"^\\textbf\{Keywords:\}\s*", "", kw).strip()

    # ---- strip manual heading numbers (1., 2.1, ...) ------------------------
    frag = re.sub(r"(\\(?:sub){0,2}section)\{\d+(?:\.\d+)*\.?\s+", r"\1{",
                  frag)

    # ---- References -> thebibliography --------------------------------------
    s, e, _ = find_block(frag, r"\\subsection\{References\}")
    assert s >= 0, "References not found"
    frag = frag[:s] + frag[e:]
    items = [p.strip() for p in re.split(r"\n\n+", pandoc(
        re.sub(r"^## References\s*", "",
               re.search(r"## References\s*\n(.*)$", pre, re.S).group(1),
               flags=re.S).strip()).strip()) if p.strip()]
    bib = ["\\begin{thebibliography}{21}"]
    for it in items:
        mref = re.match(r"\{\[\}(\d+)\{\]\}\s*(.*)", it, re.S)
        assert mref, f"bad reference item: {it[:60]!r}"
        bib.append(f"\\bibitem{{ref{mref.group(1)}}} {mref.group(2)}")
    bib.append("\\end{thebibliography}")
    bib = "\n".join(bib)

    # ---- Figure blockquotes -> figure environments --------------------------
    fig_files = {
        1: "figure1_method_schematic.pdf",
        2: "figure2_seed_effects.pdf",
        3: "figure3_exploratory_boundary.pdf",
    }

    def quote_to_figure(mq: re.Match) -> str:
        body = mq.group(0)
        mn = re.search(r"\\textbf\{Figure (\d)\.", body)
        n = int(mn.group(1))
        paras = [p.strip() for p in body.strip().split("\n\n")
                 if p.strip() and p.strip() not in (r"\begin{quote}",
                                                    r"\end{quote}")]
        head = paras[0]
        head = re.sub(r"\\textbf\{Figure \d\..*?\}\}\s*", "", head, flags=re.S)
        head = re.sub(r"^\\begin\{quote\}\s*", "", head)
        cap = head
        if len(paras) > 1:
            src = re.sub(r"^\\end\{quote\}\s*", "", paras[1])
            cap += " " + src
        cap = cap.replace(r"\begin{quote}", "").replace(r"\end{quote}", "")
        cap = cap.strip()
        f = fig_files[n]
        if f:
            graphics = (
                "\\IfFileExists{../figures/%s}\n"
                "  {\\includegraphics[width=0.88\\textwidth]{../figures/%s}}\n"
                "  {\\fbox{\\parbox[c][7cm][c]{0.9\\textwidth}{\\centering"
                " (placeholder: ../figures/%s not found)}}}\n" % (f, f, f))
        else:
            graphics = ("\\fbox{\\parbox[c][7cm][c]{0.9\\textwidth}"
                        "{\\centering (placeholder box: method schematic"
                        " pending)}}\n")
        return ("\\begin{figure}[htbp]\n\\centering\n" + graphics +
                "\\caption{" + cap + "}\n\\label{fig:figure%d}\n" % n +
                "\\end{figure}\n")

    frag = re.sub(r"\\begin\{quote\}.*?\\end\{quote\}", quote_to_figure,
                  frag, flags=re.S)

    # Allow line breaks at dots inside \texttt file names (layout only).
    frag = re.sub(r"\\texttt\{([^{}]*)\}",
                  lambda m: "\\texttt{" +
                            m.group(1).replace(".", r"\allowbreak.") + "}",
                  frag)

    # ---- assemble ------------------------------------------------------------
    tex = PREAMBLE.replace("@@TITLE@@", title)
    tex = tex.replace("@@ABSTRACT@@", abst).replace("@@KEYWORDS@@", kw)
    tex = tex.replace("@@BODY@@", frag.strip()).replace("@@BIB@@", bib)
    (ROOT / "main.tex").write_text(tex, encoding="utf-8")
    print("main.tex written:", len(tex), "bytes;",
          len(items), "references;", "figures wired:",
          [k for k, v in fig_files.items() if v])


PREAMBLE = r"""% Auto-generated by build.py from ../draft.md — do not edit prose here.
\documentclass[10pt,onecolumn]{IEEEtran}
\usepackage[no-math]{fontspec}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{etoolbox}
\usepackage{xcolor}
\usepackage[htt]{hyphenat}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3.5pt}\sloppy}
% Allow line breaks after underscores in typewriter file names (pandoc escapes
% underscores in code spans as \_; this is layout wiring, not prose editing).
\renewcommand{\_}{\textunderscore\allowbreak}
\usepackage[implicit=false,hidelinks]{hyperref}
\graphicspath{{../figures/}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\setlength{\emergencystretch}{2em}

\title{@@TITLE@@}
\author{Rongtao~Zhu\textsuperscript{1,*},
        Zhaoxia~He\textsuperscript{1},
        An~Zhao\textsuperscript{1},
        Shubin~Wei\textsuperscript{1}, and
        Qiran~Li\textsuperscript{1}%
\thanks{Yangtze University College of Arts and Science, Jingzhou, China.}%
\thanks{Corresponding author: Rongtao Zhu (181804457@qq.com).}%
\thanks{ORCID: Rongtao Zhu 0009-0000-4528-5714; Zhaoxia He 0009-0006-7400-2123;
An Zhao 0009-0003-9392-6012; Shubin Wei 0009-0007-1621-4440; Qiran Li 0000-0001-7995-8972.}%
\thanks{Empirical research article (IMRaD). Draft for author review; figures
are placeholders; author metadata and venue-specific formatting pending.}}
\markboth{Running title: Pressure Information for Independent Multi-Agent
Traffic Signal Control}{}

\begin{document}
\maketitle

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{IEEEkeywords}
@@KEYWORDS@@
\end{IEEEkeywords}

@@BODY@@

@@BIB@@

\end{document}
"""

if __name__ == "__main__":
    sys.exit(main())
