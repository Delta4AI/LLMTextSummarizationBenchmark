# Manuscript

LaTeX source for the publication:

> **A Systematic evaluation and benchmarking of text summarization methods for biomedical literature:
> From word-frequency methods to language models**
>
> Baumgärtel F, Bono E, Fillinger L, Galou L, Kęska-Izworska K, Walter S, Andorfer P, Kratochwill K, Perco P, Ley M
> bioRxiv 2026, [doi.org/10.64898/2026.01.09.697335](https://doi.org/10.64898/2026.01.09.697335)

Uses the [MDPI](https://www.mdpi.com/) journal template (IJMS).

## Prerequisites

A full TeX Live (or equivalent) installation with `pdflatex` and `bibtex`:

- **Ubuntu / Debian:** `sudo apt install texlive-full`
- **Fedora:** `sudo dnf install texlive-scheme-full`
- **macOS (Homebrew):** `brew install --cask mactex`
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

Alternatively, use [latexmk](https://ctan.org/pkg/latexmk) (included with most TeX distributions).

## Building

### With latexmk (recommended)

```bash
cd manuscript
latexmk -pdf publication.tex
```

This runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically
and produces `publication.pdf`.

To clean up intermediate files:
```bash
latexmk -C
```

### Manual build

```bash
cd manuscript
pdflatex publication.tex
bibtex publication
pdflatex publication.tex
pdflatex publication.tex
```

Two additional `pdflatex` passes are needed after `bibtex` to resolve all
cross-references and citations.

## Project structure

```
manuscript/
├── publication.tex              # Main document (includes all sections)
├── acronyms.tex                 # Acronym definitions
├── bibliography.bib             # BibTeX references
├── Definitions/                 # MDPI journal class and style files
│   ├── mdpi.cls
│   ├── mdpi.bst
│   ├── mdpi_apacite.bst / .sty
│   ├── mdpi_chicago.bst
│   ├── journalnames.tex
│   └── logo-*.eps / .pdf
├── Sections/                    # Content split into sections
│   ├── introduction.tex
│   ├── materials_methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusions.tex
└── Visualizations/              # Figures referenced by the manuscript
    ├── category_boxplot.png
    ├── category_gameshowell.png
    ├── family_boxplot.png
    ├── family_gameshowell.png
    ├── metric_correlation.png
    ├── rank_heatmap.png
    ├── supplementary1.png
    └── workflow_graphic.png
```
