# Manuscript

LaTeX source for the publication:

> **A Systematic evaluation and benchmarking of text summarization methods for biomedical literature:
> From word-frequency methods to language models**
>
> Baumgärtel F, Bono E, Fillinger L, Galou L, Kęska-Izworska K, Walter S, Andorfer P, Kratochwill K, Perco P, Ley M
> bioRxiv 2026, [doi.org/10.64898/2026.01.09.697335](https://doi.org/10.64898/2026.01.09.697335)

## Manuscript versions

| Entry point | Template | Usage |
|-------------|----------|-------|
| `publication-patterns.tex` | Patterns (Cell Press) | **Primary** — journal submission |
| `publication-biorxiv.tex` | bioRxiv two-column preprint | Preprint server |

Both versions share the same `Sections/`, `Visualizations/`, `acronyms.tex`, and `refs.bib`.

## Prerequisites

A TeX Live (or equivalent) installation with `pdflatex` and `bibtex`:

- **Ubuntu / Debian:** `sudo apt install texlive-full`
- **Fedora:** `sudo dnf install texlive-scheme-full`
- **macOS (Homebrew):** `brew install --cask mactex`
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

## Building

### Patterns version (primary)

```bash
cd manuscript
latexmk -pdf publication-patterns.tex
```

### bioRxiv version

```bash
cd manuscript
latexmk -pdf publication-biorxiv.tex
```

Both commands run the full `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically.

### Word (.docx) version

A Word build with embedded figures, resolved citations, and a table of contents can be generated from the Patterns entry point:

```bash
cd manuscript
./build-docx.sh
```

This creates `publication-patterns.docx`. Requires [Pandoc](https://pandoc.org/) ≥ 3.0.

To clean up all intermediate files:
```bash
latexmk -C
```

## Project structure

```
manuscript/
├── publication-patterns.tex     # Entry point — Patterns / Cell Press
├── publication-biorxiv.tex      # Entry point — bioRxiv preprint
├── build-docx.sh               # Build Word (.docx) with figures
├── 01_Article_MainText.tex      # Main text (used by biorxiv entry point)
├── 02_Article_Supplementary.tex # Supplementary (used by biorxiv entry point)
├── acronyms.tex                 # Acronym definitions (shared)
├── refs.bib                     # BibTeX references (shared)
├── bioRxiv.cls                  # bioRxiv document class
├── bxv_abbrvnat.bst             # bioRxiv bibliography style
├── bioRxiv_logo.png             # bioRxiv logo asset
├── orcidlink.sty                # ORCID link support
├── numbered.bst                 # Patterns bibliography style
├── numcompress.sty              # Numeric citation compression
├── Sections/                    # Content sections (shared)
│   ├── introduction.tex
│   ├── materials_methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusions.tex
└── Visualizations/              # Figures (shared)
    ├── category_boxplot.png
    ├── category_gameshowell.png
    ├── family_boxplot.png
    ├── family_gameshowell.png
    ├── metric_correlation.png
    ├── rank_heatmap.png
    ├── supplementary1.png
    └── workflow_graphic.png
```
