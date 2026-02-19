# Manuscript

LaTeX source for the publication:

> **A Systematic evaluation and benchmarking of text summarization methods for biomedical literature:
> From word-frequency methods to language models**
>
> Baumgärtel F, Bono E, Fillinger L, Galou L, Kęska-Izworska K, Walter S, Andorfer P, Kratochwill K, Perco P, Ley M
> bioRxiv 2026, [doi.org/10.64898/2026.01.09.697335](https://doi.org/10.64898/2026.01.09.697335)

## Manuscript versions

| File | Template | Status |
|------|----------|--------|
| `publication.tex` | Standalone article (bioRxiv / Patterns-compatible) | **Active** |
| `publication-mdpi.tex` | MDPI journal class (IJMS) | Obsolete |

Both versions share the same `Sections/`, `Visualizations/`, `acronyms.tex`, and `bibliography.bib`.

The MDPI version references `Sections/materials_methods-mdpi.tex` (section titled "Materials and Methods"),
while the active version references `Sections/methods.tex` (titled "Methods") with minor text updates.

## Prerequisites

A full TeX Live (or equivalent) installation with `pdflatex` and `bibtex`:

- **Ubuntu / Debian:** `sudo apt install texlive-full`
- **Fedora:** `sudo dnf install texlive-scheme-full`
- **macOS (Homebrew):** `brew install --cask mactex`
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

Alternatively, use [latexmk](https://ctan.org/pkg/latexmk) (included with most TeX distributions).

## Building

### Active version (bioRxiv / Patterns)

```bash
cd manuscript
latexmk -pdf publication.tex
```

### MDPI version (obsolete)

```bash
cd manuscript
latexmk -pdf publication-mdpi.tex
```

Both commands run the full `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically.

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

## Project structure

```
manuscript/
├── publication.tex              # Main document — bioRxiv / Patterns (active)
├── publication-mdpi.tex         # Main document — MDPI / IJMS (obsolete)
├── acronyms.tex                 # Acronym definitions (shared)
├── bibliography.bib             # BibTeX references (shared)
├── Definitions/                 # MDPI journal class and style files
│   ├── mdpi.cls
│   ├── mdpi.bst / mdpi_apacite.bst / mdpi_chicago.bst
│   └── logo-*.eps / .pdf
├── Sections/                    # Content split into sections (shared)
│   ├── introduction.tex
│   ├── methods.tex              # Used by publication.tex
│   ├── materials_methods-mdpi.tex  # Used by publication-mdpi.tex
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
