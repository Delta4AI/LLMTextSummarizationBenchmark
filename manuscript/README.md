# Manuscript

LaTeX source for the iScience submission (ISCIENCE-D-26-08490).

> Baumgärtel F, Bono E, Fillinger L, Galou L, Kęska-Izworska K, Walter S, Andorfer P,
> Kratochwill K, Perco P, Ley M
>
> Preprint: [doi.org/10.64898/2026.01.09.697335](https://doi.org/10.64898/2026.01.09.697335)

## Documents

| Entry point | Output | Purpose |
|-------------|--------|---------|
| `main.tex` | `main.pdf` | Main document |
| `supplement.tex` | `supplement.pdf` | Supplemental information |

Both share `Sections/`, `acronyms.tex`, `refs.bib`, and `Visualizations/`.

## Prerequisites

A TeX Live (or equivalent) installation with `pdflatex` and `bibtex`:

- **Fedora:** `sudo dnf install texlive-scheme-full`
- **Ubuntu / Debian:** `sudo apt install texlive-full`
- **macOS (Homebrew):** `brew install --cask mactex`
- **Windows:** [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

## Building

```bash
cd manuscript
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

Each runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically.

Clean intermediate files with `latexmk -C`.

## Structure

```
manuscript/
├── main.tex                     # Entry point — main document
├── supplement.tex               # Entry point — supplemental information
├── build-docx.sh                # Word build of the main document
├── acronyms.tex                 # Acronym definitions (shared)
├── refs.bib                     # BibTeX references (shared)
├── numbered.bst                 # Cell Press numbered bibliography style
├── Sections/                    # Content, \input by main.tex in order
│   ├── introduction.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── resource_availability.tex
│   ├── limitations.tex
│   ├── acknowledgements.tex
│   ├── author_contributions.tex
│   ├── declaration_of_interests.tex
│   ├── declaration_AI.tex
│   ├── main_figures.tex         # Figure titles and legends (no images)
│   ├── main_tables.tex
│   ├── materials_methods.tex    # STAR Methods (incl. key resources table)
│   └── supplemental_information.tex   # Titles and legends only
├── final-files/                 # Submission deliverables (see its own notes)
└── Visualizations/              # Figure sources
```
