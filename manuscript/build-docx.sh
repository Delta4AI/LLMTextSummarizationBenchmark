#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Build a Word (.docx) version of the iScience manuscript
#  with all figures, tables, and resolved citations.
#
#  Usage:   ./build-docx.sh
#
#  Requires: pandoc ≥ 3.0
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

TEX_FILE="main.tex"
OUT_FILE="main.docx"

echo "Building ${OUT_FILE} from ${TEX_FILE} …"

pandoc "$TEX_FILE" \
    -o "$OUT_FILE" \
    --resource-path=. \
    --bibliography=refs.bib \
    --citeproc \
    --number-sections \
    --toc \
    --standalone

echo "✓ Created ${OUT_FILE} ($(du -h "$OUT_FILE" | cut -f1))"
