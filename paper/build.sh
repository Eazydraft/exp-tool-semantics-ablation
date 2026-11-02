#!/usr/bin/env bash
# Rebuild figures, tables and the PDF from runs/main2/analysis.json.
#   ./paper/build.sh          -> paper/main.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --with matplotlib python paper/figures.py
python3 paper/tables.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log
echo "wrote paper/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
