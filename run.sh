#!/usr/bin/env bash
# The one entry point: run a benchmark, rebuild the index, run the analysis, build the paper, browse.
#
#   ./run.sh bfcl [args]          BFCL multi-turn base, four models x D0..D8 -> runs/main/   (ablation.run)
#   ./run.sh toolsandbox [args]   ToolSandbox, four models x D0..D5 -> runs/main/            (ablation.toolsandbox_run)
#   ./run.sh tau2 [args]          tau2-bench airline, four models x D0..D5 -> runs/main/     (ablation.tau2_run)
#   ./run.sh index                rebuild runs/main/index.json over every trajectory of every benchmark
#   ./run.sh analyze              runs/main/analysis.json, audit.json and analysis_extra.json
#   ./run.sh paper                figures, tables and paper/main.pdf
#   ./run.sh browse               trajectory browser at http://localhost:8765/
#
# [args] go to the benchmark's runner: --models ..., --conditions ..., --limit K, --task-ids ...,
# and for bfcl the subcommands generate | evaluate | index. Environment: THREADS (tasks in flight per
# model x condition), JOBS (model x condition combos in parallel), LIMIT (first K tasks only: a smoke
# test), RUN_NAME (runs/<name>/, default main), PORT (browser). Re-running resumes: tasks and
# scenarios that already have a trajectory file are skipped.
#
# BFCL and the analysis run in the uv project environment (.venv, synced on first use). ToolSandbox
# and tau2-bench pin dependency versions that conflict with it and with each other, so each gets its
# own virtualenv, created on first use from the submodule.
set -euo pipefail
cd "$(dirname "$0")"

venv_for() {  # venv_for <dir> <submodule>: create the virtualenv if missing, print its python
    if [ ! -x "$1/bin/python" ]; then
        uv venv -q --python 3.12 "$1"
        VIRTUAL_ENV="$PWD/$1" uv pip install -q -e "./$2" python-dotenv
    fi
    echo "$1/bin/python"
}

cmd="${1:-help}"; shift || true
run_name="${RUN_NAME:-main}"
case "$cmd" in
    bfcl)
        uv sync --quiet
        uv run python -m ablation.run --threads "${THREADS:-8}" --jobs "${JOBS:-4}" --run-name "$run_name" ${LIMIT:+--limit "$LIMIT"} "$@" ;;
    toolsandbox)
        "$(venv_for .venv-toolsandbox toolsandbox)" -m ablation.toolsandbox_run --jobs "${JOBS:-8}" --run-name "$run_name" ${LIMIT:+--limit "$LIMIT"} "$@" ;;
    tau2)
        "$(venv_for .venv-tau2 tau2-bench)" -m ablation.tau2_run --jobs "${JOBS:-4}" --threads "${THREADS:-8}" --run-name "$run_name" ${LIMIT:+--limit "$LIMIT"} "$@" ;;
    index)
        uv sync --quiet
        uv run python -m ablation.run index --run-name "$run_name" ;;
    analyze)
        uv sync --quiet
        venv_for .venv-toolsandbox toolsandbox > /dev/null   # ablation.audit reads ToolSandbox's ground truth through it
        uv run python -m ablation.analyze --run-name "$run_name"
        uv run python -m ablation.audit --run-name "$run_name"
        uv run python -m ablation.analyze_extra --run-name "$run_name" ;;
    paper)
        ./paper/build.sh ;;
    browse)
        python3 browser/serve.py "${PORT:-8765}" ;;
    *)
        sed -n '2,20p' "$0"; exit 1 ;;
esac
