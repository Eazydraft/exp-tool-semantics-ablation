#!/usr/bin/env bash
# One-command run of the whole study (all models x all conditions x 200 tasks).
#
#   ./run_eval.sh                       # full run, results in runs/main/
#   THREADS=16 JOBS=6 ./run_eval.sh     # more parallelism
#   LIMIT=3 ./run_eval.sh               # smoke test: first 3 tasks per combo
#   ./run_eval.sh --models qwen/qwen3-30b-a3b-instruct-2507 --conditions D0 D5
#   ./run_eval.sh evaluate              # re-run evaluation + index only
#
# THREADS = tasks run concurrently within one model x condition (API concurrency)
# JOBS    = model x condition combos run concurrently (separate processes)
# Re-running resumes: tasks that already have a result are skipped.
set -euo pipefail
cd "$(dirname "$0")"

uv sync --quiet
uv run python -m ablation.run \
    --threads "${THREADS:-8}" \
    --jobs "${JOBS:-4}" \
    --run-name "${RUN_NAME:-main}" \
    ${LIMIT:+--limit "$LIMIT"} \
    "$@"
