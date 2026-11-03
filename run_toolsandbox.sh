#!/usr/bin/env bash
# One-command ToolSandbox run of the ablation. ToolSandbox pins old openai/polars versions, so it
# gets its own virtualenv (.venv-toolsandbox, created on first use).
#
#   ./run_toolsandbox.sh                        # 4 models x D0..D5 x 91 scenarios -> runs/main/trajectories/
#   LIMIT=2 JOBS=2 ./run_toolsandbox.sh --models anthropic/claude-haiku-5.5 --conditions D0   # smoke test
#   ./run_toolsandbox.sh index                  # rebuild index.json only
# Re-running resumes: scenarios that already have a trajectory file are skipped.
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-toolsandbox
if [ ! -x "$VENV/bin/python" ]; then
    uv venv -q --python 3.12 "$VENV"
    VIRTUAL_ENV="$PWD/$VENV" uv pip install -q -e ./toolsandbox python-dotenv
fi
"$VENV/bin/python" -m ablation.toolsandbox_run \
    --jobs "${JOBS:-8}" \
    --run-name "${RUN_NAME:-main}" \
    ${LIMIT:+--limit "$LIMIT"} \
    "$@"
