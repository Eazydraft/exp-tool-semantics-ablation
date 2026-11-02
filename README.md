# Tool Semantics Documentation Ablation

Infrastructure for `experiment-design.md`: run BFCL v3 **multi_turn_base** (200 tasks) for
4 models x 6 documentation conditions (D0..D5) via OpenRouter, using BFCL's own harness,
tool implementations, execution environment and scorer unchanged.

## Run

```bash
cp .env.example .env            # put OPENROUTER_API_KEY=... in .env (already present)
./run_eval.sh                   # full study -> runs/main/
LIMIT=3 ./run_eval.sh           # smoke test: first 3 tasks per model x condition
THREADS=16 JOBS=6 ./run_eval.sh # THREADS = concurrent tasks per combo, JOBS = concurrent combos
./run_eval.sh --models qwen/qwen3-30b-a3b-instruct-2507 --conditions D0 D5
./run_eval.sh evaluate          # re-score existing results and rebuild the index
./run_eval.sh index             # rebuild index.json only
```

Re-running resumes: tasks that already have a result are skipped (BFCL behaviour).
`uv` manages the environment (`uv sync` runs automatically).

## Layout

```
gorilla/                 untouched clone of BFCL (gorilla/berkeley-function-call-leaderboard)
ablation/conditions.py   D0..D5 definitions; ablate schemas, map model calls back to original names
ablation/handler.py      BFCL handler subclass: OpenRouter client, ablation hook, raw capture, trajectory dump
ablation/run.py          registers model x condition combos in BFCL, runs generate / evaluate / index
run_eval.sh              one-command entry point
runs/<run-name>/
  trajectories/<model>-<cond>/<task_id>.json   one complete trajectory per file
  index.json             one row per trajectory (verdict, counts, tokens, cost, timing) + per-combo summary
  result/, score/        BFCL's own result and score files (score/*.csv = BFCL leaderboard tables)
  logs/                  generation log per model x condition
  run_config.json        models, conditions, temperature, BFCL commit, condition spec
```

## Trajectory file contents

`task` (prompt turns, initial backend state, involved classes), `tools_sent` (exact tool schemas
shown to the model), `name_maps` (generic -> original names), `bfcl_result` (per-turn, per-step tool
calls in original names, as BFCL scores them), `bfcl_metadata` (BFCL inference log incl. tool outputs
and backend state after each turn, per-step token counts and latencies, reasoning text),
`api_calls` (every raw HTTP request/response to OpenRouter incl. usage/cost, provider, headers,
timestamps; the API key is redacted), `totals` (tokens, cached/reasoning tokens, cost in USD,
latency), `evaluation` (BFCL verdict and error details), `error` (if inference crashed).

## How the ablation is applied

BFCL's `OpenAICompletionsHandler` is subclassed (`ablation/handler.py`). The only behavioural
differences from a stock BFCL run against an OpenAI-compatible endpoint:

- `_compile_tools` applies the condition to the task's function docs before BFCL converts them to
  OpenAI tool format. D0 produces byte-identical tools to stock BFCL.
- `_parse_query_response_FC` maps the model's tool calls back to the original tool and parameter
  names, so BFCL executes and scores them unchanged. The chat history the model sees keeps the
  names it was given. Names the model invents are passed through untouched and fail inside BFCL's
  executor as they would in the original benchmark.
- Requests go to OpenRouter with `usage.include=true` so cost is reported per call.

Ablation details: generic tool names are `tool_01..tool_NN` in the order BFCL lists the task's
functions; generic parameter names are `arg_1..arg_n` per tool, applied recursively to keys inside
object-typed parameters and arrays of objects (the model's argument values are translated back
to the original keys before execution). Removing tool descriptions also strips
the natural-language descriptions inside BFCL's `response` (return value) schema, since that
documents what the tool does; the response schema structure and all parameter types, required
lists, enums and defaults are kept in every condition.
