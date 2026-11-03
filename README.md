# Tool Semantics Documentation Ablation

Infrastructure for `experiment-design.md`: run BFCL v3 **multi_turn_base** (200 tasks) and Apple's
ToolSandbox (91 scenarios) under eight documentation conditions (D0..D7) via OpenRouter, using each
benchmark's own harness, tool implementations, execution environment and scorer unchanged. The
conditions remove tool descriptions (D1), parameter descriptions (D2), tool names (D3) or parameter
names (D4), all four at once (D5), or remove parameter descriptions while moving the prose-only
constraints into the schema as `enum`/`pattern`/renamed parameters (D6) or keeping the six
descriptions that carry the D2 effect (D7). The paper in `paper/` is built from
`runs/main/analysis.json` and `runs/main/index.json` with `./paper/build.sh`.

## One run, every trajectory

Everything is in `runs/main/trajectories/<model>-<condition>/<id>.json`, one complete trajectory
per file, in the same layout for both benchmarks:

- **BFCL** tasks (`multi_turn_base_N.json`) for the four models, the three reasoning variants and
  the repeated Claude Haiku 5.5 D0 launch (`anthropic/claude-haiku-5.5-rerun`, the run-to-run
  noise floor).
- **ToolSandbox** scenarios (`<scenario_name>.json`) for the four models under D0..D5, with
  ToolSandbox's own conversation record, execution context and pretty-printed transcript folded
  into the file under `toolsandbox`.

`runs/main/index.json` has one row per file (benchmark, model, condition, verdict, counts, tokens,
cost, timing) and a per-combo summary nested by benchmark; `run_config.json` lists every launch
that wrote into the run.

`./browse.sh` serves a small local UI at http://localhost:8765/: pick a benchmark and a model, click a
cell in the task x condition matrix, and read the transcript, the verdict (BFCL's state diff, or
ToolSandbox's milestone matches), and the tool schemas exactly as the model saw them.

## Models

`ablation/models.py` lists the model ids. `openai/gpt-6-luna` and `anthropic/claude-haiku-5.5`
reason by default and have `-nothink` variants; `google/gemma-4-26b-a4b-it` does not and has a
`-think` variant; the variants send the same model with OpenRouter's `reasoning` switch set
explicitly. Qwen3-30B-A3B-Instruct-2507 has no reasoning switch (its thinking variant is a different
checkpoint). `anthropic/claude-haiku-5.5-rerun` is the same model and settings launched a second time.

## Run

```bash
cp .env.example .env            # put OPENROUTER_API_KEY=... in .env (already present)
./run_eval.sh                   # BFCL, four models x D0..D7 -> runs/main/
./run_eval.sh --models openai/gpt-6-luna-nothink anthropic/claude-haiku-5.5-nothink google/gemma-4-26b-a4b-it-think --conditions D0 D1 D2 D3 D4 D5
./run_eval.sh --models anthropic/claude-haiku-5.5-rerun --conditions D0
./run_toolsandbox.sh            # ToolSandbox, four models x D0..D5 -> the same runs/main/
LIMIT=3 ./run_eval.sh           # smoke test: first 3 tasks per model x condition
THREADS=16 JOBS=6 ./run_eval.sh # THREADS = concurrent tasks per combo, JOBS = concurrent combos
./run_eval.sh evaluate          # re-score existing BFCL results and rebuild the index
./run_eval.sh index             # rebuild index.json only (both benchmarks)
uv run python -m ablation.analyze   # runs/main/analysis.json (BFCL trajectory analysis, ~5 min)
```

Re-running resumes: tasks and scenarios that already have a trajectory file are skipped. `uv`
manages the BFCL environment (`uv sync` runs automatically); `./run_toolsandbox.sh` creates its own
virtualenv because ToolSandbox pins old `openai`/`polars` versions.

ToolSandbox is played in its `_all_tools` variant (every sandbox tool offered, minus the five that
need a RapidAPI key) with GPT-4o-2024-05-13 as the user simulator, as in the ToolSandbox paper. One
wrinkle: ToolSandbox executes each call as a Python statement named after the tool-call id, and some
providers return ids that are not valid identifiers (a leading digit, or hyphens), which made every
call a `SyntaxError`; the runner rewrites such ids before execution.

## Layout

```
gorilla/                 untouched clone of BFCL (gorilla/berkeley-function-call-leaderboard)
toolsandbox/             untouched clone of Apple's ToolSandbox
ablation/conditions.py   D0..D7 definitions; ablate schemas, map model calls back to original names
ablation/handler.py      BFCL handler subclass: OpenRouter client, ablation hook, raw capture, trajectory dump
ablation/run.py          registers model x condition combos in BFCL, runs generate / evaluate / index
ablation/toolsandbox_run.py   the same conditions on ToolSandbox, same trajectory layout
ablation/index.py        the shared index over both benchmarks
ablation/analyze.py      BFCL trajectory analysis (tool selection vs arguments, wrong parameters, ...)
run_eval.sh, run_toolsandbox.sh, browse.sh   one-command entry points
runs/<run-name>/
  trajectories/<model>-<cond>/<id>.json   one complete trajectory per file, either benchmark
  index.json             one row per trajectory + per-combo summary, nested by benchmark
  analysis.json          output of ablation.analyze
  result/, score/        BFCL's own result and score files (score/*.csv = BFCL leaderboard tables)
  logs/                  generation log per model x condition
  run_config.json        every launch: models, conditions, temperature, BFCL commit, condition spec
```

## Trajectory file contents

BFCL: `task` (prompt turns, initial backend state, involved classes), `tools_sent` (exact tool schemas
shown to the model), `name_maps` (generic -> original names), `bfcl_result` (per-turn, per-step tool
calls in original names, as BFCL scores them), `bfcl_metadata` (BFCL inference log incl. tool outputs
and backend state after each turn, per-step token counts and latencies, reasoning text),
`api_calls` (every raw HTTP request/response to OpenRouter incl. usage/cost, provider, headers,
timestamps; the API key is redacted), `totals` (tokens, cached/reasoning tokens, cost in USD,
latency), `evaluation` (BFCL verdict and error details), `error` (if inference crashed).

ToolSandbox: the same `tools_sent`, `name_maps`, `api_calls` and `totals` for the agent,
`user_api_calls` and `user_totals` for the simulated user, `result` (similarity, milestone and
minefield mappings), `categories`, and `toolsandbox` (ToolSandbox's conversation, execution context
and pretty-printed transcript).

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

The ToolSandbox runner applies the same `apply_condition` to the schemas ToolSandbox generates
from its tools' docstrings and the same translation back on the model's calls.

Ablation details: generic tool names are `tool_01..tool_NN` in the order BFCL lists the task's
functions; generic parameter names are `arg_1..arg_n` per tool, applied recursively to keys inside
object-typed parameters and arrays of objects (the model's argument values are translated back
to the original keys before execution). Removing tool descriptions also strips
the natural-language descriptions inside BFCL's `response` (return value) schema, since that
documents what the tool does; the response schema structure and all parameter types, required
lists, enums and defaults are kept in every condition.
