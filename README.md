# Which Parts of a Tool's Documentation Does a Model Actually Need?

Code, data and paper for an ablation of tool documentation in multi-turn tool use. Four models
(GPT-6 Luna, Claude Haiku 5.5, Qwen3-30B-A3B-Instruct-2507, Gemma 4 26B-A4B) ran three benchmarks
(BFCL multi-turn base, Apple's ToolSandbox, the airline domain of τ²-bench) under conditions that
remove one piece of the documentation at a time, then all of them, then test what restores it.
Every trajectory of every run is in this repository, with the schemas exactly as the model saw them.

**Paper:** [paper/main.pdf](paper/main.pdf) (sources in `paper/`).

## The short version

Most of what you write in a tool schema, the model never needs. What it needs is each piece of
information stated **once, somewhere it can see it**. On BFCL, the only prose that mattered was a
handful of parameter descriptions that were the sole statement of a constraint ("Buy" not "buy",
a zip code not a city name, tags start with `#`). Move those constraints into `enum`, `pattern`
and a better parameter name and you can blank every description without losing a point. Remove
every carrier of what a tool *does*, and models on the airline benchmark start cancelling
reservations they meant to look up.

**If you design tools:** for every parameter description, ask whether it states a constraint the
schema does not. If it does, put it in the schema too (`enum`, `pattern`, a name that says what the
value is) and keep the prose for what structure cannot say. That is the whole difference between
a 56% and a 32% task success for one of the models here. Tool names, descriptions and parameter names
restate each other, so any one of them can go; parameter names and descriptions together cannot,
and removing everything collapses tool selection.

**If you build benchmarks:** know where your conventions live. A benchmark that keeps its enums and
formats in free text is measuring prose parsing as much as tool use, and a score can turn on
whether an allowed value was written as `[Enum]: ...` or as "(Buy/Sell)". Two benchmarks that
state the same information structurally, or let the user supply the values, showed no measurable
cost from the same ablation.

**If you evaluate agents:** one run per condition is not enough. Two identical BFCL runs eight
hours apart disagree on tens of tasks; the effects reported here are an order of magnitude larger
than that floor, and every comparison is paired at the task level with a bootstrap interval. The
repeated runs, the noise floor and the equivalence bounds for every null are all in the paper.

## What the study finds

- Removing **parameter descriptions** is the only single removal that lowers task success for every
  model on BFCL. Removing tool names or tool descriptions stays within run-to-run noise; removing
  all four pieces of prose leaves almost nothing.
- The damage is at **argument construction**, not tool selection, and comes from a few parameters
  whose description was the only statement of a constraint: an allowed-value list, a format, an
  entity type. An audit of the ground truth finds 40% of BFCL arguments in that situation.
- **Writing those constraints into the schema** (`enum`, `pattern`, a better parameter name) returns
  every model to baseline with every description blank.
- On ToolSandbox and τ²-bench, where the same audit finds nearly every such constraint stated
  structurally or the value copied from the conversation, removing the descriptions has no detectable
  cost (on τ², four trials per condition bound any effect below the smallest BFCL one).
- Removing **parameter names and descriptions together** (D8) costs more than either alone: the two
  restate what a parameter is, and losing both leaves only the type.

BFCL task success (%), each model in its default reasoning setting, 200 tasks per cell:

| | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6 Luna | 56.0 | 52.5 | 32.5 | 57.5 | 58.5 | 1.0 | 54.0 | 52.0 | 23.5 |
| Claude Haiku 5.5 | 50.5 | 48.0 | 36.5 | 55.5 | 61.5 | 0.0 | 54.0 | 55.0 | 25.5 |
| Qwen3-30B-A3B | 54.5 | 48.5 | 28.5 | 52.5 | 43.0 | 0.0 | 51.0 | 44.5 | 17.0 |
| Gemma 4 26B-A4B | 72.0 | 69.0 | 39.0 | 71.5 | 74.5 | 0.0 | 70.0 | 64.5 | 17.0 |

Paired tests, intervals, the reasoning-on/off pairs, the repeated baseline, ToolSandbox and τ²
results are in the paper and in `runs/main/analysis*.json`.

## Conditions

| | Tool names | Tool descriptions | Parameter names | Parameter descriptions |
|---|---|---|---|---|
| D0 | original | original | original | original |
| D1 | original | **removed** | original | original |
| D2 | original | original | original | **removed** |
| D3 | **generic** | original | original | original |
| D4 | original | original | **generic** | original |
| D5 | **generic** | **removed** | **generic** | **removed** |
| D6 | original | original | original | **removed**; prose-only constraints moved into the schema |
| D7 | original | original | original | **removed**; the six descriptions that carry the D2 effect kept |
| D8 | original | original | **generic** | **removed** |

Generic tool names are `tool_01..tool_NN` in listing order; generic parameter names are `arg_1..arg_n`
per tool, applied recursively to nested object keys. Structural fields (types, required lists, enums,
defaults, response schemas) are kept in every condition. Removing tool descriptions also strips the
prose inside BFCL's response schemas, since it documents what the tool does. The model's calls are
translated back to the original names before the benchmark executes and scores them, so the backend
and the evaluator see what they would in D0. Definitions: `ablation/conditions.py`.

## Setup

```bash
git clone --recursive https://github.com/Eazydraft/exp-tool-semantics-ablation.git
cd exp-tool-semantics-ablation
cp .env.example .env           # OPENROUTER_API_KEY=...  (every model, user simulator and judge is reached through OpenRouter)
```

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). The three benchmarks are pinned
submodules and are used unmodified: `gorilla/` (BFCL), `toolsandbox/`, `tau2-bench/`. `./run.sh`
creates the environments on first use (`uv sync` for BFCL and the analysis; a separate virtualenv each
for ToolSandbox and τ², whose pinned dependencies conflict). Building the paper needs a LaTeX
installation with `latexmk`.

## Running

```bash
./run.sh bfcl                       # four models x D0..D8 x 200 tasks -> runs/main/
./run.sh bfcl --models openai/gpt-6-luna-nothink anthropic/claude-haiku-5.5-nothink google/gemma-4-26b-a4b-it-think --conditions D0 D1 D2 D3 D4 D5
./run.sh bfcl --models anthropic/claude-haiku-5.5-rerun --conditions D0     # the repeated baseline
./run.sh toolsandbox                # four models x D0..D5 x 91 scenarios -> the same runs/main/
./run.sh tau2                       # four models x D0..D5 x 50 airline tasks -> the same runs/main/
./run.sh tau2 --conditions D0 D2 --trials 2 3 4    # three more trials of D0 and D2 (files <task>.t<n>.json)
LIMIT=3 ./run.sh bfcl               # smoke test: first 3 tasks per model x condition
THREADS=16 JOBS=6 ./run.sh bfcl     # tasks in flight per combo, combos in parallel
./run.sh bfcl evaluate              # re-score existing BFCL results and rebuild the index
./run.sh index                      # rebuild runs/main/index.json only
./run.sh analyze                    # analysis.json, audit.json, analysis_extra.json
./run.sh paper                      # figures, tables, paper/main.pdf
./run.sh browse                     # trajectory browser at http://localhost:8765/
```

Re-running resumes: tasks and scenarios that already have a trajectory file are skipped. The whole
study (15,784 trajectories) cost about $45 in BFCL model calls plus about $87 for ToolSandbox and τ² including their user simulators and judges.

`ablation/models.py` lists the model ids. `openai/gpt-6-luna` and `anthropic/claude-haiku-5.5`
reason by default and have `-nothink` variants; `google/gemma-4-26b-a4b-it` does not and has a
`-think` variant (the same model with OpenRouter's `reasoning` switch set explicitly). Qwen3 has no
switch. `anthropic/claude-haiku-5.5-rerun` is the same model and settings launched a second time.

## Browsing the trajectories

`./run.sh browse` serves a local page: pick a benchmark and a model, click a cell in the task ×
condition matrix, and read the transcript, the verdict (BFCL's state diff, ToolSandbox's milestone
matches, τ²'s database and communicate checks) and the tool schemas exactly as the model saw them,
with the original name beside each generic one.

## Layout

```
run.sh                   entry point (see above)
ablation/
  conditions.py          D0..D8: ablate a tool schema, map the model's calls back to original names
  handler.py             BFCL handler subclass: OpenRouter client, ablation hook, raw capture, trajectory dump
  run.py                 BFCL: register model x condition combos, generate, evaluate, index
  toolsandbox_run.py     the same conditions on ToolSandbox (stock OpenAI-API agent, patched in and out)
  tau2_run.py            the same conditions on tau2-bench airline (stock LLMAgent, patched in and out)
  models.py              model ids, reasoning variants, OpenRouter prices
  index.py               one index over every trajectory of every benchmark
  analyze.py             BFCL trajectory analysis: paired tests, selection vs construction, wrong parameters, behaviour
  audit.py               information audit: where each ground-truth argument's constraint is stated (all three benchmarks)
  toolsandbox_gt.py      dumps ToolSandbox's milestone ground truth (runs inside .venv-toolsandbox)
  analyze_extra.py       tau2 statistics; first-divergence splits for tau2 and ToolSandbox
browser/                 the trajectory browser (static page + local server)
paper/                   LaTeX sources, figures.py and tables.py (read only runs/main/*.json), build.sh, main.pdf
runs/main/
  trajectories/<model>-<condition>/<id>.json   one complete trajectory per file, every benchmark
  index.json             one row per trajectory + per-combo summary, nested by benchmark
  analysis.json, audit.json, analysis_extra.json
  result/, score/        BFCL's own result and score files
  logs/                  generation log per model x condition (written at run time, not tracked)
  run_config.json        every launch: models, conditions, temperature, BFCL commit, condition spec
```

## Trajectory files

BFCL (`multi_turn_base_N.json`): `task` (prompt turns, initial backend state, involved classes),
`tools_sent` (the exact schemas shown to the model), `name_maps` (generic → original names),
`bfcl_result` (per-turn, per-step calls in original names, as BFCL scores them), `bfcl_metadata`
(BFCL's inference log with tool outputs and backend state after each turn), `api_calls` (every raw
HTTP request and response to OpenRouter with usage, cost, provider and timestamps; the API key is
redacted), `totals`, `evaluation` (BFCL's verdict and error details), `error` if inference crashed.

ToolSandbox (`<scenario>.json`): the same `tools_sent`, `name_maps`, `api_calls` and `totals` for
the agent, `user_api_calls` and `user_totals` for the simulated user, `result` (similarity,
milestone and minefield mappings), `categories`, and `toolsandbox` (ToolSandbox's conversation,
execution context and pretty-printed transcript).

τ²-bench (`N.json`): `tools_sent`, `name_maps`, `api_calls` (agent, user simulator and judge calls),
`totals`, `user_totals`, `judge_totals`, `result` (reward, termination reason, reward breakdown) and
`tau2` (the benchmark's own simulation record with every message).

## How the ablation is applied

BFCL's `OpenAICompletionsHandler` is subclassed (`ablation/handler.py`). The only differences from
a stock BFCL run against an OpenAI-compatible endpoint: `_compile_tools` applies the condition to
the task's function docs before BFCL converts them to tool format (D0 produces byte-identical
tools); `_parse_query_response_FC` maps the model's calls back to the original names, so BFCL
executes and scores them unchanged, while the chat history keeps the names the model was given;
requests go to OpenRouter with `usage.include=true`. Names the model invents pass through untouched
and fail inside the benchmark's executor as they would in the original.

The ToolSandbox and τ² runners do the same to the schemas their benchmarks generate from tool
docstrings and signatures. ToolSandbox is played in its `_all_tools` variant for the 91 base
scenarios that need no RapidAPI key, with GPT-4o-2024-05-13 as the user as in the ToolSandbox paper
(one fix: it executes each call as a Python statement named after the tool-call id, and some
providers return ids that are not valid identifiers, which the runner rewrites). τ² runs the airline
domain in text mode, one trial, with GPT-4.1 as the user and the benchmark's pinned judge.

## Citation

```bibtex
@misc{singh2026tooldocs,
  title  = {Which Parts of a Tool's Documentation Does a Model Actually Need?},
  author = {Singh, Randeep},
  year   = {2026},
  url    = {https://github.com/Eazydraft/exp-tool-semantics-ablation}
}
```

## License

The code (`ablation/`, `browser/`, `run.sh`, the scripts in `paper/`) is under the [MIT License](LICENSE).
The data and the paper (`runs/`, the paper sources and PDF) are under the
[Creative Commons Attribution 4.0 International License](LICENSE-CC-BY-4.0) (CC BY 4.0).
Copyright 2026 Lucio AI and Randeep Singh. The benchmark submodules keep their own licenses.
