# Tool Semantics Documentation Ablation Study

## Goals

To observe the impact of different components of a tool's schema documentation on an LLM's ability to use it effectively.

# Goals
We aim to:
- Measure the contribution of individual documentation components to tool-use performance.
- Understand whether frontier and smaller models differ in their dependence on tool documentation.
- Investigate whether models can compensate for missing documentation through multi-turn interaction and exploration.

## Overview

Four models are selected for this study, all served via Openrouter:

1. openai/gpt-6-luna
2. anthropic/claude-haiku-5.5
3. qwen/qwen3-30b-a3b-instruct-2507
4. google/gemma-4-26b-a4b-it

We choose the Berkeley Function-Calling Leaderboard (BFCL) benchmark because it focuses on function calling. We will use the Base Multi-Turn section of BFCL V3, consisting of 200 tasks.

The Base Multi-Turn subset is particularly relevant because it tests foundational multi-turn interactions where the information required to complete a task is available through the user's request, previous execution results, or exploratory function calls.

We will preserve the benchmark's original tasks, tool implementations, execution environment and scoring mechanism. The primary experimental variable will be the documentation provided alongside the tools.

## Evaluation

### 1. Documentation Ablation Matrix

We will evaluate each model under the following documentation conditions:

| ID | Tool names | Tool descriptions | Parameter names | Parameter descriptions |
|---|---|---|---|---|
| D0 | Original | Original | Original | Original |
| D1 | Original | Removed | Original | Original |
| D2 | Original | Original | Original | Removed |
| D3 | Generic | Original | Original | Original |
| D4 | Original | Original | Generic | Original |
| D5 | Generic | Removed | Generic | Removed |

**Ablation details:**

- **Tool names:** Replace semantic names with opaque identifiers, e.g. `search_files` → `tool_01`.
- **Tool descriptions:** Remove natural-language descriptions of tool functionality.
- **Parameter names:** Replace semantic names with generic identifiers, e.g. `directory_path` → `arg_1`.
- **Parameter descriptions:** Remove natural-language explanations of individual parameters.

For the initial experiment, we will preserve parameter types, required/optional constraints, enums and other structural schema information.

Tool execution results and backend state will also remain unchanged since we want to measure the contribution of documentation, not remove the information an agent can obtain through interaction.

D1–D4 isolate the contribution of individual documentation components, while D5 measures the combined effect of removing semantic information.

### 2. Evaluation Metrics

For each model and documentation condition, we will measure:

| Metric | Description |
|---|---|
| Task success rate | Percentage of tasks completed successfully, according to BFCL's original evaluator. |
| Tool selection accuracy | Whether the model selects the appropriate tools for the task. |
| Argument accuracy | Whether the model supplies the correct arguments. |
| Invalid call rate | Percentage of tool calls that fail due to invalid names, parameters or types. |
| Tool calls per task | Number of calls required to complete a task. |
| Interaction turns | Number of assistant-user/tool interaction cycles required. |

Task success rate will be our primary metric. The remaining metrics will help explain the causes of performance degradation.

### 3. Experimental Procedure

For each model:

1. Initialize the BFCL environment using the original task configuration.
2. Provide the model with the tool definitions corresponding to one documentation condition.
3. Execute the task using the original multi-turn interaction protocol.
4. Record the model's responses, tool calls, execution results and final state.
5. Evaluate the final outcome using the original BFCL evaluator.
6. Repeat for every documentation condition.

Each task will be evaluated under every condition, giving us a paired comparison across documentation variants.

We will use identical system prompts, task inputs, tool implementations and execution environments across conditions. Each run will start with a fresh conversation and backend state.

We will also record token consumption to observe whether reduced documentation results in increased reasoning or interaction costs.

Results will be reported separately for each model and documentation condition, with aggregate task success rates and per-task comparisons.
