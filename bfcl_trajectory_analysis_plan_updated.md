# Analysis Plan: Visualizing BFCL Multi-Turn Tool-Use Trajectories

The raw trajectories may be more informative than the final BFCL scores
alone. Aggregate scores tell us **whether documentation matters**;
trajectory analysis can help explain **why, where, and how models
compensate when documentation is missing**.

This plan organizes the analysis into five groups.

## 1. Documentation sensitivity

These are the headline visualizations for the study.

### A. Performance degradation by ablation

A grouped bar chart showing task success rate for each model across
documentation conditions.

-   **X-axis:** Documentation condition (D0--D5).
-   **Y-axis:** Task success rate.
-   **Series:** One per model.

This is likely the most important figure in the paper.

### B. Documentation sensitivity heatmap

A heatmap with:

-   **Rows:** Models.
-   **Columns:** Documentation components removed.
-   **Values:** Percentage-point change in task success relative to
    baseline.

Illustrative layout (values below are examples, not predictions or
measured results):

  -----------------------------------------------------------------------
  Model                 Tool     Tool names      Parameter      Parameter
                descriptions                         names   descriptions
  ----------- -------------- -------------- -------------- --------------
  GPT                  -4 pp          -8 pp          -5 pp          -3 pp

  Claude               -6 pp         -11 pp          -7 pp          -4 pp

  Qwen                -15 pp         -22 pp         -18 pp         -12 pp

  Gemma               -19 pp         -25 pp         -21 pp         -16 pp
  -----------------------------------------------------------------------

This makes it easy to identify which documentation components each model
depends on most.

### C. Model robustness ranking

Plot relative degradation from baseline across ablations. This provides
a compact measure of documentation sensitivity for each model.

One possible definition is:

\[ S_m = `\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}
`\frac{P_{m,0}-P_{m,i}}{P_{m,0}}`{=tex} \]

Where:

-   (P\_{m,0}) is baseline performance for model (m).
-   (P\_{m,i}) is performance under ablation condition (i).
-   \(N\) is the number of ablation conditions included.

Lower sensitivity indicates greater robustness to missing documentation.
Report the underlying condition-level results as well; a single score
can conceal important differences between ablations.

## 2. Tool-use behaviour

Raw trajectories are particularly useful for understanding how models
change their behaviour.

### D. Tool-call sequence comparison

Visualize tool calls as sequences and compare the same task across
documentation conditions.

For example:

``` text
Task
  |
  v
search_files
  |
  v
inspect_metadata
  |
  v
read_document
  |
  v
update_document
  |
  v
Success
```

Removing descriptions may cause models to:

-   Select a different initial tool.
-   Make unnecessary exploratory calls.
-   Repeat calls.
-   Take longer paths to reach the same outcome.

Use sequence diagrams for selected individual tasks. For aggregate
transitions, consider a Sankey diagram or transition matrix. Keep the
task and initial state fixed when comparing trajectories.

### E. Tool-call distribution

Use grouped or stacked bar charts to show the frequency of:

-   Correct tool calls.
-   Incorrect tool calls.
-   Invalid tool calls.
-   Repeated calls.
-   Exploratory calls.

Compare these distributions across models and ablation conditions.

Define these categories operationally before analyzing the runs. For
example, distinguish an invalid call (which cannot be executed) from a
valid but inappropriate call (which executes but does not advance the
task).

### F. Interaction overhead

Plot average tool calls per task against task success rate, or report
calls per successful task alongside success rate.

This can reveal an important trade-off: a model may maintain its success
rate despite missing documentation, but require substantially more tool
calls.

That is still a meaningful efficiency cost.

## 3. Recovery and exploration

This may produce some of the most interesting findings.

### G. Recovery rate after first failure

Measure how often a model eventually completes a task after its first
unsuccessful tool call.

\[ `\text{Recovery rate}`{=tex} =
`\frac{\text{Tasks succeeding after an unsuccessful first call}}`{=tex}
{`\text{Tasks with an unsuccessful first call}`{=tex}} \]

Plot this for every model and documentation condition.

This distinguishes models that recover from mistakes from models that
persist with an incorrect interpretation.

Be precise about what counts as an unsuccessful call: an invalid call, a
tool error, or a valid call that does not advance the task. These can
also be reported separately.

### H. Calls until success

Plot the distribution of tool calls required to complete tasks. Box
plots, violin plots, or empirical cumulative distribution functions
(ECDFs) are suitable.

Compare baseline documentation (D0) with the more heavily ablated
conditions.

This shows whether models compensate for missing semantics through
additional exploration. Include unsuccessful and timed-out tasks in the
analysis rather than silently dropping them; otherwise, the most
difficult runs may disappear from the distribution.

### I. Exploration behaviour

Categorize calls into:

-   Task-directed calls.
-   Exploratory calls.
-   Repeated calls.
-   Recovery calls.

Visualize the proportion of each category across models and ablations.

This categorization needs a careful definition. Ideally, derive it from
the task's intended execution path, tool outputs, and trajectory
context. Avoid relying solely on an LLM judge to decide whether a call
was exploratory.

## 4. Failure analysis

Failure analysis helps explain the aggregate results.

### J. Failure taxonomy

Classify failed tasks into categories:

  -----------------------------------------------------------------------
  Failure category                    Description
  ----------------------------------- -----------------------------------
  Tool selection                      Selected an inappropriate tool.

  Argument construction               Selected an appropriate tool but
                                      supplied incorrect arguments.

  Schema violation                    Produced invalid parameter names,
                                      types, or values.

  Context loss                        Failed to use information from
                                      previous turns.

  Repetition                          Repeated an unsuccessful or
                                      unnecessary operation.

  Premature termination               Stopped before completing the task.

  State misunderstanding              Misinterpreted the environment's
                                      current state.
  -----------------------------------------------------------------------

Visualize this as a stacked bar chart for each model and documentation
condition.

A task can exhibit multiple failure modes. Decide whether to assign one
primary failure category per task or allow multiple labels, and make the
denominator clear in the chart.

### K. Failure transition analysis

For failed tasks, inspect the trajectory to identify the earliest point
at which the model diverged from a successful execution path.

This can help distinguish:

-   Initial semantic misunderstanding.
-   Incorrect inference from tool output.
-   Failure to retain previously obtained information.
-   Failure to recover from an earlier mistake.

This is especially useful for qualitative case studies. Select
representative examples systematically, rather than choosing only
unusually dramatic failures.



## 4A. Diagnostic focus: where does tool-interface meaning matter?

The preliminary GPT-6 Luna results suggest a potentially important distinction
between **tool-level semantics** and **argument-level semantics**. The analysis
should explicitly test this distinction rather than treating all documentation
as one undifferentiated source of help.

### Preliminary signal (not yet a final result)

The first completed GPT-6 Luna conditions were reported as follows:

| Condition | Ablation | Success | Steps/task | Failed tool calls/task |
|---|---|---:|---:|---:|
| D0 | None (baseline) | 56.0% | 10.2 | 0.01 |
| D1 | Tool descriptions removed | 52.5% | 10.2 | 0.01 |
| D2 | Parameter descriptions removed | 32.5% | 10.6 | 0.15 |
| D3 | Tool names made generic | 57.5% | 10.4 | 0.00 |

On this 200-task set, each task represents 0.5 percentage points. Relative
to D0, D1 is down 3.5 pp, D2 is down 23.5 pp, and D3 is up 1.5 pp. The D2
condition therefore has a large preliminary drop: 47 fewer successful tasks
than D0 if the reported percentages correspond to the same 200 tasks.

Treat these as **descriptive preliminary observations**, not inferential
conclusions. In particular, the small D1 and D3 differences may be noise.
The stronger initial signal is that removing parameter descriptions coincides
with a substantial reduction in success and an increase in failed tool calls.
The near-constant average step count does not, by itself, establish that
behaviour is unchanged; the sequence and content of calls may differ even
when their count does not.

### Working hypothesis

> Parameter descriptions may carry more operationally important semantics
> than tool descriptions: they help the agent construct arguments that are
> compatible with the intended operation and the environment's current state.

This is a hypothesis to test, not an established explanation. Parameter
descriptions can encode argument meaning, expected value formats, valid
entities, temporal or state constraints, and relationships between arguments.
The analysis should identify which of these functions explains any observed
effect.

### Separate tool selection from argument construction

For every task and condition, annotate the earliest relevant decision and
measure the following separately:

1. **Tool selection:** Was the selected tool appropriate for the next
   subgoal?
2. **Argument-role interpretation:** Did the model understand what each
   argument represented?
3. **Argument value construction:** Were the supplied values correct for
   the task and current state?
4. **State grounding:** Did the model correctly use information returned by
   earlier tool calls?
5. **Execution and task progress:** Did the call execute, and did it move the
   environment toward the required final state?

This decomposition matters because a call can select the right tool and
still fail due to an incorrect argument value. Conversely, a syntactically
valid call can execute successfully while acting on the wrong entity or
failing to advance the task.

Do not equate a benchmark state-mismatch label with an argument-construction
error without inspecting the task, call, tool result, relevant state, and
evaluator output. State mismatch is an observed outcome; its cause needs
separate diagnosis.

### Targeted analyses

#### A. First-call tool-selection accuracy

For each task, compare the first tool selected with the tool or set of tools
that could validly advance the task from its initial state.

Report:
- Appropriate first-tool rate.
- Wrong-tool rate.
- Invalid-tool/schema rate.
- No-tool or premature-termination rate.

Use task-grounded rules where possible. Some tasks may admit multiple valid
first actions, so avoid treating one canonical trajectory as the only correct
one.

#### B. Argument-construction accuracy, conditional on tool choice

Among calls that select an appropriate tool, classify the arguments:

- Correct argument roles and values.
- Correct roles, incorrect values.
- Incorrect role-to-value mapping.
- Missing required argument.
- Invalid type or format.
- Value inconsistent with the current environment state.
- Value not supported by available task/context information.

Report both call-level and task-level denominators. A task may contain
multiple calls and multiple argument errors; do not let a high-volume task
silently dominate a call-level metric.

#### C. State-mismatch investigation

For each state-mismatch failure, reconstruct the relevant state immediately
before the call and determine whether the submitted arguments were:

- Correct for the current state.
- Plausible but stale (based on an earlier state).
- Incorrectly inferred from tool output.
- Incorrectly copied or transformed from the user request.
- Underdetermined by the available context.
- Correct in isolation but inconsistent with another argument.

Compare the model's preceding observations with the actual environment state.
This helps distinguish missing parameter semantics from state-tracking,
context-retention, or task-interpretation failures.

#### D. Error amplification across turns

Find the earliest point at which D0 and an ablated run diverge for the same
task. Then trace whether the initial error:
- Is corrected on the next turn.
- Triggers a useful information-gathering call.
- Causes repeated attempts with the same incorrect value.
- Causes later calls to be grounded in an incorrect assumed state.
- Leads to premature termination.

This can reveal whether parameter documentation prevents initial argument
errors, helps diagnose tool feedback, or both.

#### E. Documentation-to-error linkage

For each argument error, record which semantic information was present in
the original parameter description and absent in the ablated condition.
Where possible, label the missing information as:
- Argument purpose / role.
- Expected entity or identifier.
- Value format or units.
- Allowed values or constraints.
- Relationship to other parameters.
- State precondition or side effect.
- Disambiguation between similar concepts.

This analysis is necessary to move from “parameter descriptions matter” to
a more precise account of *what information* matters.

### Paired task-level statistics

Because the same benchmark tasks are run under multiple documentation
conditions, use paired comparisons rather than treating condition scores as
independent samples.

For each ablation versus D0:
- Build a 2×2 task transition table: both succeed, D0-only succeeds,
  ablation-only succeeds, both fail.
- Use McNemar's test for the paired binary success outcome.
- Estimate the paired success-rate difference with a task-level paired
  bootstrap confidence interval.
- Report the two discordant cells, not only the p-value.
- For multiple ablation comparisons, state how multiplicity is handled or
  label the analyses exploratory.

For the preliminary D0–D2 comparison, the aggregate percentages alone are
not enough to calculate McNemar's test or a paired confidence interval.
The per-task outcomes are required.

### Interpreting call counts and recovery

The preliminary average steps/task (10.2 in D0 versus 10.6 in D2) is not
strong evidence either for or against exploration. Similar counts can hide
different call sequences, and “steps” may not be equivalent to tool calls.
First establish exactly what the BFCL/harness step metric counts.

In addition to mean calls or steps, compare:
- First-call correctness.
- Number of distinct tools attempted.
- Repeated calls with identical or near-identical arguments.
- Calls that gather information before acting.
- Calls after an execution error or state mismatch.
- Whether the next call changes the failed argument or repeats it.
- Task completion conditional on a first unsuccessful call.

Define these measures from the raw event stream and benchmark state, not
from aggregate step counts alone.

### Essential trajectory comparison

For every task, align D0 and D2 trajectories by task subgoal or state
transition rather than simply by turn number. A useful case-study panel
should show:

1. User request and initial environment state.
2. Original parameter description (the semantic information removed).
3. D0 call and arguments.
4. D2 call and arguments.
5. Tool outputs and state changes.
6. Earliest divergence and final evaluator outcome.

Select examples systematically: include representative cases where D0
succeeds/D2 fails, cases where both succeed, and cases where both fail.
Avoid selecting only striking anecdotes.

### What the results would and would not support

If the pattern replicates across models and paired task analysis confirms a
large D2 effect, the evidence would support the claim that parameter-level
documentation is especially important for reliable argument construction
in this benchmark setting.

It would **not**, by itself, show that:
- Tool descriptions are generally unnecessary.
- Models understand tools without documentation.
- Parameter descriptions are more important in every tool-use setting.
- The effect is caused only by argument semantics rather than state,
  formatting, or other information carried in descriptions.

The interpretation must also account for semantic information that remains
after ablation: tool outputs, user requests, parameter types, requiredness,
enum values, structural schema, and any informative naming that the
condition does not remove. D5 is therefore a test of reduced interface
semantics, not a condition with zero semantic information.


## 5. Cost and efficiency

### L. Token consumption versus success

Use a scatter plot:

-   **X-axis:** Input tokens consumed.
-   **Y-axis:** Task success (or a binary success indicator with
    jitter).
-   **Points:** Individual task executions.
-   **Groups:** Model and documentation condition.

This can show whether missing documentation leads to increased context
consumption. Consider separate plots for input tokens, output tokens,
and total tokens because they reflect different costs.

### M. Efficiency frontier

Plot task success against average tool calls per task. Models closer to
the upper-left corner achieve higher success with fewer calls.

A similar plot can use total tokens instead of tool calls.

Report latency separately where provider latency is meaningful, but be
cautious about comparing latency across providers because routing,
queueing, and infrastructure differ.

## Recommended figures for the paper

Not every analysis needs to appear in the main paper. A focused set
would be:

  -----------------------------------------------------------------------
  Figure                  Visualization           Purpose
  ----------------------- ----------------------- -----------------------
  1                       Performance degradation Main experimental
                          chart                   result.

  2                       Documentation           Identify important
                          sensitivity heatmap     schema components.

  3                       Tool-call overhead      Measure compensation
                          chart                   through interaction.

  4                       Recovery-rate chart     Measure resilience
                                                  after unsuccessful
                                                  calls.

  5                       Failure taxonomy        Explain performance
                                                  degradation.

  6                       Representative          Show concrete
                          trajectory comparison   behavioural
                                                  differences.
  -----------------------------------------------------------------------

Other plots, detailed per-task results, and additional trajectory
examples can go in an appendix or supplementary material.

## Raw trajectory data to capture

Do not save only the final BFCL result. Save a complete structured
trajectory for every execution.

A possible event record:

``` json
{
  "run_id": "uuid",
  "task_id": "multi_turn_base_001",
  "model": "provider/model-id",
  "condition": "D3",
  "turn": 2,
  "assistant_response": "...",
  "tool_calls": [
    {
      "tool_name": "tool_01",
      "arguments": {
        "arg_1": "..."
      }
    }
  ],
  "tool_results": [
    {
      "result": "..."
    }
  ],
  "input_tokens": 1200,
  "output_tokens": 150,
  "latency_ms": 850
}
```

Also capture:

-   Original and transformed tool schemas.
-   Exact prompts and model responses.
-   Raw provider responses, including malformed tool-call outputs.
-   Parsed tool calls.
-   Tool execution errors and results.
-   Backend state before and after execution, where the benchmark
    permits it.
-   Final BFCL evaluator output and intermediate scoring details, where
    available.
-   Per-task success/failure outcome for every condition, with a stable
    task identifier so paired transitions can be computed.
-   The relevant environment state immediately before and after each tool
    call, where the benchmark permits inspection.
-   A normalized record of argument roles, values, validation outcomes,
    and state-mismatch details.
-   The earliest divergence point between paired baseline and ablated
    trajectories, with the annotation rationale.
-   Provider and exact model identifiers.
-   Provider routing configuration.
-   Inference settings, including temperature, token limits, and seed
    where supported.
-   Timestamps and retry information.

Keep raw model responses separately from parsed tool calls. Otherwise,
information about malformed calls and protocol-formatting failures may
be lost.

Use a stable event schema and preserve the original benchmark artifacts.
This makes it possible to recompute behavioural metrics later without
rerunning expensive model evaluations.

## A particularly interesting research question

Investigate:

**Where in the tool-use process does interface documentation provide its
largest contribution: selecting the right tool, constructing the right
arguments, grounding those arguments in the current state, or recovering
from errors?**

The preliminary results make argument construction a particularly important
candidate. Test it by comparing:

1. Probability of selecting an appropriate first tool.
2. Argument correctness conditional on selecting an appropriate tool.
3. Frequency and causes of state-mismatch failures.
4. Probability of eventual task completion after an unsuccessful call.
5. Whether the model changes its arguments or repeats the same mistake.
6. Number and type of information-gathering, repeated, and recovery calls.

A useful outcome would be to determine whether parameter descriptions
primarily prevent incorrect arguments at the point of construction, help
the model interpret tool feedback, or support both. This is more precise
than asking only whether a model can use tools when documentation is
removed.

## Practical recommendation

Build the trajectory collection and analysis pipeline before running the
full experiment. It will be difficult to reconstruct behavioural metrics
after the fact if the harness only saves aggregate BFCL scores.

At minimum, ensure every run can be reconstructed from its task
configuration, documentation condition, model request/response sequence,
tool execution events, and final evaluator output.
