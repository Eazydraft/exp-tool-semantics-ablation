"""Dump ToolSandbox's ground truth for the 91 scenarios we play, as JSON on stdout (ablation.audit reads it).

    .venv-toolsandbox/bin/python -m ablation.toolsandbox_gt > ...

Per scenario: the user's messages (the instruction the simulator is given and any scripted user turns)
and every milestone constraint: the database it pins, the target row (a database state, a tool call in
`calls`, or the agent's reply), the similarity measure per column, and whether the target is filled in
from an earlier call's result at evaluation time (tool_trace_dependant_similarity).
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load() -> dict:
    """The dump, produced in ToolSandbox's own virtualenv (ablation.audit and analyze_extra run elsewhere)."""
    python = PROJECT_DIR / ".venv-toolsandbox" / "bin" / "python"
    out = subprocess.run([str(python), "-m", "ablation.toolsandbox_gt"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    from tool_sandbox.common.execution_context import DatabaseNamespace

    from ablation.toolsandbox_run import scenarios

    out = {}
    for name, scenario in scenarios().items():
        ctx = scenario.starting_context
        user_text = []
        for m in ctx.get_database(DatabaseNamespace.SANDBOX).to_dicts():
            sender, recipient = str(m["sender"]).split(".")[-1], str(m["recipient"]).split(".")[-1]
            if m["content"] and (sender == "USER" or (sender == "SYSTEM" and recipient == "USER")):
                user_text.append(m["content"])
        constraints = []
        for mi, milestone in enumerate(scenario.evaluation.milestone_matcher.milestones):
            for c in milestone.snapshot_constraints:
                df = c.target_dataframe
                fn = c.snapshot_constraint
                kwargs = {}
                if hasattr(fn, "keywords"):  # tool_trace_dependant_similarity is a partial carrying fill_to / extractor
                    kwargs = {k: (v.__name__ if callable(v) else v) for k, v in fn.keywords.items()}
                    fn = fn.func
                measures = {}
                for col, meas in (c.column_similarity_measure or {}).items():
                    m = meas.func if hasattr(meas, "func") else meas
                    measures[col] = {"measure": getattr(m, "__name__", str(m)),
                                     "atol": (meas.keywords or {}).get("atol_dict") if hasattr(meas, "keywords") else None}
                rows = df.to_dicts() if df is not None else []
                for row in rows:
                    trace = row.pop("tool_trace", None)
                    traces = []
                    if trace is not None:
                        traces = json.loads(trace) if isinstance(trace, str) else [json.loads(t) for t in trace]
                        traces = traces if isinstance(traces, list) else [traces]
                    constraints.append({"milestone": mi, "namespace": str(c.database_namespace).split(".")[-1],
                                        "measure": fn.__name__, "dependant": fn.__name__ == "tool_trace_dependant_similarity",
                                        "extractor": kwargs.get("extractor"), "fill_to": kwargs.get("fill_to"),
                                        "reference_milestone": c.reference_milestone_node_index,
                                        "row": {k: v for k, v in row.items() if v is not None},
                                        "calls": [{"tool": t["tool_name"], "arguments": t.get("arguments", {})} for t in traces],
                                        "column_measures": {k: v for k, v in measures.items() if k in row or k == "tool_trace"}})
        out[name] = {"user_text": user_text, "categories": [str(x).split(".")[-1] for x in scenario.categories], "constraints": constraints}
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
