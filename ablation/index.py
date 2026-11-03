"""One index over every trajectory of a run, whichever benchmark produced it.

    python -m ablation.run index        ->  runs/<run-name>/index.json

Trajectory files live at runs/<run-name>/trajectories/<model>-<condition>/<id>.json. BFCL tasks and
ToolSandbox scenarios share that layout and differ in their verdict: BFCL's is a pass/fail from its
evaluator, ToolSandbox's a similarity in [0, 1]. Every row carries `benchmark`, and the per-combo
summary is nested by benchmark.
"""

import json
from pathlib import Path


def row_for(path: Path, run_dir: Path) -> dict:
    doc = json.loads(path.read_text())
    row = {"file": str(path.relative_to(run_dir)), "benchmark": doc.get("benchmark", "bfcl"),
           "id": doc["id"], "model": doc["model"], "condition": doc["condition"]}
    if row["benchmark"] == "toolsandbox":
        row.update({
            "categories": doc["categories"], "similarity": doc["result"]["similarity"],
            "pass": doc["result"]["similarity"] == 1.0, "turn_count": doc["result"]["turn_count"],
            "error_type": doc["error"] and doc["error"]["type"],
            **doc["totals"], "user_cost_usd": doc["user_totals"]["cost_usd"],
        })
    else:
        turns = doc["bfcl_result"] if isinstance(doc["bfcl_result"], list) else []
        tool_results = [m for turn in doc["bfcl_metadata"].get("inference_log", []) if isinstance(turn, dict)
                        for step in turn.values() if isinstance(step, list) for m in step if m.get("role") == "tool"]
        ev = doc.get("evaluation") or {}
        row.update({
            "valid": ev.get("valid"),
            "error_type": (ev.get("error") or {}).get("error_type"),
            "inference_error": doc["error"] is not None,
            "num_turns": len(turns),
            "num_steps": sum(len(t) for t in turns),
            "num_tool_calls": sum(len(step) for t in turns for step in t if isinstance(step, list)),
            "num_failed_tool_calls": sum(1 for r in tool_results if str(r["content"]).startswith("Error during execution")),
            **doc["totals"],
        })
    row.update({"wall_time_s": doc["wall_time_s"], "started_at": doc["started_at"], "finished_at": doc["finished_at"]})
    return row


def summarise(rows: list) -> dict:
    summary = {"bfcl": {}, "toolsandbox": {}}
    for r in rows:
        key = f"{r['model']} {r['condition']}"
        if r["benchmark"] == "toolsandbox":
            s = summary["toolsandbox"].setdefault(key, {"scenarios": 0, "passed": 0, "similarity": 0.0, "cost_usd": 0.0, "user_cost_usd": 0.0})
            s["scenarios"] += 1
            s["passed"] += r["pass"]
            s["similarity"] += r["similarity"]
            s["cost_usd"] += r["cost_usd"]
            s["user_cost_usd"] += r["user_cost_usd"]
        else:
            s = summary["bfcl"].setdefault(key, {"tasks": 0, "evaluated": 0, "passed": 0, "cost_usd": 0.0})
            s["tasks"] += 1
            s["evaluated"] += r["valid"] is not None
            s["passed"] += r["valid"] is True
            s["cost_usd"] += r["cost_usd"]
    for s in summary["toolsandbox"].values():
        s["mean_similarity"] = s.pop("similarity") / s["scenarios"]
        s["pass_rate"] = s["passed"] / s["scenarios"]
    for s in summary["bfcl"].values():
        s["success_rate"] = s["passed"] / s["evaluated"] if s["evaluated"] else None
    return summary


def build_index(run_dir: Path):
    rows = [row_for(p, run_dir) for p in sorted((run_dir / "trajectories").glob("*/*.json"))]
    summary = summarise(rows)
    run = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    (run_dir / "index.json").write_text(json.dumps({"run": run, "summary": summary, "trajectories": rows}, indent=1))
    print(f"index: {run_dir / 'index.json'}  ({len(rows)} trajectories)")
    for name, s in summary["bfcl"].items():
        print(f"  bfcl        {name:45s} {s['passed']}/{s['evaluated']} passed  ${s['cost_usd']:.4f}")
    for name, s in summary["toolsandbox"].items():
        print(f"  toolsandbox {name:45s} similarity {s['mean_similarity']:.3f}  pass {s['passed']}/{s['scenarios']}  "
              f"agent ${s['cost_usd']:.2f} user ${s['user_cost_usd']:.2f}")
