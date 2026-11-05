"""Trajectory analysis of the BFCL run.

    python -m ablation.analyze --run-name main      ->  runs/main/analysis.json

Everything here is computed from the trajectory files plus BFCL's ground truth;
no LLM judging. Operational definitions used throughout:
  - call            one tool call the model emitted (BFCL decoded it); original names.
  - exec error      the backend returned "Error during execution: ..." for a call.
  - repeated call   identical (name, arguments) to an earlier call in the same task.
  - on-path call    the tool name occurs in the task's ground-truth call list.
  - GT call hit     for a ground-truth call in turn t: the model called the same tool
                    in turns <= t (tool hit), with identical arguments (exact hit).
"""

import argparse
import ast
import json
import os
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONDITIONS = ["D0", "D1", "D2", "D3", "D4", "D5"]  # extended at runtime with any follow-up conditions (D6, D7) present in the run
ABLATED = CONDITIONS[1:]
SINGLE = ["D1", "D2", "D3", "D4"]


# ---- ground truth ---------------------------------------------------------------

def load_ground_truth():
    """{task_id: {"turns": [[(name, {arg: value}), ...] per turn], "param_desc": {tool: {param: desc}}}}"""
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH
    from bfcl_eval.utils import load_dataset_entry, load_file

    entries = {e["id"]: e for e in load_dataset_entry("multi_turn_base")}
    answers = {a["id"]: a for a in load_file(POSSIBLE_ANSWER_PATH / "BFCL_v4_multi_turn_base.json")}
    gt = {}
    for tid, entry in entries.items():
        params = {f["name"]: list(f["parameters"]["properties"]) for f in entry["function"]}
        desc = {f["name"]: {p: v.get("description", "") for p, v in f["parameters"]["properties"].items()}
                for f in entry["function"]}
        tool_desc = {f["name"]: f.get("description", "") for f in entry["function"]}
        turns = [[parse_gt_call(c, params) for c in turn] for turn in answers[tid]["ground_truth"]]
        gt[tid] = {"turns": turns, "param_desc": desc, "tool_desc": tool_desc,
                   "question": [m["content"] for turn in entry["question"] for m in turn if m["role"] == "user"]}
    return gt


def parse_gt_call(call_str: str, params: dict) -> tuple[str, dict]:
    """'mv(source='a', destination='b')' -> ('mv', {'source': 'a', 'destination': 'b'})"""
    node = ast.parse(call_str, mode="eval").body
    name = node.func.id
    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    for i, a in enumerate(node.args):  # positional args map onto the doc's parameter order
        args[params[name][i]] = ast.literal_eval(a)
    return name, args


def same_value(a, b) -> bool:
    if a == b:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return str(a).strip() == str(b).strip()


def same_args(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(same_value(a[k], b[k]) for k in a)


# ---- per-trajectory features ------------------------------------------------------

def extract_calls(doc: dict) -> list[dict]:
    """Flat, ordered list of the model's calls with their execution results."""
    result = doc["bfcl_result"]
    if not isinstance(result, list):
        return []
    log_turns = [t for t in doc["bfcl_metadata"].get("inference_log", []) if isinstance(t, dict)]
    calls = []
    for t, turn in enumerate(result):
        for s, step in enumerate(turn):
            if not isinstance(step, list):
                continue  # plain-text reply: ends the turn
            tool_msgs = [m for m in log_turns[t].get(f"step_{s}", []) if m.get("role") == "tool"]
            for i, call in enumerate(step):
                (name, raw), = call.items()
                try:
                    args = json.loads(raw)
                except (TypeError, ValueError):
                    args = None
                out = str(tool_msgs[i]["content"]) if i < len(tool_msgs) else ""
                calls.append({"turn": t, "step": s, "name": name, "args": args, "raw": raw,
                              "result": out, "error": out.startswith("Error during execution")})
    return calls


def first_prompt_tokens(doc: dict) -> int:
    """Prompt tokens of the task's first request: the tool schemas plus the first user turn."""
    calls = doc.get("api_calls") or []
    if not calls:
        return 0
    usage = ((calls[0].get("response") or {}).get("body") or {}).get("usage") or {}
    return usage.get("prompt_tokens") or 0


def analyze_trajectory(doc: dict, gt: dict) -> dict:
    calls = extract_calls(doc)
    ev = doc.get("evaluation") or {}
    err_type = (ev.get("error") or {}).get("error_type")
    if doc["error"]:
        err_type = "inference_error"
    gt_names = {name for turn in gt["turns"] for name, _ in turn}
    seen = set()
    repeated = 0
    for c in calls:
        key = (c["name"], json.dumps(c["args"], sort_keys=True))
        repeated += key in seen
        seen.add(key)

    # Ground-truth coverage: did the right tool get called, with the right arguments?
    gt_total = tool_hits = exact_hits = 0
    wrong_params = []  # (tool, param) whose value differed in the closest same-tool call
    first_divergence = None
    for t, turn in enumerate(gt["turns"]):
        prior = [c for c in calls if c["turn"] <= t and c["args"] is not None]
        for name, args in turn:
            gt_total += 1
            same_tool = [c for c in prior if c["name"] == name]
            if not same_tool:
                if first_divergence is None:
                    first_divergence = {"turn": t, "gt": (name, args), "kind": "tool_not_called", "closest": None}
                continue
            tool_hits += 1
            if any(same_args(c["args"], args) for c in same_tool):
                exact_hits += 1
                continue
            closest = max(same_tool, key=lambda c: sum(k in c["args"] and same_value(c["args"][k], v) for k, v in args.items()))
            diff = [k for k, v in args.items() if not (k in closest["args"] and same_value(closest["args"][k], v))]
            wrong_params += [(name, k) for k in diff]
            if first_divergence is None:
                first_divergence = {"turn": t, "gt": (name, args), "kind": "wrong_arguments",
                                    "closest": {"name": closest["name"], "args": closest["args"], "result": closest["result"][:200]},
                                    "params": diff}

    first = next((c for c in calls if c["turn"] == 0), None)
    gt0_names = {n for n, _ in gt["turns"][0]} if gt["turns"] else set()
    return {
        "valid": ev.get("valid"),
        "error_type": err_type,
        "n_calls": len(calls),
        "n_steps": doc["totals"]["api_calls"],
        "n_exec_errors": sum(c["error"] for c in calls),
        "n_repeated": repeated,
        "n_off_path": sum(1 for c in calls if c["name"] not in gt_names and not c["error"]),
        "n_on_path": sum(1 for c in calls if c["name"] in gt_names and not c["error"]),
        "n_distinct_tools": len({c["name"] for c in calls}),
        "first_tool_ok": first is not None and first["name"] in gt0_names,
        "first_call_exact": first is not None and first["args"] is not None
                            and any(first["name"] == n and same_args(first["args"], a) for n, a in gt["turns"][0]),
        "no_first_call": first is None,
        "gt_total": gt_total, "gt_tool_hits": tool_hits, "gt_exact_hits": exact_hits,
        "wrong_params": wrong_params,
        "first_divergence": first_divergence,
        "first_prompt_tokens": first_prompt_tokens(doc),
        "prompt_tokens": doc["totals"]["prompt_tokens"],
        "completion_tokens": doc["totals"]["completion_tokens"],
        "cost_usd": doc["totals"]["cost_usd"],
        "calls": calls,
    }


# ---- paired statistics ----------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant cells."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap_ci(base: list[bool], abl: list[bool], n=2000, seed=0):
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(abl, base)]
    m = len(diffs)
    samples = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return samples[int(0.025 * n)] * 100, samples[int(0.975 * n)] * 100


# ---- main -------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="main")
    args = p.parse_args()
    run_dir = PROJECT_DIR / "runs" / args.run_name
    os.environ["BFCL_PROJECT_ROOT"] = str(run_dir)

    gt = load_ground_truth()
    index = json.loads((run_dir / "index.json").read_text())
    index["trajectories"] = [r for r in index["trajectories"] if r.get("benchmark", "bfcl") == "bfcl"]
    models = list(dict.fromkeys(r["model"] for r in index["trajectories"]))
    for c in sorted({r["condition"] for r in index["trajectories"]} - set(CONDITIONS)):
        CONDITIONS.append(c)
        ABLATED.append(c)

    # feat[model][cond][task_id] -> features
    feat = defaultdict(lambda: defaultdict(dict))
    for i, row in enumerate(index["trajectories"]):
        doc = json.loads((run_dir / row["file"]).read_text())
        feat[row["model"]][row["condition"]][row["id"]] = analyze_trajectory(doc, gt[row["id"]])
        if i % 500 == 0:
            print(f"  {i}/{len(index['trajectories'])}")
    with_d0 = [m for m in models if feat[m]["D0"]]  # a repeated launch of one ablation (e.g. the Luna D2 rerun) has no D0 of its own
    task_ids = sorted(feat[with_d0[0]]["D0"], key=lambda t: int(t.rsplit("_", 1)[1]))
    # Not every model has every condition (D6/D7 were run for the default reasoning setting only).
    conds = {m: [c for c in CONDITIONS if feat[m][c]] for m in models}
    abl_conds = {m: [c for c in conds[m] if c != "D0"] for m in models}
    full = [m for m in models if all(c in conds[m] for c in SINGLE + ["D5"])]  # D0..D5 present

    def mean(m, c, key):
        xs = [feat[m][c][t][key] for t in task_ids]
        return sum(xs) / len(xs)

    out = {"run": index["run"], "models": models, "conditions": CONDITIONS, "n_tasks": len(task_ids),
           "cost_usd": sum(r["cost_usd"] for r in index["trajectories"])}

    # 1. success, heatmap, robustness
    out["success"] = {m: {c: mean(m, c, "valid") for c in conds[m]} for m in models}
    out["pp_change"] = {m: {c: (out["success"][m][c] - out["success"][m]["D0"]) * 100 for c in abl_conds[m]} for m in with_d0}
    out["robustness"] = {m: {
        "D1-D4": sum((out["success"][m]["D0"] - out["success"][m][c]) / out["success"][m]["D0"] for c in SINGLE) / 4,
        "D1-D5": sum((out["success"][m]["D0"] - out["success"][m][c]) / out["success"][m]["D0"] for c in SINGLE + ["D5"]) / 5,
    } for m in full}

    # paired stats vs D0
    out["paired"] = {}
    for m in with_d0:
        base = [bool(feat[m]["D0"][t]["valid"]) for t in task_ids]
        out["paired"][m] = {}
        for c in abl_conds[m]:
            abl = [bool(feat[m][c][t]["valid"]) for t in task_ids]
            both = sum(a and b for a, b in zip(abl, base)); d0_only = sum(b and not a for a, b in zip(abl, base))
            abl_only = sum(a and not b for a, b in zip(abl, base)); neither = len(task_ids) - both - d0_only - abl_only
            lo, hi = paired_bootstrap_ci(base, abl)
            out["paired"][m][c] = {"both": both, "d0_only": d0_only, "abl_only": abl_only, "neither": neither,
                                   "p_mcnemar": mcnemar_exact(d0_only, abl_only), "diff_pp": (sum(abl) - sum(base)) / len(task_ids) * 100,
                                   "ci95": [lo, hi]}

    # 2. behaviour metrics
    keys = ["n_calls", "n_steps", "n_exec_errors", "n_repeated", "n_off_path", "n_on_path", "n_distinct_tools",
            "prompt_tokens", "completion_tokens", "cost_usd"]
    out["behaviour"] = {m: {c: {k: mean(m, c, k) for k in keys} for c in conds[m]} for m in models}
    for m in models:
        for c in conds[m]:
            fs = [feat[m][c][t] for t in task_ids]
            b = out["behaviour"][m][c]
            b["first_tool_ok"] = sum(f["first_tool_ok"] for f in fs) / len(fs)
            b["first_call_exact"] = sum(f["first_call_exact"] for f in fs) / len(fs)
            b["no_first_call"] = sum(f["no_first_call"] for f in fs) / len(fs)
            b["tool_selection_acc"] = sum(f["gt_tool_hits"] for f in fs) / sum(f["gt_total"] for f in fs)
            b["argument_acc_given_tool"] = sum(f["gt_exact_hits"] for f in fs) / max(1, sum(f["gt_tool_hits"] for f in fs))
            b["gt_exact_coverage"] = sum(f["gt_exact_hits"] for f in fs) / sum(f["gt_total"] for f in fs)
            with_err = [f for f in fs if f["n_exec_errors"] > 0]
            b["first_prompt_tokens"] = sum(f["first_prompt_tokens"] for f in fs) / len(fs)
            b["tasks_with_exec_error"] = len(with_err) / len(fs)
            b["recovery_rate"] = (sum(bool(f["valid"]) for f in with_err) / len(with_err)) if with_err else None
            succ = [f for f in fs if f["valid"]]
            b["calls_per_success"] = (sum(f["n_calls"] for f in succ) / len(succ)) if succ else None
            b["divergence_kinds"] = dict(Counter((f["first_divergence"] or {}).get("kind", "none") for f in fs if not f["valid"]))
            b["error_types"] = dict(Counter(f["error_type"] or "none" for f in fs if not f["valid"]))

    # distributions for ECDF / scatter (all tasks, failures included)
    out["per_task"] = {m: {c: [{"id": t, "valid": bool(feat[m][c][t]["valid"]), "calls": feat[m][c][t]["n_calls"],
                                "prompt_tokens": feat[m][c][t]["prompt_tokens"], "completion_tokens": feat[m][c][t]["completion_tokens"]}
                               for t in task_ids] for c in conds[m]} for m in models}

    # 3. documentation-to-error linkage: which parameters go wrong more often than in D0
    out["wrong_params"] = {}
    for m in with_d0:
        d0 = Counter(wp for t in task_ids for wp in feat[m]["D0"][t]["wrong_params"])
        out["wrong_params"][m] = {}
        for c in abl_conds[m]:
            cnt = Counter(wp for t in task_ids for wp in feat[m][c][t]["wrong_params"])
            rows = sorted(((tool, param, n, d0[(tool, param)]) for (tool, param), n in cnt.items()),
                          key=lambda r: r[2] - r[3], reverse=True)[:8]
            out["wrong_params"][m][c] = [{"tool": tool, "param": param, "n": n, "n_d0": n0,
                                          "description": gt[task_ids[0]]["param_desc"].get(tool, {}).get(param)
                                                         or next((gt[t]["param_desc"][tool][param] for t in task_ids if tool in gt[t]["param_desc"]), "")}
                                         for tool, param, n, n0 in rows]

    # full per-model, per-condition counts of wrong parameters (so runs can be pooled later without a top-k cutoff)
    out["wrong_params_full"] = {m: {c: [[tool, param, n] for (tool, param), n in
                                        Counter(wp for t in task_ids for wp in feat[m][c][t]["wrong_params"]).items()]
                                    for c in conds[m]} for m in models}

    # same linkage pooled over all models (the top parameters are shared), D2 vs D0
    pooled = {c: Counter(wp for m in full for t in task_ids for wp in feat[m][c][t]["wrong_params"]) for c in ["D0", "D2"]}
    out["wrong_params_pooled"] = []
    for (tool, param), n in sorted(pooled["D2"].items(), key=lambda kv: kv[1] - pooled["D0"][kv[0]], reverse=True)[:10]:
        desc = next((gt[t]["param_desc"][tool][param] for t in task_ids if tool in gt[t]["param_desc"]), "")
        per_model = {m: sum(1 for t in task_ids for wp in feat[m]["D2"][t]["wrong_params"] if wp == (tool, param)) for m in full}
        out["wrong_params_pooled"].append({"tool": tool, "param": param, "n_d2": n, "n_d0": pooled["D0"][(tool, param)],
                                           "per_model": per_model, "description": desc})

    # 3b. task-level attribution of the D2 drop to the six parameters of the wrong-parameters table
    SIX = {("place_order", "order_type"), ("startEngine", "ignitionMode"), ("estimate_distance", "cityA"),
           ("estimate_distance", "cityB"), ("post_tweet", "tags"), ("post_tweet", "mentions")}
    six_tasks = [t for t in task_ids if any((n, k) in SIX for turn in gt[t]["turns"] for n, a in turn for k in a)]
    other_tasks = [t for t in task_ids if t not in six_tasks]
    out["six_param_tasks"] = {"n_six": len(six_tasks), "n_other": len(other_tasks), "per_model": {}}
    for m in full:
        d0_only = [t for t in task_ids if feat[m]["D0"][t]["valid"] and not feat[m]["D2"][t]["valid"]]
        row = {"d0_only": len(d0_only), "d0_only_six": sum(t in six_tasks for t in d0_only)}
        for c in conds[m]:
            row[f"{c}_six"] = sum(bool(feat[m][c][t]["valid"]) for t in six_tasks) / len(six_tasks)
            row[f"{c}_other"] = sum(bool(feat[m][c][t]["valid"]) for t in other_tasks) / len(other_tasks)
        out["six_param_tasks"]["per_model"][m] = row

    # 3c. what the tasks lost at D2 begin with: a value differing only in case, another wrong value, or a tool never called
    out["d2_loss_kind"] = {}
    for m in full:
        kinds = Counter()
        for t in task_ids:
            if not (feat[m]["D0"][t]["valid"] and not feat[m]["D2"][t]["valid"]):
                continue
            fd = feat[m]["D2"][t]["first_divergence"]
            if not fd:
                kinds["none"] += 1
            elif fd["kind"] != "wrong_arguments":
                kinds["tool_not_called"] += 1
            else:
                gt_args, got = fd["gt"][1], fd["closest"]["args"]
                case_only = all(k in got and str(got[k]).lower() == str(v).lower() for k, v in gt_args.items() if k in fd["params"])
                kinds["case_only" if case_only else "other_value"] += 1
        out["d2_loss_kind"][m] = dict(kinds)

    # 4. case studies: per model, the simplest task that passes in D0 and fails in D2, shown in D0 / D2 / D5
    out["cases"] = {}
    for m in full:
        cands = [t for t in task_ids if feat[m]["D0"][t]["valid"] and not feat[m]["D2"][t]["valid"]]
        if not cands:
            continue
        t = min(cands, key=lambda t: (sum(len(x) for x in gt[t]["turns"]), int(t.rsplit("_", 1)[1])))
        case = {"task": t, "question": gt[t]["question"],
                "ground_truth": [[f"{n}({', '.join(f'{k}={v!r}' for k, v in a.items())})" for n, a in turn] for turn in gt[t]["turns"]],
                "conditions": {}}
        for c in ["D0", "D2", "D5"]:
            f = feat[m][c][t]
            reg = f"{m}-{c}".replace("/", "_")
            case["conditions"][c] = {
                "file": f"runs/{args.run_name}/trajectories/{reg}/{t}.json",
                "valid": f["valid"], "error_type": f["error_type"], "n_calls": f["n_calls"], "n_steps": f["n_steps"],
                "calls": [{"turn": x["turn"], "name": x["name"], "args": x["args"] if x["args"] is not None else x["raw"],
                           "result": x["result"][:160], "error": x["error"]} for x in f["calls"]],
                "divergence": f["first_divergence"],
                "final_text": last_text_reply(run_dir / "trajectories" / reg / f"{t}.json"),
            }
            dv = f["first_divergence"]
            if dv and dv.get("params"):
                tool = dv["gt"][0]
                case["conditions"][c]["missing_descriptions"] = {p: gt[t]["param_desc"][tool].get(p, "") for p in dv["params"]}
        out["cases"][m] = case

    (run_dir / "analysis.json").write_text(json.dumps(out, indent=1))
    print("wrote", run_dir / "analysis.json")


def last_text_reply(path: Path) -> str:
    """The model's last plain-text reply in the trajectory (what it told the user)."""
    doc = json.loads(path.read_text())
    texts = [m["content"] for turn in doc["bfcl_metadata"].get("inference_log", []) if isinstance(turn, dict)
             for step in turn.values() if isinstance(step, list)
             for m in step if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
    return texts[-1][:400] if texts else ""


if __name__ == "__main__":
    main()
