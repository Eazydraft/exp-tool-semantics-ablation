"""Analysis of the tau2 (airline) run and the first-divergence split for tau2 and ToolSandbox.

    ./run.sh analyze  (python -m ablation.analyze_extra [--run-name main])  ->  runs/<run-name>/analysis_extra.json

tau2: success per model x condition, each ablation against D0 on the same tasks (discordant cells, paired
difference with a bootstrap 95% CI, exact McNemar p, and the equivalence bound: the largest change the
interval allows), where the reward is lost (DB check vs communicate check, how the simulations end, tool
calls per task), the earliest divergence of every failed task from the ground-truth actions (the tool
called with other arguments, never called, or every action matched), the failed tasks with a
state-changing call the ground truth does not contain, and what the run cost.

ToolSandbox: the earliest unmet milestone of every play below similarity 1, classified by whether the
tool that milestone needs was called at all (construction failure) or never (selection failure).
"""

import argparse
import collections
import json
import random
from math import comb
from pathlib import Path

from ablation.audit import TAU2_WRITE_TOOLS, TS_ROW_TO_CALL, same_args

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONDITIONS = ["D0", "D1", "D2", "D3", "D4", "D5"]
SHORT = {"openai/gpt-6-luna": "GPT-6 Luna", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B", "google/gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}


# ---- paired statistics (as in paper/tables.py) ----

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(valid_a, valid_b, seed=0):
    """Compare condition b against a on the same tasks: discordant cells, pp difference, bootstrap CI, p."""
    both = sum(x and y for x, y in zip(valid_a, valid_b)); a_only = sum(x and not y for x, y in zip(valid_a, valid_b))
    b_only = sum(y and not x for x, y in zip(valid_a, valid_b)); neither = len(valid_a) - both - a_only - b_only
    diffs = [int(y) - int(x) for x, y in zip(valid_a, valid_b)]
    rng = random.Random(seed); n = len(diffs)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(2000))
    return dict(both=both, a_only=a_only, b_only=b_only, neither=neither, diff=sum(diffs) / n * 100,
                ci=(samples[50], samples[1949]), p=mcnemar_exact(a_only, b_only),
                bound=max(abs(samples[50]), abs(samples[1949])))  # the change is smaller than this, at 95%


# ---- tables ----

def by_combo(rows):
    out = collections.defaultdict(dict)
    for r in rows:
        out[(r["model"], r["condition"])][r["id"]] = r
    return out


def success_table(combos, models):
    return {m: {c: (sum(r["pass"] for r in combos[(m, c)].values()) / len(combos[(m, c)]) if (m, c) in combos else None)
                for c in CONDITIONS} for m in models}


def paired_table(combos, models):
    out = {}
    for m in models:
        base = combos.get((m, "D0"), {})
        for c in CONDITIONS[1:]:
            abl = combos.get((m, c), {})
            ids = sorted(set(base) & set(abl))
            if ids:
                out[(m, c)] = {"n": len(ids), **paired([base[i]["pass"] for i in ids], [abl[i]["pass"] for i in ids])}
    return out


def fmt_pct(x):
    return "   --" if x is None else f"{x * 100:5.1f}"


def print_success(title, table, models):
    print(f"\n{title}")
    print(f"{'Model':22s}" + "".join(f"{c:>8s}" for c in CONDITIONS))
    for m in models:
        print(f"{SHORT.get(m, m):22s}" + "".join(f"{fmt_pct(table[m][c]):>8s}" for c in CONDITIONS))


def print_paired(table, models):
    print(f"\n{'Model':22s} cond   n   both  D0-only  abl-only  neither   diff(pp)   95% CI          McNemar p")
    for m in models:
        for c in CONDITIONS[1:]:
            t = table.get((m, c))
            if t:
                star = " *" if t["p"] < 0.05 else ""
                print(f"{SHORT.get(m, m):22s} {c}  {t['n']:3d}   {t['both']:3d}   {t['a_only']:4d}     {t['b_only']:4d}     {t['neither']:4d}    "
                      f"{t['diff']:+6.1f}   [{t['ci'][0]:+5.1f}, {t['ci'][1]:+5.1f}]   {t['p']:.4f}{star}   |change| < {t['bound']:.1f}")


def cost_lines(rows):
    agent = sum(r["cost_usd"] + r.get("user_cost_usd", 0) for r in rows)
    judge = sum(r.get("judge_cost_usd", 0) for r in rows)
    return {"trajectories": len(rows), "agent_and_user_usd": round(agent, 2), "judges_usd": round(judge, 2)}


def tau2_calls(doc):
    return [(tc["name"], tc["arguments"]) for m in doc["tau2"]["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []]


def tau2_divergence(actions, calls):
    """Earliest ground-truth action without an exact hit, as in the BFCL analysis."""
    for a in actions:
        same = [c for c in calls if c[0] == a["name"]]
        if not same:
            return "tool_not_called"
        if not any(same_args(c[1], a.get("arguments") or {}) for c in same):
            return "wrong_arguments"
    return "none"


def tau2_off_path_write(actions, calls):
    """A state-changing call the ground truth does not contain (same tool and arguments)."""
    return any(n in TAU2_WRITE_TOOLS and not any(a["name"] == n and same_args(a.get("arguments") or {}, args) for a in actions)
               for n, args in calls)


def analyse_tau2(rows, out):
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    combos = by_combo(rows)
    succ = success_table(combos, models)
    print_success(f"tau2-bench airline: task reward = 1 (%), {len(next(iter(combos.values())))} tasks per cell", succ, models)
    pt = paired_table(combos, models)
    print_paired(pt, models)
    # where the reward is lost: DB check vs communicate check, and how runs end
    print(f"\n{'Model':22s} cond   DB fail   comm fail   both   term: max_steps  max_errors  agent/user stop   mean tool calls")
    breakdown = {}
    for m in models:
        for c in CONDITIONS:
            rs = list(combos.get((m, c), {}).values())
            if not rs:
                continue
            db = sum(1 for r in rs if not (r["reward_info"]["db_check"] or {}).get("db_match", True))
            comm = sum(1 for r in rs if any(not x["met"] for x in (r["reward_info"].get("communicate_checks") or [])))
            both = sum(1 for r in rs if not (r["reward_info"]["db_check"] or {}).get("db_match", True)
                       and any(not x["met"] for x in (r["reward_info"].get("communicate_checks") or [])))
            term = collections.Counter(r["termination_reason"].split(".")[-1].lower() for r in rs)
            calls = sum(r["num_tool_calls"] for r in rs) / len(rs)
            failed = [r for r in rs if not r["pass"]]
            div = collections.Counter(r["divergence"] for r in failed)
            writes = sum(r["off_path_write"] for r in failed)
            breakdown[f"{m} {c}"] = {"db_fail": db, "comm_fail": comm, "both": both, "termination": dict(term), "mean_tool_calls": calls,
                                     "divergence_kinds": dict(div), "failed_with_off_path_write": writes, "failed": len(failed)}
            stop = sum(v for k, v in term.items() if "stop" in k)
            print(f"{SHORT.get(m, m):22s} {c}   {db:5d}     {comm:5d}      {both:4d}         {term.get('max_steps', 0):4d}        {term.get('max_errors', 0):4d}          {stop:4d}             {calls:5.1f}")
    print(f"\nEarliest divergence of failed tasks: wrong arguments / tool never called / every action matched; failed tasks with an off-path state-changing call")
    for m in models:
        line = []
        for c in CONDITIONS:
            b = breakdown.get(f"{m} {c}")
            if b:
                d = b["divergence_kinds"]
                line.append(f"{c}: {d.get('wrong_arguments', 0):2d}/{d.get('tool_not_called', 0):2d}/{d.get('none', 0):2d} w{b['failed_with_off_path_write']:2d}")
        print(f"  {SHORT.get(m, m):22s} " + "   ".join(line))
    first_tokens = {f"{m} {c}": sum(r.get("first_prompt_tokens", 0) for r in combos[(m, c)].values()) / len(combos[(m, c)])
                    for m in models for c in CONDITIONS if (m, c) in combos}
    out["tau2"] = {"success": succ, "paired": {f"{m} {c}": v for (m, c), v in pt.items()}, "breakdown": breakdown, "cost": cost_lines(rows),
                   "first_prompt_tokens": first_tokens}
    print(f"\ncost: {out['tau2']['cost']}")


def analyse_toolsandbox_divergence(rows, run_dir, out):
    """Per play below similarity 1: the earliest unmet milestone and whether its tool was called at all."""
    from ablation.toolsandbox_gt import load as load_ts
    gt = load_ts()
    models = [m for m in SHORT if any(r["model"] == m for r in rows)]
    combos = by_combo(rows)
    result = {}
    print(f"\nToolSandbox, plays below similarity 1: earliest unmet milestone that needs a tool: that tool was called (W.A.) / never called (N.C.) / only a reply milestone unmet (tools ok) / play aborted")
    for m in models:
        line = []
        for c in CONDITIONS:
            kinds = collections.Counter()
            for sid, r in combos.get((m, c), {}).items():
                if r["similarity"] == 1.0:
                    continue
                doc = json.loads((run_dir / r["file"]).read_text())
                if doc["error"]:
                    kinds["aborted"] += 1  # ToolSandbox raised (malformed arguments); scored 0 like its own runner
                    continue
                mapping = doc["result"].get("milestone_mapping") or {}
                cons = gt[sid]["constraints"]
                n_milestones = max((x["milestone"] for x in cons), default=-1) + 1
                unmet = [i for i in range(n_milestones) if not mapping or float(mapping.get(str(i), [0, 0.0])[1]) < 1.0]
                first_tool_milestone = None
                for i in unmet:
                    needed = set()
                    for x in cons:
                        if x["milestone"] != i:
                            continue
                        needed |= {call["tool"] for call in x["calls"]}
                        if x["namespace"] == "SETTING":
                            needed |= {TS_ROW_TO_CALL[("SETTING", "snapshot_similarity")][1][col].split(".")[0] for col in x["row"]}
                        elif x["namespace"] in ("CONTACT", "REMINDER", "MESSAGING"):
                            tool = TS_ROW_TO_CALL.get((x["namespace"], x["measure"]), (None,))[0]
                            if tool:
                                needed.add(tool)
                    if needed:
                        first_tool_milestone = needed
                        break
                if first_tool_milestone is None:
                    kinds["tools_ok"] += 1
                    continue
                called = {tc["function"]["name"] for msg in doc["toolsandbox"]["conversation"] for tc in msg.get("tool_calls") or []}
                kinds["wrong_arguments" if first_tool_milestone & called else "tool_not_called"] += 1
            result[f"{m} {c}"] = dict(kinds)
            line.append(f"{c}: {kinds['wrong_arguments']:2d}/{kinds['tool_not_called']:2d}/{kinds['tools_ok']:2d}/{kinds['aborted']:2d}")
        print(f"  {SHORT.get(m, m):22s} " + "   ".join(line))
    out["toolsandbox_divergence"] = result


def sign_flip_p(diffs, n_perm=10000, seed=0):
    """Two-sided permutation p for a paired mean difference: flip the sign of each task's difference at random."""
    rng = random.Random(seed)
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def analyse_tau2_trials(rows, out):
    """Conditions run for several trials: trial-averaged success per task, D2 against D0 on the per-task
    means (bootstrap CI over tasks, equivalence bound, sign-flip p), and how much two D0 trials differ."""
    by = collections.defaultdict(lambda: collections.defaultdict(dict))  # (model, cond) -> trial -> id -> pass
    for r in rows:
        by[(r["model"], r["condition"])][r.get("trial", 1)][str(r["id"])] = bool(r["pass"])
    result = {}
    for m in sorted({m for m, _ in by}):
        trials = {c: sorted(by[(m, c)]) for c in CONDITIONS if (m, c) in by}
        multi = {c: t for c, t in trials.items() if len(t) > 1}
        if not multi:
            continue
        ids = sorted(set.intersection(*(set(by[(m, c)][t]) for c in multi for t in multi[c])), key=int)
        mean = {c: {i: sum(by[(m, c)][t][i] for t in multi[c]) / len(multi[c]) for i in ids} for c in multi}
        entry = {"n_tasks": len(ids), "n_trials": {c: len(t) for c, t in multi.items()},
                 "success": {c: sum(mean[c].values()) / len(ids) for c in multi},
                 "per_trial_success": {c: [sum(by[(m, c)][t][i] for i in ids) / len(ids) for t in multi[c]] for c in multi}}
        if "D0" in multi and "D2" in multi:
            diffs = [mean["D2"][i] - mean["D0"][i] for i in ids]
            rng = random.Random(0); n = len(diffs)
            samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(2000))
            entry["d2_vs_d0"] = {"diff": sum(diffs) / n * 100, "ci": [samples[50], samples[1949]],
                                 "bound": max(abs(samples[50]), abs(samples[1949])), "p_signflip": sign_flip_p(diffs)}
        if "D0" in multi:
            pairs = []
            t0 = multi["D0"]
            for a in range(len(t0)):
                for b in range(a + 1, len(t0)):
                    va, vb = [by[(m, "D0")][t0[a]][i] for i in ids], [by[(m, "D0")][t0[b]][i] for i in ids]
                    pairs.append({"trials": [t0[a], t0[b]], **paired(va, vb)})
            entry["d0_trial_pairs"] = pairs
        result[m] = entry
    out["tau2_trials"] = result
    for m, e in result.items():
        line = f"{SHORT.get(m, m):22s} trials {e['n_trials']}  " + "  ".join(f"{c} {e['success'][c]*100:5.1f}" for c in e["success"])
        if "d2_vs_d0" in e:
            d = e["d2_vs_d0"]; line += f"   D2-D0 {d['diff']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] bound {d['bound']:.1f} p {d['p_signflip']:.3f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="main")
    args = p.parse_args()
    run_dir = PROJECT_DIR / "runs" / args.run_name
    index = json.loads((run_dir / "index.json").read_text())
    rows = index["trajectories"]
    out = {}
    all_tau2 = [r for r in rows if r["benchmark"] == "tau2"]
    tau2_rows = [r for r in all_tau2 if r.get("trial", 1) == 1]  # the single-trial study; further trials are analysed below
    tasks = {str(t["id"]): t["evaluation_criteria"].get("actions") or [] for t in
             json.loads((PROJECT_DIR / "tau2-bench" / "data" / "tau2" / "domains" / "airline" / "tasks.json").read_text())}
    # reward_info and the calls live in the trajectory files
    for r in tau2_rows:
        doc = json.loads((run_dir / r["file"]).read_text())
        r["reward_info"] = doc["result"]["reward_info"] or {}
        calls = tau2_calls(doc)
        r["divergence"] = tau2_divergence(tasks[str(r["id"])], calls)
        # the agent's first request carries the tool schemas; user-simulator calls come first in the log
        r["first_prompt_tokens"] = next((((c.get("response") or {}).get("usage") or {}).get("prompt_tokens", 0)
                                         for c in doc.get("api_calls", []) if (c.get("request") or {}).get("model") != doc.get("user_model")), 0)
        r["off_path_write"] = tau2_off_path_write(tasks[str(r["id"])], calls)
    analyse_tau2(tau2_rows, out)
    analyse_tau2_trials(all_tau2, out)
    ts_rows = [r for r in rows if r["benchmark"] == "toolsandbox"]
    if ts_rows:
        analyse_toolsandbox_divergence(ts_rows, run_dir, out)
    (run_dir / "analysis_extra.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {run_dir / 'analysis_extra.json'}")


if __name__ == "__main__":
    main()
