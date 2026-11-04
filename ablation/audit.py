"""Information audit: where is the information a ground-truth argument needs stated?

    ./run.sh analyze  (uv run python -m ablation.audit [--run-name main])     ->  runs/<run-name>/audit.json

For every argument of every ground-truth call on BFCL (200 tasks), ToolSandbox (91 scenarios) and
tau2-bench airline (50 tasks), the audit records

  (1) whether the parameter's description states a constraint on the value: an admissible-value list
      (`enum`), a format, unit or range (`format`), or an entity type the parameter name does not give
      (`entity`). An example of a single value ("such as 'ZFA04Y'") and a restatement of the name are
      not constraints. The judgement is per (tool, parameter) and is written out in CONSTRAINT_KIND
      below, so it can be checked against the schema text;
  (2) whether a structural field states a constraint on the same parameter: an `enum`, a boolean type
      (an admissible set of two values) or a `pattern`;
  (3) whether the ground-truth value occurs verbatim in what the model had seen by then: the user's
      messages up to that turn, or the output of an earlier call. Outputs are taken from a trajectory
      that passed the task under D0 (the first of the configurations that did), so they are outputs
      the model actually received; the call is located in that trajectory and only outputs before it
      count. Booleans are never "copied": the type is the constraint. ToolSandbox's relative-date
      reminders are the one exception: their expected timestamp is computed from the clock when the
      scenario is evaluated, so the audit checks the timestamp the model passed in that trajectory.

Each argument then falls in one class: constraint stated in the description only; in the description
and a structural field; in a structural field only; no constraint stated and the value copied from
context; no constraint stated and the value not copied (free text, a name the user chose, a computed
number). The audit also counts, per tool in the ground truth, how many carriers state its purpose:
its name, its description, and its signature when no other tool offered alongside it has the same
parameter names and types.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS = ["bfcl", "toolsandbox", "tau2"]
CLASSES = ["desc_only", "desc_and_structure", "structure_only", "copied", "constructed"]

# ---- (1) what each parameter description states, per benchmark ---------------------------------------
# Listed: every ground-truth parameter whose description states a constraint. Every other ground-truth
# parameter's description restates the name, gives an example or says where the value comes from.
CONSTRAINT_KIND = {
    "bfcl": {
        # admissible-value lists
        "activateParkingBrake.mode": "enum", "authenticate_travel.grant_type": "enum",
        "compute_exchange_rate.base_currency": "enum", "compute_exchange_rate.target_currency": "enum",
        "displayCarStatus.option": "enum", "get_flight_cost.travel_class": "enum",
        "get_nearest_airport_by_city.location": "enum", "lockDoors.door": "enum", "place_order.order_type": "enum",
        "setHeadlights.mode": "enum", "startEngine.ignitionMode": "enum", "wc.mode": "enum",
        # formats, units, ranges and "no path" restrictions on a name
        "book_flight.travel_date": "format", "cat.file_name": "format", "cd.folder": "format", "cp.destination": "format",
        "create_ticket.priority": "format", "estimate_drive_feasibility_by_mileage.distance": "format",
        "fillFuelTank.fuelAmount": "format", "get_flight_cost.travel_date": "format", "get_flight_cost.travel_from": "format",
        "get_flight_cost.travel_to": "format", "grep.file_name": "format", "mkdir.dir_name": "format",
        "mv.destination": "format", "mv.source": "format", "post_tweet.mentions": "format", "post_tweet.tags": "format",
        "pressBrakePedal.pedalPosition": "format", "register_credit_card.expiration_date": "format", "rmdir.dir_name": "format",
        "setCruiseControl.distanceToNextVehicle": "format", "setCruiseControl.speed": "format",
        "set_budget_limit.budget_limit": "format", "set_navigation.destination": "format", "tail.file_name": "format",
        "touch.file_name": "format", "verify_traveler_information.date_of_birth": "format",
        # entity types the name hides
        "add_to_watchlist.stock": "entity", "estimate_distance.cityA": "entity", "estimate_distance.cityB": "entity",
    },
    "toolsandbox": {
        "add_reminder.reminder_timestamp": "format", "modify_reminder.reminder_timestamp": "format",
        "datetime_info_to_timestamp.hour": "format", "shift_timestamp.timestamp": "format",
        "timestamp_to_datetime_info.timestamp": "format",
        "search_reminder.creation_timestamp_lowerbound": "format", "search_reminder.creation_timestamp_upperbound": "format",
        "search_reminder.reminder_timestamp_lowerbound": "format", "search_reminder.reminder_timestamp_upperbound": "format",
        "search_messages.creation_timestamp_lowerbound": "format", "search_messages.creation_timestamp_upperbound": "format",
    },
    "tau2": {
        "book_reservation.cabin": "enum", "book_reservation.flight_type": "enum",
        "book_reservation.origin": "format", "book_reservation.destination": "format",
        "search_direct_flight.origin": "format", "search_direct_flight.destination": "format",
        "search_direct_flight.date": "format", "calculate.expression": "format",
    },
}

# ToolSandbox pins most of its ground truth as a database state; this maps a pinned row to the call
# that produces it (namespace, similarity measure) -> (tool, {column: parameter}).
TS_ROW_TO_CALL = {
    ("CONTACT", "addition_similarity"): ("add_contact", {"name": "name", "phone_number": "phone_number", "relationship": "relationship", "is_self": "is_self"}),
    ("CONTACT", "update_similarity"): ("modify_contact", {"person_id": "person_id", "name": "name", "phone_number": "phone_number", "relationship": "relationship", "is_self": "is_self"}),
    ("CONTACT", "removal_similarity"): ("remove_contact", {"person_id": "person_id"}),
    ("REMINDER", "addition_similarity"): ("add_reminder", {"content": "content", "reminder_timestamp": "reminder_timestamp", "latitude": "latitude", "longitude": "longitude"}),
    ("REMINDER", "update_similarity"): ("modify_reminder", {"reminder_id": "reminder_id", "content": "content", "reminder_timestamp": "reminder_timestamp"}),
    ("REMINDER", "removal_similarity"): ("remove_reminder", {"reminder_id": "reminder_id"}),
    ("MESSAGING", "addition_similarity"): ("send_message_with_phone_number", {"recipient_phone_number": "phone_number", "content": "content"}),
    ("SETTING", "snapshot_similarity"): (None, {"cellular": "set_cellular_service_status.on", "wifi": "set_wifi_status.on",
                                                "location_service": "set_location_service_status.on", "low_battery_mode": "set_low_battery_mode_status.on"}),
}
TAU2_WRITE_TOOLS = {"book_reservation", "cancel_reservation", "update_reservation_baggages", "update_reservation_flights",
                    "update_reservation_passengers", "send_certificate"}


# ---- helpers ---------------------------------------------------------------------------------------------

def leaves(v):
    """Scalar leaves of a value (lists and dicts are matched leaf by leaf)."""
    if isinstance(v, dict):
        return [x for u in v.values() for x in leaves(u)]
    if isinstance(v, list):
        return [x for u in v for x in leaves(u)] or [""]
    return [v]


def in_text(v, text: str) -> bool:
    """Does the scalar value occur verbatim in the text? Numbers and short strings need word boundaries."""
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        forms = {str(v)}
        if float(v).is_integer():
            forms |= {str(int(v)), f"{int(v)}.0"}
        return any(re.search(rf"(?<![\w.]){re.escape(f)}(?![\w.])", text) for f in forms)
    s = str(v)
    if not s.strip():
        return False
    if len(s) < 4:
        return re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text) is not None
    return s in text


def copied(value, text: str) -> bool:
    ls = leaves(value)
    return bool(ls) and all(in_text(x, text) for x in ls)


def same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    return str(a).strip() == str(b).strip()


def same_args(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(same_value(a[k], b[k]) for k in a)


def structure_of(spec: dict) -> str | None:
    """Which structural field constrains the value: enum, pattern, or a boolean type."""
    if "enum" in spec or "enum" in (spec.get("items") or {}):
        return "enum"
    if "pattern" in spec or "pattern" in (spec.get("items") or {}):
        return "pattern"
    if spec.get("type") == "boolean":
        return "type"
    return None


def classify(benchmark: str, tool: str, param: str, spec: dict, value, context: str) -> dict:
    kind = CONSTRAINT_KIND[benchmark].get(f"{tool}.{param}")
    structure = structure_of(spec)
    is_copied = copied(value, context)
    if kind and structure:
        cls = "desc_and_structure"
    elif kind:
        cls = "desc_only"
    elif structure:
        cls = "structure_only"
    elif is_copied:
        cls = "copied"
    else:
        cls = "constructed"
    return {"tool": tool, "param": param, "kind": kind, "structure": structure, "copied": is_copied, "class": cls}


def signature(fn: dict) -> tuple:
    props = (fn.get("parameters") or {}).get("properties") or {}
    return tuple(sorted((p, str(s.get("type"))) for p, s in props.items()))


def purpose_carriers(fn: dict, toolset: list[dict]) -> dict:
    """How many carriers state what this tool does: its name, its description, and a signature no
    other tool in the set shares."""
    sig = signature(fn)
    shared = [g["name"] for g in toolset if g["name"] != fn["name"] and signature(g) == sig]
    return {"name": True, "description": bool((fn.get("description") or "").strip()), "signature_unique": not shared,
            "shares_signature_with": shared}


# ---- the three benchmarks ----------------------------------------------------------------------------------

def read_doc(run_dir: Path, model_dir: str, task: str):
    p = run_dir / "trajectories" / model_dir / f"{task}.json"
    return json.loads(p.read_text()) if p.exists() else None


def audit_bfcl(run_dir: Path, index: list) -> dict:
    sys.path.insert(0, str(PROJECT_DIR))
    os.environ["BFCL_PROJECT_ROOT"] = str(run_dir)
    from bfcl_eval.utils import load_dataset_entry
    from ablation.analyze import extract_calls, load_ground_truth

    gt = load_ground_truth()
    entries = {e["id"]: e for e in load_dataset_entry("multi_turn_base")}
    passing = defaultdict(list)  # task -> model dirs whose D0 run passed
    for r in index:
        if r["benchmark"] == "bfcl" and r["condition"] == "D0" and r["valid"]:
            passing[r["id"]].append(r["file"].split("/")[1])
    args_out, tools_out, unmatched = [], {}, 0
    for tid, g in gt.items():
        fns = {f["name"]: f for f in entries[tid]["function"]}
        # outputs from the first configuration that passed the task in D0 (none: the first D0 run on disk)
        dirs = passing.get(tid) or ["openai_gpt-6-luna-D0"]
        doc = read_doc(run_dir, dirs[0], tid)
        calls = extract_calls(doc) if doc else []
        used = set()
        for t, turn in enumerate(g["turns"]):
            user_text = "\n".join(g["question"][:t + 1])
            for name, args in turn:
                match = next((i for i, c in enumerate(calls) if i not in used and c["name"] == name and c["args"] is not None
                              and same_args(c["args"], args)), None)
                if match is None:
                    unmatched += 1
                    outputs = [c["result"] for c in calls if c["turn"] <= t]
                else:
                    used.add(match)
                    outputs = [c["result"] for c in calls[:match]]
                context = user_text + "\n" + "\n".join(outputs)
                for p, v in args.items():
                    spec = fns[name]["parameters"]["properties"].get(p, {})
                    args_out.append({"task": tid, "turn": t, **classify("bfcl", name, p, spec, v, context)})
                if (tid, name) not in tools_out:
                    tools_out[(tid, name)] = purpose_carriers(fns[name], entries[tid]["function"])
    return {"arguments": args_out, "tools": [{"task": t, "tool": n, **v} for (t, n), v in tools_out.items()],
            "unmatched_calls": unmatched, "n_calls": sum(len(turn) for g in gt.values() for turn in g["turns"])}


def audit_toolsandbox(run_dir: Path, index: list) -> dict:
    from ablation.toolsandbox_gt import load as load_ts
    gt = load_ts()
    passing = defaultdict(list)
    for r in index:
        if r["benchmark"] == "toolsandbox" and r["condition"] == "D0" and r["similarity"] == 1.0:
            passing[r["id"]].append(r["file"].split("/")[1])
    # the 28 tools every scenario is shown, as sent in D0
    any_doc = next(read_doc(run_dir, r["file"].split("/")[1], r["id"]) for r in index if r["benchmark"] == "toolsandbox" and r["condition"] == "D0")
    toolset = [t["function"] for t in any_doc["tools_sent"]]
    fns = {f["name"]: f for f in toolset}
    args_out, tools_out, unmatched, dependant, n_pinned, clock = [], {}, 0, 0, 0, 0
    for sid, s in gt.items():
        dirs = passing.get(sid) or ["openai_gpt-6-luna-D0"]
        doc = read_doc(run_dir, dirs[0], sid)
        conv = doc["toolsandbox"]["conversation"] if doc else []
        calls = []  # (name, args, index in conversation)
        for i, m in enumerate(conv):
            for tc in m.get("tool_calls") or []:
                try:
                    a = json.loads(tc["function"]["arguments"])
                except (TypeError, ValueError):
                    a = {}
                calls.append((tc["function"]["name"], a, i))
        user_msgs = [m["content"] for m in conv if m["role"] == "user" and m.get("content")]
        outputs = [(i, str(m.get("content"))) for i, m in enumerate(conv) if m["role"] == "tool"]
        base_text = "\n".join(s["user_text"] + user_msgs)
        # every pinned value, with the call that produces it
        pinned = []  # (tool, {param: value}, dependant)
        for c in s["constraints"]:
            if c["calls"]:
                for call in c["calls"]:
                    pinned.append((call["tool"], call["arguments"], c["dependant"]))
            elif c["namespace"] in ("CONTACT", "REMINDER", "MESSAGING", "SETTING"):
                tool, cols = TS_ROW_TO_CALL.get((c["namespace"], c["measure"]), (None, {}))
                if c["namespace"] == "SETTING":
                    for col, v in c["row"].items():
                        t, p = cols[col].split(".")
                        pinned.append((t, {p: v}, c["dependant"]))
                elif tool:
                    pinned.append((tool, {cols[k]: v for k, v in c["row"].items() if k in cols}, c["dependant"]))
        n_pinned += len(pinned)
        for tool, args, dep in pinned:
            # locate the producing call: same tool, every pinned argument matched
            match = next((i for n, a, i in calls if n == tool and all(k in a and same_value(a[k], v) for k, v in args.items())), None)
            if match is None and any(k.endswith("timestamp") for k in args):
                # Relative-date reminders ("tomorrow 5PM"): ToolSandbox computes the expected timestamp from the
                # clock at evaluation time, so the dumped value cannot equal the one this run saw. Locate the call by
                # its other arguments and audit the timestamp the model actually passed.
                found = next(((a, i) for n, a, i in calls if n == tool
                              and all(k in a and (k.endswith("timestamp") or same_value(a[k], v)) for k, v in args.items())), None)
                if found:
                    a_model, match = found
                    args = {k: (a_model[k] if k.endswith("timestamp") else v) for k, v in args.items()}
                    clock += 1
            if match is None:
                unmatched += 1
                context = base_text + "\n" + "\n".join(o for _, o in outputs)
            else:
                context = base_text + "\n" + "\n".join(o for i, o in outputs if i < match)
            dependant += dep
            for p, v in args.items():
                spec = fns[tool]["parameters"]["properties"].get(p, {})
                row = classify("toolsandbox", tool, p, spec, v, context)
                if dep and row["class"] in ("copied", "constructed"):
                    row["class"] = "copied"  # the milestone itself fills the value in from an earlier call's result
                    row["copied"] = True
                args_out.append({"task": sid, **row})
            if (sid, tool) not in tools_out:
                tools_out[(sid, tool)] = purpose_carriers(fns[tool], toolset)
    return {"arguments": args_out, "tools": [{"task": t, "tool": n, **v} for (t, n), v in tools_out.items()],
            "unmatched_calls": unmatched, "dependant_values": dependant, "clock_relative_values": clock, "n_calls": n_pinned}


def audit_tau2(run_dir: Path, index: list) -> dict:
    tasks = json.loads((PROJECT_DIR / "tau2-bench" / "data" / "tau2" / "domains" / "airline" / "tasks.json").read_text())
    passing = defaultdict(list)
    for r in index:
        if r["benchmark"] == "tau2" and r["condition"] == "D0" and r["pass"]:
            passing[str(r["id"])].append(r["file"].split("/")[1])
    any_doc = next(read_doc(run_dir, r["file"].split("/")[1], r["id"]) for r in index if r["benchmark"] == "tau2" and r["condition"] == "D0")
    toolset = [t["function"] for t in any_doc["tools_sent"]]
    fns = {f["name"]: f for f in toolset}
    args_out, tools_out, unmatched = [], {}, 0
    for task in tasks:
        tid = str(task["id"])
        dirs = passing.get(tid) or ["openai_gpt-6-luna-D0"]
        doc = read_doc(run_dir, dirs[0], tid)
        msgs = doc["tau2"]["messages"] if doc else []
        instr = task["user_scenario"]["instructions"]
        scenario_text = "\n".join(str(instr.get(k) or "") for k in ("task_instructions", "reason_for_call", "known_info", "unknown_info"))
        calls = [(tc["name"], tc["arguments"], i) for i, m in enumerate(msgs) if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
        for action in task["evaluation_criteria"].get("actions") or []:
            name, args = action["name"], action.get("arguments") or {}
            match = next((i for n, a, i in calls if n == name and same_args(a, args)), None)
            if match is None:
                unmatched += 1
                seen = msgs
            else:
                seen = msgs[:match]
            context = scenario_text + "\n" + "\n".join(str(m.get("content") or "") for m in seen if m["role"] in ("user", "tool"))
            for p, v in args.items():
                spec = fns[name]["parameters"]["properties"].get(p, {})
                args_out.append({"task": tid, **classify("tau2", name, p, spec, v, context)})
            if (tid, name) not in tools_out:
                tools_out[(tid, name)] = purpose_carriers(fns[name], toolset)
    return {"arguments": args_out, "tools": [{"task": t, "tool": n, **v} for (t, n), v in tools_out.items()],
            "unmatched_calls": unmatched, "n_calls": sum(len(t["evaluation_criteria"].get("actions") or []) for t in tasks)}


# ---- summary -------------------------------------------------------------------------------------------------

def summarise(a: dict) -> dict:
    rows = a["arguments"]
    by_class = Counter(r["class"] for r in rows)
    kinds = Counter(r["kind"] for r in rows if r["kind"])
    params = {(r["tool"], r["param"]) for r in rows}
    desc_only_params = sorted({(r["tool"], r["param"]) for r in rows if r["class"] == "desc_only"})
    copied_in_desc = sum(1 for r in rows if r["class"] in ("desc_only", "desc_and_structure") and r["copied"])
    tools = a["tools"]
    carriers = Counter(sum((t["name"], t["description"], t["signature_unique"])) for t in tools)
    shared = Counter(t["tool"] for t in tools if not t["signature_unique"])
    return {"n_arguments": len(rows), "n_calls": a["n_calls"], "n_parameters": len(params), "classes": {c: by_class.get(c, 0) for c in CLASSES},
            "constraint_kinds": dict(kinds), "desc_only_parameters": [f"{t}.{p}" for t, p in desc_only_params],
            "desc_constraint_and_copied": copied_in_desc, "unmatched_calls": a["unmatched_calls"],
            "tool_instances": len(tools), "tools_by_carriers": {str(k): v for k, v in sorted(carriers.items())},
            "tools_sharing_signature": dict(shared.most_common()), "calls_on_shared_signature_tools": sum(not t["signature_unique"] for t in tools)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="main")
    args = p.parse_args()
    run_dir = PROJECT_DIR / "runs" / args.run_name
    index = json.loads((run_dir / "index.json").read_text())["trajectories"]
    out = {}
    for name, fn in [("bfcl", audit_bfcl), ("toolsandbox", audit_toolsandbox), ("tau2", audit_tau2)]:
        a = fn(run_dir, index)
        out[name] = {"summary": summarise(a), **a}
        s = out[name]["summary"]
        print(f"\n{name}: {s['n_calls']} ground-truth calls, {s['n_arguments']} arguments over {s['n_parameters']} parameters "
              f"({s['unmatched_calls']} calls not located in a passing D0 trajectory)")
        for c in CLASSES:
            print(f"  {c:20s} {s['classes'][c]:5d}  ({s['classes'][c] / s['n_arguments'] * 100:.1f}%)")
        print(f"  description constraints by kind: {s['constraint_kinds']};  of those arguments, value also verbatim in context: {s['desc_constraint_and_copied']}")
        print(f"  description-only parameters: {', '.join(s['desc_only_parameters'])}")
        print(f"  tool purpose stated by k carriers (tool instances in the ground truth): {s['tools_by_carriers']}; "
              f"shared signatures: {s['tools_sharing_signature']}")
    (run_dir / "audit.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {run_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
