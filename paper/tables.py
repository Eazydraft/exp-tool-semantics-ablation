"""Write the paper's numeric tables as LaTeX fragments from runs/main/analysis.json and index.json.

    python3 paper/tables.py   ->  paper/tables/*.tex

Seven model configurations are treated alike: four models, three of them in both reasoning
settings. D6 and D7 exist for the default setting of each model only; the repeated Haiku D0 run
is the model id anthropic/claude-haiku-5.5-rerun.
"""
import json
import random
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
A = json.load(open(ROOT / "runs" / "main" / "analysis.json"))
INDEX = json.load(open(ROOT / "runs" / "main" / "index.json"))["trajectories"]

# (model id, label, has D6/D7); order groups each model's two reasoning settings together
CONFIGS = [("openai/gpt-6-luna", "GPT-6 Luna, reasoning", True),
           ("openai/gpt-6-luna-nothink", "GPT-6 Luna, no reasoning", False),
           ("anthropic/claude-haiku-5.5", "Claude Haiku 5.5, reasoning", True),
           ("anthropic/claude-haiku-5.5-nothink", "Claude Haiku 5.5, no reasoning", False),
           ("qwen/qwen3-30b-a3b-instruct-2507", "Qwen3-30B-A3B, no reasoning", True),
           ("google/gemma-4-26b-a4b-it", "Gemma 4 26B-A4B, no reasoning", True),
           ("google/gemma-4-26b-a4b-it-think", "Gemma 4 26B-A4B, reasoning", False)]
MODELS = [m for m, _, _ in CONFIGS]
SHORT = {m: lab for m, lab, _ in CONFIGS}
DEFAULTS = [m for m, _, d67 in CONFIGS if d67]                 # the four runs that also have D6, D7
RERUN = "anthropic/claude-haiku-5.5-rerun"                     # second launch of Claude Haiku 5.5 D0
S, B, P, PER_TASK = A["success"], A["behaviour"], A["paired"], A["per_task"]
CONDS = ["D0", "D1", "D2", "D3", "D4", "D5"]
N_COMPARISONS = 5 * len(MODELS)
ALPHA = 0.05 / N_COMPARISONS                                   # Bonferroni over the 35 ablation-vs-D0 comparisons
tex = lambda s: s.replace("_", r"\_").replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")
pct = lambda x: f"{x*100:.1f}"


def write(name, rows):
    (OUT / f"{name}.tex").write_text("\n".join(rows) + "%")  # no trailing newline: avoids a stray space token inside tabular


def fmt_p(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


def block(rows_per_model, models=MODELS):
    """Join per-model row lists with \\addlinespace between models."""
    out = []
    for i, rows in enumerate(rows_per_model):
        out += rows
        if i < len(rows_per_model) - 1:
            out.append(r"\addlinespace")
    return out


# success rates + sensitivity score
def success_cell(m, c):
    """D0: success; D1-D5: success with the paired change versus D0, bold and starred when significant."""
    if c == "D0":
        return pct(S[m][c])
    q = P[m][c]
    cell = f"{pct(S[m][c])} (${q['diff_pp']:+.1f}$)"
    return rf"\textbf{{{cell}}}$^*$" if q["p_mcnemar"] < ALPHA else cell


write("success", [f"{SHORT[m]} & " + " & ".join(success_cell(m, c) for c in CONDS) + " \\\\" for m in MODELS])
write("robustness", [f"{SHORT[m]} & {A['robustness'][m]['D1-D4']:.3f} & {A['robustness'][m]['D1-D5']:.3f} \\\\" for m in MODELS])

# paired stats
rows = []
for m in MODELS:
    rm = []
    for i, c in enumerate(CONDS[1:]):
        p = P[m][c]
        sig = r"\textbf{yes}" if p["p_mcnemar"] < ALPHA else "no"
        first = SHORT[m] if i == 0 else ""
        rm.append(f"{first} & {c} & {p['both']} & {p['d0_only']} & {p['abl_only']} & {p['neither']} & {p['diff_pp']:+.1f} & [{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}] & {fmt_p(p['p_mcnemar'])} & {sig} \\\\")
    rows.append(rm)
write("paired", block(rows))

# wrong parameters, pooled over the seven configurations (full counts, no top-k cutoff per run)
d2, d0, per_model, desc = Counter(), Counter(), {}, {}
for m in MODELS:
    full = A["wrong_params_full"][m]
    per_model[m] = {(t, p): n for t, p, n in full["D2"]}
    for t, p, n in full["D2"]:
        d2[(t, p)] += n
    for t, p, n in full["D0"]:
        d0[(t, p)] += n
for r in A["wrong_params_pooled"]:
    desc[(r["tool"], r["param"])] = r["description"]
for m in MODELS:
    for r in A["wrong_params"][m]["D2"]:
        desc.setdefault((r["tool"], r["param"]), r["description"])
rows = []
for key, n in sorted(d2.items(), key=lambda kv: kv[1] - d0[kv[0]], reverse=True)[:10]:
    t, p = key
    pm = " & ".join(str(per_model[m].get(key, 0)) for m in MODELS)
    rows.append(f"\\code{{{tex(t)}.\\allowbreak {tex(p)}}} & \\emph{{{tex(desc.get(key, '').strip())}}} & \\textbf{{{n}}} & {d0[key]} & {pm} \\\\")
write("wrongparams", rows)

# selection vs arguments
rows = []
for m in MODELS:
    cells = []
    for c in ["D0", "D2", "D5"]:
        b = B[m][c]
        cells += [pct(b["tool_selection_acc"]), pct(b["argument_acc_given_tool"]), pct(b["first_tool_ok"]), pct(b["first_call_exact"])]
    rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
write("selection", rows)

# divergence kinds among failed tasks
rows = []
for m in MODELS:
    cells = []
    for c in ["D0", "D2", "D5"]:
        d = B[m][c]["divergence_kinds"]
        cells += [str(d.get("wrong_arguments", 0)), str(d.get("tool_not_called", 0)), str(d.get("none", 0))]
    rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
write("divergence", rows)

# behaviour (appendix)
rows = []
for m in MODELS:
    rm = []
    for i, c in enumerate(CONDS):
        b = B[m][c]
        first = SHORT[m] if i == 0 else ""
        cps = "--" if b["calls_per_success"] is None else f"{b['calls_per_success']:.1f}"
        rm.append(f"{first} & {c} & {b['n_calls']:.1f} & {b['n_steps']:.1f} & {b['n_on_path']:.1f} & {b['n_off_path']:.1f} & {b['n_exec_errors']:.2f} & {b['n_repeated']:.2f} & {b['n_distinct_tools']:.1f} & {cps} \\\\")
    rows.append(rm)
write("behaviour", block(rows))

# recovery
rows = []
for m in MODELS:
    cells = []
    for c in CONDS:
        b = B[m][c]
        n = round(b["tasks_with_exec_error"] * A["n_tasks"])
        cells.append("-- (0)" if b["recovery_rate"] is None else f"{b['recovery_rate']*100:.0f}\\% ({n})")
    rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
write("recovery", rows)

# failure outcomes (appendix)
ERR = ["multi_turn:instance_state_mismatch", "multi_turn:execution_response_mismatch", "multi_turn:empty_turn_model_response", "multi_turn:force_terminated", "inference_error"]
rows = []
for m in MODELS:
    rm = []
    for i, c in enumerate(CONDS):
        e = B[m][c]["error_types"]
        first = SHORT[m] if i == 0 else ""
        rm.append(f"{first} & {c} & " + " & ".join(str(e.get(k, 0)) for k in ERR) + f" & {sum(e.values())} \\\\")
    rows.append(rm)
write("failures", block(rows))

# tokens and cost (appendix)
rows = []
for m in MODELS:
    rm = []
    for i, c in enumerate(CONDS):
        b = B[m][c]
        first = SHORT[m] if i == 0 else ""
        rm.append(f"{first} & {c} & {b['prompt_tokens']:,.0f} & {b['completion_tokens']:,.0f} & {b['cost_usd']:.4f} \\\\")
    rows.append(rm)
write("cost", block(rows))

# reasoning tokens per task by condition, from the index
by = {}
for r in INDEX:
    if r["benchmark"] == "bfcl":
        by.setdefault((r["model"], r["condition"]), []).append(r.get("reasoning_tokens") or 0)
rows = []
for m in MODELS:
    cells = [f"{sum(by[(m, c)]) / len(by[(m, c)]):,.0f}" for c in CONDS]
    rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
write("tokens", rows)

# a few scalars used in the text
macros = [f"\\newcommand{{\\totalcost}}{{{A['cost_usd']:.2f}}}", f"\\newcommand{{\\ntasks}}{{{A['n_tasks']}}}",
          f"\\newcommand{{\\bfclcommit}}{{{A['run']['bfcl_commit'][:10]}}}", f"\\newcommand{{\\ncomparisons}}{{{N_COMPARISONS}}}",
          f"\\newcommand{{\\alphabonf}}{{{ALPHA:.4f}}}"]
write("macros", macros)
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


# Appendix: full case-study trajectories (one longtable per default run), from analysis.json "cases"
def texs(s):
    out = []
    for ch in str(s):
        out.append({"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_",
                    "%": r"\%", "^": r"\^{}", "~": r"\~{}"}.get(ch, ch))
    return "".join(out)


def brk(s):
    """Insert break opportunities after punctuation so long code strings wrap inside narrow cells."""
    for ch in [",", "(", "{", ":", "/", "=", "'", "-", "."]:
        s = s.replace(ch, ch + "\\allowbreak{}")
    return s.replace("\\_", "\\_\\allowbreak{}")


def call_cell(x):
    """One call and its result as a LaTeX cell."""
    if x is None:
        return ""
    args = json.dumps(x["args"], ensure_ascii=False) if isinstance(x["args"], dict) else str(x["args"])
    if len(args) > 150:
        args = args[:147] + "..."
    res = str(x["result"]).replace("\n", " ")
    if len(res) > 100:
        res = res[:97] + "..."
    colour = r"\color{red!70!black}" if x["error"] else r"\color{black!55}"
    return f"\\footnotesize\\code{{{brk(texs(x['name']) + '(' + texs(args) + ')')}}}\\newline{{\\scriptsize\\ttfamily{colour}-> {brk(texs(res))}}}"


lines = []
ERRLABEL = {"multi_turn:instance_state_mismatch": "wrong final state", "multi_turn:execution_response_mismatch": "required output missing",
            "multi_turn:empty_turn_model_response": "turn ended without a call", "multi_turn:force_terminated": "20-step limit", None: ""}
COLS = ["D0", "D2", "D5"]
for m in DEFAULTS:
    cs = A["cases"].get(m)
    if not cs:
        continue
    tid = cs["task"]
    lines.append(f"\\subsection*{{{SHORT[m]}, task \\code{{{texs(tid)}}}}}")
    lines.append(r"\begin{enumerate}[label=Turn~\arabic*:,leftmargin=*,itemsep=0pt,start=0]\small")
    for q in cs["question"]:
        lines.append(f"  \\item {texs(q)}")
    lines.append(r"\end{enumerate}")
    gt = "; ".join(f"t{i}: " + (", ".join(f"\\code{{{brk(texs(c))}}}" for c in turn) or r"\emph{no call}") for i, turn in enumerate(cs["ground_truth"]))
    lines.append(f"{{\\sloppy\\noindent\\small\\textbf{{Ground truth.}} {gt}\\par}}\\medskip")
    verd = []
    for c in COLS:
        v = cs["conditions"][c]
        verd.append(("\\textbf{pass}" if v["valid"] else "\\textbf{fail}, " + ERRLABEL.get(v["error_type"], texs(v["error_type"] or ""))) + f" ({v['n_steps']} steps, {v['n_calls']} calls)")
    lines.append(r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.5cm} >{\raggedright\arraybackslash}p{4.55cm} >{\raggedright\arraybackslash}p{4.55cm} >{\raggedright\arraybackslash}p{4.55cm}}")
    lines.append(r"\toprule")
    lines.append(f" & D0 original \\newline {verd[0]} & D2 no parameter descriptions \\newline {verd[1]} & D5 all removed \\newline {verd[2]} \\\\")
    lines.append(r"\midrule\endhead")
    lines.append(r"\midrule\multicolumn{4}{r}{\scriptsize\emph{continued on next page}}\\\endfoot\bottomrule\endlastfoot")
    nturns = len(cs["question"])
    for t in range(nturns):
        per = {c: [x for x in cs["conditions"][c]["calls"] if x["turn"] == t] for c in COLS}
        nrows = max(1, max(len(v) for v in per.values()))
        for k in range(nrows):
            cells = []
            for c in COLS:
                if per[c]:
                    cells.append(call_cell(per[c][k]) if k < len(per[c]) else "")
                else:
                    cells.append(r"\emph{(no tool call)}" if k == 0 else "")
            label = f"t{t}" if k == 0 else ""
            lines.append(f"{label} & " + " & ".join(cells) + r" \\[2pt]")
        if t < nturns - 1:
            lines.append(r"\hline\addlinespace[2pt]")
    finals = []
    for c in COLS:
        ft = (cs["conditions"][c].get("final_text") or "").replace("\n", " ")
        finals.append(("\\footnotesize\\emph{Final reply:} ``" + texs(ft[:300]) + ("\\ldots" if len(ft) > 300 else "") + "''") if ft else "")
    lines.append(r"\hline\addlinespace[2pt]")
    lines.append(" & " + " & ".join(finals) + r" \\")
    lines.append(r"\end{longtable}")
    lines.append("")
(OUT / "cases.tex").write_text("\n".join(lines) + "%")
print("wrote cases.tex")


# ---- D6, D7 and the repeated D0 run: tables computed from per-task verdicts --------------------------------

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
                bound=max(abs(samples[50]), abs(samples[1949])))  # equivalence bound: the change is smaller than this


def valid_vec(m, c):
    return [t["valid"] for t in PER_TASK[m][c]]


if "D6" in A["conditions"] and "D7" in A["conditions"]:
    rows = []
    for m in DEFAULTS:
        rm, first = [], True
        for c in ["D6", "D7"]:
            for base in ["D0", "D2"]:
                r = paired(valid_vec(m, base), valid_vec(m, c))
                label = SHORT[m] if first else ""
                first = False
                rm.append(f"{label} & {c} vs {base} & {pct(S[m][base])} & {pct(S[m][c])} & {r['a_only']} & {r['b_only']} & {r['diff']:+.1f} & [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {fmt_p(r['p'])} \\\\")
        rows.append(rm)
    write("followup", block(rows))
    # six-parameter vs other tasks: D0 and D2 for every configuration, D6 to D8 where they exist
    rows = []
    for m in MODELS:
        r = A["six_param_tasks"]["per_model"][m]
        cells = [f"{r['d0_only_six']}/{r['d0_only']}"]
        for c in ["D0", "D2", "D6", "D7", "D8"]:
            cells += [pct(r[f"{c}_six"]), pct(r[f"{c}_other"])] if f"{c}_six" in r else ["--", "--"]
        rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    write("sixparam", rows)
    shares = [A["six_param_tasks"]["per_model"][m]["d0_only_six"] / A["six_param_tasks"]["per_model"][m]["d0_only"] * 100 for m in MODELS]
    write("sixparammacros", [f"\\newcommand{{\\sixsharemin}}{{{min(shares):.0f}}}", f"\\newcommand{{\\sixsharemax}}{{{max(shares):.0f}}}"])

# D8 (generic parameter names and blank descriptions together) against D0, D2 and D4
if "D8" in A["conditions"]:
    rows, macros = [], []
    for m in DEFAULTS:
        rm, first = [], True
        for base in ["D0", "D2", "D4"]:
            r = paired(valid_vec(m, base), valid_vec(m, "D8"))
            label = SHORT[m] if first else ""
            first = False
            rm.append(f"{label} & D8 vs {base} & {pct(S[m][base])} & {pct(S[m]['D8'])} & {r['a_only']} & {r['b_only']} & {r['diff']:+.1f} & [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {r['bound']:.1f} & {fmt_p(r['p'])} \\\\")
        rows.append(rm)
    write("names", block(rows))
    vs = {b: [paired(valid_vec(m, b), valid_vec(m, "D8")) for m in DEFAULTS] for b in ["D0", "D2", "D4"]}
    diffs = lambda b: ", ".join(f"{r['diff']:+.1f}" for r in vs[b])
    rng8 = lambda xs, f="{:.1f}": f"{f.format(min(xs))} to {f.format(max(xs))}"
    macros = [f"\\newcommand{{\\deight}}{{{', '.join(f'{S[m]['D8']*100:.1f}' for m in DEFAULTS)}}}",
              f"\\newcommand{{\\deightrange}}{{{rng8([S[m]['D8']*100 for m in DEFAULTS])}}}",
              f"\\newcommand{{\\deightvsdtworange}}{{{rng8([-r['diff'] for r in vs['D2']])}}}",
              f"\\newcommand{{\\deightvsdfourrange}}{{{rng8([-r['diff'] for r in vs['D4']])}}}",
              f"\\newcommand{{\\deightvsdzero}}{{{diffs('D0')}}}",
              f"\\newcommand{{\\deightvsdtwo}}{{{diffs('D2')}}}",
              f"\\newcommand{{\\deightvsdfour}}{{{diffs('D4')}}}",
              f"\\newcommand{{\\deightvsdtwobound}}{{{min(r['bound'] for r in vs['D2']):.1f} to {max(r['bound'] for r in vs['D2']):.1f}}}",
              f"\\newcommand{{\\deightvsdtwopmax}}{{{fmt_p(max(r['p'] for r in vs['D2']))}}}"]
    six = A["six_param_tasks"]["per_model"]
    rng = lambda xs: f"{min(xs):.1f} to {max(xs):.1f}"
    macros += [f"\\newcommand{{\\deightsixmax}}{{{max(six[m]['D8_six'] for m in DEFAULTS)*100:.1f}}}",
               f"\\newcommand{{\\deightotherdrop}}{{{rng([(six[m]['D0_other'] - six[m]['D8_other'])*100 for m in DEFAULTS])}}}",
               f"\\newcommand{{\\dtwootherabs}}{{{max(abs(six[m]['D0_other'] - six[m]['D2_other'])*100 for m in DEFAULTS):.1f}}}",
               f"\\newcommand{{\\deightargacc}}{{{rng([B[m]['D8']['argument_acc_given_tool']*100 for m in DEFAULTS])}}}",
               f"\\newcommand{{\\dtwoargacc}}{{{rng([B[m]['D2']['argument_acc_given_tool']*100 for m in DEFAULTS])}}}",
               f"\\newcommand{{\\dzeroargacc}}{{{rng([B[m]['D0']['argument_acc_given_tool']*100 for m in DEFAULTS])}}}"]
    LUNA, LUNA_RERUN = "openai/gpt-6-luna", "openai/gpt-6-luna-rerun"
    if LUNA_RERUN in PER_TASK and "D2" in PER_TASK[LUNA_RERUN]:
        second = {t["id"]: bool(t["valid"]) for t in PER_TASK[LUNA_RERUN]["D2"]}
        ids = [t["id"] for t in PER_TASK[LUNA]["D2"]]
        rr = paired(valid_vec(LUNA, "D2"), [second[i] for i in ids])          # D2 run 1 vs D2 run 2
        r8 = paired([second[i] for i in ids], valid_vec(LUNA, "D8"))          # D2 run 2 vs D8 (launched the same day)
        macros += [f"\\newcommand{{\\lunadtworerun}}{{{sum(second.values())/len(ids)*100:.1f}}}",
                   f"\\newcommand{{\\lunadtwodiff}}{{{rr['diff']:+.1f}}}", f"\\newcommand{{\\lunadtwodiffci}}{{[{rr['ci'][0]:.1f}, {rr['ci'][1]:.1f}]}}",
                   f"\\newcommand{{\\lunadtwodiscordant}}{{{rr['a_only']} and {rr['b_only']}}}",
                   f"\\newcommand{{\\deightvslunarerun}}{{{r8['diff']:+.1f}}}", f"\\newcommand{{\\deightvslunarerunci}}{{[{r8['ci'][0]:.1f}, {r8['ci'][1]:.1f}]}}",
                   f"\\newcommand{{\\deightvslunarerunp}}{{{fmt_p(r8['p'])}}}"]
    write("namesmacros", macros)
    print("D8 table done")

# repeated D0 run (Claude Haiku 5.5 with reasoning, launched twice): run-to-run noise
if RERUN in PER_TASK:
    m = "anthropic/claude-haiku-5.5"
    second = {t["id"]: bool(t["valid"]) for t in PER_TASK[RERUN]["D0"]}
    ids = [t["id"] for t in PER_TASK[m]["D0"]]
    first = {t["id"]: t["valid"] for t in PER_TASK[m]["D0"]}
    r = paired([first[i] for i in ids], [second[i] for i in ids])
    rows = [f"D0 run 1 vs D0 run 2 & {r['both']} & {r['a_only']} & {r['b_only']} & {r['neither']} & {r['diff']:+.1f} & [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {fmt_p(r['p'])} \\\\"]
    for c in ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]:
        if c not in A["conditions"]:
            continue
        rows.append(f"D0 run 1 vs {c} & " + " & ".join(str(P[m][c][k]) if c in P[m] else "" for k in ["both", "d0_only", "abl_only", "neither"]) +
                    (f" & {P[m][c]['diff_pp']:+.1f} & [{P[m][c]['ci95'][0]:.1f}, {P[m][c]['ci95'][1]:.1f}] & {fmt_p(P[m][c]['p_mcnemar'])} \\\\" if c in P[m] else " & & & \\\\"))
        r2 = paired([second[i] for i in ids], valid_vec(m, c))
        rows.append(f"D0 run 2 vs {c} & {r2['both']} & {r2['a_only']} & {r2['b_only']} & {r2['neither']} & {r2['diff']:+.1f} & [{r2['ci'][0]:.1f}, {r2['ci'][1]:.1f}] & {fmt_p(r2['p'])} \\\\")
    write("rerun", rows)
    macros = [f"\\newcommand{{\\rerunsecond}}{{{sum(second.values())/len(ids)*100:.1f}}}", f"\\newcommand{{\\rerunboth}}{{{r['both']}}}",
              f"\\newcommand{{\\rerunaonly}}{{{r['a_only']}}}", f"\\newcommand{{\\rerunbonly}}{{{r['b_only']}}}", f"\\newcommand{{\\rerundiff}}{{{r['diff']:+.1f}}}",
              f"\\newcommand{{\\rerunp}}{{{fmt_p(r['p'])}}}"]
    write("rerunmacros", macros)
print("D6/D7 and rerun tables done")


# ---- ToolSandbox: the four models, default reasoning setting ----------------------------------------------
TS_SHORT = {"openai/gpt-6-luna": "GPT-6 Luna", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
            "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B", "google/gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}
T = [r for r in INDEX if r["benchmark"] == "toolsandbox"]
if T:
    sims = {}
    for r in T:
        sims.setdefault((r["model"], r["condition"]), {})[r["id"]] = r
    names = sorted(sims[("openai/gpt-6-luna", "D0")])
    cats = {r["id"]: r["categories"] for r in T}

    def sim(m, c):
        return [sims[(m, c)][n]["similarity"] for n in names]

    def passed(m, c):
        return [sims[(m, c)][n]["similarity"] == 1.0 for n in names]

    def signflip(diffs, seed=0, n_perm=10000):
        """Paired permutation (sign-flip) test on the mean difference."""
        rng = random.Random(seed); obs = abs(sum(diffs))
        hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n_perm))
        return (hits + 1) / (n_perm + 1)

    def boot_ci(diffs, seed=0):
        rng = random.Random(seed); n = len(diffs)
        s = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(2000))
        return s[50], s[1949]

    rows, prows = [], []
    for m in DEFAULTS:
        cells = [f"{sum(sim(m, c)) / len(names) * 100:.1f}" for c in CONDS] + [pct(sum(passed(m, c)) / len(names)) for c in CONDS]
        rows.append(f"{TS_SHORT[m]} & " + " & ".join(cells) + " \\\\")
        first = True
        for c in CONDS[1:]:
            diffs = [b - a for a, b in zip(sim(m, "D0"), sim(m, c))]
            lo, hi = boot_ci(diffs); r = paired(passed(m, "D0"), passed(m, c))
            prows.append(f"{TS_SHORT[m] if first else ''} & {c} & {sum(sim(m, 'D0')) / len(names) * 100:.1f} & {sum(sim(m, c)) / len(names) * 100:.1f}"
                         f" & {sum(diffs) / len(names) * 100:+.1f} & [{lo:.1f}, {hi:.1f}] & {fmt_p(signflip(diffs))}"
                         f" & {pct(sum(passed(m, 'D0')) / len(names))} & {pct(sum(passed(m, c)) / len(names))} & {r['a_only']}/{r['b_only']} & {fmt_p(r['p'])} \\\\")
            first = False
        if m != DEFAULTS[-1]:
            prows.append(r"\addlinespace")
    write("toolsandbox", rows)
    write("toolsandboxpaired", prows)
    # category breakdown: D0 and D2 mean similarity per category
    CATS = [("SINGLE_TOOL_CALL", "single tool call"), ("MULTIPLE_TOOL_CALL", "multiple tool calls"), ("SINGLE_USER_TURN", "single user turn"),
            ("MULTIPLE_USER_TURN", "multiple user turns"), ("STATE_DEPENDENCY", "state dependency"), ("CANONICALIZATION", "canonicalization"),
            ("INSUFFICIENT_INFORMATION", "insufficient information")]
    crows = []
    for key, label in CATS:
        sub = [n for n in names if key in cats[n]]
        cells = []
        for m in DEFAULTS:
            for c in ["D0", "D2"]:
                cells.append(f"{sum(sims[(m, c)][n]['similarity'] for n in sub) / len(sub) * 100:.1f}")
        crows.append(f"{label} & {len(sub)} & " + " & ".join(cells) + " \\\\")
    write("toolsandboxcats", crows)
    print("toolsandbox tables done")


# ---- tau2-bench airline, the information audit and the divergence splits -------------------------------------
X = json.load(open(ROOT / "runs" / "main" / "analysis_extra.json"))
AU = json.load(open(ROOT / "runs" / "main" / "audit.json"))
T2 = [r for r in INDEX if r["benchmark"] == "tau2" and r.get("trial", 1) == 1]  # the single-trial study; extra trials below
if T2:
    passed = {}
    for r in T2:
        passed.setdefault((r["model"], r["condition"]), {})[str(r["id"])] = bool(r["pass"])
    ids = sorted(passed[("openai/gpt-6-luna", "D0")], key=int)
    N_T2 = 5 * len(DEFAULTS)
    ALPHA_T2 = 0.05 / N_T2  # Bonferroni over the 20 tau2 comparisons
    rows, prows = [], []
    for m in DEFAULTS:
        base = [passed[(m, "D0")][i] for i in ids]
        cells = [pct(sum(base) / len(ids))]
        first = True
        for c in CONDS[1:]:
            abl = [passed[(m, c)][i] for i in ids]
            r = paired(base, abl)
            cell = f"{pct(sum(abl) / len(ids))} (${r['diff']:+.1f}$)"
            cells.append(rf"\textbf{{{cell}}}$^*$" if r["p"] < ALPHA_T2 else cell)
            sig = r"\textbf{yes}" if r["p"] < ALPHA_T2 else "no"
            prows.append(f"{TS_SHORT[m] if first else ''} & {c} & {r['both']} & {r['a_only']} & {r['b_only']} & {r['neither']} & {r['diff']:+.1f}"
                         f" & [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {r['bound']:.1f} & {fmt_p(r['p'])} & {sig} \\\\")
            first = False
        rows.append(f"{TS_SHORT[m]} & " + " & ".join(cells) + " \\\\")
        if m != DEFAULTS[-1]:
            prows.append(r"\addlinespace")
    write("tau2", rows)
    write("tau2paired", prows)
    pr = {(m, c): paired([passed[(m, "D0")][i] for i in ids], [passed[(m, c)][i] for i in ids]) for m in DEFAULTS for c in CONDS[1:]}
    single = [pr[(m, c)]["bound"] for m in DEFAULTS for c in ["D1", "D2", "D3", "D4"]]
    others = [pr[(m, c)]["diff"] for m in DEFAULTS for c in ["D1", "D3", "D4"]]
    d2b = [pr[(m, "D2")]["bound"] for m in DEFAULTS]
    d2d = [pr[(m, "D2")]["diff"] for m in DEFAULTS]
    write("tau2macros", [f"\\newcommand{{\\tautwodtwobounds}}{{{', '.join(f'{b:.0f}' for b in d2b[:-1])} and {d2b[-1]:.0f}}}",
                         f"\\newcommand{{\\tautwodtworange}}{{${min(d2d):+.1f}$ to ${max(d2d):+.1f}$}}",
                         f"\\newcommand{{\\tautwosinglebounds}}{{{min(single):.0f} to {max(single):.0f}}}",
                         f"\\newcommand{{\\tautwootherrange}}{{${min(others):+.1f}$ to ${max(others):+.1f}$}}"])
    # D0 and D2 run for several trials: trial-averaged success, D2 against D0 on per-task means, D0 trial-to-trial noise
    TT = X.get("tau2_trials", {})
    if TT:
        rows, macros = [], []
        for m in DEFAULTS:
            e = TT[m]; d = e["d2_vs_d0"]; pairs = e["d0_trial_pairs"]
            noise = [abs(pr["diff"]) for pr in pairs]
            rows.append(f"{TS_SHORT[m]} & {e['n_trials']['D0']} & {pct(e['success']['D0'])} & {pct(e['success']['D2'])} & {d['diff']:+.1f}"
                        f" & [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}] & {d['bound']:.1f} & {d['p_signflip']:.3f} & {min(noise):.1f} to {max(noise):.1f} \\\\")
        bounds = [TT[m]["d2_vs_d0"]["bound"] for m in DEFAULTS]
        noise = [abs(pr["diff"]) for m in DEFAULTS for pr in TT[m]["d0_trial_pairs"]]
        macros = [f"\\newcommand{{\\tautrials}}{{{TT[DEFAULTS[0]]['n_trials']['D0']}}}",
                  f"\\newcommand{{\\tautrialdiff}}{{{', '.join(f'{TT[m]['d2_vs_d0']['diff']:+.1f}' for m in DEFAULTS)}}}",
                  f"\\newcommand{{\\tautrialbound}}{{{min(bounds):.1f} to {max(bounds):.1f}}}",
                  f"\\newcommand{{\\tautrialpmin}}{{{min(TT[m]['d2_vs_d0']['p_signflip'] for m in DEFAULTS):.2f}}}",
                  f"\\newcommand{{\\tautrialdiffrange}}{{${min(TT[m]['d2_vs_d0']['diff'] for m in DEFAULTS):+.1f}$ to ${max(TT[m]['d2_vs_d0']['diff'] for m in DEFAULTS):+.1f}$}}",
                  f"\\newcommand{{\\taunoise}}{{{min(noise):.1f} to {max(noise):.1f}}}"]
        write("tau2trials", rows)
        write("tau2trialsmacros", macros)
        print("tau2 trials table done")
    # where the reward is lost and how the simulations end (appendix)
    rows = []
    for m in DEFAULTS:
        rm = []
        for i, c in enumerate(CONDS):
            b = X["tau2"]["breakdown"][f"{m} {c}"]
            t = b["termination"]
            rm.append(f"{TS_SHORT[m] if i == 0 else ''} & {c} & {b['failed']} & {b['db_fail']} & {b['comm_fail']} & {b['both']} & "
                      f"{t.get('max_steps', 0)} & {t.get('too_many_errors', 0)} & {b['mean_tool_calls']:.1f} \\\\")
        rows.append(rm)
    write("tau2breakdown", block(rows))
    # earliest divergence of failed tasks, D0 / D2 / D5, with the failed tasks that made an off-path state-changing call
    rows = []
    for m in DEFAULTS:
        cells = []
        for c in ["D0", "D2", "D5"]:
            b = X["tau2"]["breakdown"][f"{m} {c}"]
            d = b["divergence_kinds"]
            cells += [str(d.get("wrong_arguments", 0)), str(d.get("tool_not_called", 0)), str(d.get("none", 0)), str(b["failed_with_off_path_write"])]
        rows.append(f"{TS_SHORT[m]} & " + " & ".join(cells) + " \\\\")
    write("tau2divergence", rows)
    rows = []
    for m in DEFAULTS:
        cells = []
        for c in ["D0", "D2", "D5"]:
            d = X["toolsandbox_divergence"][f"{m} {c}"]
            cells += [str(d.get("wrong_arguments", 0)), str(d.get("tool_not_called", 0)), str(d.get("tools_ok", 0)), str(d.get("aborted", 0))]
        rows.append(f"{TS_SHORT[m]} & " + " & ".join(cells) + " \\\\")
    write("toolsandboxdivergence", rows)
    print("tau2 tables done")

# the information audit: where each ground-truth argument's constraint is stated, per benchmark
LABELS = [("bfcl", "BFCL multi-turn"), ("toolsandbox", "ToolSandbox"), ("tau2", r"$\tau^2$-bench airline")]
rows, prows = [], []
for b, label in LABELS:
    s = AU[b]["summary"]
    cl, k = s["classes"], s["constraint_kinds"]
    rows.append(f"{label} & {s['n_calls']} & {s['n_arguments']} & {s['n_parameters']} & {cl['desc_only']} & {cl['desc_and_structure']} & {cl['structure_only']} "
                f"& {cl['copied']} & {cl['constructed']} & {s['desc_constraint_and_copied']} & {k.get('enum', 0)} & {k.get('format', 0)} & {k.get('entity', 0)} & {len(s['desc_only_parameters'])} \\\\")
    carriers = s["tools_by_carriers"]
    n3, n2 = carriers.get("3", 0), carriers.get("2", 0)
    prows.append(f"{label} & {s['tool_instances']} & {n3} & {n2} & {n2 / s['tool_instances'] * 100:.0f}\\% & "
                 + ", ".join(f"\\code{{{tex(t)}}}" for t, _ in sorted(s["tools_sharing_signature"].items(), key=lambda kv: -kv[1])[:4]) + " \\\\")
write("audit", rows)
write("purpose", prows)
au_macros = []
for b, _ in LABELS:
    s = AU[b]["summary"]
    cl = s["classes"]
    key = {"bfcl": "bfcl", "toolsandbox": "ts", "tau2": "tau"}[b]
    au_macros += [f"\\newcommand{{\\au{key}args}}{{{s['n_arguments']}}}", f"\\newcommand{{\\au{key}desconly}}{{{cl['desc_only']}}}",
                  f"\\newcommand{{\\au{key}desconlypct}}{{{cl['desc_only'] / s['n_arguments'] * 100:.1f}}}",
                  f"\\newcommand{{\\au{key}copiedpct}}{{{(cl['copied']) / s['n_arguments'] * 100:.0f}}}",
                  f"\\newcommand{{\\au{key}desconlyparams}}{{{len(s['desc_only_parameters'])}}}",
                  f"\\newcommand{{\\au{key}sharedpct}}{{{s['calls_on_shared_signature_tools'] / s['tool_instances'] * 100:.0f}}}",
                  f"\\newcommand{{\\au{key}descverbatim}}{{{s['desc_constraint_and_copied']}}}"]
write("auditmacros", au_macros)
print("audit tables done")
# the parameters whose description is the only statement of a constraint, per benchmark (appendix)
rows = []
for b, label in LABELS:
    ps = AU[b]["summary"]["desc_only_parameters"]
    rows.append(f"\\item[{label}:] " + ", ".join(f"\\code{{{tex(p).replace(chr(92) + '_', chr(92) + '_' + chr(92) + 'allowbreak{}').replace('.', '.' + chr(92) + 'allowbreak{}')}}}" for p in ps) + ".")
write("auditparams", rows)
