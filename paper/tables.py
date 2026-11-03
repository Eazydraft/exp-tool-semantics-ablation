"""Write the paper's numeric tables as LaTeX fragments from runs/main/analysis.json.

    python3 paper/tables.py   ->  paper/tables/*.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
A = json.load(open(ROOT / "runs" / "main" / "analysis.json"))
S, B, P = A["success"], A["behaviour"], A["paired"]
MODELS = ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "qwen/qwen3-30b-a3b-instruct-2507", "google/gemma-4-26b-a4b-it"]
SHORT = {"openai/gpt-6-luna": "GPT-6 Luna", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B", "google/gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}
CONDS = ["D0", "D1", "D2", "D3", "D4", "D5"]
tex = lambda s: s.replace("_", r"\_").replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")
pct = lambda x: f"{x*100:.1f}"


def write(name, rows):
    (OUT / f"{name}.tex").write_text("\n".join(rows) + "%")  # no trailing newline: avoids a stray space token inside tabular


# success rates + sensitivity score
rows = []
for m in MODELS:
    rows.append(f"{SHORT[m]} & " + " & ".join(pct(S[m][c]) for c in CONDS) + " \\\\")
write("success", rows)
write("robustness", [f"{SHORT[m]} & {A['robustness'][m]['D1-D4']:.3f} & {A['robustness'][m]['D1-D5']:.3f} \\\\" for m in MODELS])

# paired stats
rows = []
for m in MODELS:
    for i, c in enumerate(CONDS[1:]):
        p = P[m][c]
        pv = "$<$0.0001" if p["p_mcnemar"] < 1e-4 else f"{p['p_mcnemar']:.4f}"
        sig = r"\textbf{yes}" if p["p_mcnemar"] < 0.0025 else "no"
        first = SHORT[m] if i == 0 else ""
        rows.append(f"{first} & {c} & {p['both']} & {p['d0_only']} & {p['abl_only']} & {p['neither']} & {p['diff_pp']:+.1f} & [{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}] & {pv} & {sig} \\\\")
    if m != MODELS[-1]:
        rows.append(r"\addlinespace")
write("paired", rows)

# wrong parameters (pooled)
rows = []
for r in A["wrong_params_pooled"]:
    pm = " & ".join(str(r["per_model"][m]) for m in MODELS)
    rows.append(f"\\code{{{tex(r['tool'])}.\\allowbreak {tex(r['param'])}}} & \\emph{{{tex(r['description'].strip())}}} & \\textbf{{{r['n_d2']}}} & {r['n_d0']} & {pm} \\\\")
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
    for i, c in enumerate(CONDS):
        b = B[m][c]
        first = SHORT[m] if i == 0 else ""
        cps = "--" if b["calls_per_success"] is None else f"{b['calls_per_success']:.1f}"
        rows.append(f"{first} & {c} & {b['n_calls']:.1f} & {b['n_steps']:.1f} & {b['n_on_path']:.1f} & {b['n_off_path']:.1f} & {b['n_exec_errors']:.2f} & {b['n_repeated']:.2f} & {b['n_distinct_tools']:.1f} & {cps} \\\\")
    if m != MODELS[-1]:
        rows.append(r"\addlinespace")
write("behaviour", rows)

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
    for i, c in enumerate(CONDS):
        e = B[m][c]["error_types"]
        first = SHORT[m] if i == 0 else ""
        rows.append(f"{first} & {c} & " + " & ".join(str(e.get(k, 0)) for k in ERR) + f" & {sum(e.values())} \\\\")
    if m != MODELS[-1]:
        rows.append(r"\addlinespace")
write("failures", rows)

# tokens and cost (appendix)
rows = []
for m in MODELS:
    for i, c in enumerate(CONDS):
        b = B[m][c]
        first = SHORT[m] if i == 0 else ""
        rows.append(f"{first} & {c} & {b['prompt_tokens']:,.0f} & {b['completion_tokens']:,.0f} & {b['cost_usd']:.4f} \\\\")
    if m != MODELS[-1]:
        rows.append(r"\addlinespace")
write("cost", rows)

# a few scalars used in the text
macros = [f"\\newcommand{{\\totalcost}}{{{A['cost_usd']:.2f}}}", f"\\newcommand{{\\ntasks}}{{{A['n_tasks']}}}",
          f"\\newcommand{{\\bfclcommit}}{{{A['run']['bfcl_commit'][:10]}}}"]
write("macros", macros)
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))

# Appendix: full case-study trajectories (one longtable per model), from analysis.json "cases"
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
for m in MODELS:
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

# ---- Follow-up conditions (D6, D7) and the D0 rerun: tables computed from per-task verdicts -----------
from math import comb
import random


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
                ci=(samples[50], samples[1949]), p=mcnemar_exact(a_only, b_only))


def fmt_p(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


def valid_vec(m, c):
    return [t["valid"] for t in A["per_task"][m][c]]


if "D6" in A["conditions"] and "D7" in A["conditions"]:
    rows = []
    for m in MODELS:
        first = True
        for c in ["D6", "D7"]:
            for base in ["D0", "D2"]:
                r = paired(valid_vec(m, base), valid_vec(m, c))
                label = SHORT[m] if first else ""
                first = False
                rows.append(f"{label} & {c} vs {base} & {pct(S[m][base])} & {pct(S[m][c])} & {r['a_only']} & {r['b_only']} & {r['diff']:+.1f} & [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {fmt_p(r['p'])} \\\\")
        if m != MODELS[-1]:
            rows.append(r"\addlinespace")
    write("followup", rows)
    # six-parameter vs other tasks, all relevant conditions
    sp = A["six_param_tasks"]["per_model"]
    rows = []
    for m in MODELS:
        r = sp[m]
        cells = [f"{r['d0_only_six']}/{r['d0_only']}"]
        for c in ["D0", "D2", "D6", "D7"]:
            cells += [pct(r[f"{c}_six"]), pct(r[f"{c}_other"])]
        rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
    write("sixparam", rows)

# D0 rerun (Claude Haiku 5.5): run-to-run noise, from the rerun's index.json
rerun = ROOT / "runs" / "main" / "rerun" / "index.json"
if rerun.exists():
    R = json.load(open(rerun))
    m = "anthropic/claude-haiku-5.5"
    second = {r["id"]: bool(r["valid"]) for r in R["trajectories"] if r["model"] == m and r["condition"] == "D0"}
    ids = [t["id"] for t in A["per_task"][m]["D0"]]
    first = {t["id"]: t["valid"] for t in A["per_task"][m]["D0"]}
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
print("follow-up tables done")


# ---- Reasoning-controlled pairs (runs/main/reasoning) and ToolSandbox (runs/main/toolsandbox) ---------------
PAIRS = [("openai/gpt-6-luna", "openai/gpt-6-luna-nothink", "on", "off"),
         ("anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5-nothink", "on", "off"),
         ("google/gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b-it-think", "off", "on")]
reasoning = ROOT / "runs" / "main" / "reasoning" / "index.json"
if reasoning.exists():
    main_rows = json.load(open(ROOT / "runs" / "main" / "index.json"))["trajectories"]
    var_rows = json.load(open(reasoning))["trajectories"]
    by = {}
    for r in main_rows + var_rows:
        by.setdefault((r["model"], r["condition"]), {})[r["id"]] = r
    ids = sorted(by[("openai/gpt-6-luna", "D0")])

    def vec(m, c):
        return [bool(by[(m, c)][i]["valid"]) for i in ids]

    def tok(m):
        return sum(by[(m, "D0")][i]["reasoning_tokens"] for i in ids) / len(ids)

    rows = []
    for base, var, lab_base, lab_var in PAIRS:
        for m, lab, first in [(base, lab_base + "$^{\\ast}$", True), (var, lab_var, False)]:
            if (m, "D0") not in by:
                continue
            succ = [pct(sum(vec(m, c)) / len(ids)) for c in CONDS]
            d2, d4 = paired(vec(m, "D0"), vec(m, "D2")), paired(vec(m, "D0"), vec(m, "D4"))
            rows.append(f"{SHORT[base] if first else ''} & {lab} & {tok(m):.0f} & " + " & ".join(succ) +
                        f" & {d2['diff']:+.1f} [{d2['ci'][0]:.1f}, {d2['ci'][1]:.1f}] & {fmt_p(d2['p'])}"
                        f" & {d4['diff']:+.1f} [{d4['ci'][0]:.1f}, {d4['ci'][1]:.1f}] & {fmt_p(d4['p'])} \\\\")
        if base != PAIRS[-1][0]:
            rows.append(r"\addlinespace")
    write("reasoning", rows)
    print("reasoning table:", *rows, sep="\n")

toolsandbox = ROOT / "runs" / "main" / "toolsandbox" / "index.json"
if toolsandbox.exists():
    T = json.load(open(toolsandbox))["trajectories"]
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
    for m in MODELS:
        cells = [f"{sum(sim(m, c)) / len(names) * 100:.1f}" for c in CONDS] + [pct(sum(passed(m, c)) / len(names)) for c in CONDS]
        rows.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
        first = True
        for c in CONDS[1:]:
            diffs = [b - a for a, b in zip(sim(m, "D0"), sim(m, c))]
            lo, hi = boot_ci(diffs); r = paired(passed(m, "D0"), passed(m, c))
            prows.append(f"{SHORT[m] if first else ''} & {c} & {sum(sim(m, 'D0')) / len(names) * 100:.1f} & {sum(sim(m, c)) / len(names) * 100:.1f}"
                         f" & {sum(diffs) / len(names) * 100:+.1f} & [{lo:.1f}, {hi:.1f}] & {fmt_p(signflip(diffs))}"
                         f" & {pct(sum(passed(m, 'D0')) / len(names))} & {pct(sum(passed(m, c)) / len(names))} & {r['a_only']}/{r['b_only']} & {fmt_p(r['p'])} \\\\")
            first = False
        if m != MODELS[-1]:
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
        for m in MODELS:
            for c in ["D0", "D2"]:
                cells.append(f"{sum(sims[(m, c)][n]['similarity'] for n in sub) / len(sub) * 100:.1f}")
        crows.append(f"{label} & {len(sub)} & " + " & ".join(cells) + " \\\\")
    write("toolsandboxcats", crows)
    print("toolsandbox tables:", *rows, *prows, sep="\n")
