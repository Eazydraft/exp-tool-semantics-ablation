"""Build the paper's figures from runs/main2/analysis.json.

    uv run --with matplotlib python paper/figures.py      ->  paper/figures/*.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
A = json.load(open(ROOT / "runs" / "main2" / "analysis.json"))
S, B, P = A["success"], A["behaviour"], A["paired"]

MODELS = ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "qwen/qwen3-30b-a3b-instruct-2507", "google/gemma-4-26b-a4b-it"]
SHORT = {"openai/gpt-6-luna": "GPT-6 Luna", "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
         "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B", "google/gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}
CONDS = ["D0", "D1", "D2", "D3", "D4", "D5"]
CLAB = {"D0": "D0\noriginal", "D1": "D1\nno tool\ndesc.", "D2": "D2\nno param.\ndesc.", "D3": "D3\ngeneric\ntool names",
        "D4": "D4\ngeneric\nparam. names", "D5": "D5\nall\nremoved"}
# Validated categorical palette, fixed order (one hue per model), ordinal blues for D0/D2/D5.
MCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CCOL = {"D0": "#1c5cab", "D2": "#3987e5", "#D5": "#86b6ef", "D5": "#86b6ef"}
ONPATH, OFFPATH, FAILED = "#008300", "#6250d6", "#e34948"
ERR = [("multi_turn:instance_state_mismatch", "wrong final state", "#e87ba4"),
       ("multi_turn:execution_response_mismatch", "required output missing", "#6250d6"),
       ("multi_turn:empty_turn_model_response", "turn ended without a call", "#008300"),
       ("multi_turn:force_terminated", "20-step limit", "#e34948"),
       ("inference_error", "provider error", "#898781")]
GRID = "#e1e0d9"
TXT = "#52514e"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT, "text.color": "#0b0b0b",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42})


def tidy(ax, ymax=None, pct=False):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure 1: success rate by condition, grouped by model -----------------------------------
fig, ax = plt.subplots(figsize=(6.6, 2.7))
x, w = np.arange(len(CONDS)), 0.19
for i, m in enumerate(MODELS):
    vals = [S[m][c] * 100 for c in CONDS]
    ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=MCOL[i], label=SHORT[m], linewidth=0)
ax.set_xticks(x, [CLAB[c] for c in CONDS])
ax.set_ylabel("task success (200 tasks)")
tidy(ax, 100, pct=True)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
save(fig, "fig1_success")

# Figure 2: paired change vs D0, heatmap (diverging blue / grey / red) ----------------------
fig, ax = plt.subplots(figsize=(4.0, 1.9))
abl = CONDS[1:]
M = np.array([[A["pp_change"][m][c] for c in abl] for m in MODELS])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])
im = ax.imshow(M, cmap=cmap, vmin=-40, vmax=40, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        sig = P[MODELS[i]][abl[j]]["p_mcnemar"] < 0.0025
        ax.text(j, i, f"{v:+.1f}" + ("*" if sig else ""), ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 20 else "#0b0b0b", fontweight="bold" if sig else "normal")
ax.set_xticks(range(len(abl)), abl)
ax.set_yticks(range(len(MODELS)), [SHORT[m] for m in MODELS])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("pp vs D0", fontsize=7)
cb.ax.tick_params(labelsize=7, length=0)
cb.outline.set_visible(False)
save(fig, "fig2_heatmap")

# Figure 3: tool selection vs argument construction -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.28})
for ax, key, title in [(axes[0], "tool_selection_acc", "(a) tool selection"),
                       (axes[1], "argument_acc_given_tool", "(b) arguments | right tool")]:
    for i, m in enumerate(MODELS):
        ax.plot(range(6), [B[m][c][key] * 100 for c in CONDS], marker="o", ms=3.5, lw=1.4, color=MCOL[i], label=SHORT[m])
    ax.set_xticks(range(6), CONDS)
    ax.set_title(title, loc="left")
    tidy(ax, 100, pct=True)
ax = axes[2]
x, w = np.arange(len(MODELS)), 0.2
for k, (c, alpha) in enumerate([("D0", 1.0), ("D2", 1.0), ("D5", 1.0)]):
    tool = [B[m][c]["first_tool_ok"] * 100 for m in MODELS]
    exact = [B[m][c]["first_call_exact"] * 100 for m in MODELS]
    ax.bar(x + (k - 1) * w, tool, w * 0.9, color=CCOL[c], alpha=0.35, linewidth=0)
    ax.bar(x + (k - 1) * w, exact, w * 0.9, color=CCOL[c], linewidth=0, label=f"{c}")
ax.set_xticks(x, ["Luna", "Haiku", "Qwen3", "Gemma"])
ax.set_title("(c) first call of the task", loc="left")
tidy(ax, 100, pct=True)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), handlelength=1.0, columnspacing=0.8, title="light = right tool, solid = right tool and arguments", title_fontsize=7)
axes[0].legend(loc="lower left", handlelength=1.2)
save(fig, "fig3_selection_vs_args")

# Figure 4: earliest divergence in failed tasks, D0 / D2 / D5 -------------------------------
fig, ax = plt.subplots(figsize=(6.6, 2.2))
labels, wa, tnc, none = [], [], [], []
for m in MODELS:
    for c in ["D0", "D2", "D5"]:
        d = B[m][c]["divergence_kinds"]
        labels.append(f"{c}")
        wa.append(d.get("wrong_arguments", 0)); tnc.append(d.get("tool_not_called", 0)); none.append(d.get("none", 0))
x = np.array([i + (i // 3) * 0.8 for i in range(len(labels))])
ax.bar(x, wa, 0.8, color="#e87ba4", label="first unmatched call: wrong arguments", linewidth=0)
ax.bar(x, tnc, 0.8, bottom=wa, color="#6250d6", label="first unmatched call: tool never called", linewidth=0)
ax.bar(x, none, 0.8, bottom=np.array(wa) + np.array(tnc), color="#898781", label="all calls matched, still failed", linewidth=0)
ax.set_xticks(x, labels)
for i, m in enumerate(MODELS):
    ax.text(x[i * 3 + 1], 205, SHORT[m], ha="center", fontsize=7.5, color=TXT)
ax.set_ylabel("failed tasks (of 200)")
tidy(ax, 230)
ax.set_yticks([0, 50, 100, 150, 200])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
save(fig, "fig4_divergence")

# Figure 5: composition of calls per task (on-path / off-path / failed) ----------------------
fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
for ax, m, col in zip(axes, MODELS, MCOL):
    on = [B[m][c]["n_on_path"] for c in CONDS]; off = [B[m][c]["n_off_path"] for c in CONDS]; er = [B[m][c]["n_exec_errors"] for c in CONDS]
    ax.bar(range(6), on, 0.7, color=ONPATH, linewidth=0, label="on-path")
    ax.bar(range(6), off, 0.7, bottom=on, color=OFFPATH, linewidth=0, label="off-path")
    ax.bar(range(6), er, 0.7, bottom=np.array(on) + np.array(off), color=FAILED, linewidth=0, label="execution error")
    ax.set_xticks(range(6), CONDS)
    ax.set_title(SHORT[m], loc="left")
    tidy(ax, 21)
axes[0].set_ylabel("tool calls per task")
axes[0].legend(loc="upper left", handlelength=1.0)
save(fig, "fig5_calls")

# Figure 6: failure outcomes per condition --------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
for ax, m in zip(axes, MODELS):
    bottom = np.zeros(6)
    for key, label, col in ERR:
        vals = np.array([B[m][c]["error_types"].get(key, 0) for c in CONDS])
        ax.bar(range(6), vals, 0.7, bottom=bottom, color=col, linewidth=0, label=label)
        bottom += vals
    ax.set_xticks(range(6), CONDS)
    ax.set_title(SHORT[m], loc="left")
    tidy(ax, 200)
axes[0].set_ylabel("failed tasks (of 200)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
save(fig, "fig6_failures")

# Figure 7: ECDF of calls per task, D0 vs D2 vs D5 --------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9), sharey=True)
for ax, m in zip(axes, MODELS):
    for c in ["D0", "D2", "D5"]:
        xs = np.sort([t["calls"] for t in A["per_task"][m][c]])
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax.step(np.concatenate([[0], xs]), np.concatenate([[0], ys]), where="post", color=CCOL[c], lw=1.4, label=c)
    ax.set_xlim(0, 40); ax.set_title(SHORT[m], loc="left")
    ax.set_xlabel("tool calls per task")
    tidy(ax, 100, pct=True)
axes[0].set_ylabel("share of tasks")
axes[0].legend(loc="lower right", handlelength=1.2)
save(fig, "fig7_ecdf")

# Figure 8: efficiency frontier -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for i, m in enumerate(MODELS):
    xs = [B[m][c]["n_calls"] for c in CONDS]; ys = [S[m][c] * 100 for c in CONDS]
    ax.scatter(xs, ys, s=22, color=MCOL[i], label=SHORT[m], edgecolor="white", linewidth=0.6, zorder=3)
    for c, xx, yy in zip(CONDS, xs, ys):
        if c in ("D2", "D5"):
            ax.annotate(c, (xx, yy), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=TXT)
ax.set_xlabel("mean tool calls per task"); ax.set_ylabel("task success")
ax.set_xlim(0, 22)
tidy(ax, 100, pct=True)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.legend(loc="upper right", handlelength=1.0)
save(fig, "fig8_frontier")

# Figure 9: recovery after an execution error (D2 only has usable denominators) ------------------
fig, ax = plt.subplots(figsize=(3.4, 2.0))
x = np.arange(len(MODELS))
for k, c in enumerate(["D2"]):
    rates = [(B[m][c]["recovery_rate"] or 0) * 100 for m in MODELS]
    ns = [round(B[m][c]["tasks_with_exec_error"] * A["n_tasks"]) for m in MODELS]
    bars = ax.bar(x, rates, 0.5, color=CCOL[c], linewidth=0, label=c)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"n={n}", ha="center", fontsize=6.5, color=TXT)
ax.set_xticks(x, ["Luna", "Haiku", "Qwen3", "Gemma 4"])
ax.set_ylabel("tasks passed after\nan execution error (D2)")
tidy(ax, 40, pct=True)
save(fig, "fig9_recovery")
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))

# Figure 0: one task, three conditions, as a trajectory comparison (Claude Haiku 5.5, task 78) -------
import textwrap
from matplotlib.patches import FancyBboxPatch

W = 6.8 * 72  # figure width in points; the axes use point coordinates so text sizes map directly
fig = plt.figure(figsize=(6.8, 5.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
COLW = W / 3
BOXW, PAD = COLW - 12, 6
HEAD = {"D0": ("D0  original", "#0ca30c", "PASS"), "D2": ("D2  no parameter descriptions", "#d03b3b", "FAIL  wrong final state"),
        "D5": ("D5  all removed", "#d03b3b", "FAIL  no tool call")}
maxy = [0]


def box(x, y, text, kind, size=6.4, extra=None):
    """Draw a wrapped text box with its top-left at (x, y) in points; return the y below it."""
    cpl = int(BOXW / (size * (0.61 if kind == "call" else 0.50))) - 2   # characters per line
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, cpl) or [""]
    h = len(lines) * size * 1.25 + 8
    style = {"user": dict(fc="#f0efec", ec="#c3c2b7", ls="-"), "call": dict(fc="#eef4fc", ec="#2a78d6", ls="-"),
             "text": dict(fc="#ffffff", ec="#c3c2b7", ls="--"), "note": dict(fc="#fdeeee", ec="#d03b3b", ls="-")}[kind]
    if extra:
        style.update(extra)
    ax.add_patch(FancyBboxPatch((x, y), BOXW, h, boxstyle="round,pad=1.5,rounding_size=3", lw=0.9, **style))
    ax.text(x + 5, y + 4, "\n".join(lines), va="top", ha="left", fontsize=size, family="monospace" if kind == "call" else "sans-serif",
            color="#0b0b0b", linespacing=1.15)
    maxy[0] = max(maxy[0], y + h)
    return y + h + 6


for i, c in enumerate(["D0", "D2", "D5"]):
    x = i * COLW + PAD
    title, col, verdict = HEAD[c]
    ax.text(x, 4, title, fontsize=7.8, fontweight="bold", va="top", color="#0b0b0b")
    ax.text(x, 15, verdict, fontsize=6.8, fontweight="bold", va="top", color=col)
    ax.plot([x, x + BOXW], [26, 26], color=col, lw=2.2, solid_capstyle="butt")
    y = 32
    y = box(x, y, "User, turn 0: verify my tire pressure; if it is under 37.5 PSI, find the nearest tire shop.", "user")
    if c == "D5":
        y = box(x, y, "no tool call", "text", extra=dict(ec="#d03b3b"))
        y = box(x, y, "Model: I can't post that yet. None of my tools are labeled as social media or tweeting functions, so I'd be guessing which one to use, and posting is a public action I don't want to get wrong.", "text")
        ax.text(x + BOXW / 2, y + 4, "Turn 1 receives the same answer.\nBFCL: turn ended without a call.", ha="center", va="top", fontsize=6.4, color=TXT, style="italic")
        maxy[0] = max(maxy[0], y + 24)
        continue
    y = box(x, y, "check_tire_pressure()\n-> 35.0 / 35.0 / 33.0 / 33.0 PSI", "call")
    y = box(x, y, "find_nearest_tire_shop()\n-> 456 Oakwood Avenue, Rivermist, 83214", "call")
    y = box(x, y, "User, turn 1: post the tweet 'Ensuring my wheels are well-maintained. Maintenance is key to success!' with the hashtag 'BusinessOnTheMove'.", "user")
    if c == "D0":
        y = box(x, y, 'post_tweet(\n  content="Ensuring my wheels ... success!",\n  tags=["#BusinessOnTheMove"])\n-> tweet 10 stored with\n   tags=["#BusinessOnTheMove"]', "call")
        y = box(x, y, "Backend state matches the ground truth after both turns.", "text")
    else:
        y = box(x, y, 'post_tweet(\n  content="Ensuring my wheels ... success!\n           #BusinessOnTheMove")\n-> tweet 10 stored with tags=[]', "call", extra=dict(ec="#d03b3b", fc="#fdf1f1"))
        y = box(x, y, "Removed description of tags: \"List of tags for the tweet. Tag name should start with #.\" This is the only difference between the two runs.", "note")
H = maxy[0] + 8
ax.set_xlim(0, W); ax.set_ylim(H, 0)
fig.set_size_inches(6.8, H / 72)
fig.savefig(OUT / "fig0_trajectory.pdf")
plt.close(fig)
print("wrote fig0_trajectory.pdf")

# Figure 10: follow-up conditions, if present in the analysis ---------------------------------------------
if "D6" in A["conditions"] and "D7" in A["conditions"]:
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    conds = ["D0", "D2", "D6", "D7"]
    lab = {"D0": "D0\noriginal", "D2": "D2\nno param. desc.", "D6": "D6\nno param. desc.,\nconstraints in schema", "D7": "D7\nno param. desc.,\nexcept six params"}
    x, w = np.arange(len(conds)), 0.19
    for i, m in enumerate(MODELS):
        vals = [S[m][c] * 100 for c in conds]
        bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=MCOL[i], label=SHORT[m], linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=5.8, color=TXT)
    ax.set_xticks(x, [lab[c] for c in conds])
    ax.set_ylabel("task success (200 tasks)")
    tidy(ax, 100, pct=True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, "fig10_followup")
    print("wrote fig10_followup.pdf")
