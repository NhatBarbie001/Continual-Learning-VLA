"""Build machine-readable result tables and the CL figure from the downloaded Kaggle outputs.

Usage: python analysis/make_tables.py
Reads results/<run>/cl/summary.md (written by the 03_cl* notebooks) and the R1/R1b numbers below,
writes results/tables/*.csv and results/figures/cl_per_task.png.
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
# results/ of the full project (sibling of this code folder), or $RESULTS_DIR
_code = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("RESULTS_DIR") or (_code.parent / "results" if (_code.parent / "results").is_dir() else _code / "results"))
TAB, FIG = HERE / "tables", HERE / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

GOAL = {0: "open the middle drawer", 1: "bowl on stove", 2: "wine bottle on cabinet", 3: "open top drawer + bowl inside",
        4: "bowl on cabinet", 5: "push plate to stove", 6: "cream cheese in bowl", 7: "turn on stove",
        8: "bowl on plate", 9: "wine bottle on rack"}
R1 = {0: 8, 1: 9, 2: 9, 3: 1, 4: 9, 5: 7, 6: 10, 7: 10, 8: 8, 9: 5}      # HuggingFaceVLA/smolvla_libero, nas 1
R1B = {0: 10, 1: 10, 2: 8, 3: 6, 4: 9, 5: 10, 6: 7, 7: 9, 8: 10, 9: 9}   # lerobot/smolvla_libero, nas 10

# ---- reproduction table (Goal; other suites appended when results/<run>/r1b/<suite>/summary.csv exist)
with open(TAB / "reproduce_libero.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["suite", "task", "instruction", "checkpoint", "n_action_steps", "successes", "episodes"])
    for t in range(10):
        w.writerow(["libero_goal", t, GOAL[t], "HuggingFaceVLA/smolvla_libero", 1, R1[t], 10])
    for t in range(10):
        w.writerow(["libero_goal", t, GOAL[t], "lerobot/smolvla_libero", 10, R1B[t], 10])
    for p in sorted(HERE.glob("r1b_rerun_kaggle/r1b/*/summary.csv")):   # full 30-task rerun (an earlier partial run is superseded)
        for row in csv.DictReader(open(p)):
            w.writerow([p.parent.name, row.get("task"), row.get("instruction", ""), "lerobot/smolvla_libero", 10,
                        row.get("success", row.get("n_success")), row.get("episodes", row.get("n_episodes"))])

# ---- CL runs
ROW = re.compile(r"^\| (\d+) \| (new|old) \| (\d+)/10 \| (.+) \|$")
HEAD = re.compile(r"^\| task \| role \| stage 0 \(R1b\) \| (.+) \|$")
runs = []   # (run, method, task, role, stage0, successes)
for p in sorted(HERE.glob("*/cl/summary.md")):
    methods = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if (m := HEAD.match(line)):
            methods = [c.strip() for c in m.group(1).split("|")]
        elif (m := ROW.match(line)) and methods:
            cells = [c.strip() for c in m.group(4).split("|")]
            for meth, c in zip(methods, cells):
                k = re.match(r"(\d+)/(\d+)", c)
                if k:
                    runs.append((p.parent.parent.name, meth, int(m.group(1)), m.group(2), int(m.group(3)), int(k.group(1))))
with open(TAB / "cl_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run", "method", "task", "role", "stage0_successes", "successes", "episodes"])
    for r in runs:
        w.writerow([*r, 10])

with open(TAB / "cl_resources.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run", "method", "steps", "s_per_step", "train_minutes", "peak_gpu_mib", "stored_old_task_data"])
    w.writerow(["cl_run1_kaggle", "SeqFT", 600, 4.77, 48, 5300, "none"])
    w.writerow(["cl_run1_kaggle", "CIVD", 600, 8.91, 90, 7400, "9 instruction strings"])
    w.writerow(["cl_run2_kaggle", "SeqFT", 1200, 4.58, 92, 5293, "none"])
    w.writerow(["cl_run2_kaggle", "CIVD", 1200, 8.61, 173, 7391, "9 instruction strings"])
    w.writerow(["cl_seed2_kaggle", "SeqFT", 1200, 5.19, 104, 5293, "none"])
    w.writerow(["cl_seed2_kaggle", "CIVD", 1200, 9.08, 182, 7401, "9 instruction strings"])
    w.writerow(["cl_seed3_kaggle", "SeqFT", 1200, 4.81, 97, 5293, "none"])
    w.writerow(["cl_seed3_kaggle", "CIVD", 1200, 8.17, 164, 7399, "9 instruction strings"])
    w.writerow(["cl_ablation_kaggle", "CIVD λ=0.3", 1200, 7.87, None, None, "9 instruction strings"])
    w.writerow(["cl_ablation_kaggle", "M3b new-instruction distillation", 1200, 8.03, None, None, "none"])
    w.writerow(["cl_er_kaggle", "M2 ER (5 ep/old task)", 1200, 4.71, 95, 5245, "45 demonstrations"])
    w.writerow(["cl_er_kaggle", "M4 CIVD + ER-tiny (1 ep/old task)", 1200, 8.69, 174, 7293, "9 demonstrations + 9 instruction strings"])
    w.writerow(["cl_lambda_kaggle", "CIVD λ=0.1", 1200, 8.68, 174, 7391, "9 instruction strings"])
    w.writerow(["cl_lambda_kaggle", "CIVD λ=0.03", 1200, 9.27, 186, 7391, "9 instruction strings"])
    w.writerow(["cl_seed2_ext_kaggle", "M3b new-instruction distillation (seed 2000)", 1200, 8.05, 162, 7515, "none"])
    w.writerow(["cl_seed2_ext_kaggle", "M4 CIVD + ER-tiny (seed 2000)", 1200, 8.22, 165, 7347, "9 demonstrations + 9 instruction strings"])
    w.writerow(["cl_lambda01_seeds_kaggle", "CIVD λ=0.1 (seed 2000)", 1200, 8.61, 173, 7503, "9 instruction strings"])
    w.writerow(["cl_lambda01_seeds_kaggle", "CIVD λ=0.1 (seed 3000)", 1200, 7.73, 155, 7399, "9 instruction strings"])
    w.writerow(["cl_er_controls_kaggle", "M1 ER-1 (seed 1000)", 1200, 4.59, 92, 5329, "9 demonstrations"])
    w.writerow(["cl_er_controls_kaggle", "M2 ER-5 (seed 2000)", 1200, 4.44, 89, 5245, "45 demonstrations"])


# ---- figure: per-task success after one CL stage, mean of 3 training seeds: stage 0, SeqFT, CIVD λ = 1, CIVD λ = 0.1
MAIN = ("cl_run2_kaggle", "cl_seed2_kaggle", "cl_seed3_kaggle")


def series(run, header):
    h = header.lower()
    if run in MAIN and "seqft" in h:
        return "SeqFT"
    if run in MAIN and "civd" in h:
        return "CIVD λ = 1"
    if (run == "cl_lambda_kaggle" and "λ=0.1" in h) or run == "cl_lambda01_seeds_kaggle":
        return "CIVD λ = 0.1"
    return None


NAMES = ["SeqFT", "CIVD λ = 1", "CIVD λ = 0.1"]
main = [(series(r[0], r[1]), r) for r in runs if series(r[0], r[1])]
if main:
    order = [3, 8, 4, 6, 9, 2, 7, 0, 1, 5]
    mean, nseed = {}, {}
    for meth in NAMES:
        for t in order:
            v = [r[5] for m, r in main if m == meth and r[2] == t]
            mean[(meth, t)] = sum(v) / max(1, len(v))
            nseed[meth] = len(v)
    INK, MUTED, GRID = "#1b2430", "#5b6676", "#e3e6ea"
    COL = {"Stage 0": "#b9bec6", "SeqFT": "#eb6834", "CIVD λ = 1": "#93bbea", "CIVD λ = 0.1": "#1d5fb4"}
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    x = list(range(len(order)))
    bw = 0.2
    groups = [("Stage 0", [R1B[t] * 10 for t in order])] + [(m, [mean[(m, t)] * 10 for t in order]) for m in NAMES]
    for i, (name, vals) in enumerate(groups):
        ax.bar([xi + (i - 1.5) * (bw + 0.015) for xi in x], vals, width=bw, color=COL[name], label=name, zorder=2)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(x)
    import textwrap
    ax.set_xticklabels([("TARGET · " if t == 3 else "") + f"task {t}\n" + "\n".join(textwrap.wrap(GOAL[t], 13)) for t in order],
                       fontsize=7.5, color=INK)
    for i, t in enumerate(order):   # zero-height bars are invisible: label them
        for j, m in enumerate(NAMES, start=1):
            if mean[(m, t)] == 0:
                ax.text(i + (j - 1.5) * (bw + 0.015), 1.5, "0", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylabel("success (%)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.13), fontsize=9)
    ax.set_title("Success per task after one CL stage · mean of 3 training seeds", loc="left", fontsize=10.5, color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(FIG / "cl_per_task.png")
    print("figure:", FIG / "cl_per_task.png", "| seeds per series:", nseed)
print("tables:", sorted(p.name for p in TAB.glob("*.csv")), "| CL rows:", len(runs))
