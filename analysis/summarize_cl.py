"""Summarize one CL stage: success per task for each run, new-task gain and mean forgetting vs stage 0.

Usage: python summarize_cl.py <cl_dir> --runs m0 m3 [--new-task 3] [--old-tasks 8 4 0 1 5 7 9 2 6]
<cl_dir> holds eval/<run>_task<T>/eval_info.json (written by 03_eval_cl.sh or the 03_cl* notebooks).
Writes <cl_dir>/summary.md and prints it.

Stage 0 = R1b (lerobot/smolvla_libero, n_action_steps 10, seed 1000, 10 episodes per task).
Forgetting = stage-0 success − success after the stage, averaged over the old tasks (gains count as negative).
"""
import argparse
import json
import math
from pathlib import Path

R1B = {0: 10, 1: 10, 2: 8, 3: 6, 4: 9, 5: 10, 6: 7, 7: 9, 8: 10, 9: 9}


def wilson(k, n, z=1.96):
    q = k / n
    d = 1 + z * z / n
    c = (q + z * z / (2 * n)) / d
    h = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d
    return f"{100 * (c - h):.0f}–{100 * (c + h):.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cl_dir")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--new-task", type=int, default=3)
    ap.add_argument("--old-tasks", type=int, nargs="+", default=[8, 4, 0, 1, 5, 7, 9, 2, 6])
    a = ap.parse_args()
    out = Path(a.cl_dir)

    def sr(name):
        f = out / "eval" / name / "eval_info.json"
        if not f.exists():
            return None
        o = json.load(open(f))["overall"]
        return int(o["n_success"]), int(o["n_episodes"])

    new, old = a.new_task, a.old_tasks
    lines = [f"# One CL stage on LIBERO-Goal · new task {new}", "",
             "| task | role | stage 0 (R1b) | " + " | ".join(a.runs) + " |", "|---|---|---|" + "---|" * len(a.runs)]
    res = {}
    for t in [new] + old:
        cells = []
        for m in a.runs:
            r = res[(m, t)] = sr(f"{m}_task{t}")
            cells.append("missing" if r is None else f"{r[0]}/{r[1]} ({wilson(*r)})")
        lines.append(f"| {t} | {'new' if t == new else 'old'} | {R1B[t]}/10 | " + " | ".join(cells) + " |")
    lines += ["", "| run | new-task gain (pp) | old tasks | mean forgetting (pp) |", "|---|---|---|---|"]
    for m in a.runs:
        rn = res[(m, new)]
        ro = [(t, res[(m, t)]) for t in old if res[(m, t)]]
        gain = f"{10 * (rn[0] * 10 / rn[1] - R1B[new]):+.0f}" if rn else "n/a"
        if ro:
            fg = sum(10 * (R1B[t] - r[0] * 10 / r[1]) for t, r in ro) / len(ro)
            tot = f"{sum(r[0] for _, r in ro)}/{sum(r[1] for _, r in ro)}"
            lines.append(f"| {m} | {gain} | {tot} | {fg:.1f} ({len(ro)} tasks) |")
        else:
            lines.append(f"| {m} | {gain} | n/a | n/a |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
