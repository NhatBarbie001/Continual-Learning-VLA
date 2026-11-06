"""Summarize a LIBERO evaluation folder (one task_<id>/eval_info.json per task) with Wilson 95 % CIs.

Usage: python summarize_eval.py <suite_dir> [--title TEXT]
Writes <suite_dir>/summary.md and summary.csv (same format as the 01_r1_eval notebook) and prints the table.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    out = Path(a.suite_dir)
    rows = []
    for f in sorted(out.glob("task_*/eval_info.json"), key=lambda p: int(p.parent.name.split("_")[1])):
        o = json.load(open(f))["overall"]
        k, n = int(o["n_success"]), int(o["n_episodes"])
        lo, hi = wilson(k, n)
        rows.append(dict(task=int(f.parent.name.split("_")[1]), success=k, episodes=n, pc_success=100 * k / n,
                         ci_low=lo, ci_high=hi, sec_per_episode=o.get("eval_ep_s", float("nan"))))
    if not rows:
        print("no eval_info.json found in", out)
        return
    K, N = sum(r["success"] for r in rows), sum(r["episodes"] for r in rows)
    lo, hi = wilson(K, N)
    lines = [f"# {a.title or out.name}", "", "| task | success | SR % | 95% Wilson CI | s/episode |", "|---|---|---|---|---|"]
    lines += [f"| {r['task']} | {r['success']}/{r['episodes']} | {r['pc_success']:.0f} | {r['ci_low']:.0f}–{r['ci_high']:.0f} | "
              f"{r['sec_per_episode']:.0f} |" for r in rows]
    lines += [f"| **suite** | **{K}/{N}** | **{100 * K / N:.1f}** | **{lo:.1f}–{hi:.1f}** | |"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    print(f"tasks finished: {len(rows)}")


if __name__ == "__main__":
    main()
