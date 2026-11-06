"""Aggregate CL runs across training seeds and test SeqFT vs CIVD (or any two methods) with Fisher exact tests.

Usage: python compare_runs.py results/cl_run2_kaggle results/cl_seed2_kaggle results/cl_seed3_kaggle \
           [--a seqft] [--b civd]
Each argument is a run folder containing cl/summary.md (written by the 03_cl* notebooks or summarize_cl.py).
--a / --b pick the two methods by a substring of their column header (case-insensitive).
Prints per-task pooled successes with Fisher p, pooled old-task success, and forgetting mean ± std over runs.
"""
import argparse
import re
import statistics as st
from pathlib import Path

from scipy.stats import fisher_exact

R1B = {0: 10, 1: 10, 2: 8, 3: 6, 4: 9, 5: 10, 6: 7, 7: 9, 8: 10, 9: 9}
ROW = re.compile(r"^\| (\d+) \| (new|old) \| \d+/10 \| (.+) \|$")
HEAD = re.compile(r"^\| task \| role \| stage 0 \(R1b\) \| (.+) \|$")


def parse(path):
    cols, rows = None, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if (m := HEAD.match(line)):
            cols = [c.strip().lower() for c in m.group(1).split("|")]
        elif (m := ROW.match(line)) and cols:
            vals = [re.match(r"(\d+)/(\d+)", c.strip()) for c in m.group(3).split("|")]
            rows[int(m.group(1))] = (m.group(2), {c: int(v.group(1)) for c, v in zip(cols, vals) if v})
    return cols, rows


def pick(cols, key):
    hits = [c for c in cols if key.lower() in c]
    if len(hits) != 1:
        raise SystemExit(f"--a/--b '{key}' must match exactly one column of {cols}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--a", default="seqft")
    ap.add_argument("--b", default="civd")
    a = ap.parse_args()
    data = []
    for r in a.runs:
        cols, rows = parse(Path(r) / "cl" / "summary.md")
        data.append((Path(r).name, rows, pick(cols, a.a), pick(cols, a.b)))
    tasks = list(data[0][1])
    new = next(t for t in tasks if data[0][1][t][0] == "new")
    old = [t for t in tasks if t != new]
    n = 10 * len(data)
    print(f"| task | stage 0 | {a.a} Σ/{n} | {a.b} Σ/{n} | Fisher p |\n|---|---|---|---|---|")
    for t in tasks:
        x = sum(rows[t][1][ca] for _, rows, ca, _ in data)
        y = sum(rows[t][1][cb] for _, rows, _, cb in data)
        print(f"| {t}{' (new)' if t == new else ''} | {R1B[t]} | {x} | {y} | {fisher_exact([[x, n - x], [y, n - y]])[1]:.2g} |")
    for label, sel in ((a.a, 2), (a.b, 3)):
        fg = [sum(R1B[t] - rows[t][1][d[sel]] for t in old) / len(old) * 10 for d in data for rows in [d[1]]]
        tot = sum(d[1][t][1][d[sel]] for d in data for t in old)
        print(f"{label}: old tasks {tot}/{n * len(old)}, forgetting per run {[round(f, 1) for f in fg]}"
              + (f", mean ± std {st.mean(fg):.1f} ± {st.stdev(fg):.1f} pp" if len(fg) > 1 else ""))
    xa = sum(d[1][t][1][d[2]] for d in data for t in old)
    xb = sum(d[1][t][1][d[3]] for d in data for t in old)
    N = n * len(old)
    print(f"pooled old-task Fisher p ({a.a} vs {a.b}): {fisher_exact([[xa, N - xa], [xb, N - xb]])[1]:.2g}")


if __name__ == "__main__":
    main()
