"""Pick the training episodes for one CL stage from the offline LIBERO-Goal episode map.

Usage: python select_episodes.py libero_goal_episodes.json --new-task 3 [--replay K]

Prints three lines (read by scripts/02_train_cl.sh):
  1. the episode list for --dataset.episodes (new task, plus the first K episodes of each old task when K > 0)
  2. the old-task instructions joined with "|" (CIVD_OLD_TASKS)
  3. the dataset revision the map was built from (lerobot/libero @ a1aaacb...)

The map (kaggle/libero_goal_episodes.json) was built from the per-frame `task_index` of lerobot/libero at that
revision; the Kaggle notebooks also assert every instruction against LIBERO's own task list at run time.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("episode_map")
ap.add_argument("--new-task", type=int, default=3)
ap.add_argument("--replay", type=int, default=0, help="demos replayed per old task (0 = none)")
a = ap.parse_args()

m = json.load(open(a.episode_map, encoding="utf-8"))
tasks = {int(t): e for t, e in m["tasks"].items()}
new = tasks[a.new_task]
old = {t: e for t, e in tasks.items() if t != a.new_task}
episodes = sorted(new["episodes"] + [ep for e in old.values() for ep in e["episodes"][:a.replay]])

print(json.dumps(episodes, separators=(",", ":")))
print("|".join(e["instruction"] for e in old.values()))
print(m["revision"])
