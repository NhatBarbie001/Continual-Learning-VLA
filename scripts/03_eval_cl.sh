#!/usr/bin/env bash
# Part 4: evaluate a CL checkpoint on the new task and every old LIBERO-Goal task (10 episodes each, seed 1000,
# n_action_steps 10, i.e. the same initial states as stage 0 / R1b), then write the per-run summary.
#
# Usage: bash scripts/03_eval_cl.sh NAME [NAME ...]
#   NAME  run name(s) trained by 02_train_cl.sh; the newest checkpoint of each is evaluated
# Env: NEW_TASK (3), OLD_TASKS ("8 4 0 1 5 7 9 2 6"), EPISODES (10), GPU, OUT_ROOT.
set -uo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 1; }
: "${NEW_TASK:=3}" "${OLD_TASKS:=8 4 0 1 5 7 9 2 6}" "${EPISODES:=10}"
activate_venv
OUT="$OUT_ROOT/cl"; mkdir -p "$OUT/eval"

for NAME in "$@"; do
  CK=$(ls -d "$OUT/train_$NAME"/checkpoints/*/pretrained_model | sort | tail -1)
  [ -n "$CK" ] || { echo "no checkpoint for $NAME"; continue; }
  for T in $NEW_TASK $OLD_TASKS; do
    DIR="$OUT/eval/${NAME}_task$T"
    [ -f "$DIR/eval_info.json" ] && continue
    S=$(date +%s)
    timeout -k 30 2700 "$PY" -m lerobot.scripts.lerobot_eval \
      --policy.path="$CK" --policy.device=cuda:"$GPU" --policy.n_action_steps=10 "--rename_map=$RENAME_MAP" \
      --env.type=libero --env.task=libero_goal "--env.task_ids=[$T]" --env.max_parallel_tasks=1 \
      --eval.n_episodes="$EPISODES" --eval.batch_size=5 --seed=1000 \
      --output_dir="$DIR" > "$DIR.log" 2>&1
    echo "eval ${NAME}_task$T: exit=$? seconds=$(( $(date +%s) - S )) $(grep -o '"pc_success": [0-9.]*' "$DIR/eval_info.json" 2>/dev/null | tail -1)" | tee -a "$OUT/progress.log"
  done
done
"$PY" "$CODE_DIR/analysis/summarize_cl.py" "$OUT" --new-task "$NEW_TASK" --old-tasks $OLD_TASKS --runs "$@"
