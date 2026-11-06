#!/usr/bin/env bash
# Part 2 (Reproduce): evaluate a SmolVLA checkpoint on LIBERO, one lerobot-eval call per task.
#
# Usage: bash code/scripts/01_eval_libero.sh POLICY SUITE [TASKS] [NAS]
#   POLICY  lerobot/smolvla_libero (R1b, paper config) | HuggingFaceVLA/smolvla_libero (R1) | a local pretrained_model dir
#   SUITE   libero_spatial | libero_object | libero_goal | libero_10
#   TASKS   task ids, default "0 1 2 3 4 5 6 7 8 9"
#   NAS     n_action_steps; default 10 (R1b). 0 = checkpoint default (R1 used 1). R3 = 1 / 10 / 50.
# Env: EPISODES (10), BATCH (3 parallel envs), SEED (1000), TASK_TIMEOUT (2700 s), GPU, OUT_ROOT.
#
# Examples (the runs in the report):
#   R1b, 4 suites:  for s in libero_spatial libero_object libero_goal libero_10; do bash code/scripts/01_eval_libero.sh lerobot/smolvla_libero $s; done
#   R1, Goal:       bash code/scripts/01_eval_libero.sh HuggingFaceVLA/smolvla_libero libero_goal "" 0
#   R3, chunks:     for n in 1 10 50; do bash code/scripts/01_eval_libero.sh HuggingFaceVLA/smolvla_libero libero_goal "0 9" $n; done
set -uo pipefail
source "$(dirname "$0")/common.sh"
POLICY=${1:?policy}; SUITE=${2:?suite}; TASKS=${3:-}; NAS=${4:-10}
[ -z "$TASKS" ] && TASKS="0 1 2 3 4 5 6 7 8 9"
: "${EPISODES:=10}" "${BATCH:=3}" "${SEED:=1000}" "${TASK_TIMEOUT:=2700}"
activate_venv

TAG=$(basename "$POLICY")_nas${NAS}
OUT="$OUT_ROOT/eval/$TAG/$SUITE"; mkdir -p "$OUT"
ARGS=()
[ "$NAS" != "0" ] && ARGS+=(--policy.n_action_steps="$NAS")
[ "$(needs_rename "$POLICY")" = "1" ] && ARGS+=("--rename_map=$RENAME_MAP")

for T in $TASKS; do
  DIR="$OUT/task_$T"
  if [ -f "$DIR/eval_info.json" ]; then echo "task $T: already done"; continue; fi
  for TRY in 0 1; do   # a deadlocked env pool is killed by the timeout and retried once
    rm -rf "$DIR"; S=$(date +%s)
    timeout -k 30 "$TASK_TIMEOUT" "$PY" -m lerobot.scripts.lerobot_eval \
      --policy.path="$POLICY" --policy.device=cuda:"$GPU" "${ARGS[@]}" \
      --env.type=libero --env.task="$SUITE" "--env.task_ids=[$T]" --env.max_parallel_tasks=1 \
      --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" --seed="$SEED" \
      --output_dir="$DIR" > "$DIR.log" 2>&1
    RC=$?
    echo "task $T try $TRY: exit=$RC seconds=$(( $(date +%s) - S )) $(grep -o '"pc_success": [0-9.]*' "$DIR/eval_info.json" 2>/dev/null | tail -1)" | tee -a "$OUT/progress.log"
    [ -f "$DIR/eval_info.json" ] && break
  done
done
"$PY" "$CODE_DIR/analysis/summarize_eval.py" "$OUT" --title "$TAG · $SUITE · seed $SEED"
