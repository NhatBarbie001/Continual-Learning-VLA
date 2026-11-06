#!/usr/bin/env bash
# Part 4 (Improve): one continual-learning stage. Fine-tune the stage-0 model on ONE new LIBERO-Goal task.
#
# Usage: bash code/scripts/02_train_cl.sh METHOD NAME [LAMBDA]
#   METHOD  seqft    M0  plain fine-tune on the new task
#           civd     M3  CIVD: + λ·distillation on new-task observations paired with OLD instructions
#           civdnew  M3b ablation: distillation paired with the NEW instruction (LwF-style)
#           erK      M2  SeqFT + K replayed demos per old task (e.g. er5)
#           civderK  M4  CIVD + K replayed demos per old task (e.g. civder1)
#   NAME    run name; the checkpoint goes to $OUT_ROOT/cl/train_NAME
#   LAMBDA  distillation weight (default 1.0; ignored for seqft / erK)
# Env: BASE (lerobot/smolvla_libero), NEW_TASK (3), STEPS (1200), BATCH (16), SEED (1000), DISTILL_BS (4), GPU.
set -euo pipefail
source "$(dirname "$0")/common.sh"
METHOD=${1:?method}; NAME=${2:?name}; LAM=${3:-1.0}
: "${BASE:=lerobot/smolvla_libero}" "${NEW_TASK:=3}" "${STEPS:=1200}" "${BATCH:=16}" "${SEED:=1000}" "${DISTILL_BS:=4}"
activate_venv
OUT="$OUT_ROOT/cl"; mkdir -p "$OUT"

# Episode list (new task, plus K demos per old task for replay) and the old instructions, from the offline map.
REPLAY=0; MODE=$METHOD
case "$METHOD" in
  er*)     REPLAY=${METHOD#er};     MODE=seqft ;;
  civder*) REPLAY=${METHOD#civder}; MODE=civd ;;
esac
SEL=$("$PY" "$CODE_DIR/analysis/select_episodes.py" "$EPISODE_MAP" --new-task "$NEW_TASK" --replay "$REPLAY")
EPS=$(echo "$SEL" | sed -n 1p); OLD=$(echo "$SEL" | sed -n 2p); REV=$(echo "$SEL" | sed -n 3p)

if [ "$MODE" = "civd" ] || [ "$MODE" = "civdnew" ]; then
  ENTRY=("$CODE_DIR/civd/civd.py")
  CM=old; [ "$MODE" = "civdnew" ] && CM=new
  export CIVD_TEACHER=$BASE CIVD_OLD_TASKS="$OLD" CIVD_LAMBDA=$LAM CIVD_DISTILL_BS=$DISTILL_BS CIVD_MODE=$CM CIVD_LOG_EVERY=25
else
  ENTRY=(-m lerobot.scripts.lerobot_train)
fi

echo "[$NAME] method=$METHOD mode=$MODE λ=$LAM replay=$REPLAY steps=$STEPS batch=$BATCH seed=$SEED gpu=$GPU"
S=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU "$PY" "${ENTRY[@]}" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false --policy.train_expert_only=true \
  --dataset.repo_id=lerobot/libero --dataset.revision="$REV" "--dataset.episodes=$EPS" \
  "--rename_map=$RENAME_MAP" \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="$STEPS" --log_freq=25 --seed="$SEED" \
  --output_dir="$OUT/train_$NAME" 2>&1 | tee "$OUT/train_$NAME.log"
echo "[$NAME] train done in $(( ($(date +%s) - S) / 60 )) min" | tee -a "$OUT/progress.log"
