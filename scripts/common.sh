#!/usr/bin/env bash
# Shared settings for the run scripts. Source it; do not run it.
# Every value can be overridden from the environment, e.g. `GPU=1 SEED=2000 bash scripts/run_cl.sh ...`.

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${VENV:=/tmp/venv}"                       # Python 3.12 venv created by 00_setup_env.sh
: "${LEROBOT_DIR:=/tmp/lerobot}"
: "${LEROBOT_COMMIT:=ba83e4cf43737e2665be0aafcb5f513b83c7d0f3}"
: "${OUT_ROOT:=$CODE_DIR/outputs}"           # where checkpoints, eval dirs and summaries go
: "${GPU:=0}"                                # CUDA device for the policy
: "${EGL_DEVICE:=0}"                         # MuJoCo EGL device (Kaggle exposes only 0)

PY="$VENV/bin/python"
EPISODE_MAP="$CODE_DIR/kaggle/libero_goal_episodes.json"   # LIBERO-Goal task -> episodes in lerobot/libero @ a1aaacb

# lerobot/smolvla_libero expects the cameras as camera1/camera2; the dataset and env call them image/image2.
RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  unset PYTHONPATH UV_SYSTEM_PYTHON
  export MPLBACKEND=Agg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$EGL_DEVICE
}

needs_rename() {   # $1 = policy path; prints 1 if the checkpoint expects camera1/camera2
  case "$1" in
    lerobot/smolvla_libero|*/train_*/checkpoints/*) echo 1 ;;   # CL checkpoints are fine-tuned from it
    *) echo 0 ;;
  esac
}
