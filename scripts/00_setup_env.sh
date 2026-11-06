#!/usr/bin/env bash
# Install everything needed to run SmolVLA + LIBERO (≈ 10 min). Tested on Kaggle (Ubuntu, 2× T4, CUDA 12).
# Usage: bash code/scripts/00_setup_env.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "== system libraries (headless MuJoCo rendering, video)"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq libegl1 libgl1 libglvnd0 libosmesa6 libosmesa6-dev ffmpeg cmake build-essential git

echo "== Python 3.12 venv at $VENV (LeRobot needs >= 3.12)"
pip install -q -U uv
uv python install 3.12
rm -rf "$VENV" && uv venv "$VENV" --python 3.12

echo "== LeRobot @ $LEROBOT_COMMIT"
rm -rf "$LEROBOT_DIR"
git clone -q https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
git -C "$LEROBOT_DIR" checkout -q "$LEROBOT_COMMIT"
(cd "$LEROBOT_DIR" && uv pip install --python "$PY" -e ".[smolvla,libero]")

echo "== import check"
activate_venv
"$PY" - <<'EOF'
import importlib, torch
for m in ["torch", "lerobot", "transformers", "mujoco", "libero"]:
    mod = importlib.import_module(m)
    print(f"{m}: ok {getattr(mod, '__version__', '')}")
print("cuda:", torch.version.cuda, "|", torch.cuda.device_count(), "GPU(s)")
EOF
# LIBERO asks for a dataset path on its first import; answer "N" to keep the defaults.
yes N | "$PY" -c "from libero.libero import benchmark; print('libero suites:', len(benchmark.get_benchmark_dict()))" | tail -1
echo "setup: ok"
