"""Generate the Kaggle notebooks in this folder from one shared setup recipe.

Usage: python kaggle/build_notebooks.py
Writes 00_smoke_test/00_smoke_test.ipynb and 01_r1_eval/01_r1_eval.ipynb.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEROBOT_COMMIT = "ba83e4cf43737e2665be0aafcb5f513b83c7d0f3"
# Kaggle sets PYTHONPATH/UV_SYSTEM_PYTHON for its own Python and an inline matplotlib backend
# that the venv does not have; clear both in every cell that uses the venv.
VENV = "source /tmp/venv/bin/activate; unset PYTHONPATH UV_SYSTEM_PYTHON; export MPLBACKEND=Agg"
PY = "/tmp/venv/bin/python"
# LIBERO-Goal task id -> instruction, dataset task_index and episode indices in lerobot/libero (pinned revision)
GOAL_MAP = json.loads((HERE / "libero_goal_episodes.json").read_text())


def setup_cells(report: str, strict: bool) -> list[tuple[str, str]]:
    """Shared install recipe. strict=True stops the run on failure (saves GPU quota)."""
    fail = "exit 1" if strict else "true"
    return [
        ("code", f"""%%bash
echo "=== ENV ===" >> {report}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a {report}
echo "CPUs: $(nproc)" | tee -a {report}
free -g | head -2 | tee -a {report}"""),
        ("code", f"""%%bash
# System libraries for headless MuJoCo rendering and video decoding
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq libegl1 libgl1 libglvnd0 libosmesa6 libosmesa6-dev ffmpeg cmake build-essential > /dev/null 2>&1 \\
  && echo "apt: ok" | tee -a {report} || {{ echo "apt: FAILED" | tee -a {report}; {fail}; }}
# LeRobot needs Python >= 3.12; build an isolated venv with uv
pip install -q -U uv
uv python install 3.12 > /dev/null 2>&1
rm -rf /tmp/venv && uv venv /tmp/venv --python 3.12 > /dev/null 2>&1
/tmp/venv/bin/python --version > /dev/null 2>&1 \\
  && echo "venv: ok ($(/tmp/venv/bin/python --version))" | tee -a {report} || {{ echo "venv: FAILED" | tee -a {report}; {fail}; }}"""),
        ("code", f"""%%bash
{VENV}
LOG=/kaggle/working/install.log
cd /tmp && rm -rf lerobot && git clone -q https://github.com/huggingface/lerobot.git && cd lerobot && git checkout -q {LEROBOT_COMMIT}
echo "lerobot commit: $(git rev-parse HEAD)" | tee -a {report}
unset UV_PYTHON UV_PROJECT_ENVIRONMENT PIP_TARGET
# Target the venv explicitly: Kaggle's environment otherwise sends uv to the system Python
uv pip install --python /tmp/venv/bin/python -e ".[smolvla,libero]" > $LOG 2>&1
RC=$?
echo "install exit: $RC ($(head -1 $LOG))" | tee -a {report}
if [ $RC -ne 0 ]; then tail -40 $LOG | tee -a {report}; {fail}; fi
python - <<'EOF' 2>&1 | tee -a {report}
import importlib, torch
for m in ["torch", "lerobot", "transformers", "mujoco", "libero"]:
    mod = importlib.import_module(m)
    print(f"import {{m}}: ok {{getattr(mod, '__version__', '')}}")
print("cuda:", torch.version.cuda, "|", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0))
EOF
[ ${{PIPESTATUS[0]}} -eq 0 ] || {fail}
# LIBERO asks for a dataset path on first import; answer "N" to keep defaults
(yes N | python -c "from libero.libero import benchmark; print('libero:', len(benchmark.get_benchmark_dict()), 'suites')") 2>&1 | tail -1 | tee -a {report}
# Kaggle ships its own /usr/local/bin/lerobot-* that shadow the venv's; every call below uses `{PY} -m ...`
echo "lerobot-eval on PATH: $(which -a lerobot-eval | tr '\\n' ' ')" | tee -a {report}
{PY} -c "import lerobot, libero, mujoco" && touch /tmp/setup_ok && echo "setup: ok" | tee -a {report}"""),
    ]


def smoke_notebook() -> list[tuple[str, str]]:
    R = "/kaggle/working/smoke_report.txt"
    guard = f'python -c "import torch" 2>/dev/null || {{ echo "SKIP: install failed, see install.log" | tee -a {R}; exit 0; }}'
    return [
        ("md", """# 00 · Smoke test: SmolVLA + LIBERO on Kaggle

**Before running:** Accelerator **GPU T4 x2**, Internet **On**. Then *Run All*.
Every step appends to `/kaggle/working/smoke_report.txt`; the last cell prints it."""),
        *setup_cells(R, strict=False),
        ("code", f"""%%bash
{VENV}
{guard}
# Try GPU (EGL) rendering first, fall back to CPU (OSMesa)
for GL in egl osmesa; do
  if MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL python - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 128, 128); r.update_scene(d); img = r.render()
assert img.mean() > 0
EOF
  then echo "render backend: $GL" | tee -a {R}; echo $GL > /tmp/gl.txt; break
  else echo "render $GL: FAILED" | tee -a {R}; fi
done"""),
        ("code", f"""%%bash
{VENV}
{guard}
GL=$(cat /tmp/gl.txt 2>/dev/null || echo egl); export MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 1000 > /tmp/mem_eval.txt &
MON=$!
START=$(date +%s)
{PY} -m lerobot.scripts.lerobot_eval \\
  --policy.path=HuggingFaceVLA/smolvla_libero \\
  --env.type=libero --env.task=libero_goal '--env.task_ids=[0]' \\
  --eval.n_episodes=2 --eval.batch_size=1 \\
  --policy.device=cuda \\
  --output_dir=/kaggle/working/smoke_eval 2>&1 | tail -25
EXIT=${{PIPESTATUS[0]}}
kill $MON
echo "=== EVAL (2 episodes, libero_goal task 0) exit=$EXIT seconds=$(( $(date +%s) - START )) peakMiB=$(sort -n /tmp/mem_eval.txt | tail -1)" | tee -a {R}"""),
        ("code", f"""%%bash
{VENV}
{guard}
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 1000 > /tmp/mem_train.txt &
MON=$!
START=$(date +%s)
{PY} -m lerobot.scripts.lerobot_train \\
  --policy.path=lerobot/smolvla_base \\
  --dataset.repo_id=lerobot/libero \\
  '--rename_map={{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}}' \\
  --batch_size=8 --steps=50 --log_freq=10 --save_freq=100000 \\
  --policy.device=cuda --policy.push_to_hub=false \\
  --output_dir=/tmp/smoke_train 2>&1 | tail -30
EXIT=${{PIPESTATUS[0]}}
kill $MON
echo "=== TRAIN (50 steps, bs 8, incl. dataset download) exit=$EXIT seconds=$(( $(date +%s) - START )) peakMiB=$(sort -n /tmp/mem_train.txt | tail -1)" | tee -a {R}"""),
        ("code", f"""print(open("{R}").read())"""),
    ]


def r1_notebook() -> list[tuple[str, str]]:
    R = "/kaggle/working/r1_report.txt"
    return [
        ("md", """# 01 · R1: evaluate an official SmolVLA LIBERO checkpoint on LIBERO suites

Runs `POLICY` on every suite in `SUITES`, **10 episodes per task**, fixed seed and initial states.
Tasks are split over the **two T4s** (even/odd ids) and run one task per `lerobot-eval` call,
so finished tasks survive if the session stops; re-running skips them.

- `HuggingFaceVLA/smolvla_libero` (R1): cameras `image`/`image2`, checkpoint default `n_action_steps` = 1.
- `lerobot/smolvla_libero` (R1b, paper architecture): cameras `camera1`/`camera2` → `RENAME = True`; `n_action_steps` = 10.

**How to run:** `kaggle kernels push -p kaggle/01_r1_eval --accelerator NvidiaTeslaT4 -t <seconds>`.
**Outputs** (`/kaggle/working/<TAG>/<suite>/`): `task_<id>/eval_info.json`, `task_<id>/videos/`, `summary.md`, `summary.csv`, `progress.log`."""),
        ("code", """# ---- parameters ----
SUITES = ["libero_spatial", "libero_object", "libero_10"]   # libero_goal done (R1 76 %, R1b 88 %)
POLICY, TAG = "lerobot/smolvla_libero", "r1b"
RENAME = True                  # lerobot/smolvla_libero expects camera1/camera2
NAS = 10                       # n_action_steps; 0 = checkpoint default
N_EPISODES = 10
BATCH = 3                      # parallel envs per GPU; 5 x 2 GPUs on 4 CPUs / 31 GB hung or was OOM-killed
SEED = 1000
TASKS = list(range(10))
TASK_TIMEOUT, RETRIES = 2700, 1  # seconds per lerobot-eval call; a hung env pool is killed and retried
open("/tmp/params.sh", "w").write(
    f'SUITES="{" ".join(SUITES)}"\\nPOLICY={POLICY}\\nTAG={TAG}\\nRENAME={int(RENAME)}\\nNAS={NAS}\\n'
    f'N_EPISODES={N_EPISODES}\\nBATCH={BATCH}\\nSEED={SEED}\\nTASK_TIMEOUT={TASK_TIMEOUT}\\nRETRIES={RETRIES}\\n'
    f'TASKS_GPU0="{" ".join(map(str, TASKS[0::2]))}"\\nTASKS_GPU1="{" ".join(map(str, TASKS[1::2]))}"\\n')
print(open("/tmp/params.sh").read())"""),
        *setup_cells(R, strict=True),
        ("code", f"""%%bash
# One lerobot-eval call per task so that partial results survive a stopped session
cat > /tmp/run_tasks.sh <<'EOF'
#!/usr/bin/env bash
{VENV}
source /tmp/params.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
GPU=$1; shift
ARGS=()
[ "$NAS" != "0" ] && ARGS+=(--policy.n_action_steps=$NAS)
[ "$RENAME" = "1" ] && ARGS+=({RENAME})
for SUITE in $SUITES; do
  OUT=/kaggle/working/$TAG/$SUITE
  mkdir -p $OUT
  for T in "$@"; do
    DIR=$OUT/task_$T
    if [ -f $DIR/eval_info.json ]; then echo "[gpu$GPU] task $T: already done" >> $OUT/progress.log; continue; fi
    for TRY in $(seq 0 $RETRIES); do
      rm -rf $DIR
      S=$(date +%s)
      # Kaggle's EGL exposes a single device (id 0) even with both T4s visible, so all rendering
      # goes to EGL device 0; only the policy is placed on cuda:$GPU.
      # timeout: a deadlocked env pool once blocked a GPU for an hour; kill it and retry.
      MUJOCO_EGL_DEVICE_ID=0 timeout -k 30 $TASK_TIMEOUT {PY} -m lerobot.scripts.lerobot_eval \\
        --policy.path=$POLICY --policy.device=cuda:$GPU "${{ARGS[@]}}" \\
        --env.type=libero --env.task=$SUITE "--env.task_ids=[$T]" --env.max_parallel_tasks=1 \\
        --eval.n_episodes=$N_EPISODES --eval.batch_size=$BATCH --seed=$SEED \\
        --output_dir=$DIR > $DIR.log 2>&1
      RC=$?
      echo "[gpu$GPU] task $T try $TRY: exit=$RC seconds=$(( $(date +%s) - S )) $(grep -o '"pc_success": [0-9.]*' $DIR/eval_info.json 2>/dev/null | tail -1)" >> $OUT/progress.log
      # results are written before env.close(); a crash on close still leaves a valid eval_info.json
      [ -f $DIR/eval_info.json ] && break
      tail -5 $DIR.log >> $OUT/progress.log
      pkill -f "cuda:$GPU.*task_ids=\\[$T\\]" 2>/dev/null; sleep 10
    done
  done
done
EOF
chmod +x /tmp/run_tasks.sh
echo runner ready"""),
        ("code", f"""%%bash
source /tmp/params.sh
if [ ! -f /tmp/setup_ok ]; then echo "Setup did not finish; see r1_report.txt and install.log"; cat {R}; exit 1; fi
START=$(date +%s)
mkdir -p /kaggle/working/$TAG
# memory/CPU trace every 60 s to diagnose OOM kills (exit 137)
(while true; do echo "$(date +%H:%M:%S) $(free -m | awk '/Mem:/{{print "used",$3,"avail",$7}}') load $(cut -d' ' -f1 /proc/loadavg)"; sleep 60; done) > /kaggle/working/$TAG/mem.log 2>&1 &
MEMMON=$!
/tmp/run_tasks.sh 0 $TASKS_GPU0 &
P0=$!
sleep 90   # stagger: both runs starting LIBERO at once hung the first pair of tasks
/tmp/run_tasks.sh 1 $TASKS_GPU1 &
P1=$!
wait $P0 $P1
kill $MEMMON 2>/dev/null
echo "=== $TAG done in $(( ($(date +%s) - START) / 60 )) min" | tee -a {R}
for SUITE in $SUITES; do echo "--- $SUITE"; cat /kaggle/working/$TAG/$SUITE/progress.log; done | tee -a {R}"""),
        ("code", """import json, math, csv
from pathlib import Path

params = {k: v.strip('"') for k, v in (l.strip().split("=", 1) for l in open("/tmp/params.sh") if "=" in l)}

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))

for suite in params["SUITES"].split():
    out = Path(f"/kaggle/working/{params['TAG']}/{suite}")
    rows = []
    for f in sorted(out.glob("task_*/eval_info.json"), key=lambda p: int(p.parent.name.split("_")[1])):
        o = json.load(open(f))["overall"]
        k, n = int(o["n_success"]), int(o["n_episodes"])
        lo, hi = wilson(k, n)
        rows.append(dict(task=int(f.parent.name.split("_")[1]), success=k, episodes=n, pc_success=100 * k / n,
                         ci_low=lo, ci_high=hi, sec_per_episode=o.get("eval_ep_s", float("nan"))))
    K, N = sum(r["success"] for r in rows), sum(r["episodes"] for r in rows)
    lo, hi = wilson(K, N)
    lines = [f"# {params['TAG']} · {suite} · {params['POLICY']} · n_action_steps {params['NAS']} · seed {params['SEED']} · {params['N_EPISODES']} ep/task", "",
             "| task | success | SR % | 95% Wilson CI | s/episode |", "|---|---|---|---|---|"]
    lines += [f"| {r['task']} | {r['success']}/{r['episodes']} | {r['pc_success']:.0f} | {r['ci_low']:.0f}–{r['ci_high']:.0f} | {r['sec_per_episode']:.0f} |" for r in rows]
    lines += [f"| **suite** | **{K}/{N}** | **{100 * K / max(N, 1):.1f}** | **{lo:.1f}–{hi:.1f}** | |"]
    if rows:
        (out / "summary.md").write_text("\\n".join(lines))
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("\\n".join(lines))
    print(f"Tasks finished: {len(rows)}/10\\n")"""),
    ]


RENAME = ('\'--rename_map={"observation.images.image": "observation.images.camera1", '
          '"observation.images.image2": "observation.images.camera2"}\'')


def r2_notebook() -> list[tuple[str, str]]:
    R = "/kaggle/working/r2_report.txt"
    OUT = "/kaggle/working/r2"
    return [
        ("md", """# 02 · R2: calibration fine-tune for Phase 4 (+ R3 on the idle GPU)

**GPU0:** fine-tune `lerobot/smolvla_base` (paper architecture, VLM frozen) on **LIBERO-Goal task 0 only**,
saving a checkpoint every `SAVE_FREQ` steps; then each checkpoint is evaluated on task 0 (10 episodes).
This tells Phase 4 how many steps one stage needs to reach ≥ 70 % success (gate G1).

**GPU1 (while GPU0 trains):**
1. **R1b** — `lerobot/smolvla_libero`, the LeRobot checkpoint with the *paper's* architecture (16 VLM layers, 0.75× expert,
   fine-tuned from `smolvla_base`), on all 10 Goal tasks × 10 episodes with `n_action_steps = 10` (best value in paper Table 13).
2. **R3** — `HuggingFaceVLA/smolvla_libero` with `n_action_steps` = 10 and 50 on Goal tasks 0 and 9
   (R1 already has `n_action_steps` = 1), evidence for the open-loop-chunk limitation.

Run with **Save Version → Save & Run All (Commit)**, GPU T4 x2, Internet On.
Outputs in `/kaggle/working/r2/`: `summary.md`, `progress.log`, `train.log`, `eval/*/eval_info.json`, `train/checkpoints/*/pretrained_model`."""),
        ("code", """# ---- parameters ----
SUITE, TASK = "libero_goal", 0
STEPS, SAVE_FREQ, BATCH = 1500, 500, 32   # T4 ~9.4 s/step at batch 32 (no native bf16) -> ~4 h
EVAL_EPISODES = 10
EVAL_NAS = 10                  # n_action_steps for the checkpoint sweep (= Phase 4 eval setting)
FINAL_NAS1 = True              # also evaluate the last checkpoint with n_action_steps = 1
R1B = False                    # done in run 2 (88/100)
R1B_CKPT, R1B_NAS = "lerobot/smolvla_libero", 10
R3 = False                     # done in run 2
R3_TASKS, R3_NAS = [0, 9], [10, 50]
open("/tmp/params.sh", "w").write(
    f'SUITE={SUITE}\\nTASK={TASK}\\nSTEPS={STEPS}\\nSAVE_FREQ={SAVE_FREQ}\\nBATCH={BATCH}\\n'
    f'EVAL_EPISODES={EVAL_EPISODES}\\nEVAL_NAS={EVAL_NAS}\\nFINAL_NAS1={int(FINAL_NAS1)}\\nR3={int(R3)}\\n'
    f'R1B={int(R1B)}\\nR1B_CKPT={R1B_CKPT}\\nR1B_NAS={R1B_NAS}\\n'
    f'R3_TASKS="{" ".join(map(str, R3_TASKS))}"\\nR3_NAS="{" ".join(map(str, R3_NAS))}"\\n')
print(open("/tmp/params.sh").read())"""),
        *setup_cells(R, strict=True),
        ("code", f"""%%bash
{VENV}
# Episodes of each LIBERO-Goal task in lerobot/libero at a pinned revision. Episode metadata carries no task,
# so the map was built offline from the per-frame `task_index` (kaggle/libero_goal_episodes.json).
mkdir -p {OUT}
cat > /tmp/libero_goal_episodes.json <<'EOF'
{json.dumps(GOAL_MAP)}
EOF
cat > /tmp/select_episodes.py <<'EOF'
import json
from libero.libero import benchmark

p = {{k: v.strip('"') for k, v in (l.strip().split("=", 1) for l in open("/tmp/params.sh") if "=" in l)}}
m = json.load(open("/tmp/libero_goal_episodes.json"))
entry = m["tasks"][p["TASK"]]
instr = benchmark.get_benchmark_dict()[p["SUITE"]]().get_task(int(p["TASK"])).language
assert instr.strip().lower() == entry["instruction"], (instr, entry["instruction"])
print(f"instruction: {{instr}} | dataset task_index {{entry['task_index']}}")
print(f"episodes: {{len(entry['episodes'])}} | dataset revision: {{m['revision']}}")
json.dump(entry["episodes"], open("/tmp/r2_episodes.json", "w"))
open("/tmp/ds_rev", "w").write(m["revision"])
EOF
python /tmp/select_episodes.py 2>&1 | tail -3 | tee -a {R}
[ -s /tmp/r2_episodes.json ] && [ "$(cat /tmp/r2_episodes.json)" != "[]" ] || {{ echo "no episodes selected" | tee -a {R}; exit 1; }}"""),
        ("code", f"""%%bash
{VENV}
source /tmp/params.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p {OUT}/eval
# Evaluate one policy on one LIBERO task: eval_one GPU POLICY_PATH NAS TASK NAME [extra args]
cat > /tmp/eval_one.sh <<'EOF'
#!/usr/bin/env bash
{VENV}
source /tmp/params.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
GPU=$1; POL=$2; NAS=$3; T=$4; NAME=$5; shift 5
DIR={OUT}/eval/$NAME
[ -f $DIR/eval_info.json ] && exit 0
S=$(date +%s)
MUJOCO_EGL_DEVICE_ID=0 {PY} -m lerobot.scripts.lerobot_eval \\
  --policy.path=$POL --policy.device=cuda:$GPU --policy.n_action_steps=$NAS "$@" \\
  --env.type=libero --env.task=$SUITE "--env.task_ids=[$T]" --env.max_parallel_tasks=1 \\
  --eval.n_episodes=$EVAL_EPISODES --eval.batch_size=5 --seed=1000 \\
  --output_dir=$DIR > $DIR.log 2>&1
RC=$?
echo "[gpu$GPU] $NAME: exit=$RC seconds=$(( $(date +%s) - S )) $(grep -o '"pc_success": [0-9.]*' $DIR/eval_info.json 2>/dev/null | tail -1)" >> {OUT}/progress.log
[ $RC -ne 0 ] && tail -15 $DIR.log >> {OUT}/progress.log
exit 0
EOF
chmod +x /tmp/eval_one.sh
EPS=$(tr -d ' ' < /tmp/r2_episodes.json)

# GPU0: training (no env during training, so pin it to GPU0 only)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 2000 > /tmp/mem_train.txt &
MON=$!
S=$(date +%s)
CUDA_VISIBLE_DEVICES=0 {PY} -m lerobot.scripts.lerobot_train \\
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \\
  --dataset.repo_id=lerobot/libero --dataset.revision=$(cat /tmp/ds_rev) "--dataset.episodes=$EPS" \\
  {RENAME} \\
  --batch_size=$BATCH --steps=$STEPS --save_freq=$SAVE_FREQ --log_freq=100 --seed=1000 \\
  --output_dir={OUT}/train > {OUT}/train.log 2>&1 &
TRAIN=$!

# GPU1 meanwhile: R1b (paper-architecture checkpoint, needs the camera rename), then R3
(
  if [ "$R1B" = "1" ]; then
    for T in 0 1 2 3 4 5 6 7 8 9; do
      /tmp/eval_one.sh 1 $R1B_CKPT $R1B_NAS $T r1b_task${{T}}_nas${{R1B_NAS}} {RENAME}
    done
  fi
  if [ "$R3" = "1" ]; then
    for T in $R3_TASKS; do for N in $R3_NAS; do
      /tmp/eval_one.sh 1 HuggingFaceVLA/smolvla_libero $N $T r3_task${{T}}_nas${{N}}
    done; done
  fi
) &
R3PID=$!

wait $TRAIN; RC=$?
kill $MON
echo "=== TRAIN exit=$RC minutes=$(( ($(date +%s) - S) / 60 )) peakMiB=$(sort -n /tmp/mem_train.txt | tail -1)" | tee -a {R}
grep -E "step:" {OUT}/train.log | tail -3 | tee -a {R}
[ $RC -ne 0 ] && tail -30 {OUT}/train.log | tee -a {R}
[ -n "$R3PID" ] && wait $R3PID
echo "R3 done" | tee -a {R}"""),
        ("code", f"""%%bash
{VENV}
source /tmp/params.sh
# Checkpoint sweep on task 0, split over both GPUs; the last checkpoint also with n_action_steps = 1
CKS=$(ls -d {OUT}/train/checkpoints/*/pretrained_model 2>/dev/null | sort)
[ -n "$CKS" ] || {{ echo "no checkpoints" | tee -a {R}; exit 1; }}
LAST=$(echo "$CKS" | tail -1)
i=0
for CK in $CKS; do
  STEP=$(basename $(dirname $CK))
  /tmp/eval_one.sh $((i % 2)) $CK $EVAL_NAS $TASK ckpt${{STEP}}_nas${{EVAL_NAS}} {RENAME} &
  i=$((i + 1))
  [ $((i % 2)) -eq 0 ] && wait
done
wait
if [ "$FINAL_NAS1" = "1" ]; then
  /tmp/eval_one.sh 0 $LAST 1 $TASK ckpt$(basename $(dirname $LAST))_nas1 {RENAME}
fi
cat {OUT}/progress.log | tee -a {R}
# Keep weights, drop optimizer state of all but the last checkpoint (output size)
for D in $(ls -d {OUT}/train/checkpoints/*/training_state 2>/dev/null | sort | head -n -1); do rm -rf $D; done
du -sh {OUT}/train | tee -a {R}"""),
        ("code", """import json, math, re
from pathlib import Path

out = Path("/kaggle/working/r2")

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))

def row(name):
    f = out / "eval" / name / "eval_info.json"
    if not f.exists():
        return f"| {name} | failed / missing | | | |"
    o = json.load(open(f))["overall"]
    k, n = int(o["n_success"]), int(o["n_episodes"])
    lo, hi = wilson(k, n)
    return f"| {name} | {k}/{n} | {100 * k / n:.0f} | {lo:.0f}–{hi:.0f} | {o.get('eval_ep_s', float('nan')):.0f} |"

log = (out / "train.log").read_text(errors="ignore") if (out / "train.log").exists() else ""
# LeRobot log order: step … loss … data_s … updt_s … step_s
steps = re.findall(r"step:(\\S+).*?loss:(\\S+).*?data_s:(\\S+).*?step_s:(\\S+)", log)
lines = ["# R2 · smolvla_base fine-tuned on LIBERO-Goal task 0", ""]
if steps:
    dat = [float(s[2]) for s in steps]; stp = [float(s[3]) for s in steps]
    lines += [f"- train log points: {len(steps)}; last: step {steps[-1][0]}, loss {steps[-1][1]}",
              f"- mean step time {sum(stp) / len(stp):.3f} s/step (data loading {sum(dat) / len(dat):.3f} s)", ""]
lines += ["| run | success | SR % | 95% Wilson CI | s/episode |", "|---|---|---|---|---|"]
names = sorted(p.name for p in (out / "eval").iterdir() if p.is_dir())
lines += [row(n) for n in names if n.startswith("ckpt")]
lines += ["", "**R3 · official smolvla_libero, Goal (n_action_steps = 1 is in R1)**", "",
          "| run | success | SR % | 95% Wilson CI | s/episode |", "|---|---|---|---|---|"]
lines += [row(n) for n in names if n.startswith("r3_")]
r1b = [n for n in names if n.startswith("r1b_")]
if r1b:
    K = N = 0
    for n in r1b:
        f = out / "eval" / n / "eval_info.json"
        if f.exists():
            o = json.load(open(f))["overall"]; K += int(o["n_success"]); N += int(o["n_episodes"])
    lo, hi = wilson(K, N)
    lines += ["", "**R1b · lerobot/smolvla_libero (paper architecture), Goal**", "",
              "| run | success | SR % | 95% Wilson CI | s/episode |", "|---|---|---|---|---|"]
    lines += [row(n) for n in sorted(r1b, key=lambda s: int(s.split("_")[1][4:]))]
    lines += [f"| **suite** | **{K}/{N}** | **{100 * K / max(N, 1):.1f}** | **{lo:.1f}–{hi:.1f}** | |"]
(out / "summary.md").write_text("\\n".join(lines))
print("\\n".join(lines))"""),
    ]


def cl_notebook() -> list[tuple[str, str]]:
    R = "/kaggle/working/cl_report.txt"
    OUT = "/kaggle/working/cl"
    civd_src = (HERE.parent / "civd" / "civd.py").read_text()
    return [
        ("md", """# 03 · Phase 4: SeqFT vs CIVD (one continual-learning stage)

**Stage 0 (already measured, R1b):** `lerobot/smolvla_libero`, trained on all 10 LIBERO-Goal tasks.
**Stage 1 (this notebook):** fine-tune it further on **one** Goal task (`NEW_TASK`, its weakest task in R1b), then
re-evaluate the new task and old tasks to measure forgetting.

- **GPU0 · M0 SeqFT:** plain fine-tune on the new task.
- **GPU1 · M3 CIVD:** same, plus velocity distillation from the frozen stage-0 model on the *same* new-task
  observations paired with the *other* tasks' instructions (`code/civd/civd.py`). No old demonstrations are used.

Both runs use identical steps, batch, lr, seed. Training is a short burst (T4 has no native bf16, ~9 s/step at batch 32),
so the question is whether a short single-task fine-tune already damages old tasks, and whether CIVD reduces it.

Run with **Save Version → Save & Run All (Commit)**, GPU T4 x2, Internet On.
Outputs in `/kaggle/working/cl/`: `summary.md`, `progress.log`, `train_m0.log`, `train_m3.log`, `eval/*/eval_info.json`."""),
        ("code", """# ---- parameters ----
SUITE, BASE = "libero_goal", "lerobot/smolvla_libero"
NEW_TASK = 3                   # weakest Goal task in R1b (60 %): open the top drawer and put the bowl inside
OLD_TASKS = [0, 1, 8]          # old tasks at 100 % in R1b, evaluated after the stage
STEPS, BATCH = 600, 16
CIVD_LAMBDA, CIVD_DISTILL_BS = 1.0, 4
TRAIN_SEED = 1000              # training seed; evaluation always uses seed 1000 (same initial states as stage 0)
# one run per GPU: (name, mode, lambda); mode = seqft | civd (old instructions) | civdnew (M3b: new instruction)
RUNS = [("m0", "seqft", 0.0), ("m3", "civd", CIVD_LAMBDA)]
EVAL_EPISODES, NAS = 10, 10    # same as R1b
SMOKE_STEPS = 3                # short CIVD run first: distill loss must be ~0 at step 1 (student == teacher)
open("/tmp/params.sh", "w").write(
    f'SUITE={SUITE}\\nBASE={BASE}\\nNEW_TASK={NEW_TASK}\\nOLD_TASKS="{" ".join(map(str, OLD_TASKS))}"\\n'
    f'STEPS={STEPS}\\nBATCH={BATCH}\\nCIVD_LAMBDA={CIVD_LAMBDA}\\nCIVD_DISTILL_BS={CIVD_DISTILL_BS}\\n'
    f'EVAL_EPISODES={EVAL_EPISODES}\\nNAS={NAS}\\nSMOKE_STEPS={SMOKE_STEPS}\\nTRAIN_SEED={TRAIN_SEED}\\n'
    + "".join(f'RUN{g}="{n} {m} {l}"\\n' for g, (n, m, l) in enumerate(RUNS)))
print(open("/tmp/params.sh").read())"""),
        *setup_cells(R, strict=True),
        ("code", f"""%%bash
{VENV}
mkdir -p {OUT}/eval /tmp/cl
cat > /tmp/cl/civd.py <<'CIVD_EOF'
{civd_src}CIVD_EOF
cat > /tmp/libero_goal_episodes.json <<'EOF'
{json.dumps(GOAL_MAP)}
EOF
cat > /tmp/select.py <<'EOF'
import json
from libero.libero import benchmark

p = {{k: v.strip('"') for k, v in (l.strip().split("=", 1) for l in open("/tmp/params.sh") if "=" in l)}}
m = json.load(open("/tmp/libero_goal_episodes.json"))
suite = benchmark.get_benchmark_dict()[p["SUITE"]]()
for t, e in m["tasks"].items():
    assert suite.get_task(int(t)).language.strip().lower() == e["instruction"], t
new = m["tasks"][p["NEW_TASK"]]
old = [e["instruction"] for t, e in m["tasks"].items() if t != p["NEW_TASK"]]
print(f"new task: {{new['instruction']}} | {{len(new['episodes'])}} episodes | revision {{m['revision']}}")
print(f"distillation instructions: {{len(old)}} other Goal tasks")
json.dump(new["episodes"], open("/tmp/cl_episodes.json", "w"))
# experience replay variants: new-task episodes + the first K episodes of every old Goal task
for k in (1, 5):
    er = sorted(new["episodes"] + [ep for t, e in m["tasks"].items() if t != p["NEW_TASK"] for ep in e["episodes"][:k]])
    json.dump(er, open(f"/tmp/cl_episodes_er{{k}}.json", "w"))
    print(f"ER-{{k}}: {{len(er)}} episodes ({{len(er) - len(new['episodes'])}} replayed from old tasks)")
open("/tmp/ds_rev", "w").write(m["revision"])
open("/tmp/old_tasks", "w").write("|".join(old))
EOF
python /tmp/select.py 2>&1 | tail -5 | tee -a {R}
[ -s /tmp/cl_episodes.json ] || {{ echo "no episodes selected" | tee -a {R}; exit 1; }}
{PY} -m py_compile /tmp/cl/civd.py && echo "civd.py: ok" | tee -a {R}"""),
        ("code", f"""%%bash
{VENV}
source /tmp/params.sh
# Train one method on one GPU: train_one GPU NAME STEPS [civd]
cat > /tmp/train_one.sh <<'EOF'
#!/usr/bin/env bash
{VENV}
source /tmp/params.sh
GPU=$1; NAME=$2; N=$3; MODE=$4; LAM=${{5:-$CIVD_LAMBDA}}
# erK = SeqFT with K replayed episodes per old task; civderK = CIVD plus the same replay
EPF=/tmp/cl_episodes.json
case "$MODE" in
  er*) EPF=/tmp/cl_episodes_${{MODE}}.json; MODE=seqft ;;
  civder*) EPF=/tmp/cl_episodes_er${{MODE#civder}}.json; MODE=civd ;;
esac
EPS=$(tr -d ' ' < $EPF)
if [ "$MODE" = "civd" ] || [ "$MODE" = "civdnew" ]; then
  ENTRY="/tmp/cl/civd.py"
  CM=old; [ "$MODE" = "civdnew" ] && CM=new
  export CIVD_TEACHER=$BASE CIVD_OLD_TASKS="$(cat /tmp/old_tasks)" CIVD_LAMBDA=$LAM CIVD_DISTILL_BS=$CIVD_DISTILL_BS CIVD_MODE=$CM
  [ "$N" -le 10 ] && export CIVD_LOG_EVERY=1 || export CIVD_LOG_EVERY=25
else
  ENTRY="-m lerobot.scripts.lerobot_train"
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU -lms 2000 > /tmp/mem_$NAME.txt &
MON=$!
S=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU {PY} $ENTRY \\
  --policy.path=$BASE --policy.device=cuda --policy.push_to_hub=false --policy.train_expert_only=true \\
  --dataset.repo_id=lerobot/libero --dataset.revision=$(cat /tmp/ds_rev) "--dataset.episodes=$EPS" \\
  {RENAME} \\
  --batch_size=$BATCH --steps=$N --save_freq=$N --log_freq=25 --seed=$TRAIN_SEED \\
  --output_dir=/tmp/cl/train_$NAME > {OUT}/train_$NAME.log 2>&1
RC=$?
kill $MON
echo "[gpu$GPU] train $NAME: exit=$RC steps=$N minutes=$(( ($(date +%s) - S) / 60 )) peakMiB=$(sort -n /tmp/mem_$NAME.txt | tail -1)" >> {OUT}/progress.log
[ $RC -ne 0 ] && tail -30 {OUT}/train_$NAME.log >> {OUT}/progress.log
exit $RC
EOF
chmod +x /tmp/train_one.sh

# Smoke: a few CIVD steps; checks the launcher, teacher loading, memory, and distill ~0 at step 1
/tmp/train_one.sh 1 smoke $SMOKE_STEPS civd
RC=$?
grep "\\[civd\\]" {OUT}/train_smoke.log | tee -a {R}
tail -2 {OUT}/progress.log | tee -a {R}
rm -rf /tmp/cl/train_smoke
[ $RC -eq 0 ] || {{ echo "CIVD smoke failed; M0 only" | tee -a {R}; touch /tmp/civd_failed; }}"""),
        ("code", f"""%%bash
{VENV}
source /tmp/params.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cat > /tmp/eval_one.sh <<'EOF'
#!/usr/bin/env bash
{VENV}
source /tmp/params.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
GPU=$1; POL=$2; T=$3; NAME=$4
DIR={OUT}/eval/$NAME
[ -f $DIR/eval_info.json ] && exit 0
S=$(date +%s)
MUJOCO_EGL_DEVICE_ID=0 {PY} -m lerobot.scripts.lerobot_eval \\
  --policy.path=$POL --policy.device=cuda:$GPU --policy.n_action_steps=$NAS {RENAME} \\
  --env.type=libero --env.task=$SUITE "--env.task_ids=[$T]" --env.max_parallel_tasks=1 \\
  --eval.n_episodes=$EVAL_EPISODES --eval.batch_size=5 --seed=1000 \\
  --output_dir=$DIR > $DIR.log 2>&1
RC=$?
echo "[gpu$GPU] eval $NAME: exit=$RC seconds=$(( $(date +%s) - S )) $(grep -o '"pc_success": [0-9.]*' $DIR/eval_info.json 2>/dev/null | tail -1)" >> {OUT}/progress.log
[ $RC -ne 0 ] && tail -15 $DIR.log >> {OUT}/progress.log
exit 0
EOF
chmod +x /tmp/eval_one.sh

# Each GPU: train its method, then evaluate it on the new task first, then the old tasks
run() {{
  GPU=$1; NAME=$2; MODE=$3; LAM=$4
  /tmp/train_one.sh $GPU $NAME $STEPS $MODE $LAM || return
  CK=$(ls -d /tmp/cl/train_$NAME/checkpoints/*/pretrained_model | sort | tail -1)
  for T in $NEW_TASK $OLD_TASKS; do /tmp/eval_one.sh $GPU $CK $T ${{NAME}}_task$T; done
}}
START=$(date +%s)
for G in 0 1; do
  V=RUN$G; read NAME MODE LAM <<< "${{!V}}"
  [ -z "$NAME" ] && continue
  if [ "$MODE" != "seqft" ] && [ -f /tmp/civd_failed ]; then continue; fi
  run $G $NAME $MODE $LAM &
done
wait
echo "=== CL done in $(( ($(date +%s) - START) / 60 )) min" | tee -a {R}
for f in {OUT}/train_*.log; do grep "\\[civd\\]" $f | tail -2 | sed "s|^|$(basename $f): |"; done | tee -a {R}
cat {OUT}/progress.log | tee -a {R}"""),
        ("code", """import json, math, re
from pathlib import Path

out = Path("/kaggle/working/cl")
p = {k: v.strip('"') for k, v in (l.strip().split("=", 1) for l in open("/tmp/params.sh") if "=" in l)}
new, old = int(p["NEW_TASK"]), [int(t) for t in p["OLD_TASKS"].split()]
runs = [p[k].split() for k in ("RUN0", "RUN1") if p.get(k)]
names = [r[0] for r in runs]
labels = {r[0]: f"{r[0]} {r[1]}" + (f" λ={r[2]}" if "civd" in r[1] else "") for r in runs}
# Stage 0 = R1b (lerobot/smolvla_libero, n_action_steps 10, seed 1000, 10 episodes), measured in 02_r2_calibration
R1B = {0: 10, 1: 10, 2: 8, 3: 6, 4: 9, 5: 10, 6: 7, 7: 9, 8: 10, 9: 9}

def wilson(k, n, z=1.96):
    q = k / n; d = 1 + z * z / n; c = (q + z * z / (2 * n)) / d
    h = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d
    return f"{100 * (c - h):.0f}–{100 * (c + h):.0f}"

def sr(name):
    f = out / "eval" / name / "eval_info.json"
    if not f.exists():
        return None
    o = json.load(open(f))["overall"]
    return int(o["n_success"]), int(o["n_episodes"])

lines = [f"# Phase 4 · one CL stage on LIBERO-Goal · base {p['BASE']} · new task {new} · {p['STEPS']} steps × batch {p['BATCH']} · train seed {p['TRAIN_SEED']}", "",
         "| task | role | stage 0 (R1b) | " + " | ".join(labels[n] for n in names) + " |", "|---|---|---|" + "---|" * len(names)]
summary = {}
for t in [new] + old:
    cells = []
    for m in names:
        r = sr(f"{m}_task{t}")
        cells.append("missing" if r is None else f"{r[0]}/{r[1]} ({wilson(*r)})")
        summary[(m, t)] = r
    lines.append(f"| {t} | {'new' if t == new else 'old'} | {R1B[t]}/10 | " + " | ".join(cells) + " |")
lines += ["", "| method | new-task gain (pp) | mean forgetting on old tasks (pp) |", "|---|---|---|"]
for m, label in [(n, labels[n]) for n in names]:
    rn = summary.get((m, new))
    ro = [summary.get((m, t)) for t in old]
    gain = f"{10 * (rn[0] * 10 / rn[1] - R1B[new]):+.0f}" if rn else "n/a"
    fg = [10 * (R1B[t] - r[0] * 10 / r[1]) for t, r in zip(old, ro) if r]
    lines.append(f"| {label} | {gain} | {sum(fg) / len(fg):.0f} ({len(fg)} tasks) |" if fg else f"| {label} | {gain} | n/a |")
for m in names:
    log = (out / f"train_{m}.log").read_text(errors="ignore") if (out / f"train_{m}.log").exists() else ""
    st = re.findall(r"step:(\\S+).*?loss:(\\S+).*?updt_s:(\\S+).*?mem_gb:(\\S+)", log)
    if st:
        u = [float(s[2]) for s in st]
        lines.append(f"- {m}: last loss {st[-1][1]} at step {st[-1][0]}; update {sum(u) / len(u):.2f} s/step; mem {st[-1][3]} GB")
(out / "summary.md").write_text("\\n".join(lines))
print("\\n".join(lines))"""),
    ]


def write(name: str, cells: list[tuple[str, str]]) -> None:
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(cells):
        lines = src.split("\n")
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i}", "metadata": {},
                "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    path = HERE / name / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(HERE.parent)} ({len(cells)} cells)")


if __name__ == "__main__":
    write("00_smoke_test", smoke_notebook())
    write("01_r1_eval", r1_notebook())
    write("02_r2_calibration", r2_notebook())
    write("03_cl", cl_notebook())
    # Run 2: longer stage and every old Goal task (closest to the new task first, so partial results are useful)
    run2 = [(k, c.replace("STEPS, BATCH = 600, 16", "STEPS, BATCH = 1200, 16")
                 .replace("OLD_TASKS = [0, 1, 8]          # old tasks at 100 % in R1b, evaluated after the stage",
                          "OLD_TASKS = [8, 4, 0, 1, 5, 7, 9, 2, 6]   # all old Goal tasks; bowl tasks (8, 4) first"))
            for k, c in cl_notebook()]
    assert run2 != cl_notebook()
    write("03_cl_run2", run2)
    # Seeds 2 and 3 of run 2 (training seed only; evaluation seed stays 1000)
    for k, seed in [("03_cl_seed2", 2000), ("03_cl_seed3", 3000)]:
        nb = [(a, c.replace("TRAIN_SEED = 1000 ", f"TRAIN_SEED = {seed} ")) for a, c in run2]
        assert nb != run2
        write(k, nb)
    # Ablations at run-2 settings: CIVD with lambda 0.3 (GPU0) and M3b = distillation with the new instruction (GPU1)
    abl = [(a, c.replace('RUNS = [("m0", "seqft", 0.0), ("m3", "civd", CIVD_LAMBDA)]',
                         'RUNS = [("m3l03", "civd", 0.3), ("m3b", "civdnew", 1.0)]')) for a, c in run2]
    assert abl != run2
    write("03_cl_ablation", abl)
    # Replay baselines at run-2 settings: M2 ER (5 episodes per old task, GPU0) and M4 CIVD + ER-tiny (1 per old task, GPU1)
    er = [(a, c.replace('RUNS = [("m0", "seqft", 0.0), ("m3", "civd", CIVD_LAMBDA)]',
                        'RUNS = [("m2", "er5", 0.0), ("m4", "civder1", CIVD_LAMBDA)]')) for a, c in run2]
    assert er != run2
    write("03_cl_er", er)
    # Plasticity sweep at run-2 settings: CIVD with lambda 0.1 (GPU0) and 0.03 (GPU1)
    lam = [(a, c.replace('RUNS = [("m0", "seqft", 0.0), ("m3", "civd", CIVD_LAMBDA)]',
                         'RUNS = [("m3l01", "civd", 0.1), ("m3l003", "civd", 0.03)]')) for a, c in run2]
    assert lam != run2
    write("03_cl_lambda", lam)
    # Second training seed (2000) for the two key comparisons: M3b (GPU0) and M4 CIVD + ER-tiny (GPU1)
    s2x = [(a, c.replace('RUNS = [("m0", "seqft", 0.0), ("m3", "civd", CIVD_LAMBDA)]',
                         'RUNS = [("m3b", "civdnew", 1.0), ("m4", "civder1", CIVD_LAMBDA)]')
                .replace("TRAIN_SEED = 1000 ", "TRAIN_SEED = 2000 ")) for a, c in run2]
    assert sum(x != y for x, y in zip(s2x, run2)) >= 1 and "TRAIN_SEED = 2000" in "".join(c for _, c in s2x)
    write("03_cl_seed2_ext", s2x)
