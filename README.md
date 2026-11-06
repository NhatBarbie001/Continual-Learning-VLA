# Code

This folder holds all the code behind the report: the CIVD method, ready-to-run scripts for every experiment, the analysis scripts that produce the tables, and the Kaggle notebooks the reported numbers came from.

```
code/
├── civd/civd.py              CIVD (Part 4): drop-in launcher for LeRobot's lerobot_train (≈140 lines)
├── scripts/                  run any experiment on a Linux machine with an NVIDIA GPU (no Kaggle needed)
│   ├── common.sh             shared settings (paths, pinned LeRobot commit, camera rename map)
│   ├── 00_setup_env.sh       install system libs, Python 3.12 venv, LeRobot @ ba83e4c with [smolvla,libero]
│   ├── 01_eval_libero.sh     Part 2: evaluate a checkpoint on a LIBERO suite (R1, R1b, R3)
│   ├── 02_train_cl.sh        Part 4: one CL stage with SeqFT / CIVD / M3b / ER / CIVD+ER
│   ├── 03_eval_cl.sh         Part 4: evaluate CL checkpoints on the new task and all old tasks
│   └── run_cl.sh             Part 4: 02 + 03 end to end
├── analysis/
│   ├── select_episodes.py    training episode list (new task, plus K replayed demos per old task)
│   ├── summarize_eval.py     per-task success and Wilson 95 % CI for one suite
│   ├── summarize_cl.py       one CL stage: success per task, new-task gain, forgetting vs stage 0
│   ├── compare_runs.py       pool runs across seeds; Fisher exact tests; forgetting mean ± std
│   └── make_tables.py        results/tables/*.csv and results/figures/cl_per_task.png
└── kaggle/                   the notebooks that produced every number in the report (2× T4)
    ├── build_notebooks.py    generates all notebooks below (single source of truth)
    ├── kgl.sh                push / status / logs / output around the Kaggle CLI
    ├── libero_goal_episodes.json   LIBERO-Goal task → episodes in lerobot/libero @ a1aaacb
    ├── 00_smoke_test         install + 1-episode eval + 50 training steps
    ├── 01_r1_eval            R1 / R1b on LIBERO suites
    ├── 02_r2_calibration     R1b on Goal, R3 (n_action_steps 1/10/50), fine-tuning cost
    ├── 03_cl, 03_cl_run2     CL run 1 (600 steps) and run 2 (1200 steps, main result)
    ├── 03_cl_seed2, _seed3   run 2 with training seeds 2000, 3000
    ├── 03_cl_ablation        CIVD λ = 0.3 and M3b
    ├── 03_cl_er              M2 ER-5 and M4 CIVD + ER-1
    ├── 03_cl_lambda          CIVD λ = 0.1 and 0.03
    └── 03_cl_seed2_ext       M3b and M4 with training seed 2000
```

## Which code belongs to which part

| Part of the assignment | Code | Output in `results/` |
|---|---|---|
| 2 · Reproduce SmolVLA on LIBERO | `scripts/01_eval_libero.sh`, `kaggle/01_r1_eval`, `kaggle/02_r2_calibration` | `r1/`, `r2_run2_kaggle/`, `r1b_rerun_kaggle/` |
| 3 · Analyse (failure taxonomy) | labels in `results/failure_labels.csv`, read from the eval videos | `phase3_failures.md` |
| 4 · Improve (CIVD and baselines) | `civd/civd.py`, `scripts/02–03`, `run_cl.sh`, `kaggle/03_cl*` | `cl_*_kaggle/`, `CL_SEEDS_SUMMARY.md` |
| Tables and figure | `analysis/*` | `tables/`, `figures/` |

## Option A: run on your own GPU machine

Requirements: Linux, an NVIDIA GPU with ≥ 16 GB memory (CIVD peaks at 7.4 GB on a T4), and internet access to the Hugging Face Hub.

```bash
bash code/scripts/00_setup_env.sh                                       # ≈ 10 min, venv in /tmp/venv

# Part 2 · reproduce (≈ 1–1.5 h per suite on one T4)
bash code/scripts/01_eval_libero.sh lerobot/smolvla_libero libero_goal  # R1b, paper configuration
#   → outputs/eval/smolvla_libero_nas10/libero_goal/summary.md

# Part 4 · one CL stage (base model → new Goal task 3, 1200 steps)
bash code/scripts/run_cl.sh seqft m0                                    # SeqFT, ≈ 1.6 h + 1 h eval
bash code/scripts/run_cl.sh civd  m3 1.0                                # CIVD,  ≈ 3 h + 1 h eval
python code/analysis/summarize_cl.py outputs/cl --runs m0 m3            # table for both runs
```

- **Run two methods at once on two GPUs:** `GPU=0 bash code/scripts/run_cl.sh seqft m0 & GPU=1 bash code/scripts/run_cl.sh civd m3 1.0 &`.
- **Change the settings with environment variables:** `SEED`, `STEPS`, `BATCH`, `NEW_TASK`, `OLD_TASKS`, `EPISODES`, `OUT_ROOT`.
- **Full list of reported runs:** in the header of `scripts/run_cl.sh`.

## Option B: rerun the exact Kaggle notebooks

```bash
pip install kaggle                                   # plus an API token in ~/.kaggle/
python code/kaggle/build_notebooks.py                # regenerate notebooks (optional, they are committed)
bash code/kaggle/kgl.sh push code/kaggle/03_cl_run2 43200
bash code/kaggle/kgl.sh status code/kaggle/03_cl_run2
bash code/kaggle/kgl.sh output code/kaggle/03_cl_run2      # → results/03_cl_run2/
```

The account needs a verified phone number, otherwise Kaggle runs the notebook without GPU and internet. Each notebook installs its own environment, runs one method per T4 in parallel, and writes `summary.md`.

## Reproduce the tables from the downloaded results

```bash
python code/analysis/make_tables.py                                    # results/tables/*.csv, results/figures/cl_per_task.png
python code/analysis/compare_runs.py results/cl_run2_kaggle results/cl_seed2_kaggle results/cl_seed3_kaggle
#   → SeqFT 154/270 vs CIVD 228/270, forgetting 34.1 ± 2.8 vs 6.7 ± 2.9 pp, Fisher p = 2.5e-12
python code/analysis/compare_runs.py results/cl_ablation_kaggle --a m3b --b l03
```

## Using CIVD directly

```bash
export CIVD_TEACHER=lerobot/smolvla_libero          # frozen previous-stage model
export CIVD_OLD_TASKS="open the middle drawer of the cabinet|put the bowl on the stove|..."
export CIVD_LAMBDA=1.0 CIVD_DISTILL_BS=4 CIVD_MODE=old   # CIVD_MODE=new → M3b ablation
python code/civd/civd.py <any lerobot_train arguments>
```

CIVD patches `SmolVLAPolicy.forward` to add `λ · ‖v_student − v_teacher‖²`. Both velocities are computed on the current batch's observations paired with old instructions, at an `x_t` built from an action chunk the teacher samples. At step 1 the distillation loss is exactly 0, because the student equals the teacher. That is the built-in sanity check.

## Notes

- **Pins:**
  - LeRobot `ba83e4cf43737e2665be0aafcb5f513b83c7d0f3`
  - dataset `lerobot/libero` @ `a1aaacb`
  - evaluation seed 1000 everywhere (fixed LIBERO initial states)
- **Camera names:** `lerobot/smolvla_libero` expects `camera1` / `camera2`, so the scripts pass the `--rename_map` from `common.sh`.
- **Rendering:** MuJoCo renders through EGL on device 0, and the policy goes on `cuda:$GPU`. On Kaggle only EGL device 0 works.
- **Episode map:** `libero_goal_episodes.json` was built once from the per-frame `task_index` of the dataset. The notebooks assert every instruction against LIBERO's task list.
- **Testing status:**
  - The notebooks in `kaggle/` are what produced the reported numbers.
  - `scripts/` reproduces the same commands outside Kaggle. Their syntax and episode selection were checked locally, but they have not been run end to end on a non-Kaggle machine.
  - The analysis scripts were run on the downloaded results.
