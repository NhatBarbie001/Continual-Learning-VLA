#!/usr/bin/env bash
# Part 4, end to end: train one CL stage with one method, then evaluate it.
#
# Usage: bash code/scripts/run_cl.sh METHOD NAME [LAMBDA]       (same arguments as 02_train_cl.sh)
#
# The runs reported (each ≈ 1.6 h for SeqFT / ER, ≈ 3 h for CIVD on one T4, plus ≈ 1 h evaluation):
#   main result, 3 seeds:  for s in 1000 2000 3000; do SEED=$s bash code/scripts/run_cl.sh seqft m0_s$s;
#                                                        SEED=$s bash code/scripts/run_cl.sh civd  m3_s$s 1.0; done
#   ablation:              bash code/scripts/run_cl.sh civd    m3l03 0.3
#                          bash code/scripts/run_cl.sh civdnew m3b   1.0
#   replay baselines:      bash code/scripts/run_cl.sh er5     m2
#                          bash code/scripts/run_cl.sh civder1 m4    1.0
#   λ sweep:               bash code/scripts/run_cl.sh civd    m3l01 0.1;  bash code/scripts/run_cl.sh civd m3l003 0.03
# Two GPUs: run two methods in parallel with GPU=0 / GPU=1 (EGL rendering stays on device 0).
set -euo pipefail
DIR="$(dirname "$0")"
bash "$DIR/02_train_cl.sh" "$@"
bash "$DIR/03_eval_cl.sh" "$2"
