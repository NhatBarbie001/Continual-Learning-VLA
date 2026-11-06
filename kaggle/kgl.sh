#!/usr/bin/env bash
# Drive Kaggle kernels from the local terminal.
# Usage: bash code/kaggle/kgl.sh {whoami|quota|push|status|logs|output} [kernel-folder]
#   push    <dir> [sec]  fill username into kernel-metadata.json and run on GPU T4 x2
#   status  <dir>  state of the latest run
#   logs    <dir>  execution log of the latest run
#   output  <dir>  download outputs to results/<dir-name>/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root (this file is code/kaggle/kgl.sh)
export PYTHONIOENCODING=utf-8

user() { kaggle config view 2>/dev/null | sed -n 's/^- username: //p' | head -1; }

slug() {
  python -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" "$1/kernel-metadata.json"
}

prepare() {
  local dir="$1" u
  u="$(user)"
  [ -n "$u" ] || { echo "Not logged in: run 'kaggle auth login' first." >&2; exit 1; }
  sed "s/__USER__/$u/" "$dir/kernel-metadata.template.json" > "$dir/kernel-metadata.json"
}

cmd="${1:-}"; dir="${2:-}"
case "$cmd" in
  whoami) user ;;
  quota)  kaggle quota ;;
  push)   # optional 3rd arg: max run time in seconds (caps quota use)
          prepare "$dir"
          kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4 ${3:+-t "$3"} ;;
  status) prepare "$dir"; kaggle kernels status "$(slug "$dir")" ;;
  logs)   prepare "$dir"; kaggle kernels logs "$(slug "$dir")" ;;
  output)
    prepare "$dir"
    out="$ROOT/results/$(basename "$dir")"; mkdir -p "$out"
    kaggle kernels output "$(slug "$dir")" -p "$out"
    echo "Saved to $out" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
