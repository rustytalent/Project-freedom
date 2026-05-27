#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data}"
RESAMPLED_DIR="${RESAMPLED_DIR:-$DATA_ROOT/resampled}"
RAW_1M_DIR="${RAW_1M_DIR:-$DATA_ROOT/raw_1m}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ASSET_WORKERS="${ASSET_WORKERS:-2}"
CORE25_CHECKPOINT_DIR="${CORE25_CHECKPOINT_DIR:-output_checkpoints/core25}"
LOG_DIR="${LOG_DIR:-logs}"

usage() {
  cat <<'EOF'
Phase 4 V2 terminal runbook

Usage:
  scripts/phase4_v2_runbook.sh setup
  scripts/phase4_v2_runbook.sh smoke
  scripts/phase4_v2_runbook.sh neutral
  scripts/phase4_v2_runbook.sh generous
  scripts/phase4_v2_runbook.sh conservative
  scripts/phase4_v2_runbook.sh compare
  scripts/phase4_v2_runbook.sh status
  scripts/phase4_v2_runbook.sh all

Environment overrides:
  DATA_ROOT=/path/to/kite_indian_market_data
  RESAMPLED_DIR=/path/to/resampled
  RAW_1M_DIR=/path/to/raw_1m
  ASSET_WORKERS=2
  NO_CAFFEINATE=1

Recommended order:
  setup -> smoke -> neutral -> generous -> conservative -> compare
EOF
}

require_paths() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found/executable: $PYTHON_BIN" >&2
    exit 1
  fi
  if [[ ! -d "$RESAMPLED_DIR" ]]; then
    echo "ERROR: RESAMPLED_DIR not found: $RESAMPLED_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$RAW_1M_DIR" ]]; then
    echo "ERROR: RAW_1M_DIR not found: $RAW_1M_DIR" >&2
    exit 1
  fi
}

run_with_caffeinate() {
  mkdir -p "$LOG_DIR"
  if [[ "${NO_CAFFEINATE:-0}" == "1" ]] || ! command -v caffeinate >/dev/null 2>&1; then
    env PYTHONPATH=. "$@"
  else
    caffeinate -dimsu env PYTHONPATH=. "$@"
  fi
}

show_setup() {
  echo "Project:       $PROJECT_DIR"
  echo "Branch:        $(git branch --show-current)"
  echo "Latest commit: $(git log --oneline -1)"
  echo "Python:        $PYTHON_BIN"
  echo "DATA_ROOT:     $DATA_ROOT"
  echo "RESAMPLED_DIR: $RESAMPLED_DIR"
  echo "RAW_1M_DIR:    $RAW_1M_DIR"
  echo
  require_paths
  echo "[resampled]"
  ls -lh "$RESAMPLED_DIR"
  echo
  echo "[raw_1m sample]"
  ls -lh "$RAW_1M_DIR" | head
  echo
  echo "[v2 flags]"
  PYTHONPATH=. "$PYTHON_BIN" examples/multi_asset_run.py --help \
    | grep -E "run-execution-backtest-v2|use-1m-resolution|fill-policy|slippage-model" || true
}

run_smoke() {
  require_paths
  run_with_caffeinate "$PYTHON_BIN" -u examples/multi_asset_run.py \
    --symbols HDFCBANK,TCS \
    --data-source parquet \
    --data-dir "$RESAMPLED_DIR" \
    --mode train \
    --model-dir output_models/phase4_v2_smoke_user \
    --folds 2 \
    --iters 1 \
    --final-iters 1 \
    --asset-workers 1 \
    --checkpoint-dir output_checkpoints/phase4_v2_smoke_user \
    --resume \
    --regularization-preset conservative_finml \
    --feature-store-dir output_feature_store/phase4_v2_smoke_user \
    --run-execution-backtest-v2 \
    --use-1m-resolution \
    --raw-1m-dir "$RAW_1M_DIR" \
    --fill-policy neutral \
    --slippage-model state_dependent \
    --skip-policy-model \
    --out output_phase4_v2_smoke_user \
    2>&1 | tee "$LOG_DIR/phase4_v2_smoke_user.log"
}

run_core25_policy() {
  local policy="$1"
  local out_dir="output_core25_phase4_v2_${policy}"
  local model_dir="output_models/core25_phase4_v2_${policy}"
  local fs_dir="output_feature_store/core25_phase4_v2_${policy}"
  local log_file="$LOG_DIR/core25_phase4_v2_${policy}.log"

  require_paths
  run_with_caffeinate "$PYTHON_BIN" -u examples/multi_asset_run.py \
    --universe core25 \
    --data-source parquet \
    --data-dir "$RESAMPLED_DIR" \
    --mode train \
    --model-dir "$model_dir" \
    --asset-workers "$ASSET_WORKERS" \
    --checkpoint-dir "$CORE25_CHECKPOINT_DIR" \
    --resume \
    --regularization-preset conservative_finml \
    --feature-store-dir "$fs_dir" \
    --run-execution-backtest-v2 \
    --use-1m-resolution \
    --raw-1m-dir "$RAW_1M_DIR" \
    --fill-policy "$policy" \
    --slippage-model state_dependent \
    --skip-policy-model \
    --out "$out_dir" \
    2>&1 | tee "$log_file"
}

show_status() {
  ps aux | grep multi_asset_run | grep -v grep || true
  echo
  for d in output_core25_phase4_v2_neutral output_core25_phase4_v2_generous output_core25_phase4_v2_conservative; do
    if [[ -d "$d" ]]; then
      echo "[$d]"
      ls -lh "$d" | tail -20
      echo
    fi
  done
}

show_compare() {
  for policy in neutral generous conservative; do
    local fp="output_core25_phase4_v2_${policy}/execution_backtest_v2_summary.csv"
    local label
    label="$(printf '%s' "$policy" | tr '[:lower:]' '[:upper:]')"
    echo
    echo "===== ${label} ====="
    if [[ -f "$fp" ]]; then
      column -s, -t < "$fp"
    else
      echo "Missing: $fp"
    fi
  done
}

cmd="${1:-}"
case "$cmd" in
  setup) show_setup ;;
  smoke) run_smoke ;;
  neutral) run_core25_policy neutral ;;
  generous) run_core25_policy generous ;;
  conservative) run_core25_policy conservative ;;
  status) show_status ;;
  compare) show_compare ;;
  all)
    show_setup
    run_smoke
    run_core25_policy neutral
    run_core25_policy generous
    run_core25_policy conservative
    show_compare
    ;;
  ""|-h|--help|help) usage ;;
  *)
    echo "ERROR: unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
