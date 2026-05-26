# Phase 4 Execution Simulator V2 Foundation

Status: implemented foundation slice

## What Changed

- Added `liqpool/execution_simulator_v2.py`.
- Added optional CLI flags in `examples/multi_asset_run.py`:
  - `--run-execution-backtest-v2`
  - `--fill-policy generous|neutral|conservative`
  - `--use-1m-resolution`
  - `--raw-1m-dir`
  - `--slippage-model state_dependent|flat`
  - `--v2-base-slippage-bps`
  - `--execution-exchange NSE|BSE`
- Added v2 artifacts when the v2 run flag is enabled:
  - `execution_backtest_v2_summary.csv`
  - `execution_backtest_v2_trades.csv`
  - `execution_backtest_v2_vs_v1_delta.csv`
- Added focused simulator tests in `tests/test_execution_simulator_v2.py`.

## Simulator Capabilities

The v2 foundation now supports:

- 1-minute stop/target path resolution, with conservative stop-first handling when a single 1-minute candle contains both stop and target.
- Adverse-selection-aware fill policy for blind limit entries:
  - `generous`: touch fills immediately.
  - `neutral`: requires price to trade through the level by 0.10 ATR.
  - `conservative`: requires 0.15 ATR through-trade and can worsen fills.
- Confirmation entries now use next-bar entry semantics in v2 instead of same-bar close execution.
- State-dependent slippage that increases near liquidity levels and during open/close session phases.
- Itemized Zerodha intraday equity costs:
  - brokerage buy/sell/total
  - STT
  - exchange transaction charge
  - SEBI charge
  - stamp duty
  - GST
  - total cost bps
  - breakeven move
- Pre-touch directional simulator hook for Track A.

## Guardrails

- V2 is optional and does not change the existing v1 execution reports unless `--run-execution-backtest-v2` is passed.
- If `--use-1m-resolution` is enabled and `--raw-1m-dir` is omitted, the CLI tries the sibling `raw_1m` directory next to `--data-dir`.
- If raw 1-minute data is unavailable, the v2 simulator falls back to 5-minute resolution for that symbol.
- Strategy profitability is still unproven. This is execution infrastructure, not a green light for live trading.

## Validation Run

Commands run:

```bash
python3 -m compileall liqpool examples tests
.venv/bin/python -m unittest tests.test_execution_simulator_v2 -v
.venv/bin/python -m unittest discover -s tests/leakage -v
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py --help
git diff --check
```

Result:

- Compile: pass
- Execution simulator v2 tests: 5 passed
- Leakage fast tests: 4 passed
- CLI parse check: pass
- Diff whitespace check: pass

## Smoke Test

After the foundation commit, a two-symbol smoke run exposed an important v2
geometry bug: next-bar confirmation entries could occur after price had already
moved beyond the valid stop/target geometry, collapsing risk to near zero and
printing impossible huge R values. The simulator now rejects those invalid
trades, and a regression test covers the failure mode.

Smoke command shape:

```bash
PYTHONPATH=. .venv/bin/python -u examples/multi_asset_run.py \
  --symbols HDFCBANK,TCS \
  --data-source parquet \
  --data-dir ".../kite_indian_market_data/resampled" \
  --mode train \
  --model-dir output_models/phase4_v2_smoke \
  --folds 2 \
  --iters 1 \
  --final-iters 1 \
  --asset-workers 1 \
  --checkpoint-dir output_checkpoints/phase4_v2_smoke \
  --resume \
  --regularization-preset conservative_finml \
  --feature-store-dir output_feature_store/phase4_v2_smoke \
  --run-execution-backtest-v2 \
  --use-1m-resolution \
  --raw-1m-dir ".../kite_indian_market_data/raw_1m" \
  --fill-policy neutral \
  --slippage-model state_dependent \
  --skip-policy-model \
  --out output_phase4_v2_smoke
```

Corrected v2 smoke result:

| mode | trades | win rate | net R | profit factor |
|---|---:|---:|---:|---:|
| displacement_confirmed | 505 | 40.6% | -0.65 | 0.25 |
| touch_confirmed | 1366 | 33.7% | -0.73 | 0.20 |
| blind_limit | 1629 | 16.7% | -1.27 | 0.10 |
| reclaim_confirmed | 625 | 13.4% | -1.45 | 0.06 |

V2 is stricter than v1 on this smoke sample, as expected:

| mode | v1 net R | v2 net R | delta |
|---|---:|---:|---:|
| blind_limit | -0.94 | -1.27 | -0.33 |
| displacement_confirmed | -0.47 | -0.65 | -0.17 |
| reclaim_confirmed | -1.07 | -1.45 | -0.38 |
| touch_confirmed | -0.59 | -0.73 | -0.14 |

## Next Required Step

Run the full Core25 v2 baseline across all four existing execution modes under
the neutral fill policy. Then repeat with generous and conservative fill policies
before trusting any Track A or Track B sweep.
