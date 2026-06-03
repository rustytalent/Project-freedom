# Phase 4 V2 Terminal Runbook

Use the helper script so the long v2 commands stay consistent.

## Start

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
git checkout claude/liquidity-pool-backtester-1uskb
git pull origin claude/liquidity-pool-backtester-1uskb
```

## Verify Paths And Flags

```bash
scripts/phase4_v2_runbook.sh setup
```

The script defaults to:

```bash
DATA_ROOT="/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data"
RESAMPLED_DIR="$DATA_ROOT/resampled"
RAW_1M_DIR="$DATA_ROOT/raw_1m"
ASSET_WORKERS=2
```

Override them only if your data moved:

```bash
DATA_ROOT="/some/other/kite_indian_market_data" scripts/phase4_v2_runbook.sh setup
```

## Run Order

Run the two-stock smoke first:

```bash
scripts/phase4_v2_runbook.sh smoke
```

Then run the three Core25 fill policies one at a time:

```bash
scripts/phase4_v2_runbook.sh neutral
scripts/phase4_v2_runbook.sh generous
scripts/phase4_v2_runbook.sh conservative
```

Compare all three:

```bash
scripts/phase4_v2_runbook.sh compare
```

Check whether a run is still alive:

```bash
scripts/phase4_v2_runbook.sh status
```

## Output Locations

- Smoke: `output_phase4_v2_smoke_user/`
- Neutral: `output_core25_phase4_v2_neutral/`
- Generous: `output_core25_phase4_v2_generous/`
- Conservative: `output_core25_phase4_v2_conservative/`
- Logs: `logs/`

The most important files are:

```bash
execution_backtest_v2_summary.csv
execution_backtest_v2_vs_v1_delta.csv
execution_backtest_v2_trades.csv
```

## Notes

- The script uses `caffeinate` by default on macOS so the Mac does not sleep mid-run.
- To disable that behavior:

```bash
NO_CAFFEINATE=1 scripts/phase4_v2_runbook.sh smoke
```

- If the Mac starts lagging, close Discord, YouTube, Arc extra tabs, and WhatsApp.
- Do not quit Google Drive while this is running because the parquet data lives there.
- If a run stops, rerun the same command. `--resume` is already included.
