# Equity Vertical Runbook

This is the operational path while historical option intraday OHLCV data
is unavailable. It turns the latest trained core25 equity bundle into the
commercial research artifact stack.

## 1. Train or Reuse Latest Equity Bundle

Use the standard multi-asset train path. The bundle must include:

- Daily Brief generation,
- outcome-log integration,
- V2 execution artifacts,
- policy labels,
- policy return model,
- feature store.

The latest successful run should contain:

```text
output_models/core25_<RUN_TAG>/multi_asset_report.pkl
output_core25_<RUN_TAG>/daily_brief.json
output_core25_<RUN_TAG>/daily_brief.txt
output_feature_store/core25_<RUN_TAG>/
```

## 2. Generate Equity Daily Brief / Prediction Artifact

If the train run already emitted `daily_brief.json` and
`daily_brief.txt`, use that output directory directly. Otherwise run
predict mode on the saved bundle.

```bash
cd /root/Project-freedom/liquidity_backtester

RUN_TAG="$(cat logs/latest_run_tag.txt)"
PREDICT_TAG="predict_${RUN_TAG}_$(date +%Y%m%d_%H%M)"

export DATA_ROOT="/root/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"
export RAW_1M_DIR="$DATA_ROOT/raw_1m"

nohup env PYTHONPATH=. \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -u examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode predict \
  --model-dir "output_models/core25_${RUN_TAG}" \
  --asset-workers 15 \
  --feature-store-dir "output_feature_store/core25_${RUN_TAG}" \
  --run-execution-backtest-v2 \
  --use-1m-resolution \
  --raw-1m-dir "$RAW_1M_DIR" \
  --fill-policy neutral \
  --slippage-model state_dependent \
  --skip-policy-model \
  --out "output_${PREDICT_TAG}" \
  > "logs/${PREDICT_TAG}.log" 2>&1 &

echo $! > "logs/${PREDICT_TAG}.pid"
echo "$PREDICT_TAG" | tee logs/latest_predict_tag.txt
cat "logs/${PREDICT_TAG}.pid"
```

## 3. Watch Predict Job

```bash
cd /root/Project-freedom/liquidity_backtester
PREDICT_TAG="$(cat logs/latest_predict_tag.txt)"

watch -n 60 '
cd /root/Project-freedom/liquidity_backtester
PREDICT_TAG="$(cat logs/latest_predict_tag.txt)"
PID="$(cat logs/${PREDICT_TAG}.pid 2>/dev/null)"

date
echo "PREDICT_TAG=$PREDICT_TAG"
ps -p "$PID" -o pid,stat,pcpu,pmem,rss,etime,args 2>/dev/null || echo "predict stopped/finished"
echo
tail -100 "logs/${PREDICT_TAG}.log" 2>/dev/null
echo
ls -lh "output_${PREDICT_TAG}/daily_brief.json" "output_${PREDICT_TAG}/daily_brief.txt" 2>/dev/null || true
'
```

## 4. Pack Customer Delivery

Use this after an output directory has a brief.

```bash
cd /root/Project-freedom/liquidity_backtester
PREDICT_TAG="$(cat logs/latest_predict_tag.txt)"

mkdir -p "packed/${PREDICT_TAG}"

PYTHONPATH=. .venv/bin/python -u analysis/pack_artifacts.py \
  --input "output_${PREDICT_TAG}" \
  --output "packed/${PREDICT_TAG}" \
  --customer-delivery \
  --customer-id pilot_equity_customer \
  --customer-tier paid \
  --zip \
  2>&1 | tee "logs/pack_${PREDICT_TAG}.log"
```

Expected pack:

```text
packed/<PREDICT_TAG>/daily_brief.json
packed/<PREDICT_TAG>/daily_brief.txt
packed/<PREDICT_TAG>/consolidated_summary.json
packed/<PREDICT_TAG>/consolidated_models.json
packed/<PREDICT_TAG>/consolidated_backtest.json
packed/<PREDICT_TAG>/README.md
packed/<PREDICT_TAG>/manifest.json
packed/<PREDICT_TAG>.zip
```

## 5. Backfill / Continue Outcome Log

```bash
cd /root/Project-freedom/liquidity_backtester
RUN_TAG="$(cat logs/latest_run_tag.txt)"

nohup env PYTHONPATH=. .venv/bin/python -u analysis/backfill_outcome_log.py \
  --bundle "output_models/core25_${RUN_TAG}/multi_asset_report.pkl" \
  --output-root "data/outcome_log_${RUN_TAG}" \
  --days 90 \
  > "logs/backfill_outcome_${RUN_TAG}.log" 2>&1 &

echo $! > "logs/backfill_outcome_${RUN_TAG}.pid"
cat "logs/backfill_outcome_${RUN_TAG}.pid"
```

Watcher:

```bash
cd /root/Project-freedom/liquidity_backtester
RUN_TAG="$(cat logs/latest_run_tag.txt)"

watch -n 60 '
cd /root/Project-freedom/liquidity_backtester
RUN_TAG="$(cat logs/latest_run_tag.txt)"
PID="$(cat logs/backfill_outcome_${RUN_TAG}.pid 2>/dev/null)"

date
ps -p "$PID" -o pid,stat,pcpu,pmem,rss,etime,args 2>/dev/null || echo "backfill stopped/finished"
echo
tail -80 "logs/backfill_outcome_${RUN_TAG}.log" 2>/dev/null
echo
echo "predictions:"
find "data/outcome_log_${RUN_TAG}/predictions" -type f -name "*.parquet" 2>/dev/null | wc -l
echo "resolutions:"
find "data/outcome_log_${RUN_TAG}/resolutions" -type f -name "*.parquet" 2>/dev/null | wc -l
'
```

## 6. Optional Website Push

The brief generator can push artifacts automatically when website
environment variables are configured. For manual customer delivery,
the packed zip is enough.

Website push requires:

```text
CRUX_ARTIFACT_API_URL
CRUX_ARTIFACT_API_TOKEN
```

Use exact levels only for licensed/paid research tiers. Demo, sample,
free, and public tiers must remain abstracted and watermarked.

## 7. What Not To Run Yet

Do not run options Gate-1 again until
`/root/kite_indian_market_data/options_active` contains historical
intraday option OHLCV for NIFTY and BANKNIFTY across old expiries. The
current warehouse has Greeks and active contracts, but not enough
historical option bars to train the options model.
