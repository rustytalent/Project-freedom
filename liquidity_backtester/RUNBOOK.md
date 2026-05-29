# End-to-End Runbook

Full pipeline: setup → train → predict → SaaS feed → distribute. Run everything
from the `liquidity_backtester/` directory (it is a subfolder of the repo, so
`cd` into the repo first). `PYTHONPATH=.` is required so the `liqpool`/`service`
packages import.

> macOS note: use `python3` to create the venv (macOS has no bare `python`).
> After that, every command below calls the venv's interpreter explicitly as
> `.venv/bin/python`, so you do NOT need to `source .venv/bin/activate` — it
> works the same in zsh, bash, and non-interactive shells.

> NOTE: Steps 2–4 need the Indian market parquet data and produce a real model.
> If you just want to see the SaaS feed work end-to-end with NO data/model, jump
> to **Step 7 (synthetic smoke)**.

---

## 0. One-time setup

```bash
# go to the liquidity_backtester folder inside the cloned repo, e.g. on Mac:
#   cd ~/Projects/Project-freedom/liquidity_backtester
# if unsure where it is:  find ~ -type d -name liquidity_backtester 2>/dev/null
cd /path/to/Project-freedom/liquidity_backtester

# virtualenv with python3 (macOS/Linux)
python3 -m venv .venv

# install deps via the venv's own pip (no activation needed)
.venv/bin/pip install -r requirements.txt

# extra deps only needed to serve the HTTP API (not for file export)
.venv/bin/pip install -r service/requirements.txt
```


---

## 1. (If needed) Prepare market data

You need resampled parquet under a directory, e.g. `$RESAMPLED_DIR`, with
per-timeframe files (5m/15m/...). If you pull from Kite/Zerodha:

```bash
export DATA_ROOT="/path/to/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"

# (optional) authenticate + resample raw kite parquet -> resampled timeframes
PYTHONPATH=. .venv/bin/python examples/zerodha_login.py
PYTHONPATH=. .venv/bin/python examples/resample_kite_parquet.py \
  --in "$DATA_ROOT/raw" --out "$RESAMPLED_DIR"
```

---

## 2. Train the model

```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode train \
  --asset-workers 4 \
  --model-dir output_models/core25_latest \
  --out output_core25_train
```

`--asset-workers` controls how many symbols train in parallel (default 4; 4–5 is
the sweet spot on a 16GB machine, use 5 if you have more RAM/cores, 1 to
disable). Each worker is pinned to 1 BLAS/LightGBM thread to avoid
oversubscription. Produces `output_models/core25_latest/multi_asset_report.pkl`
(+ `metadata.json`). This pickle is the IP — it stays server-side and is NEVER
distributed.

---

## 3. Predict (writes the raw outputs the SaaS engine ingests)

```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode predict \
  --model-dir output_models/core25_latest \
  --out output_core25_predict_latest
```

Key raw files written to `output_core25_predict_latest/`:
`live_gate_decisions.csv` (all candidate levels — the richest source),
`live_plan.json`, `tradeable_setups.csv`, `watchlist.csv`,
`track_a_pretouch_setups.csv`.

---

## 4. Generate a distributable SaaS data file (per customer)

Converts the raw predict outputs into the opaque, watermarked feed. Touches no
model code. Run once per customer (watermark = customer id + date).

```bash
PYTHONPATH=. .venv/bin/python scripts/export_saas_feed.py \
  --raw output_core25_predict_latest \
  --customer acme-capital \
  --date 2026-05-29 \
  --out dist/feed_acme_2026-05-29
```

Writes `dist/feed_acme_2026-05-29.json` and `.csv`. Output fields are only:
`feature_intensity_score` (0–100 opaque rank), `feature_state` (state_1/2/3),
`level_zone`, usage/compliance notes. A built-in leak guard fails if any
internal field would be emitted.

---

## 5. (Alternative to step 4) Serve the feed over HTTP

Same engine, served live instead of as a file.

```bash
GFEED_API_KEYS="acme-key:acme-capital,beta-key:beta-fund" \
GFEED_RAW_DIR="output_core25_predict_latest" \
GFEED_RATE_PER_MIN=120 \
PYTHONPATH=. .venv/bin/python -m uvicorn service.app:app --host 0.0.0.0 --port 8080

# customer call:
curl -H "X-API-Key: acme-key" \
  "http://localhost:8080/v1/levels?symbol=HDFCBANK.NS&date=2026-05-29"
```

---

## 6. Distribute

- One file per customer (`--customer <id>`) so a leak is traceable.
- Hand over ONLY `dist/*.json` / `dist/*.csv` — never the `.pkl`, the predict
  outputs, or any `liqpool/` code.
- Include `docs/data_product_spec.md` (customer doc) and
  `docs/legal/compliance.md` (disclaimer/ToS basis).
- ⚠ Before any PAID launch: obtain SEBI-specialist legal opinion + evaluate RA
  registration. See `docs/sebi_compliance_notes.md`.

---

## 7. Synthetic smoke (no data/model needed)

Generates fake raw predict output, runs the full ingest→export pipeline, and
prints a watermark + leak-guard proof. Use to demo the feed or hand a sample to
a reviewer.

```bash
PYTHONPATH=. .venv/bin/python scripts/synthetic_smoke.py \
  --out-dir /tmp/saas_smoke --customer reviewer --date 2026-05-29
# -> /tmp/saas_smoke/saas_feed_reviewer_2026-05-29.{json,csv}
```

---

## 8. Quality gates (run before committing / deploying)

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q                   # full test suite
.venv/bin/python scripts/compliance_lint.py                  # CI gate (exit!=0 on hard hits)
.venv/bin/python scripts/compliance_lint.py --report docs/compliance_audit_report.md  # full audit
```
