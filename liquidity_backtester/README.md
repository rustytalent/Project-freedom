# Liquidity-Pool Backtester

A research-grade backtester focused on **detecting liquidity pools** on a 5-minute chart by
combining features from multiple higher timeframes, then **walk-forward testing** whether each
pool is respected or broken, and finally **searching** the parameter / factor-weight space to
converge on the configurations that produce the most reliable pools.

The name "liquidity pool" here refers to **price zones where resting orders / stops likely sit**
— not DeFi pools. Concepts are inspired by ICT/SMC: external liquidity (swing high/low extremes,
EQH/EQL, prior D/W/M extremes) and internal liquidity (FVGs, order blocks, single-bar rejection
imbalances, volume-by-price nodes, ORB extremes).

## Layout

```
liquidity_backtester/
├── liqpool/
│   ├── config.py        # Config / DetectionParams / FactorWeights
│   ├── data.py          # yfinance fetch + multi-TF resampling (synthetic fallback)
│   ├── indicators.py    # ATR, EMA, RSI, momentum, per-bar candle features
│   ├── features.py      # swings, EQH/EQL, prev D/W/M, FVG, order blocks, wick imbalance, ORB, HVN
│   ├── pools.py         # cluster candidates → Pool zones, score with FactorWeights + TF confluence
│   ├── tester.py        # walk-forward respect / break classification
│   ├── optimizer.py     # explore + refine search over (weights, params)
│   ├── plotting.py      # plotly 5m chart with zone overlays and outcome markers
│   └── cli.py           # `python -m liqpool.cli run|optimize`
├── examples/example_run.py
├── output/              # html charts + json dumps land here
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Single-run on default weights
python -m liqpool.cli run --symbol AAPL --base 5m --period 60d \
    --tfs 15min,60min,240min,1D,1W --horizon 200 --out output

# Optimise weights + detection params, then plot the best config
python -m liqpool.cli optimize --symbol AAPL --base 5m --period 60d \
    --iters 80 --out output
```

Outputs:
- `output/<SYMBOL>_<TF>_pools.html` — interactive 5m chart with zones (green=respected,
  red=broken, grey=untouched, orange/blue=untested supply/demand)
- `output/<SYMBOL>_pools.json` — pool definitions
- `output/<SYMBOL>_results.json` — per-pool outcomes
- `output/<SYMBOL>_opt_trials.json` — full optimiser trial log

`yfinance` intraday limits: `1m` is ~7 trading days back; `5m`/`15m`/`30m` ~60 days; `1h` ~730d.
Use `--start/--end` for daily history. If yfinance returns nothing the loader silently falls back
to a synthetic OHLCV generator so the pipeline is exercisable offline.

## What goes into a pool

A `Pool` is a price zone `(price_low, price_high)` anchored to a formation time. Each pool keeps
its list of **contributors** — the individual `LevelCandidate`s that voted for it. Contributors
come from these detectors, each computed on every requested timeframe:

| Detector              | What it captures                                              | Factor weight key      |
|-----------------------|---------------------------------------------------------------|------------------------|
| `SWING_H` / `SWING_L` | Fractal swings (the bedrock)                                  | (baseline, fixed 0.4)  |
| `EQH` / `EQL`         | Multiple swings within ATR tolerance — heaviest stop clusters | `eqhl`                 |
| `PDH` / `PDL`         | Previous-day high / low                                       | `prev_day`             |
| `PWH` / `PWL`         | Previous-week high / low                                      | `prev_week`            |
| `PMH` / `PML`         | Previous-month high / low                                     | `prev_month`           |
| `FVG_bull/bear`       | Three-candle imbalance (gap)                                  | `fvg`                  |
| `OB_bull/bear`        | Order block: last opposite candle before a displacement       | `order_block`          |
| `REJ_high/low`        | Dominant wick + small body — single-bar rejection / grab      | `in_candle_imbalance`  |
| `HVN`                 | Volume-by-price peak                                          | `volume_node`          |
| `ORB_H/L`             | Session opening-range extreme                                 | `orb_extreme`          |

Pool score combines contributor `strength` with the factor weight, with diminishing returns for
repeated factors, then applies a **multi-TF confluence multiplier** for every additional distinct
timeframe that agrees. A 5m EQH that overlaps a 1D PDH and a 1H FVG outscores any of them alone.

## Pool test

For each pool we walk forward up to `test_horizon_bars` bars on the base 5m TF and classify:

- **respected** — price entered the zone and then reversed by `respect_reaction_atr` * ATR within
  `respect_within_bars` bars, *without* a bar closing through the zone.
- **broken** — a bar closed beyond the zone by `break_close_buffer_atr` * ATR (we stop at the
  first close-through).
- **untouched** — price never entered the zone in the horizon.

We also record `bars_to_touch`, `bars_to_break`, `max_excursion_through` (in ATR units past the
zone), `reaction_atr`, and `n_touches`.

## Optimizer

The optimiser does **random exploration** for the first `opt_explore_frac` of trials, then
**local refinement** (gaussian perturbation with shrinking sigma) around the running-best config.
Objective:

```
J = respect_rate * log(1 + tested_n) - lambda * ||weights||^2 / 100
```

Higher tested-pool count is rewarded, but only if respect-rate doesn't collapse. The l2 penalty
keeps the weights from running away. Tweak `Config.opt_iterations`, `opt_explore_frac`, `opt_seed`.

## Extending

- Add a detector: implement it in `features.py` returning `LevelCandidate`s, then hook it into
  `detect_all()` and add a weight key in `FactorWeights` + `_FACTOR_OF_SOURCE` in `pools.py`.
- Add a TF: pass it in `Config.higher_tfs` (use pandas offsets, e.g. `"4H"`).
- Swap data source: replace `liqpool.data.fetch`; just return a UTC, lower-cased OHLCV DataFrame.

## Caveats / honest limitations

- yfinance 5m data is **delayed and capped at ~60 days**. For deeper history use a paid source
  or daily TFs.
- The "learning" loop is a random + local search — fine for tens of factors, not a Bayesian
  optimiser. Drop in scikit-optimize or Optuna if you need it.
- `respected` is defined by reaction magnitude, not P&L. This is a pool-quality metric, not a
  trading-strategy backtest. Pairing it with an entry/exit model is the next step.
