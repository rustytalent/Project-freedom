"""Minimal end-to-end smoke test. Uses synthetic data if yfinance is unavailable."""
from pathlib import Path
import json

from liqpool import Config, fetch, multi_timeframe, build_pools, test_pools, plot_chart
from liqpool.pools import project_to_base
from liqpool.tester import summarise
from liqpool.optimizer import optimize


def main():
    cfg = Config(symbol="AAPL", base_interval="5m", period="60d",
                 higher_tfs=["15min", "60min", "240min", "1D", "1W"],
                 test_horizon_bars=120, opt_iterations=40, min_pool_score=1.2)

    base = fetch(cfg.symbol, cfg.base_interval, cfg.period)
    tf_data = multi_timeframe(base, cfg.higher_tfs)
    print("bars:", {tf: len(df) for tf, df in tf_data.items()})

    # Baseline run
    pools = project_to_base(build_pools(tf_data, cfg), tf_data["base"].index)
    results = test_pools(tf_data["base"], pools, cfg)
    print("baseline:", json.dumps(summarise(results), default=str))

    out = Path("output"); out.mkdir(exist_ok=True)
    plot_chart(tf_data["base"], pools, results=results,
               title=f"{cfg.symbol} baseline pools",
               out_path=str(out / f"{cfg.symbol}_baseline.html"), top_n=20)

    # Optimised run
    best, log = optimize(tf_data, cfg)
    pools_b = project_to_base(build_pools(tf_data, best), tf_data["base"].index)
    results_b = test_pools(tf_data["base"], pools_b, best)
    print("optimised:", json.dumps(summarise(results_b), default=str))

    plot_chart(tf_data["base"], pools_b, results=results_b,
               title=f"{cfg.symbol} optimised pools",
               out_path=str(out / f"{cfg.symbol}_optimised.html"), top_n=20)

    (out / f"{cfg.symbol}_opt_trials.json").write_text(json.dumps(log, indent=2, default=str))
    (out / "best_weights.json").write_text(json.dumps(best.weights.as_dict(), indent=2))
    print("done. open output/*.html")


if __name__ == "__main__":
    main()
