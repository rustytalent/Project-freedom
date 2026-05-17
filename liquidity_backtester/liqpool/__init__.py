from .config import Config, FactorWeights, DetectionParams
from .data import fetch, resample, multi_timeframe
from .pools import build_pools, Pool
from .tester import test_pools, PoolResult, summarise
from .optimizer import optimize
from .plotting import plot_chart
from . import correlation
from . import walkforward
from . import stats
from . import stratified

__all__ = [
    "Config", "FactorWeights", "DetectionParams",
    "fetch", "resample", "multi_timeframe",
    "build_pools", "Pool",
    "test_pools", "PoolResult", "summarise",
    "optimize",
    "plot_chart",
    "correlation",
    "walkforward",
    "stats",
    "stratified",
]
