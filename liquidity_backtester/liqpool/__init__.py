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
from . import regime
from . import featurize
from . import ml_model
from . import directional
from . import timing
from . import multi_asset
from . import sectors
from . import journal as journal_module
from . import sizing
from . import drift
# broker_zerodha is imported on demand (it has an optional kiteconnect dep)

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
    "regime",
    "featurize",
    "ml_model",
    "directional",
    "timing",
    "multi_asset",
    "sectors",
    "journal_module",
    "sizing",
    "drift",
]
