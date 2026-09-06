"""quantgauntlet: backtesting with built-in defenses against look-ahead bias,
survivorship bias, and overfitting.

Typical use::

    import quantgauntlet as qg

    panel = qg.load_panel(["AAPL", "MSFT"], "2018-01-01", "2024-12-31", market="US")
    arena = qg.Arena(panel, qg.EngineConfig(cost_model=qg.CostModel.bps(5, 5)))
    arena.add(qg.RuleBasedStrategy(my_signal, warmup=21))
    print(arena.run().summary())
"""

from quantgauntlet.arena import Arena, ArenaResult
from quantgauntlet.data.cache import ParquetCache
from quantgauntlet.data.loader import load_bars, load_panel
from quantgauntlet.data.panel import Panel, PanelView
from quantgauntlet.data.quality import check_data_quality
from quantgauntlet.data.schema import SchemaError, validate_bars
from quantgauntlet.data.sources import (
    DataSource,
    LocalSource,
    NSEBhavcopySource,
    YFinanceSource,
    available_sources,
    get_source,
)
from quantgauntlet.execution import (
    BpsCommission,
    CostModel,
    EngineComparison,
    EngineConfig,
    EventDrivenEngine,
    FillTiming,
    FixedSlippage,
    Order,
    OrderType,
    PerUnitCommission,
    TimeInForce,
    VectorizedEngine,
    VolumeShareSlippage,
    compare_engines,
)
from quantgauntlet.flags import Flag, Severity
from quantgauntlet.markets import CRYPTO, INDIA, US, Frequency, Market, get_market
from quantgauntlet.metrics import (
    CSCVResult,
    PerformanceMetrics,
    compute_metrics,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quantgauntlet.result import BacktestResult
from quantgauntlet.strategy import (
    BarContext,
    BatchRuleStrategy,
    BatchWeightStrategy,
    RuleBasedStrategy,
    Strategy,
    WeightStrategy,
)
from quantgauntlet.strategy.ml import (
    WalkForwardMLStrategy,
    forward_returns,
    long_short_top_k,
    signed_equal_weight,
)
from quantgauntlet.validation import (
    FutureLeakTester,
    PurgedKFold,
    StaticScanner,
    ValidationReport,
    validate,
)

__version__ = "0.1.0"

__all__ = [
    "CRYPTO",
    "INDIA",
    "US",
    "Arena",
    "ArenaResult",
    "BacktestResult",
    "BarContext",
    "BatchRuleStrategy",
    "BatchWeightStrategy",
    "BpsCommission",
    "CSCVResult",
    "CostModel",
    "DataSource",
    "EngineComparison",
    "EngineConfig",
    "EventDrivenEngine",
    "FillTiming",
    "FixedSlippage",
    "Flag",
    "Frequency",
    "FutureLeakTester",
    "LocalSource",
    "Market",
    "NSEBhavcopySource",
    "Order",
    "OrderType",
    "Panel",
    "PanelView",
    "ParquetCache",
    "PerUnitCommission",
    "PerformanceMetrics",
    "PurgedKFold",
    "RuleBasedStrategy",
    "SchemaError",
    "Severity",
    "StaticScanner",
    "Strategy",
    "TimeInForce",
    "ValidationReport",
    "VectorizedEngine",
    "VolumeShareSlippage",
    "WalkForwardMLStrategy",
    "WeightStrategy",
    "YFinanceSource",
    "__version__",
    "available_sources",
    "check_data_quality",
    "compare_engines",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "forward_returns",
    "get_market",
    "get_source",
    "load_bars",
    "load_panel",
    "long_short_top_k",
    "minimum_backtest_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "signed_equal_weight",
    "validate",
    "validate_bars",
]
