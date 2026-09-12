"""fullbacktester: backtesting with built-in defenses against look-ahead bias,
survivorship bias, and overfitting.

Typical use::

    import fullbacktester as fbt

    panel = fbt.load_panel(["AAPL", "MSFT"], "2018-01-01", "2024-12-31", market="US")
    arena = fbt.Arena(panel, fbt.EngineConfig(cost_model=fbt.CostModel.bps(5, 5)))
    arena.add(fbt.RuleBasedStrategy(my_signal, warmup=21))
    print(arena.run().summary())
"""

from fullbacktester.arena import Arena, ArenaResult
from fullbacktester.data.cache import ParquetCache
from fullbacktester.data.loader import load_bars, load_panel
from fullbacktester.data.panel import Panel, PanelView
from fullbacktester.data.quality import check_data_quality
from fullbacktester.data.schema import SchemaError, validate_bars
from fullbacktester.data.sources import (
    AlpacaSource,
    CredentialError,
    DataSource,
    LocalSource,
    NSEBhavcopySource,
    UpstoxSource,
    YFinanceSource,
    available_sources,
    get_source,
)
from fullbacktester.execution import (
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
from fullbacktester.flags import Flag, Severity
from fullbacktester.live import (
    ExpectationGap,
    PaperSession,
    ReplayCheck,
    SessionLockError,
    expectation_gap,
    replay_check,
)
from fullbacktester.markets import CRYPTO, INDIA, US, Frequency, Market, get_market
from fullbacktester.metrics import (
    CSCVResult,
    PerformanceMetrics,
    compute_metrics,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from fullbacktester.result import BacktestResult
from fullbacktester.strategy import (
    BarContext,
    BatchRuleStrategy,
    BatchWeightStrategy,
    RuleBasedStrategy,
    Strategy,
    WeightStrategy,
)
from fullbacktester.strategy.ml import (
    WalkForwardMLStrategy,
    forward_returns,
    long_short_top_k,
    signed_equal_weight,
)
from fullbacktester.validation import (
    FutureLeakTester,
    PurgedKFold,
    StaticScanner,
    ValidationReport,
    validate,
)

__version__ = "0.3.0"

__all__ = [
    "CRYPTO",
    "INDIA",
    "US",
    "AlpacaSource",
    "Arena",
    "ArenaResult",
    "BacktestResult",
    "BarContext",
    "BatchRuleStrategy",
    "BatchWeightStrategy",
    "BpsCommission",
    "CSCVResult",
    "CostModel",
    "CredentialError",
    "DataSource",
    "EngineComparison",
    "EngineConfig",
    "EventDrivenEngine",
    "ExpectationGap",
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
    "PaperSession",
    "ParquetCache",
    "PerUnitCommission",
    "PerformanceMetrics",
    "PurgedKFold",
    "ReplayCheck",
    "RuleBasedStrategy",
    "SchemaError",
    "SessionLockError",
    "Severity",
    "StaticScanner",
    "Strategy",
    "TimeInForce",
    "UpstoxSource",
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
    "expectation_gap",
    "forward_returns",
    "get_market",
    "get_source",
    "load_bars",
    "load_panel",
    "long_short_top_k",
    "minimum_backtest_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "replay_check",
    "signed_equal_weight",
    "validate",
    "validate_bars",
]
