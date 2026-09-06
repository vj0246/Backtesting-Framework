"""Engine configuration shared by both execution engines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from quantgauntlet.execution.costs import CostModel


class FillTiming(StrEnum):
    """When a market order placed after bar ``t`` executes.

    ``NEXT_OPEN``: at bar ``t + 1``'s open, the first price that exists after the
    signal does. ``SAME_CLOSE``: at bar ``t``'s close, which assumes you can trade at a
    price you have already observed. The latter is optimistic and is flagged by the
    validation report; it exists for market-on-close strategies that genuinely can.
    Limit and stop orders always evaluate against the next bar regardless of timing.
    """

    NEXT_OPEN = "next_open"
    SAME_CLOSE = "same_close"


@dataclass(frozen=True)
class EngineConfig:
    """Execution assumptions. The same object drives both engines.

    Attributes:
        initial_cash: Starting cash in the panel's quote currency.
        cost_model: Commission and slippage applied to every fill.
        fill_timing: See ``FillTiming``.
        fractional_shares: Allow non-integer position sizes (crypto, idealized runs).
        lot_size: Smallest tradeable increment when ``fractional_shares`` is False.
        allow_short: Whether negative positions may be opened. Target-weight orders
            with negative weights are clipped to zero when False; explicit sell orders
            that would go short are rejected.
        max_gross_leverage: Cap on sum(|position notional|) / equity after a fill.
            None means unconstrained (cash may go negative, i.e. margin is free).
            1.0 means fully funded, no margin.
        max_participation: Cap on |traded units| / bar volume per fill. The
            unfilled remainder stays open for GTC orders and is cancelled for DAY.
            None disables the cap.
    """

    initial_cash: float = 100_000.0
    cost_model: CostModel = field(default_factory=CostModel.zero)
    fill_timing: FillTiming = FillTiming.NEXT_OPEN
    fractional_shares: bool = False
    lot_size: float = 1.0
    allow_short: bool = True
    max_gross_leverage: float | None = None
    max_participation: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.initial_cash) and self.initial_cash > 0):
            raise ValueError("initial_cash must be positive")
        if not self.fractional_shares and self.lot_size <= 0:
            raise ValueError("lot_size must be positive when fractional_shares is False")
        if self.max_gross_leverage is not None and self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive or None")
        if self.max_participation is not None and not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1] or None")

    @classmethod
    def idealized(
        cls, cost_model: CostModel | None = None, initial_cash: float = 100_000.0
    ) -> EngineConfig:
        """Frictionless-execution settings under which both engines must agree exactly."""
        return cls(
            initial_cash=initial_cash,
            cost_model=cost_model if cost_model is not None else CostModel.zero(),
            fill_timing=FillTiming.NEXT_OPEN,
            fractional_shares=True,
            allow_short=True,
            max_gross_leverage=None,
            max_participation=None,
        )

    @property
    def is_idealized(self) -> bool:
        return (
            self.fractional_shares
            and self.allow_short
            and self.max_gross_leverage is None
            and self.max_participation is None
        )

    def round_quantity(self, quantity: float) -> float:
        """Round toward zero to the lot grid (identity when fractional shares are allowed)."""
        if self.fractional_shares:
            return float(quantity)
        lots = math.trunc(quantity / self.lot_size + 1e-12 * math.copysign(1.0, quantity))
        return float(lots * self.lot_size)
