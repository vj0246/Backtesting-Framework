"""Transaction cost models shared by both engines.

Every model works on NumPy arrays as well as scalars, so the event-driven
engine (one fill at a time) and the vectorized engine (one bar of trades at a
time) apply *identical* arithmetic. That is what makes strategy comparisons
and engine cross-checks meaningful: nobody gets a cheaper broker.

Slippage is expressed as an adverse price move in basis points as a function
of participation (traded quantity / bar volume). Commission is a cash amount
per fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class CommissionModel(Protocol):
    def commission(self, quantity: ArrayLike, reference_price: ArrayLike) -> FloatArray:
        """Cash commission for trading ``|quantity|`` units at ``reference_price``."""
        ...


@runtime_checkable
class SlippageModel(Protocol):
    def slippage_bps(self, participation: ArrayLike) -> FloatArray:
        """Adverse price move in basis points given participation of bar volume."""
        ...


@dataclass(frozen=True)
class ZeroCommission:
    def commission(self, quantity: ArrayLike, reference_price: ArrayLike) -> FloatArray:
        return np.zeros_like(np.asarray(quantity, dtype=np.float64))


@dataclass(frozen=True)
class BpsCommission:
    """Commission as basis points of traded notional, with an optional per-order minimum."""

    bps: float
    minimum: float = 0.0

    def commission(self, quantity: ArrayLike, reference_price: ArrayLike) -> FloatArray:
        notional = np.abs(np.asarray(quantity, dtype=np.float64)) * np.asarray(
            reference_price, dtype=np.float64
        )
        fee = notional * (self.bps / 10_000.0)
        if self.minimum > 0:
            fee = np.where(notional > 0, np.maximum(fee, self.minimum), 0.0)
        return np.asarray(fee, dtype=np.float64)


@dataclass(frozen=True)
class PerUnitCommission:
    """Commission per unit traded (per share / per contract), with an optional per-order minimum."""

    per_unit: float
    minimum: float = 0.0

    def commission(self, quantity: ArrayLike, reference_price: ArrayLike) -> FloatArray:
        qty = np.abs(np.asarray(quantity, dtype=np.float64))
        fee = qty * self.per_unit
        if self.minimum > 0:
            fee = np.where(qty > 0, np.maximum(fee, self.minimum), 0.0)
        return np.asarray(fee, dtype=np.float64)


@dataclass(frozen=True)
class ZeroSlippage:
    def slippage_bps(self, participation: ArrayLike) -> FloatArray:
        return np.zeros_like(np.asarray(participation, dtype=np.float64))


@dataclass(frozen=True)
class FixedSlippage:
    """Constant adverse move in basis points, independent of size."""

    bps: float

    def slippage_bps(self, participation: ArrayLike) -> FloatArray:
        return np.full_like(np.asarray(participation, dtype=np.float64), self.bps)


@dataclass(frozen=True)
class VolumeShareSlippage:
    """Quadratic price impact in participation, after Zipline's volume-share model.

    impact_bps = base_bps + impact_coefficient * participation**2 * 10_000

    ``participation`` is clipped at ``max_participation`` for the *price* calculation;
    whether an order can actually trade more than that in one bar is the fill
    model's decision (``EngineConfig.max_participation``).
    """

    impact_coefficient: float = 0.1
    max_participation: float = 0.25
    base_bps: float = 0.0

    def slippage_bps(self, participation: ArrayLike) -> FloatArray:
        share = np.clip(np.asarray(participation, dtype=np.float64), 0.0, self.max_participation)
        share = np.where(np.isfinite(share), share, 0.0)
        return np.asarray(self.base_bps + self.impact_coefficient * share**2 * 10_000.0)


@dataclass(frozen=True)
class CostModel:
    """Commission + slippage bundle applied identically by every engine."""

    commission: CommissionModel
    slippage: SlippageModel

    @classmethod
    def zero(cls) -> CostModel:
        return cls(commission=ZeroCommission(), slippage=ZeroSlippage())

    @classmethod
    def bps(cls, commission_bps: float = 0.0, slippage_bps: float = 0.0) -> CostModel:
        """Flat-rate model: ``commission_bps`` of notional plus ``slippage_bps`` adverse move."""
        return cls(commission=BpsCommission(commission_bps), slippage=FixedSlippage(slippage_bps))

    def participation(self, quantity: ArrayLike, bar_volume: ArrayLike) -> FloatArray:
        qty = np.abs(np.asarray(quantity, dtype=np.float64))
        vol = np.asarray(bar_volume, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(vol > 0, qty / vol, np.where(qty > 0, np.inf, 0.0))
        return np.asarray(share, dtype=np.float64)

    def fill_price(
        self, quantity: ArrayLike, reference_price: ArrayLike, participation: ArrayLike
    ) -> FloatArray:
        """Price paid per unit: reference moved against the trade by the slippage model."""
        qty = np.asarray(quantity, dtype=np.float64)
        ref = np.asarray(reference_price, dtype=np.float64)
        move = self.slippage.slippage_bps(participation) / 10_000.0
        return np.asarray(ref * (1.0 + np.sign(qty) * move), dtype=np.float64)

    def slippage_cost(
        self, quantity: ArrayLike, reference_price: ArrayLike, participation: ArrayLike
    ) -> FloatArray:
        qty = np.asarray(quantity, dtype=np.float64)
        ref = np.asarray(reference_price, dtype=np.float64)
        move = self.slippage.slippage_bps(participation) / 10_000.0
        return np.asarray(np.abs(qty) * ref * move, dtype=np.float64)

    def total_cost(
        self, quantity: ArrayLike, reference_price: ArrayLike, bar_volume: ArrayLike
    ) -> FloatArray:
        """Commission plus slippage cash cost for a trade, vectorized."""
        part = self.participation(quantity, bar_volume)
        return np.asarray(
            self.commission.commission(quantity, reference_price)
            + self.slippage_cost(quantity, reference_price, part),
            dtype=np.float64,
        )
