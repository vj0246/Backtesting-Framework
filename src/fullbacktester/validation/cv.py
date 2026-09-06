"""Cross-validation that respects time.

``PurgedKFold`` gives contiguous test folds and removes from the training set
every sample whose label window would overlap the test window (purging), plus
a further ``embargo`` of samples after the test fold whose features are still
correlated with the test labels (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 7).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedKFold:
    """K contiguous folds with purge and embargo gaps.

    Attributes:
        n_splits: Number of folds.
        purge: Label horizon in samples. Training samples within ``purge`` of
            either edge of the test fold are dropped, because their label windows
            overlap the test period.
        embargo: Extra samples dropped after the test fold.
    """

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be non-negative")

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` for each fold, in time order."""
        if n_samples < self.n_splits:
            raise ValueError(f"{n_samples} samples cannot form {self.n_splits} folds")
        indices = np.arange(n_samples)
        edges = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        for k in range(self.n_splits):
            start, stop = int(edges[k]), int(edges[k + 1])
            test = indices[start:stop]
            keep = np.ones(n_samples, dtype=bool)
            keep[max(0, start - self.purge) : min(n_samples, stop + self.purge + self.embargo)] = (
                False
            )
            yield indices[keep], test

    def get_n_splits(self) -> int:
        return self.n_splits
