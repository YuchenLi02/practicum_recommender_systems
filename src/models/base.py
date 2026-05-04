from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        n_users: int,
        n_items: int,
        user_train: dict[int, set[int]],
        interactions=None,
        **kwargs,
    ) -> None:
        """Train on per-user training sets."""

    @abstractmethod
    def recommend(
        self, users: Iterable[int], top_n: int, exclude_train: bool = True
    ) -> dict[int, list[int]]:
        """Return a top-N candidate list per user."""

    def score(self, users: Iterable[int], items: Iterable[int]) -> np.ndarray:
        """Return a (len(users), len(items)) score matrix."""
        raise NotImplementedError(
            f"{self.name} does not expose explicit scores"
        )
