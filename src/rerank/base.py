from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self,
        user: int,
        candidates: list[int],
        relevance: np.ndarray,
        item_content: np.ndarray,
        top_k: int,
        **context,
    ) -> list[int]:
        """Return a top_k re-ordered subset of candidates."""
