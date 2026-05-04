from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from .base import Recommender


class ALSRecommender(Recommender):
    name = "als"

    def __init__(self, factors: int = 64, iterations: int = 30, regularization: float = 0.01):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_train: dict[int, set[int]] | None = None

    def fit(self, n_users, n_items, user_train, interactions=None, **kwargs):
        import implicit

        self.user_train = user_train
        rows, cols = [], []
        for u, items in user_train.items():
            rows.extend([u] * len(items))
            cols.extend(items)
        train_matrix = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False,
        )
        model.fit(train_matrix)
        self.user_factors = np.asarray(model.user_factors)
        self.item_factors = np.asarray(model.item_factors)

    def recommend(self, users: Iterable[int], top_n: int, exclude_train: bool = True):
        assert self.user_factors is not None and self.item_factors is not None
        out: dict[int, list[int]] = {}
        for u in users:
            scores = self.user_factors[u] @ self.item_factors.T
            if exclude_train and self.user_train is not None:
                for it in self.user_train.get(int(u), set()):
                    scores[it] = -1e9
            out[int(u)] = np.argsort(-scores)[:top_n].tolist()
        return out

    def score(self, users, items):
        assert self.user_factors is not None and self.item_factors is not None
        users = list(users)
        items = list(items)
        return self.user_factors[users] @ self.item_factors[items].T
