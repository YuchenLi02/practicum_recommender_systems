from __future__ import annotations

from typing import Iterable

import numpy as np

from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self.popular: np.ndarray | None = None
        self.user_train: dict[int, set[int]] | None = None
        self.item_pop: np.ndarray | None = None

    def fit(self, n_users, n_items, user_train, interactions=None, **kwargs):
        from ..data import item_popularity

        self.user_train = user_train
        self.item_pop = item_popularity(user_train)
        self.popular = np.argsort(-self.item_pop)

    def recommend(self, users: Iterable[int], top_n: int, exclude_train: bool = True):
        assert self.popular is not None and self.user_train is not None
        out: dict[int, list[int]] = {}
        for u in users:
            seen = self.user_train.get(int(u), set()) if exclude_train else set()
            picks = [int(it) for it in self.popular if int(it) not in seen][:top_n]
            out[int(u)] = picks
        return out

    def score(self, users, items):
        assert self.item_pop is not None
        users = list(users)
        items = list(items)
        row = self.item_pop[items].astype(np.float32)
        return np.tile(row, (len(users), 1))
