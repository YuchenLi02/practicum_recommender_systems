from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    n_users: int
    n_items: int
    interactions: pd.DataFrame
    item_content: np.ndarray
    item_meta: pd.DataFrame
    user2idx: dict
    item2idx: dict
    idx2item: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.idx2item:
            self.idx2item = {i: it for it, i in self.item2idx.items()}

    @property
    def density(self) -> float:
        return len(self.interactions) / (self.n_users * self.n_items)


def chronological_split(
    bundle: DatasetBundle, train_frac: float = 0.8
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Per-user chronological train and test split."""
    interactions = bundle.interactions.sort_values(["uid", "timestamp"])
    user_train: dict[int, set[int]] = defaultdict(set)
    user_test: dict[int, set[int]] = defaultdict(set)

    for uid, grp in interactions.groupby("uid"):
        iids = grp["iid"].tolist()
        n = len(iids)
        if n < 2:
            user_train[int(uid)].update(iids)
            continue
        sp = max(1, int(n * train_frac))
        user_train[int(uid)].update(iids[:sp])
        user_test[int(uid)].update(iids[sp:])

    return user_train, user_test


def item_popularity(user_train: dict[int, set[int]]) -> np.ndarray:
    """Per-item training-set interaction count."""
    counts: dict[int, int] = defaultdict(int)
    for items in user_train.values():
        for it in items:
            counts[int(it)] += 1
    if not counts:
        return np.zeros(0, dtype=np.int64)
    n_items = max(counts) + 1
    pop = np.zeros(n_items, dtype=np.int64)
    for it, c in counts.items():
        pop[it] = c
    return pop


def encode_text(texts: list[str], n_components: int = 128) -> np.ndarray:
    """L2-normalized TF-IDF plus truncated SVD embedding of item text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    X = vec.fit_transform(texts)
    n_components = min(n_components, X.shape[1] - 1, len(texts) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Z = svd.fit_transform(X)
    return normalize(Z, norm="l2", axis=1).astype(np.float32)
