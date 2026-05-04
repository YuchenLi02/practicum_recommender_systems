from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data import (
    DatasetBundle,
    chronological_split,
    item_popularity,
    load_amazon_beauty,
    load_movielens_1m,
)
from src.evaluation.runner import EvaluationRunner
from src.models.als import ALSRecommender
from src.models.base import Recommender
from src.models.neumf import NeuMFRecommender
from src.models.popularity import PopularityRecommender
from src.models.sasrec import SASRecRecommender
from src.models.tiger import TIGERRecommender
from src.rerank.base import Reranker
from src.rerank.dpp import DPPReranker
from src.rerank.identity import IdentityReranker
from src.rerank.mmr import MMRReranker

DATASET_LOADERS = {
    "load_movielens_1m": load_movielens_1m,
    "load_amazon_beauty": load_amazon_beauty,
}

BASE_RANKER_CLASSES = {
    "popularity": PopularityRecommender,
    "als": ALSRecommender,
    "neumf": NeuMFRecommender,
    "sasrec": SASRecRecommender,
    "tiger": TIGERRecommender,
}

RERANKER_CLASSES = {
    "identity": IdentityReranker,
    "mmr": MMRReranker,
    "dpp": DPPReranker,
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict[str, Any]) -> DatasetBundle:
    """Instantiate a dataset bundle from the dataset config block."""
    loader = DATASET_LOADERS[cfg["loader"]]
    return loader(data_dir=cfg["data_dir"])


def build_base_ranker(spec: dict[str, Any]) -> Recommender:
    """Instantiate a base ranker from its config block."""
    spec = dict(spec)
    cls = BASE_RANKER_CLASSES[spec.pop("name")]
    return cls(**spec)


def build_reranker(spec: dict[str, Any]) -> Reranker:
    """Instantiate a reranker from its config block."""
    spec = dict(spec)
    cls = RERANKER_CLASSES[spec.pop("name")]
    return cls(**spec)


def build_runner(bundle: DatasetBundle, user_train: dict[int, set[int]]) -> EvaluationRunner:
    """Build the evaluation runner with item popularity from the training split."""
    item_pop = item_popularity(user_train)
    if len(item_pop) < bundle.n_items:
        padded = np.zeros(bundle.n_items, dtype=np.int64)
        padded[: len(item_pop)] = item_pop
        item_pop = padded
    return EvaluationRunner(
        n_items=bundle.n_items,
        item_content=bundle.item_content,
        item_pop=item_pop,
    )


def split_dataset(bundle: DatasetBundle, train_frac: float = 0.8):
    """Per-user chronological train and test split."""
    return chronological_split(bundle, train_frac=train_frac)
