from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import DatasetBundle


def load_movielens_1m(data_dir: str = "data/ml-1m") -> DatasetBundle:
    """Load MovieLens 1M with rating-greater-equal-4 implicit feedback."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        _download(data_dir.parent)

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=["user", "item", "rating", "timestamp"],
        engine="python",
    )
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=["item", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    pos = ratings[ratings["rating"] >= 4].copy()
    user_ids = pos["user"].unique()
    item_ids = pos["item"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    pos["uid"] = pos["user"].map(user2idx)
    pos["iid"] = pos["item"].map(item2idx)

    movies = movies[movies["item"].isin(item2idx)].copy()
    movies["iid"] = movies["item"].map(item2idx)
    movies["glist"] = movies["genres"].str.split("|")
    all_genres = sorted({g for gl in movies["glist"] for g in gl})
    g2i = {g: i for i, g in enumerate(all_genres)}

    item_content = np.zeros((len(item2idx), len(all_genres)), dtype=np.float32)
    for _, row in movies.iterrows():
        for g in row["glist"]:
            if g in g2i:
                item_content[row["iid"], g2i[g]] = 1.0

    return DatasetBundle(
        name="ml-1m",
        n_users=len(user_ids),
        n_items=len(item_ids),
        interactions=pos[["uid", "iid", "timestamp"]],
        item_content=item_content,
        item_meta=movies[["iid", "title", "genres"]],
        user2idx=user2idx,
        item2idx=item2idx,
    )


def _download(parent: Path) -> None:
    """Download and extract the MovieLens 1M zip from grouplens."""
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "ml-1m.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            zip_path,
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(parent)
