from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import DatasetBundle, encode_text


def load_amazon_beauty(data_dir: str = "data/amazon_beauty") -> DatasetBundle:
    """Load Amazon Beauty 5-core with rating-greater-equal-4 implicit feedback."""
    data_dir = Path(data_dir)
    reviews_path = data_dir / "reviews_Beauty_5.json.gz"
    meta_path = data_dir / "meta_Beauty.json.gz"
    if not reviews_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"download reviews_Beauty_5.json.gz + meta_Beauty.json.gz to {data_dir}/"
        )

    rows = []
    with gzip.open(reviews_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("overall", 0) >= 4:
                rows.append((r["reviewerID"], r["asin"], int(r.get("unixReviewTime", 0))))
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"])

    while True:
        u_counts = df.groupby("user").size()
        i_counts = df.groupby("item").size()
        keep_u = u_counts[u_counts >= 5].index
        keep_i = i_counts[i_counts >= 5].index
        new_df = df[df["user"].isin(keep_u) & df["item"].isin(keep_i)]
        if len(new_df) == len(df):
            break
        df = new_df

    user_ids = df["user"].unique()
    item_ids = df["item"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}
    df["uid"] = df["user"].map(user2idx)
    df["iid"] = df["item"].map(item2idx)

    titles: dict[str, str] = {}
    descs: dict[str, str] = {}
    cats: dict[str, str] = {}
    keep_set = set(item_ids)
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            try:
                m = eval(line)
            except Exception:
                continue
            asin = m.get("asin")
            if asin in keep_set:
                titles[asin] = (m.get("title") or "")[:200]
                d = m.get("description")
                descs[asin] = (d or "")[:200] if isinstance(d, str) else ""
                cats[asin] = " > ".join(m.get("categories", [[]])[0]) if m.get("categories") else ""

    cache_path = data_dir / "items.npy"
    if cache_path.exists():
        item_content = np.load(cache_path).astype(np.float32)
    else:
        item_content = encode_text(
            [
                (titles.get(it, "") + " " + descs.get(it, "")).strip() or it
                for it in item_ids
            ]
        ).astype(np.float32)
        np.save(cache_path, item_content)

    item_meta = pd.DataFrame({
        "iid": [item2idx[it] for it in item_ids],
        "title": [titles.get(it, "") for it in item_ids],
        "genres": [cats.get(it, "") for it in item_ids],
    })

    return DatasetBundle(
        name="amazon-beauty",
        n_users=len(user_ids),
        n_items=len(item_ids),
        interactions=df[["uid", "iid", "timestamp"]],
        item_content=item_content,
        item_meta=item_meta,
        user2idx=user2idx,
        item2idx=item2idx,
    )
