from __future__ import annotations

import numpy as np


def ndcg_at_k(recs: list[int], rel: set[int], k: int = 10) -> float:
    recs = recs[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(recs) if int(it) in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(recs: list[int], rel: set[int], k: int = 10) -> float:
    if not rel:
        return 0.0
    return len(set(int(x) for x in recs[:k]) & rel) / len(rel)


def hit_rate_at_k(recs: list[int], rel: set[int], k: int = 10) -> float:
    return 1.0 if set(int(x) for x in recs[:k]) & rel else 0.0


def intra_list_diversity(recs: list[int], item_content: np.ndarray, k: int = 10) -> float:
    recs = [int(x) for x in recs[:k]]
    if len(recs) < 2:
        return 0.0
    vecs = item_content[recs]
    norms = np.linalg.norm(vecs, axis=1) + 1e-9
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j])))
    return float(1.0 - np.mean(sims)) if sims else 0.0


def catalog_coverage(all_recs: list[list[int]], n_items: int, k: int = 10) -> float:
    seen = {int(it) for r in all_recs for it in r[:k]}
    return len(seen) / max(1, n_items)


def average_recommended_popularity(
    all_recs: list[list[int]], item_pop: np.ndarray, k: int = 10
) -> float:
    flat = [int(it) for r in all_recs for it in r[:k]]
    if not flat:
        return 0.0
    return float(np.mean(item_pop[flat]))
