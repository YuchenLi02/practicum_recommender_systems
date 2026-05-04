from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics as M


@dataclass
class EvalOutput:
    summary: dict[str, float]
    per_user: dict[str, np.ndarray]


class EvaluationRunner:
    def __init__(self, n_items: int, item_content: np.ndarray, item_pop: np.ndarray, k: int = 10):
        self.n_items = n_items
        self.item_content = item_content
        self.item_pop = item_pop
        self.k = k

    def evaluate(
        self, recs_dict: dict[int, list[int]], user_test: dict[int, set[int]]
    ) -> EvalOutput:
        """Score recommendations on six top-k metrics."""
        ndcg, recall, hr, ild = [], [], [], []
        all_recs = []
        for u, recs in recs_dict.items():
            rel = user_test.get(int(u), set())
            if not rel:
                continue
            ndcg.append(M.ndcg_at_k(recs, rel, self.k))
            recall.append(M.recall_at_k(recs, rel, self.k))
            hr.append(M.hit_rate_at_k(recs, rel, self.k))
            ild.append(M.intra_list_diversity(recs, self.item_content, self.k))
            all_recs.append(list(recs[: self.k]))

        ndcg = np.array(ndcg, dtype=np.float32)
        recall = np.array(recall, dtype=np.float32)
        hr = np.array(hr, dtype=np.float32)
        ild = np.array(ild, dtype=np.float32)

        summary = {
            "NDCG@10": float(ndcg.mean()) if ndcg.size else 0.0,
            "Recall@10": float(recall.mean()) if recall.size else 0.0,
            "HR@10": float(hr.mean()) if hr.size else 0.0,
            "Diversity": float(ild.mean()) if ild.size else 0.0,
            "Coverage": M.catalog_coverage(all_recs, self.n_items, self.k),
            "AvgPop": M.average_recommended_popularity(all_recs, self.item_pop, self.k),
        }
        per_user = {
            "NDCG@10": ndcg,
            "Recall@10": recall,
            "HR@10": hr,
            "Diversity": ild,
        }
        return EvalOutput(summary=summary, per_user=per_user)
