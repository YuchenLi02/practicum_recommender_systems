from __future__ import annotations

import numpy as np

from .base import Reranker


class MMRReranker(Reranker):
    name = "mmr"

    def __init__(self, lam: float):
        assert 0.0 <= lam <= 1.0
        self.lam = lam

    @property
    def name(self):  # noqa: F811
        return f"mmr_lam={self.lam}"

    def rerank(self, user, candidates, relevance, item_content, top_k, **context):
        finite = np.isfinite(relevance)
        candidates = [c for c, ok in zip(candidates, finite) if ok]
        rel_raw = relevance[finite]
        if len(candidates) == 0:
            return []

        lo, hi = float(rel_raw.min()), float(rel_raw.max())
        rel_norm = (rel_raw - lo) / (hi - lo + 1e-9)
        rel_dict = {it: rel_norm[i] for i, it in enumerate(candidates)}

        norms = np.linalg.norm(item_content, axis=1) + 1e-9

        selected: list[int] = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            if not selected:
                best = max(remaining, key=lambda x: rel_dict[x])
            else:
                sel_vecs = item_content[selected]
                sel_norms = norms[selected]
                best, best_score = None, -np.inf
                for it in remaining:
                    v = item_content[it]
                    sims = (sel_vecs @ v) / (sel_norms * norms[it])
                    max_sim = float(sims.max()) if sims.size else 0.0
                    score = self.lam * rel_dict[it] - (1.0 - self.lam) * max_sim
                    if score > best_score:
                        best_score, best = score, it
            selected.append(int(best))
            remaining.remove(best)

        return selected
