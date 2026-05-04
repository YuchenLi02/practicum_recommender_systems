from __future__ import annotations

import numpy as np

from .base import Reranker


class IdentityReranker(Reranker):
    name = "none"

    def rerank(self, user, candidates, relevance, item_content, top_k, **context):
        order = np.argsort(-relevance)
        return [candidates[i] for i in order[:top_k]]
