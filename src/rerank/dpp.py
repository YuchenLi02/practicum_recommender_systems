from __future__ import annotations

import numpy as np

from .base import Reranker


class DPPReranker(Reranker):
    """Greedy MAP inference over a DPP kernel L_ij = q_i * S_ij * q_j with q_i = exp(alpha * Rel)."""

    name = "dpp"

    def __init__(self, alpha: float = 1.0, eps: float = 1e-9):
        self.alpha = alpha
        self.eps = eps

    @property
    def name(self):  # noqa: F811
        return f"dpp_alpha={self.alpha}"

    def rerank(self, user, candidates, relevance, item_content, top_k, **context):
        finite = np.isfinite(relevance)
        candidates = [c for c, ok in zip(candidates, finite) if ok]
        rel = relevance[finite]
        if len(candidates) == 0:
            return []

        rel = rel - rel.max()
        q = np.exp(self.alpha * rel).astype(np.float64)

        vecs = item_content[candidates]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + self.eps
        vecs_n = vecs / norms
        S = vecs_n @ vecs_n.T
        S = np.clip(S, 0.0, 1.0)

        L = (q[:, None] * S) * q[None, :]
        N = len(candidates)
        top_k = min(top_k, N)

        c = np.zeros((N, top_k))
        d2 = np.diag(L).copy().astype(np.float64) + self.eps
        selected_idx: list[int] = []

        for j in range(top_k):
            i = int(np.argmax(d2))
            selected_idx.append(i)
            if j == top_k - 1:
                break
            num = L[i, :] - c[:, :j] @ c[i, :j]
            ei = num / np.sqrt(d2[i] + self.eps)
            c[:, j] = ei
            d2 = d2 - ei**2
            d2[i] = -np.inf

        return [int(candidates[i]) for i in selected_idx]
