from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .base import Recommender


class _SASRecModule(nn.Module):
    def __init__(self, n_items: int, hidden_units: int, num_blocks: int,
                 num_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        self.attn_norm = nn.ModuleList([nn.LayerNorm(hidden_units) for _ in range(num_blocks)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_blocks)
        ])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(hidden_units) for _ in range(num_blocks)])
        self.ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, hidden_units),
                nn.Dropout(dropout),
            ) for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_units)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """Embed input ids and run causal self-attention; return per-position hidden."""
        B, L = seqs.shape
        positions = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seqs) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (seqs == 0)
        causal = torch.triu(torch.ones(L, L, device=seqs.device, dtype=torch.bool), diagonal=1)

        for blk in range(len(self.attn)):
            q = self.attn_norm[blk](x)
            attn_out, _ = self.attn[blk](q, x, x, attn_mask=causal, need_weights=False)
            x = x + attn_out
            x = x + self.ff[blk](self.ff_norm[blk](x))
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return self.out_norm(x)

    def score_all_items(self, hidden_last: torch.Tensor) -> torch.Tensor:
        """Project the final hidden state onto the item embedding table."""
        return hidden_last @ self.item_emb.weight.T


def _build_sequences(user_train: dict[int, set[int]],
                     interactions, max_seq_len: int):
    """Build right-padded chronological item sequences per user."""
    train_set = {int(u): set(map(int, items)) for u, items in user_train.items()}
    if interactions is None:
        seqs = {}
        for u, its in train_set.items():
            arr = np.zeros(max_seq_len, dtype=np.int64)
            its_list = list(its)[-max_seq_len:]
            arr[-len(its_list):] = np.array(its_list, dtype=np.int64) + 1
            seqs[u] = arr
        return seqs

    df = interactions.sort_values(["uid", "timestamp"])
    seqs: dict[int, np.ndarray] = {}
    for uid, grp in df.groupby("uid"):
        u = int(uid)
        items = [int(it) for it in grp["iid"].tolist() if int(it) in train_set.get(u, set())]
        items = items[-max_seq_len:]
        arr = np.zeros(max_seq_len, dtype=np.int64)
        if items:
            arr[-len(items):] = np.array(items, dtype=np.int64) + 1
        seqs[u] = arr
    return seqs


class SASRecRecommender(Recommender):
    name = "sasrec"

    def __init__(
        self,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_neg: int = 1,
        device: str | None = None,
    ):
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_neg = n_neg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._scores: dict[int, np.ndarray] | None = None
        self.user_train: dict[int, set[int]] | None = None
        self.n_items: int = 0

    def fit(self, n_users, n_items, user_train, interactions=None, **kwargs):
        self.user_train = {int(u): set(map(int, v)) for u, v in user_train.items()}
        self.n_items = n_items

        seqs = _build_sequences(self.user_train, interactions, self.max_seq_len)
        users = sorted(seqs)
        if not users:
            self._scores = {}
            return

        all_input = []
        all_target = []
        for u in users:
            full = seqs[u]
            all_input.append(full[:-1])
            all_target.append(full[1:])

        all_input = np.stack(all_input).astype(np.int64)
        all_target = np.stack(all_target).astype(np.int64)
        U = len(users)

        device = torch.device(self.device)
        model = _SASRecModule(
            n_items, self.hidden_units, self.num_blocks, self.num_heads,
            self.dropout, max_seq_len=self.max_seq_len,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay, betas=(0.9, 0.98))
        bce = nn.BCEWithLogitsLoss(reduction="none")

        rng = np.random.default_rng(0)
        all_input_t = torch.from_numpy(all_input).to(device)
        all_target_t = torch.from_numpy(all_target).to(device)

        for _ in range(self.epochs):
            perm = rng.permutation(U)
            model.train()
            for start in range(0, U, self.batch_size):
                idx = perm[start:start + self.batch_size]
                seq = all_input_t[idx]
                pos = all_target_t[idx]
                neg = torch.from_numpy(
                    rng.integers(1, n_items + 1, size=pos.shape).astype(np.int64)
                ).to(device)
                same = (neg == pos) & (pos != 0)
                if same.any():
                    neg = torch.where(same, (neg % n_items) + 1, neg)

                hidden = model(seq)
                pos_emb = model.item_emb(pos)
                neg_emb = model.item_emb(neg)
                pos_logit = (hidden * pos_emb).sum(-1)
                neg_logit = (hidden * neg_emb).sum(-1)

                pos_label = torch.ones_like(pos_logit)
                neg_label = torch.zeros_like(neg_logit)
                mask = (pos != 0).float()
                loss_pos = (bce(pos_logit, pos_label) * mask).sum() / mask.sum().clamp(min=1)
                loss_neg = (bce(neg_logit, neg_label) * mask).sum() / mask.sum().clamp(min=1)
                loss = loss_pos + loss_neg
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._cache_scores(model, seqs, users, n_items, device)

    def _cache_scores(self, model, seqs, users, n_items, device):
        model.eval()
        scores: dict[int, np.ndarray] = {}
        BATCH = 256
        with torch.no_grad():
            for start in range(0, len(users), BATCH):
                batch_users = users[start:start + BATCH]
                batch = np.stack([seqs[u] for u in batch_users])
                seq_t = torch.from_numpy(batch).to(device)
                hidden = model(seq_t)
                last_h = hidden[:, -1, :]
                logits = model.score_all_items(last_h)
                sc = logits[:, 1:].cpu().numpy().astype(np.float32)
                for i, u in enumerate(batch_users):
                    scores[u] = sc[i]
        self._scores = scores

    def recommend(self, users: Iterable[int], top_n: int, exclude_train: bool = True):
        assert self._scores is not None and self.user_train is not None
        out: dict[int, list[int]] = {}
        for u in users:
            sc = self._scores.get(int(u))
            if sc is None:
                out[int(u)] = []
                continue
            sc = sc.copy()
            if exclude_train:
                for it in self.user_train.get(int(u), set()):
                    sc[it] = -1e9
            out[int(u)] = np.argsort(-sc)[:top_n].tolist()
        return out

    def score(self, users, items):
        assert self._scores is not None
        users = list(users)
        items = list(items)
        return np.stack([self._scores[int(u)][items] for u in users])
