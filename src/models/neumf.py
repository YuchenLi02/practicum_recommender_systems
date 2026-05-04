from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import Recommender


class _NCFDataset(Dataset):
    def __init__(self, train_pairs, user_train, n_items, n_neg=4, rng=None):
        self.samples = []
        rng = rng or np.random
        all_items = set(range(n_items))
        for u, pos_it in train_pairs:
            self.samples.append((u, pos_it, 1.0))
            neg_pool = list(all_items - user_train[u])
            negs = rng.choice(neg_pool, min(n_neg, len(neg_pool)), replace=False)
            for neg in negs:
                self.samples.append((u, int(neg), 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, it, label = self.samples[idx]
        return (
            torch.tensor([u], dtype=torch.long),
            torch.tensor([it], dtype=torch.long),
            torch.tensor([label], dtype=torch.float),
        )


class _NeuMFModule(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, mlp_dims=(64, 32, 16, 8)):
        super().__init__()
        self.gmf_u = nn.Embedding(n_users, emb_dim)
        self.gmf_i = nn.Embedding(n_items, emb_dim)
        self.mlp_u = nn.Embedding(n_users, emb_dim)
        self.mlp_i = nn.Embedding(n_items, emb_dim)

        layers, in_dim = [], emb_dim * 2
        for out_dim in mlp_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(emb_dim + mlp_dims[-1], 1)

        for emb in [self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u, it):
        u, it = u.squeeze(1), it.squeeze(1)
        gmf = self.gmf_u(u) * self.gmf_i(it)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(it)], dim=1))
        return torch.sigmoid(self.out(torch.cat([gmf, mlp], dim=1))).squeeze(1)


class NeuMFRecommender(Recommender):
    name = "neumf"

    def __init__(
        self,
        emb_dim: int = 32,
        mlp_dims: tuple = (64, 32, 16, 8),
        epochs: int = 20,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        n_neg: int = 4,
        device: str | None = None,
    ):
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_neg = n_neg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _NeuMFModule | None = None
        self.user_train: dict[int, set[int]] | None = None
        self._scores: dict[int, np.ndarray] | None = None
        self.n_items: int = 0

    def fit(self, n_users, n_items, user_train, interactions=None, **kwargs):
        self.user_train = user_train
        self.n_items = n_items

        train_pairs = [(u, it) for u, items in user_train.items() for it in items]
        dataset = _NCFDataset(train_pairs, user_train, n_items, n_neg=self.n_neg)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        self.model = _NeuMFModule(n_users, n_items, self.emb_dim, self.mlp_dims).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCELoss()

        for _ in range(self.epochs):
            self.model.train()
            for u, it, label in loader:
                u, it, label = u.to(self.device), it.to(self.device), label.to(self.device)
                opt.zero_grad()
                pred = self.model(u, it)
                loss = loss_fn(pred, label.squeeze(1))
                loss.backward()
                opt.step()

        self._cache_scores()

    def _cache_scores(self) -> None:
        assert self.model is not None and self.user_train is not None
        self.model.eval()
        self._scores = {}
        all_items = torch.arange(self.n_items, device=self.device)
        users = sorted(self.user_train.keys())
        BATCH = 64
        with torch.no_grad():
            for start in range(0, len(users), BATCH):
                batch = users[start : start + BATCH]
                u_t = (
                    torch.tensor(batch, device=self.device)
                    .repeat_interleave(self.n_items)
                    .unsqueeze(1)
                )
                i_t = all_items.repeat(len(batch)).unsqueeze(1)
                sc = self.model(u_t, i_t).cpu().numpy().reshape(len(batch), self.n_items)
                for i, u in enumerate(batch):
                    self._scores[u] = sc[i]

    def recommend(self, users, top_n, exclude_train=True):
        assert self._scores is not None and self.user_train is not None
        out: dict[int, list[int]] = {}
        for u in users:
            sc = self._scores[u].copy()
            if exclude_train:
                for it in self.user_train.get(int(u), set()):
                    sc[it] = -1e9
            out[int(u)] = np.argsort(-sc)[:top_n].tolist()
        return out

    def score(self, users, items):
        assert self._scores is not None
        users = list(users)
        items = list(items)
        return np.stack([self._scores[u][items] for u in users])
