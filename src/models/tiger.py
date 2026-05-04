from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Recommender


class _RQVAE(nn.Module):
    def __init__(self, content_dim: int, latent_dim: int = 64,
                 codebook_size: int = 64, n_codewords: int = 3,
                 commitment_weight: float = 0.25):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.encoder = nn.Sequential(
            nn.Linear(content_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, content_dim),
        )
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim) for _ in range(n_codewords)
        ])
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.n_codewords = n_codewords
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

    def quantize(self, z: torch.Tensor):
        """Residual vector quantize z; return codeword indices, quantized z, and commitment loss."""
        residual = z
        codes = []
        quantized = torch.zeros_like(z)
        commit_loss = z.new_zeros(())
        for cb in self.codebooks:
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ cb.weight.T
                + cb.weight.pow(2).sum(dim=1)
            )
            idx = dist.argmin(dim=1)
            q = cb(idx)
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())
            q_st = residual + (q - residual).detach()
            quantized = quantized + q_st
            residual = residual - q.detach()
            codes.append(idx)
        return torch.stack(codes, dim=1), quantized, commit_loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        codes, q, commit = self.quantize(z)
        x_hat = self.decoder(q)
        recon = F.mse_loss(x_hat, x)
        loss = recon + self.commitment_weight * commit
        return loss, codes, recon, commit


def _train_rqvae(content: np.ndarray, *, latent_dim: int, codebook_size: int,
                 n_codewords: int, epochs: int, batch_size: int, lr: float,
                 device: torch.device) -> tuple[np.ndarray, _RQVAE]:
    """Train the RQ-VAE on item content and return per-item codeword tuples."""
    n_items, content_dim = content.shape
    rqvae = _RQVAE(content_dim, latent_dim, codebook_size, n_codewords).to(device)
    opt = torch.optim.Adam(rqvae.parameters(), lr=lr)
    x = torch.from_numpy(content.astype(np.float32)).to(device)

    for _ in range(epochs):
        perm = torch.randperm(n_items, device=device)
        rqvae.train()
        for start in range(0, n_items, batch_size):
            idx = perm[start:start + batch_size]
            loss, _, _, _ = rqvae(x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    rqvae.eval()
    with torch.no_grad():
        codes, _, _ = rqvae.quantize(rqvae.encoder(x))
    return codes.cpu().numpy().astype(np.int64), rqvae


def _add_disambiguation(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """Append a disambiguation codeword so each item has a unique semantic ID."""
    n_items, k = codes.shape
    out = np.zeros((n_items, k + 1), dtype=np.int64)
    out[:, :k] = codes
    seen: dict[tuple, int] = {}
    for i in range(n_items):
        key = tuple(codes[i].tolist())
        out[i, k] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    return out


class _SemIDTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.tok_emb(x) + self.pos_emb(positions))
        causal = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1,
        )
        h = self.blocks(h, mask=causal, src_key_padding_mask=key_pad_mask)
        return self.head(self.norm(h))


def _build_sid_vocab(codes: np.ndarray) -> tuple[int, np.ndarray, dict, int, int, int]:
    """Flatten codeword positions into a single vocabulary and add PAD/SOS/EOS tokens."""
    k = codes.shape[1]
    block_sizes = [int(codes[:, j].max()) + 1 for j in range(k)]
    offsets = np.cumsum([0] + block_sizes)[:-1]
    sid = codes + offsets[None, :]
    base_vocab = int(sum(block_sizes))
    PAD, SOS, EOS = base_vocab, base_vocab + 1, base_vocab + 2
    vocab_size = base_vocab + 3
    sid_to_iid: dict[tuple, int] = {tuple(sid[i].tolist()): i for i in range(sid.shape[0])}
    return vocab_size, sid, sid_to_iid, PAD, SOS, EOS


def _build_seqs(user_train, interactions, sid: np.ndarray, max_history: int):
    """Per-user chronological list of training item ids, truncated to max_history."""
    train_set = {int(u): set(map(int, items)) for u, items in user_train.items()}
    df = interactions.sort_values(["uid", "timestamp"])
    out: dict[int, list[int]] = {}
    for uid, grp in df.groupby("uid"):
        u = int(uid)
        items = [int(it) for it in grp["iid"].tolist() if int(it) in train_set.get(u, set())]
        items = items[-max_history:]
        out[u] = items
    return out


class _SemIDDataset(torch.utils.data.Dataset):
    def __init__(self, seqs: dict[int, list[int]], sid: np.ndarray,
                 PAD: int, SOS: int, EOS: int, max_len: int, k: int):
        self.examples: list[tuple[int, int]] = []
        for u, items in seqs.items():
            if len(items) < 2:
                continue
            self.examples.append((u, len(items)))
        self.seqs = seqs
        self.sid = sid
        self.PAD, self.SOS, self.EOS = PAD, SOS, EOS
        self.max_len = max_len
        self.k = k

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        u, n = self.examples[i]
        items = self.seqs[u]
        history = items[:-1]
        target = items[-1]
        h_sids = self.sid[history].reshape(-1)
        max_hist_tokens = self.max_len - self.k - 2
        if len(h_sids) > max_hist_tokens:
            h_sids = h_sids[-max_hist_tokens:]
        t_sid = self.sid[target]
        seq = np.concatenate([[self.SOS], h_sids, t_sid, [self.EOS]]).astype(np.int64)
        x = seq[:-1]
        y = seq[1:]
        return x, y


def _pad_collate(batch, PAD):
    L = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = L - len(x)
        xs.append(np.concatenate([x, [PAD] * pad]))
        ys.append(np.concatenate([y, [PAD] * pad]))
    return torch.from_numpy(np.stack(xs).astype(np.int64)), torch.from_numpy(np.stack(ys).astype(np.int64))


class TIGERRecommender(Recommender):
    name = "tiger"

    def __init__(
        self,
        codebook_size: int = 64,
        n_codewords: int = 3,
        rqvae_epochs: int = 200,
        rqvae_lr: float = 1e-3,
        latent_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        transformer_epochs: int = 30,
        transformer_lr: float = 1e-3,
        batch_size: int = 64,
        beam_width: int = 50,
        max_history_items: int = 30,
        device: str | None = None,
    ):
        self.codebook_size = codebook_size
        self.n_codewords = n_codewords
        self.rqvae_epochs = rqvae_epochs
        self.rqvae_lr = rqvae_lr
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.transformer_epochs = transformer_epochs
        self.transformer_lr = transformer_lr
        self.batch_size = batch_size
        self.beam_width = beam_width
        self.max_history_items = max_history_items
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.user_train: dict[int, set[int]] | None = None
        self.n_items: int = 0
        self._candidates: dict[int, list[int]] | None = None
        self._candidate_log_probs: dict[int, np.ndarray] | None = None

    def fit(self, n_users, n_items, user_train, interactions=None, **kwargs):
        assert interactions is not None, "TIGER requires interaction timestamps"
        item_content = kwargs.get("item_content")
        if item_content is None:
            raise ValueError("TIGER requires item_content; pass via kwargs")
        self.user_train = {int(u): set(map(int, v)) for u, v in user_train.items()}
        self.n_items = n_items
        device = torch.device(self.device)

        codes, _ = _train_rqvae(
            item_content,
            latent_dim=self.latent_dim,
            codebook_size=self.codebook_size,
            n_codewords=self.n_codewords,
            epochs=self.rqvae_epochs,
            batch_size=256,
            lr=self.rqvae_lr,
            device=device,
        )
        codes = _add_disambiguation(codes, self.codebook_size)

        vocab_size, sid, sid_to_iid, PAD, SOS, EOS = _build_sid_vocab(codes)
        K = sid.shape[1]
        max_len = K * (self.max_history_items + 1) + 2

        seqs = _build_seqs(self.user_train, interactions, sid, self.max_history_items)
        if not seqs:
            self._candidates = {}
            self._candidate_log_probs = {}
            return

        ds = _SemIDDataset(seqs, sid, PAD, SOS, EOS, max_len, K)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=lambda b: _pad_collate(b, PAD), num_workers=0,
        )
        model = _SemIDTransformer(
            vocab_size, d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, max_len=max_len,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.transformer_lr)
        for _ in range(self.transformer_epochs):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, key_pad_mask=(x == PAD))
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=PAD,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        self._beam_search(model, sid, sid_to_iid, PAD, SOS, EOS, K, max_len, device, seqs)

    def _beam_search(self, model, sid: np.ndarray, sid_to_iid: dict, PAD: int,
                     SOS: int, EOS: int, K: int, max_len: int, device,
                     user_histories: dict[int, list[int]]):
        """Top-W beam search over the next K semantic-ID tokens for each user."""
        model.eval()
        users = sorted(user_histories.keys())
        cands: dict[int, list[int]] = {}
        cand_logp: dict[int, np.ndarray] = {}
        W = self.beam_width

        for u in users:
            history = user_histories.get(u, [])[-self.max_history_items:]
            if not history:
                cands[u] = []
                cand_logp[u] = np.array([], dtype=np.float32)
                continue
            h_sids = sid[history].reshape(-1)
            seq = np.concatenate([[SOS], h_sids]).astype(np.int64)
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)

            beams = [(seq_t, 0.0)]
            for _ in range(K):
                next_beams: list[tuple[torch.Tensor, float]] = []
                with torch.no_grad():
                    for s, lp in beams:
                        logits = model(s)
                        log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                        topw = torch.topk(log_probs, W)
                        for tok, sc in zip(topw.indices.tolist(), topw.values.tolist()):
                            new_s = torch.cat(
                                [s, torch.tensor([[tok]], device=device, dtype=torch.long)], dim=1,
                            )
                            next_beams.append((new_s, lp + sc))
                next_beams.sort(key=lambda b: -b[1])
                beams = next_beams[:W]

            seen: set[int] = set()
            picked_iid: list[int] = []
            picked_lp: list[float] = []
            for s, lp in beams:
                tail = tuple(s[0, -K:].tolist())
                iid = sid_to_iid.get(tail)
                if iid is None:
                    continue
                if iid in self.user_train[u]:
                    continue
                if iid in seen:
                    continue
                seen.add(iid)
                picked_iid.append(int(iid))
                picked_lp.append(float(lp))
                if len(picked_iid) >= W:
                    break
            cands[u] = picked_iid
            cand_logp[u] = np.array(picked_lp, dtype=np.float32)

        self._candidates = cands
        self._candidate_log_probs = cand_logp

    def recommend(self, users: Iterable[int], top_n: int, exclude_train: bool = True):
        assert self._candidates is not None
        out: dict[int, list[int]] = {}
        for u in users:
            cands = self._candidates.get(int(u), [])
            out[int(u)] = list(cands[:top_n])
        return out

    def score(self, users, items):
        assert self._candidates is not None and self._candidate_log_probs is not None
        users = list(users)
        items = list(items)
        out = np.full((len(users), len(items)), -np.inf, dtype=np.float32)
        for r, u in enumerate(users):
            cand_to_logp = dict(zip(self._candidates.get(int(u), []),
                                    self._candidate_log_probs.get(int(u), np.array([]))))
            for c, it in enumerate(items):
                if it in cand_to_logp:
                    out[r, c] = cand_to_logp[it]
        return out


