from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import yaml  # noqa: F401

from experiments._common import (
    build_base_ranker,
    build_dataset,
    load_config,
    split_dataset,
)
from scripts.job_index import decode
from src.utils.seeding import set_seed

CONFIG_FOR = {
    "ml-1m": "configs/ml_1m.yaml",
    "amazon-beauty": "configs/amazon_beauty.yaml",
}


def find_base_spec(cfg: dict, name: str) -> dict:
    """Look up the base ranker config block by name."""
    for spec in cfg["base_rankers"]:
        if spec["name"] == name:
            return spec
    raise KeyError(f"base ranker {name!r} not found in config")


def main() -> None:
    """Train one (base, dataset, seed) cell and cache the top-N candidate list."""
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, required=True)
    p.add_argument("--out-dir", default="results/base_cache")
    p.add_argument("--n-cand", type=int, default=50)
    args = p.parse_args()

    base_name, dataset_name, seed = decode(args.idx)
    cfg = load_config(CONFIG_FOR[dataset_name])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}__{dataset_name}__seed{seed}.pkl"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print(f"[idx={args.idx}] base={base_name} dataset={dataset_name} seed={seed}")

    set_seed(seed)
    bundle = build_dataset(cfg["dataset"])
    user_train, user_test = split_dataset(bundle, cfg["dataset"]["train_frac"])
    test_users = sorted(u for u in user_test if user_train.get(u))

    spec = find_base_spec(cfg, base_name)
    base = build_base_ranker(spec)
    t0 = time.time()
    base.fit(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        user_train=user_train,
        interactions=bundle.interactions,
        item_content=bundle.item_content,
    )
    fit_sec = time.time() - t0
    print(f"  fit done in {fit_sec:.1f}s")

    cands_dict = base.recommend(test_users, top_n=args.n_cand, exclude_train=True)

    scores_dict: dict[int, np.ndarray] = {}
    for u in test_users:
        cs = cands_dict[int(u)]
        try:
            sc = base.score([u], cs).reshape(-1).astype(np.float32)
        except NotImplementedError:
            sc = np.linspace(1.0, 0.0, len(cs), dtype=np.float32)
        scores_dict[int(u)] = sc

    payload = {
        "base": base_name,
        "dataset": dataset_name,
        "seed": int(seed),
        "n_users": bundle.n_users,
        "n_items": bundle.n_items,
        "test_users": test_users,
        "candidates": {int(u): list(map(int, cands_dict[u])) for u in test_users},
        "scores": scores_dict,
        "user_train": {int(u): set(map(int, v)) for u, v in user_train.items()},
        "user_test": {int(u): set(map(int, v)) for u, v in user_test.items()},
        "wallclock_sec": fit_sec,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[idx={args.idx}] cached -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
