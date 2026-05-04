from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from experiments._common import (
    build_dataset,
    build_reranker,
    build_runner,
    load_config,
)
from scripts.train_base import CONFIG_FOR
from src.utils.io import save_json
from src.utils.seeding import set_seed


def rerank_cell(cache_path: str) -> tuple[str, dict]:
    """Read one base-cache pickle, run every reranker in the config, return metrics."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    base_name = cache["base"]
    dataset_name = cache["dataset"]
    seed = cache["seed"]
    test_users = cache["test_users"]
    cands_dict = cache["candidates"]
    scores_dict = cache["scores"]
    user_train = cache["user_train"]
    user_test = cache["user_test"]

    cfg = load_config(CONFIG_FOR[dataset_name])
    bundle = build_dataset(cfg["dataset"])
    runner = build_runner(bundle, user_train)
    K = cfg["evaluation"]["k"]

    sorted_users = sorted(test_users, key=lambda u: len(user_train.get(int(u), set())))
    n = len(sorted_users)
    terciles = {
        "low": set(sorted_users[: n // 3]),
        "medium": set(sorted_users[n // 3 : 2 * n // 3]),
        "high": set(sorted_users[2 * n // 3 :]),
    }

    out: dict[str, dict] = {}
    for rr_spec in cfg["rerankers"]:
        if rr_spec.get("name") == "llm":
            continue
        set_seed(seed)
        rr = build_reranker(rr_spec)
        t0 = time.time()
        reranked: dict[int, list[int]] = {}
        for u in test_users:
            cands = list(cands_dict[int(u)])
            rel = np.asarray(scores_dict[int(u)], dtype=np.float32)
            reranked[int(u)] = rr.rerank(
                user=int(u),
                candidates=cands,
                relevance=rel,
                item_content=bundle.item_content,
                top_k=K,
            )
        ev = runner.evaluate(reranked, user_test)
        wallclock = time.time() - t0

        by_tercile: dict[str, dict] = {}
        for tname, tset in terciles.items():
            sub = {u: r for u, r in reranked.items() if u in tset}
            by_tercile[tname] = runner.evaluate(sub, user_test).summary

        out[rr.name] = {
            "summary": ev.summary,
            "per_user": {m: arr.tolist() for m, arr in ev.per_user.items()},
            "by_tercile": by_tercile,
            "wallclock_sec": wallclock,
        }
        print(
            f"[{base_name}/{dataset_name}/s{seed}/{rr.name}] "
            f"NDCG={ev.summary['NDCG@10']:.4f} ILD={ev.summary['Diversity']:.4f} "
            f"({wallclock:.1f}s)"
        )
    key = f"{base_name}__{dataset_name}__seed{seed}"
    return key, out


def _worker(cache_path):
    return rerank_cell(cache_path)


def main() -> None:
    """Sweep every cached base-ranker output through every reranker."""
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="results/base_cache")
    p.add_argument("--out-dir", default="results/rerank")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cache_paths = sorted(str(p) for p in Path(args.cache_dir).glob("*.pkl"))
    if not cache_paths:
        raise SystemExit(f"no cache found under {args.cache_dir}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.force:
        before = len(cache_paths)
        cache_paths = [
            cp for cp in cache_paths
            if not (Path(args.out_dir) / f"{Path(cp).stem}.json").exists()
        ]
        skipped = before - len(cache_paths)
        if skipped:
            print(f"[skip] {skipped} cells already in {args.out_dir}")

    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_worker, cache_paths)
    else:
        results = [_worker(cp) for cp in cache_paths]

    for key, out in results:
        save_json(out, Path(args.out_dir) / f"{key}.json")
    print(f"wrote {len(results)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
