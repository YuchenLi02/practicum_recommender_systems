from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.statistics import bootstrap_ci

METRICS = ["NDCG@10", "Recall@10", "HR@10", "Diversity", "Coverage", "AvgPop"]


def main() -> None:
    """Collapse per-cell rerank JSONs into long and wide summary CSVs."""
    p = argparse.ArgumentParser()
    p.add_argument("--rerank-dir", default="results/rerank")
    p.add_argument("--out-dir", default="results/aggregate")
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_long: list[dict] = []
    grouped_per_user: dict[tuple, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    grouped_summary: dict[tuple, list[dict]] = defaultdict(list)

    for path in sorted(Path(args.rerank_dir).glob("*.json")):
        key = path.stem
        base, dataset, seed_str = key.split("__")
        seed = int(seed_str.replace("seed", ""))
        with open(path) as f:
            data = json.load(f)
        for rr_name, payload in data.items():
            for m, v in payload["summary"].items():
                rows_long.append({
                    "base": base, "dataset": dataset, "reranker": rr_name,
                    "seed": seed, "metric": m, "value": v,
                })
            grouped_summary[(base, dataset, rr_name)].append(payload["summary"])
            for m, arr in payload["per_user"].items():
                grouped_per_user[(base, dataset, rr_name)][m].append(np.asarray(arr, dtype=np.float32))

    with open(out_dir / "summary_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["base", "dataset", "reranker", "seed", "metric", "value"])
        w.writeheader()
        w.writerows(rows_long)

    wide_rows: list[dict] = []
    for (base, dataset, rr), summaries in grouped_summary.items():
        row = {"base": base, "dataset": dataset, "reranker": rr, "n_seeds": len(summaries)}
        per_user_metric_arrays = grouped_per_user[(base, dataset, rr)]
        for m in METRICS:
            vs = np.array([s.get(m, np.nan) for s in summaries], dtype=np.float64)
            row[f"{m}_mean"] = float(np.nanmean(vs))
            row[f"{m}_seed_std"] = float(np.nanstd(vs, ddof=1)) if len(vs) > 1 else 0.0
            if m in per_user_metric_arrays:
                concat = np.concatenate(per_user_metric_arrays[m])
                mean, lo, hi = bootstrap_ci(concat, n_boot=args.n_boot)
                row[f"{m}_ci_lo"] = lo
                row[f"{m}_ci_hi"] = hi
        wide_rows.append(row)

    if wide_rows:
        cols = sorted({k for r in wide_rows for k in r.keys()})
        with open(out_dir / "summary_wide.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(wide_rows)

    print(f"long rows: {len(rows_long)}, wide rows: {len(wide_rows)}")


if __name__ == "__main__":
    main()
