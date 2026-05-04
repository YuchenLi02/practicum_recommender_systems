from __future__ import annotations

BASES = ["popularity", "als", "neumf", "sasrec", "tiger"]
DATASETS = ["ml-1m", "amazon-beauty"]
SEEDS = [0, 1, 2, 3, 4]


def n_jobs() -> int:
    """Total number of (base, dataset, seed) cells."""
    return len(BASES) * len(DATASETS) * len(SEEDS)


def decode(idx: int) -> tuple[str, str, int]:
    """Map a linear job index to a (base, dataset, seed) tuple."""
    n_seed = len(SEEDS)
    n_ds = len(DATASETS)
    base_i = idx // (n_ds * n_seed)
    rem = idx % (n_ds * n_seed)
    ds_i = rem // n_seed
    seed_i = rem % n_seed
    return BASES[base_i], DATASETS[ds_i], SEEDS[seed_i]


def all_jobs():
    """Iterate over every (idx, base, dataset, seed) tuple."""
    for i in range(n_jobs()):
        yield (i, *decode(i))


if __name__ == "__main__":
    for i, b, d, s in all_jobs():
        print(f"{i:3d}  base={b:10s}  dataset={d:14s}  seed={s}")
