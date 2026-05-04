# Modeling and Optimizing Large-Scale Recommender Systems for Dynamic User Engagement

The pipeline trains five base rankers on two catalogs over five seeds, sweeps two production re-rankers (MMR, DPP) over their tuning parameter, and reports six metrics with paired bootstrap confidence intervals.

## Layout

    src/          library code: data loaders, base rankers, re-rankers, evaluation
    scripts/      pipeline entry points (train, rerank, aggregate)
    experiments/  shared dataset and model factories
    configs/      per-dataset YAML
    slurm/        sbatch scripts
    paper/        compiled PDF and LaTeX source

## Install

    pip install -r requirements.txt

## Data

MovieLens 1M is downloaded automatically into data/ml-1m on first run.

Amazon Beauty 5-core requires manual download of reviews_Beauty_5.json.gz and meta_Beauty.json.gz from the Stanford SNAP mirror into data/amazon_beauty.

## Run

Local single-cell:

    PYTHONPATH=. python -m scripts.train_base --idx 0
    PYTHONPATH=. python -m scripts.run_rerank --workers 4
    PYTHONPATH=. python -m scripts.aggregate

The matrix is 5 base rankers x 2 datasets x 5 seeds = 50 cells. Linear job indices are produced by scripts.job_index.

SLURM array (one cell per task):

    sbatch --array=0-49 slurm/train_base.sbatch
    sbatch slurm/rerank.sbatch

## Output

    results/base_cache/   per-cell pickles with top-50 candidates and base scores
    results/rerank/       per-cell JSONs with summary, per-user, by-tercile metrics
    results/aggregate/    long and wide CSVs of every metric across the matrix

## Metrics

NDCG@10, Recall@10, HR@10, intra-list diversity, catalog coverage, average recommended popularity. All computed at top-10. Statistical tests use percentile bootstrap on per-user values with Bonferroni correction across pairwise comparisons within a cell.
