# Decentralized Learning Stress Test

[![CI](https://github.com/Aleck-Tao/decentralized-learning-stress-test/actions/workflows/ci.yml/badge.svg)](https://github.com/Aleck-Tao/decentralized-learning-stress-test/actions/workflows/ci.yml)

How much clean-data performance is worth giving up for resistance to poisoned updates? This CPU PyTorch benchmark compares mean, coordinate-median, and trimmed-mean aggregation across eight simulated data owners using UCI Adult.

Each peer trains locally and exchanges model deltas. The main experiment uses synchronous all-to-all communication; a separate ring fixture exercises peers with different local views. The implementation, data splits, and measurements are available alongside the results.

## What the results show

The [Adult benchmark](results/benchmark/report.md) covers six scenarios, three aggregators, and three seeds. The table below shows mean ensemble balanced accuracy; the clean synthetic mix is the reference for both attacks.

| Training condition | Peer mean | Coordinate median | Trimmed mean |
|---|---:|---:|---:|
| IID real rows | 0.758 | 0.758 | 0.758 |
| Non-IID, scarce real rows | 0.720 | 0.667 | 0.678 |
| Non-IID, scarce + local synthetic rows | 0.729 | 0.667 | 0.683 |
| Sign-flipped updates | 0.484 | 0.605 | 0.677 |
| Poisoned synthetic labels | 0.707 | 0.628 | 0.654 |

- Trimming has a visible trade-off. Under sign-flip, its accuracy falls by 0.006 versus 0.245 for the mean. On the clean synthetic mix, however, it starts 0.046 below the mean. The robust rule helps against this attack at the cost of lower clean-data accuracy in this comparison.
- Synthetic augmentation gives a small, uneven gain. The mean rises from 0.720 to 0.729, while coordinate median is essentially unchanged. Resampling scarce local rows changes their training weight and adds numeric jitter; it cannot supply missing classes or new source populations.
- The loss-based membership audit remains close to chance in several conditions: ROC-AUC is 0.532 for scarce real data and 0.535 after augmentation with peer mean. That is a measurement of one attack, not evidence that the synthetic rows protect privacy.
- Zero disagreement in the all-to-all run follows from common initialization and common messages. The [ring fixture](results/ring-smoke/report.md) has 0.12–0.32 prediction disagreement. It tests local-view behavior on generated data; it is not another Adult benchmark.

[Analysis notes](docs/result_analysis.md) explain the update arithmetic and the clean-data/attack trade-off. [Per-run measurements](results/benchmark/benchmark_runs.csv), [round histories](results/benchmark/round_metrics.csv), and the [complete summary](results/benchmark/benchmark_summary.json) retain the values behind the table, including the larger synthetic-mix scenario.

## Run it

Use Python 3.11+ in a virtual environment. The pinned Adult archive is included under `data/raw/`.

```bash
python -m pip install -e .
dlstress verify --root . --result-dir results/benchmark
dlstress smoke --root . --output-dir results/local/quick-check
dlstress verify --root . --result-dir results/local/quick-check
```

The smoke command uses a generated fixture. To rerun the Adult experiment and tests:

```bash
python -m unittest discover -s tests -v
dlstress benchmark --root . --output-dir results/local/adult-rerun
dlstress verify --root . --result-dir results/local/adult-rerun
```

The [configuration](configs/benchmark.json) fixes 12 training rounds, the peer partitions, attack settings, and seeds. Verification checks source/config hashes, dataset bytes, partition and synthetic lineage, and result files. Floating-point metrics may differ slightly across PyTorch/BLAS environments.

## Data and scope

The [source lock](data/source_lock.json) pins the official UCI archive by SHA-256. The original training file supplies peer train/holdout partitions; the official test file is split into disjoint utility, membership-calibration, and membership-evaluation sets. Source-row IDs are checked for overlap. Per-peer class counts are recorded because a single-class holdout reports that class's recall, not two-class balanced performance. See [data provenance](docs/data_provenance.md).

This is a single-process simulation on a public dataset. The generated peer partitions represent experimental data owners, and the local bootstrap generator is non-private. The results concern this MLP, topology, and attack setup; they do not establish a network deployment, formal Byzantine tolerance, or a privacy guarantee. [Methodology](docs/methodology.md), [threat model](docs/threat_model.md), and [privacy audit](docs/privacy_boundary.md) give the details.

Code: MIT. Dataset: Becker, B. and Kohavi, R. (1996), *Adult*, UCI Machine Learning Repository, [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20), CC BY 4.0.
