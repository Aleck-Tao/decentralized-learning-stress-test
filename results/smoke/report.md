# Decentralized learning stress-test report

> This report is generated from a peer-to-peer simulation. It is not a network deployment, Byzantine-tolerance proof, privacy guarantee, or public-sector pilot.

Dataset class: `generated_test_fixture`  
Peers: 4  
Runs: 15  
Topology: `all_to_all`

| Scenario | Aggregator | Runs | Ensemble balanced accuracy | Worst-peer holdout balanced accuracy | Prediction disagreement | MIA ROC-AUC | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| iid_real | coordinate_median | 1 | 0.612 | 0.506 | 0.000 | 0.485 | 0.023 |
| iid_real | peer_mean | 1 | 0.606 | 0.506 | 0.000 | 0.477 | 0.023 |
| iid_real | trimmed_mean | 1 | 0.612 | 0.506 | 0.000 | 0.485 | 0.023 |
| noniid_scarce_real | coordinate_median | 1 | 0.580 | 0.421 | 0.000 | 0.442 | 0.023 |
| noniid_scarce_real | peer_mean | 1 | 0.611 | 0.421 | 0.000 | 0.419 | 0.023 |
| noniid_scarce_real | trimmed_mean | 1 | 0.580 | 0.421 | 0.000 | 0.442 | 0.023 |
| noniid_scarce_synthetic | coordinate_median | 1 | 0.580 | 0.421 | 0.000 | 0.442 | 0.023 |
| noniid_scarce_synthetic | peer_mean | 1 | 0.611 | 0.421 | 0.000 | 0.419 | 0.023 |
| noniid_scarce_synthetic | trimmed_mean | 1 | 0.580 | 0.421 | 0.000 | 0.442 | 0.023 |
| noniid_sign_flip | coordinate_median | 1 | 0.508 | 0.333 | 0.000 | 0.363 | 0.023 |
| noniid_sign_flip | peer_mean | 1 | 0.499 | 0.030 | 0.000 | 0.358 | 0.023 |
| noniid_sign_flip | trimmed_mean | 1 | 0.508 | 0.333 | 0.000 | 0.363 | 0.023 |
| noniid_synthetic_poison | coordinate_median | 1 | 0.572 | 0.421 | 0.000 | 0.443 | 0.023 |
| noniid_synthetic_poison | peer_mean | 1 | 0.611 | 0.421 | 0.000 | 0.420 | 0.023 |
| noniid_synthetic_poison | trimmed_mean | 1 | 0.572 | 0.421 | 0.000 | 0.443 | 0.023 |

## Interpretation boundary

The client partitions are generated from a public benchmark and do not represent real organisations. Synthetic augmentation is a non-private local bootstrap with bounded jitter. Membership inference is one loss-based black-box attack; a low score does not establish privacy. Group metrics are diagnostics, not a fairness certification.
