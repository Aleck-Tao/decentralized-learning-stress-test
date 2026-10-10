# Decentralized learning stress-test report

> This report is generated from a peer-to-peer simulation. It is not a network deployment, Byzantine-tolerance proof, privacy guarantee, or public-sector pilot.

Dataset class: `external_real_dataset`  
Peers: 8  
Runs: 18  
Topology: `all_to_all`

| Scenario | Aggregator | Runs | Ensemble balanced accuracy | Worst-peer holdout balanced accuracy | Prediction disagreement | MIA ROC-AUC | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| noniid_scarce_synthetic | coordinate_median | 3 | 0.667 | 0.618 | 0.000 | 0.545 | 3.284 |
| noniid_scarce_synthetic | peer_mean | 3 | 0.729 | 0.689 | 0.000 | 0.535 | 3.284 |
| noniid_scarce_synthetic | trimmed_mean | 3 | 0.683 | 0.628 | 0.000 | 0.544 | 3.284 |
| noniid_sign_flip | coordinate_median | 3 | 0.605 | 0.470 | 0.000 | 0.496 | 3.284 |
| noniid_sign_flip | peer_mean | 3 | 0.484 | 0.175 | 0.000 | 0.406 | 3.284 |
| noniid_sign_flip | trimmed_mean | 3 | 0.677 | 0.630 | 0.000 | 0.514 | 3.284 |

## Interpretation boundary

The client partitions are generated from a public benchmark and do not represent real organisations. Synthetic augmentation is a non-private local bootstrap with bounded jitter. Membership inference is one loss-based black-box attack; a low score does not establish privacy. Group metrics are diagnostics, not a fairness certification.
