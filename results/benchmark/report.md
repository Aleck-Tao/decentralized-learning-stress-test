# Decentralized learning stress-test report

> This report is generated from a peer-to-peer simulation. It is not a network deployment, Byzantine-tolerance proof, privacy guarantee, or public-sector pilot.

Dataset class: `external_real_dataset`  
Peers: 8  
Runs: 54  
Topology: `all_to_all`

| Scenario | Aggregator | Runs | Ensemble balanced accuracy | Worst-peer holdout balanced accuracy | Prediction disagreement | MIA ROC-AUC | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| iid_real | coordinate_median | 3 | 0.758 | 0.736 | 0.000 | 0.485 | 3.284 |
| iid_real | peer_mean | 3 | 0.758 | 0.735 | 0.000 | 0.485 | 3.284 |
| iid_real | trimmed_mean | 3 | 0.758 | 0.735 | 0.000 | 0.485 | 3.284 |
| noniid_scarce_real | coordinate_median | 3 | 0.667 | 0.631 | 0.000 | 0.532 | 3.284 |
| noniid_scarce_real | peer_mean | 3 | 0.720 | 0.664 | 0.000 | 0.532 | 3.284 |
| noniid_scarce_real | trimmed_mean | 3 | 0.678 | 0.636 | 0.000 | 0.533 | 3.284 |
| noniid_scarce_synthetic | coordinate_median | 3 | 0.667 | 0.618 | 0.000 | 0.545 | 3.284 |
| noniid_scarce_synthetic | peer_mean | 3 | 0.729 | 0.689 | 0.000 | 0.535 | 3.284 |
| noniid_scarce_synthetic | trimmed_mean | 3 | 0.683 | 0.628 | 0.000 | 0.544 | 3.284 |
| noniid_sign_flip | coordinate_median | 3 | 0.605 | 0.470 | 0.000 | 0.496 | 3.284 |
| noniid_sign_flip | peer_mean | 3 | 0.484 | 0.175 | 0.000 | 0.406 | 3.284 |
| noniid_sign_flip | trimmed_mean | 3 | 0.677 | 0.630 | 0.000 | 0.514 | 3.284 |
| noniid_synthetic_poison | coordinate_median | 3 | 0.628 | 0.553 | 0.000 | 0.506 | 3.284 |
| noniid_synthetic_poison | peer_mean | 3 | 0.707 | 0.652 | 0.000 | 0.529 | 3.284 |
| noniid_synthetic_poison | trimmed_mean | 3 | 0.654 | 0.607 | 0.000 | 0.517 | 3.284 |
| noniid_synthetic_privacy_audit | coordinate_median | 3 | 0.666 | 0.618 | 0.000 | 0.545 | 3.284 |
| noniid_synthetic_privacy_audit | peer_mean | 3 | 0.741 | 0.695 | 0.000 | 0.531 | 3.284 |
| noniid_synthetic_privacy_audit | trimmed_mean | 3 | 0.689 | 0.639 | 0.000 | 0.544 | 3.284 |

## Interpretation boundary

The client partitions are generated from a public benchmark and do not represent real organisations. Synthetic augmentation is a non-private local bootstrap with bounded jitter. Membership inference is one loss-based black-box attack; a low score does not establish privacy. Group metrics are diagnostics, not a fairness certification.
