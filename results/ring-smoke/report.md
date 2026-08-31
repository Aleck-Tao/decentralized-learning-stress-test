# Decentralized learning stress-test report

> This report is generated from a peer-to-peer simulation. It is not a network deployment, Byzantine-tolerance proof, privacy guarantee, or public-sector pilot.

Dataset class: `generated_test_fixture`  
Peers: 5  
Runs: 6  
Topology: `ring`

| Scenario | Aggregator | Runs | Ensemble balanced accuracy | Worst-peer holdout balanced accuracy | Prediction disagreement | MIA ROC-AUC | Communication MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| noniid_scarce_synthetic | coordinate_median | 1 | 0.510 | 0.500 | 0.157 | 0.586 | 0.019 |
| noniid_scarce_synthetic | peer_mean | 1 | 0.517 | 0.414 | 0.121 | 0.586 | 0.019 |
| noniid_scarce_synthetic | trimmed_mean | 1 | 0.510 | 0.500 | 0.157 | 0.586 | 0.019 |
| noniid_sign_flip | coordinate_median | 1 | 0.496 | 0.118 | 0.199 | 0.533 | 0.019 |
| noniid_sign_flip | peer_mean | 1 | 0.529 | 0.000 | 0.324 | 0.347 | 0.019 |
| noniid_sign_flip | trimmed_mean | 1 | 0.496 | 0.118 | 0.199 | 0.533 | 0.019 |

## Interpretation boundary

The client partitions are generated from a public benchmark and do not represent real organisations. Synthetic augmentation is a non-private local bootstrap with bounded jitter. Membership inference is one loss-based black-box attack; a low score does not establish privacy. Group metrics are diagnostics, not a fairness certification.
