# What coordinate trimming removes

Passive diagnostics accompany matched clean/sign-flip runs. The targeted-sender label uses the attacked run's sender IDs in both conditions; all senders are honest in the clean run. The aggregation rule never reads that evaluator label.

| Scenario | Sender group | Tie-averaged rejection | Boundary ambiguity | Mean update L2 norm |
|---|---|---:|---:|---:|
| noniid_scarce_synthetic | other_sender | 0.4629 | 0.3445 | 0.04541 |
| noniid_scarce_synthetic | targeted_sender | 0.6114 | 0.3377 | 0.07155 |
| noniid_sign_flip | other_sender | 0.4232 | 0.3362 | 0.04427 |
| noniid_sign_flip | targeted_sender | 0.7303 | 0.3353 | 0.35735 |

Exact rank-based rejection and tie-averaged rejection are both retained. If identical values straddle a trimming boundary, the exact sender assignment depends on sorting; the tie-averaged measure divides rejected slots equally among equal values. Each eight-peer receiver removes half the coordinates overall (two at each end).

Rows average sender summaries across the same three seeds. Rounds and receivers are repeated observations, not independent replicates. Per-sender class proportions, sample counts, and rejection histories are in per_sender_diagnostics.csv and aggregation_diagnostics.csv.

| Aggregator | Seed | Clean balanced accuracy | Attacked balanced accuracy | Attack minus clean |
|---|---:|---:|---:|---:|
| coordinate_median | 101 | 0.73823 | 0.69757 | -0.04066 |
| coordinate_median | 307 | 0.76192 | 0.61523 | -0.14669 |
| coordinate_median | 809 | 0.50003 | 0.50112 | 0.00109 |
| peer_mean | 101 | 0.76660 | 0.50000 | -0.26660 |
| peer_mean | 307 | 0.76391 | 0.50000 | -0.26391 |
| peer_mean | 809 | 0.65655 | 0.45064 | -0.20591 |
| trimmed_mean | 101 | 0.73656 | 0.71902 | -0.01753 |
| trimmed_mean | 307 | 0.76031 | 0.72998 | -0.03033 |
| trimmed_mean | 809 | 0.55100 | 0.58112 | 0.03012 |

A rejected coordinate is not automatically harmful or useful. These measurements show whose values are removed and how often; they do not prove that minority-label information caused the clean-accuracy gap. That causal question needs a further intervention on partitions or training, not a correlation between label skew and rejection.
