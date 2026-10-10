# Reproducing the sender rejection audit

Run `python -m decentralized_stress benchmark --root . --config configs/aggregation_diagnostics.json --output-dir results/local-aggregation-diagnostics`, then verify that output with the usual `verify` command. The 18-run matrix holds the original Adult splits, three seeds, training settings, and attack fixed. Instrumentation is enabled only for this configuration and only emits coordinate-rejection diagnostics for trimmed mean.

For each parameter coordinate, the actual rank-based rejected sender index is counted. Equal values that straddle a trimming boundary are also counted separately. Tie-averaged rejection divides the rejected quota equally among all equal values: with updates `[1, 2, 2, 2]` and one value trimmed at each end, rejection probabilities are `[1, 1/3, 1/3, 1/3]`. This is an attribution convention; aggregation itself continues to use the exact original sorted values.

`aggregation_diagnostics.csv` contains every receiver/round/sender observation. `per_sender_diagnostics.csv` averages those observations without treating them as independent samples. `diagnostic_summary.json` includes each paired clean/attack accuracy delta, while `DIAGNOSTICS.md` presents the selected comparisons. No post-hoc run selection is applied.

CI runs the full benchmark and this diagnostic comparison every Monday at 01:29 UTC, on changes, and on manual dispatch. Artifacts are retained for 90 days. The committed outputs remain available after an Actions artifact expires. Dependabot proposes GitHub Actions updates weekly; it does not merge them automatically.
