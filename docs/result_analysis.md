# Reading the aggregation results

These notes interpret the [committed Adult run](../results/benchmark/report.md) using the implemented update rules. Accuracy below means ensemble balanced accuracy. The algebra below is a simplified explanation, not an additional experiment or a robustness theorem.

## Why the sign-flip treatment is severe

The [benchmark configuration](../configs/benchmark.json) uses eight peers, two malicious senders, and a sign-flip multiplier of −5. [Peer mean](../src/decentralized_stress/aggregation.py) gives every sender equal weight. In the illustrative case where all local deltas would have been the same vector `g`, the attacked aggregate is

```text
(6g + 2 × (−5g)) / 8 = −0.5g.
```

Thus two senders can reverse the mean update even though six are honest. Actual non-IID peers produce different deltas, so this calculation explains the attack's leverage without predicting the observed accuracy of 0.484.

With `trim_count = 2`, trimmed mean sorts each coordinate, removes the two largest and two smallest values, and averages the remaining four. Extreme malicious coordinates can be removed, but an attacker need not be extreme on every coordinate. The retained coordinates can also come from different senders. The measured 0.006 accuracy drop is specific to this attack and setting.

## Robustness has a clean-data cost

On the clean synthetic mix, peer mean reaches 0.729 and trimmed mean 0.683. Under sign-flip the ordering reverses: 0.484 versus 0.677. Reporting only the attacked case would hide the starting-point difference.

Label-skewed honest peers can generate different useful update directions. Coordinate trimming has no way to label an extreme value as either a minority-data contribution or a poisoned contribution. This is a plausible mechanism for the clean-data cost; the current run does not log coordinate-by-coordinate rejection identities, so it does not directly attribute the loss to particular peers. That attribution would require an instrumented follow-up.

The synthetic generator in [data.py](../src/decentralized_stress/data.py) resamples a scarce peer's own rows and jitters numeric features. It increases repeated exposure to local examples, but it does not add a class absent from that peer. The small gain for peer mean and near-zero gain for coordinate median are therefore compatible with augmentation changing optimization without resolving the underlying label skew.

## What consensus means in this experiment

The [protocol](../src/decentralized_stress/protocol.py) applies an aggregate delta to each recipient's previous model:

```text
theta_i(t + 1) = theta_i(t) + Aggregate({delta_j(t): j in N_i plus i}).
```

For all-to-all communication, every recipient has the same message set. With identical initial models, order-independent aggregation gives identical next models by induction. Floating-point reduction order can introduce tiny differences; the committed metrics report zero disagreement and zero distance at their stored precision. Consequently, the reported ensemble does not demonstrate a benefit from model diversity in this run.

The ring has only three values per receiver, including its own. Trimming two from each end would leave no values and is rejected by configuration preflight. With one trimmed from each end, the surviving value is the coordinate median. That explains the matching median/trimmed-mean rows in the [five-peer ring smoke report](../results/ring-smoke/report.md), rather than treating them as two independent robust mechanisms.

## Evidence to inspect

The [run table](../results/benchmark/benchmark_runs.csv) retains each seed rather than only the means. The [partition manifest](../data/generated/benchmark/partition_manifest.json) records per-peer class counts and source-row hashes; this matters when interpreting worst-peer holdout scores. The [round table](../results/benchmark/round_metrics.csv) records local loss and communication over time. The [membership audit definition](methodology.md#membership-inference) explains its disjoint calibration/evaluation samples and signed operating-point measurements.
