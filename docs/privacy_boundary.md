# Privacy boundary

Version 0.1 measures one privacy risk and implements no privacy defence.

The loss-based membership audit is useful because it forces member/nonmember split discipline and produces an interpretable ROC-AUC and operating point. It is insufficient to show privacy: attack power depends on model access, calibration data, sample selection, training regime, and attacker knowledge. An AUC near or below 0.5 for this attack does not rule out a stronger or inverted attack.

The local synthetic generator is a seeded bootstrap with numeric jitter. It has no privacy accounting, can duplicate or closely approximate source rows, and must never be called differentially private, anonymised, or safe to release.

Not implemented:

- differential privacy or an epsilon/delta accountant;
- secure aggregation or encryption;
- reconstruction, gradient inversion, property inference, or attribute inference;
- protection against repeated-query or composition leakage;
- a release policy for genuinely private records.

A future DP experiment must define adjacency, clipping, mechanism, accountant, composition, randomness handling, and the unit of protection before exposing epsilon. It should not publish a fixed random seed while claiming that the realised noise remains secret.

