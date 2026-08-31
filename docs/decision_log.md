# Decision log

This log records design choices made for the initial public prototype. It does not reconstruct earlier research activity.

## D-001: New bounded repository

Decentralized learning was kept separate from the UAV runtime-evidence repository. The two projects share reproducibility methods but answer different research questions.

## D-002: Peer-to-peer simulator before networking

The first release makes model exchange and local aggregation executable without adding sockets, containers, cryptography, or orchestration. This isolates learning and adversarial-evaluation decisions while keeping the absence of a real deployment explicit.

## D-003: Adult as a small real-data benchmark

The pinned UCI Adult archive is small enough for CPU CI and includes mixed numeric/categorical features and an evaluation-only demographic attribute. Its client boundaries are simulated and it is not treated as contemporary administrative data.

## D-004: Fixed preprocessing instead of pooled fitting

Fixed numeric transforms and category hashing avoid learning a shared encoder from pooled client rows. The collision and scaling limitations remain visible.

## D-005: Simple non-private synthetic treatment

A local bootstrap with jitter was chosen so its lineage and failure modes are explainable. Differential privacy was not added without a complete adjacency/accounting design.

## D-006: Pre-register scenarios, retain negative results

The benchmark compares clean, non-IID/scarce, synthetic-mix, sign-flip, and synthetic-poisoning scenarios. Robust aggregation is not required to outperform mean aggregation as a release gate; unexpected or negative outcomes remain in the report.

## D-007: Separate membership calibration and evaluation

Attack thresholds are selected on calibration examples and evaluated on disjoint source rows. Synthetic training rows are excluded as membership targets.

