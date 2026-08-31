# Methodology

## Research boundary

The implementation is an in-process simulator of synchronous peer-to-peer model exchange. A peer owns its `Records` object, performs local PyTorch training, and emits a `PeerMessage` containing a model delta, sender/round identity, sample count, and an attack marker used only by the evaluator. No raw feature matrix or label vector is a message field.

The orchestrating Python process can inspect all peers because this is an experiment harness. Therefore the absence of raw rows from the protocol object is a software boundary, not isolation, access control, or confidential computing.

## Preprocessing without pooled fitted statistics

Six numeric Adult fields use fixed, documented clipping/log transforms. Seven categorical fields use per-field SHA-256 hashing into eight buckets. Categories and normalization statistics are therefore not fitted on a pooled client dataset. The `sex` field is excluded from training features and retained as an evaluation-only group label.

Hash collisions are possible and form part of the model limitation. Missing categorical values are explicit hashed tokens rather than rows silently discarded.

## Simulated peers and non-IID data

The UCI training split is assigned to eight peers. IID experiments split each class evenly after shuffling. Non-IID experiments draw class-allocation probabilities from a Dirichlet distribution with configured alpha, retrying until every peer has enough rows for train and holdout sets. A fixed subset of peers then retains only the configured fraction of local training rows to simulate scarcity.

The resulting peer IDs do not correspond to workclass, geography, agency, or any real organisation.

## Local synthetic augmentation

Only designated low-data peers generate synthetic rows. The generator bootstraps local examples with replacement and adds Gaussian jitter equal to 2% of local numeric-feature standard deviation, with bounded output. Hashed categorical indicators remain unchanged. The generator records the peer, ratio, seed, parent-ID-set hash, output count, and poisoning status.

This mechanism is not differentially private and can duplicate source rows. It is used to study a controlled real-synthetic mixture, not to protect data.

## Model and protocol

Every peer starts from the same seeded two-layer MLP. In each synchronous round it performs one or more local SGD epochs and calculates a delta from its current local state. Binary cross-entropy uses a positive-class weight calculated only from that peer's local rows; no pooled label prevalence is used. The simulator sends the delta to graph neighbours. Each recipient also includes its own local delta, aggregates the declared set, and applies that aggregate to its own previous state.

The all-to-all experiment has no distinguished aggregator process, but every honest peer sees the same sender set. A ring produces local views and non-zero consensus dynamics, but offers fewer values for robust coordinate trimming.

## Aggregation

- `peer_mean`: unweighted coordinate mean. Sample counts are reported but not trusted as aggregation weights.
- `coordinate_median`: coordinate-wise middle value, averaging the two central values for even message counts.
- `trimmed_mean`: removes the configured number of largest and smallest values independently for every coordinate, then averages the remainder.

Configuration preflight evaluates trimming against the smallest receiver-local message set, including the receiver's own update. All-to-all receivers see all configured peers; ring receivers see only themselves and two neighbours. A configuration is rejected before data loading if trimming would remove every local value.

These are empirical baselines. No convergence or Byzantine-resilience theorem is asserted for the implemented topology, optimiser, or attacker.

## Attacks

`sign_flip` multiplies every outgoing delta coordinate from declared malicious peers by `-scale`. `synthetic_label_poisoning` applies only to generated rows at malicious low-data peers and flips their binary labels. Real rows are not altered.

The malicious peer set is static and known to the evaluator but not used by the aggregation functions. There is no adaptive attacker, collusion protocol, Sybil identity, model replacement, backdoor trigger, or message delay/drop attack in v0.1.

## Membership inference

The audit uses negative per-example binary cross-entropy as a membership score: lower loss produces a higher score. Each peer contributes equal counts of members and nonmembers to calibration, and equal counts to evaluation. A threshold is selected on disjoint calibration members/nonmembers by maximising TPR-FPR. It is then evaluated on separate member/nonmember sets. ROC-AUC, signed evaluation advantage at the calibrated threshold, and TPR at 1% FPR are reported.

Only original real training rows are eligible as members; generated rows are excluded from the attack sample. Calibration and evaluation source IDs are checked for overlap.

## Evaluation

Global test predictions are evaluated per peer and as a probability ensemble. Local holdout balanced accuracy measures performance on each peer's retained distribution. The worst peer and standard deviation expose distributional disparity. Prediction disagreement is the mean pairwise difference in thresholded peer predictions. Parameter consensus distance is the mean per-coordinate RMS distance across peer state pairs.

The implementation averages recall over the classes actually present in an evaluated split. If a local holdout contains only one class, its reported balanced accuracy therefore equals that class's recall; it is not evidence of balanced two-class performance. Per-peer train and holdout class counts are recorded in the partition manifest so this condition can be audited alongside the metric.

The subgroup TPR gap uses the two Adult `sex` categories only when each contains positive examples. It is a diagnostic on one attribute and threshold, not a normative fairness conclusion.

Communication counts transmitted tensor payload bytes for every directed graph edge. It excludes transport framing, encryption, retransmission, serialisation, and protocol headers.
