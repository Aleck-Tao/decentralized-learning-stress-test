# Threat model

## Assets and boundaries

The protected conceptual asset is each peer's local row set. The experiment checks that the declared message schema contains model deltas rather than raw features or labels. It also measures one model-level membership signal.

Because all peers run in one Python process, this is not an enforcement boundary. The simulator, operating system, and experiment operator are trusted and can access all rows.

## Adversaries represented

1. A static Byzantine peer emits a scaled sign-flipped model delta.
2. A malicious synthetic-data source at a declared low-data peer flips labels on that peer's generated augmentation rows.
3. A black-box membership attacker observes model losses and calibrates a threshold using separate reference samples.

## Trusted assumptions

- Peer identities and graph membership are fixed.
- Rounds are synchronous and messages arrive once, in order, without corruption.
- Honest local training code and preprocessing are identical.
- The experiment configuration, Adult archive hash, and evaluator are trusted.
- The attack fraction and trimmed-mean parameter are known before the run.

## Outside the model

- Sybil, eclipse, replay, delay, omission, equivocation, and network-partition attacks;
- adaptive/colluding attackers and stealthy backdoors;
- dishonest sample-count or topology metadata;
- secure aggregation, encryption, authentication, remote attestation, or trusted hardware;
- malicious coordinator behaviour, because there is no coordinator protocol;
- reconstruction, property inference, gradient inversion, or white-box membership attacks;
- formal safety, liveness, convergence, or Byzantine thresholds.

Passing the benchmark means only that the implemented experiment produced its recorded metrics under this threat model.

