from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from .aggregation import aggregate_deltas
from .attacks import sign_flip
from .data import Records
from .model import (
    ModelState,
    apply_delta,
    clone_state,
    delta_payload_bytes,
    make_initial_state,
    mean_pairwise_state_distance,
    train_local,
    validate_state,
)


@dataclass(frozen=True)
class Topology:
    neighbours: dict[int, tuple[int, ...]]

    @property
    def peer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.neighbours))

    def validate(self) -> None:
        peer_ids = set(self.neighbours)
        if peer_ids != set(range(len(peer_ids))):
            raise ValueError("Peer IDs must be contiguous integers beginning at zero")
        if len(peer_ids) < 3:
            raise ValueError("Topology requires at least three peers")
        for peer_id, neighbours in self.neighbours.items():
            if peer_id in neighbours:
                raise ValueError("Self edges are not allowed")
            if len(neighbours) != len(set(neighbours)):
                raise ValueError("Duplicate topology edge")
            if not set(neighbours).issubset(peer_ids):
                raise ValueError("Topology refers to an unknown peer")
            for neighbour in neighbours:
                if peer_id not in self.neighbours[neighbour]:
                    raise ValueError("Topology must be symmetric")
        visited = {0}
        queue = [0]
        while queue:
            current = queue.pop(0)
            for neighbour in self.neighbours[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        if visited != peer_ids:
            raise ValueError("Topology must be connected")


def make_topology(kind: str, peers: int) -> Topology:
    if peers < 3:
        raise ValueError("At least three peers are required")
    if kind == "all_to_all":
        neighbours = {peer: tuple(other for other in range(peers) if other != peer) for peer in range(peers)}
    elif kind == "ring":
        neighbours = {
            peer: tuple(sorted({(peer - 1) % peers, (peer + 1) % peers})) for peer in range(peers)
        }
    else:
        raise ValueError(f"Unknown topology: {kind}")
    topology = Topology(neighbours)
    topology.validate()
    return topology


@dataclass(frozen=True)
class PeerMessage:
    sender_id: int
    round_index: int
    sample_count: int
    delta: Mapping[str, Tensor]
    attacked: bool

    def __post_init__(self) -> None:
        if self.sender_id < 0 or self.round_index < 0 or self.sample_count < 1:
            raise ValueError("Invalid peer-message metadata")
        validate_state(self.delta)

    @property
    def payload_bytes(self) -> int:
        return delta_payload_bytes(self.delta)


@dataclass(frozen=True)
class SimulationResult:
    states: dict[int, ModelState]
    round_records: tuple[dict[str, object], ...]
    communication_bytes: int
    malicious_peer_ids: tuple[int, ...]


def simulate(
    train_records: dict[int, Records],
    *,
    topology: Topology,
    aggregator: str,
    rounds: int,
    hidden_dim: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    trim_count: int,
    seed: int,
    malicious_peer_ids: tuple[int, ...] = (),
    update_attack: str | None = None,
    sign_flip_scale: float = 5.0,
) -> SimulationResult:
    topology.validate()
    if set(train_records) != set(topology.peer_ids):
        raise ValueError("Training records and topology peers differ")
    if not train_records:
        raise ValueError("No peers supplied")
    feature_counts = {records.feature_count for records in train_records.values()}
    if len(feature_counts) != 1:
        raise ValueError("All peers require the same feature dimension")
    malicious = tuple(sorted(set(malicious_peer_ids)))
    if not set(malicious).issubset(topology.peer_ids):
        raise ValueError("Unknown malicious peer")
    if update_attack not in {None, "sign_flip"}:
        raise ValueError("Unknown update attack")

    initial = make_initial_state(next(iter(feature_counts)), hidden_dim, seed + 77_777)
    states = {peer_id: clone_state(initial) for peer_id in topology.peer_ids}
    records: list[dict[str, object]] = []
    communication_total = 0
    for round_index in range(rounds):
        messages: dict[int, PeerMessage] = {}
        local_losses: list[float] = []
        for peer_id in topology.peer_ids:
            delta, loss = train_local(
                states[peer_id],
                train_records[peer_id],
                hidden_dim=hidden_dim,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed * 100_000 + round_index * 1_000 + peer_id,
            )
            attacked = peer_id in malicious and update_attack is not None
            if attacked and update_attack == "sign_flip":
                delta = sign_flip(delta, sign_flip_scale)
            messages[peer_id] = PeerMessage(
                sender_id=peer_id,
                round_index=round_index,
                sample_count=len(train_records[peer_id]),
                delta=delta,
                attacked=attacked,
            )
            local_losses.append(loss)

        updated_states: dict[int, ModelState] = {}
        round_bytes = 0
        for receiver in topology.peer_ids:
            sender_ids = (receiver,) + topology.neighbours[receiver]
            received = [messages[sender] for sender in sender_ids]
            aggregate = aggregate_deltas(
                [message.delta for message in received], aggregator, trim_count=trim_count
            )
            updated_states[receiver] = apply_delta(states[receiver], aggregate)
            round_bytes += sum(messages[sender].payload_bytes for sender in topology.neighbours[receiver])
        states = updated_states
        communication_total += round_bytes
        records.append(
            {
                "round": round_index,
                "mean_local_loss": sum(local_losses) / len(local_losses),
                "parameter_consensus_distance": mean_pairwise_state_distance(states),
                "communication_bytes": round_bytes,
                "malicious_peer_ids": list(malicious),
                "update_attack": update_attack,
            }
        )
    return SimulationResult(
        states=states,
        round_records=tuple(records),
        communication_bytes=communication_total,
        malicious_peer_ids=malicious,
    )
