from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AGGREGATORS = {"peer_mean", "coordinate_median", "trimmed_mean"}
TOPOLOGIES = {"all_to_all", "ring"}
SCENARIOS = {
    "iid_real",
    "noniid_scarce_real",
    "noniid_scarce_synthetic",
    "noniid_sign_flip",
    "noniid_synthetic_poison",
    "noniid_synthetic_privacy_audit",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    clients: int
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    partition_alpha: float
    scarcity_fraction: float
    scarce_client_fraction: float
    synthetic_ratio: float
    byzantine_fraction: float
    sign_flip_scale: float
    trim_count: int
    topology: str
    seeds: tuple[int, ...]
    aggregators: tuple[str, ...]
    scenarios: tuple[str, ...]

    @property
    def byzantine_peer_count(self) -> int:
        return max(1, int(round(self.clients * self.byzantine_fraction)))

    @property
    def scarce_peer_count(self) -> int:
        if self.scarce_client_fraction <= 0:
            return 0
        return max(1, int(round(self.clients * self.scarce_client_fraction)))

    @property
    def minimum_receiver_message_count(self) -> int:
        """Smallest local aggregation set, including the receiver's own update."""
        if self.topology == "all_to_all":
            return self.clients
        if self.topology == "ring":
            return 3
        raise ValueError(f"Unknown topology: {self.topology}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkConfig":
        if raw.get("schema_version") != "1.0":
            raise ValueError("Unsupported benchmark schema_version")
        config = cls(
            dataset=str(raw["dataset"]),
            clients=int(raw["clients"]),
            rounds=int(raw["rounds"]),
            local_epochs=int(raw["local_epochs"]),
            batch_size=int(raw["batch_size"]),
            learning_rate=float(raw["learning_rate"]),
            hidden_dim=int(raw["hidden_dim"]),
            partition_alpha=float(raw["partition_alpha"]),
            scarcity_fraction=float(raw["scarcity_fraction"]),
            scarce_client_fraction=float(raw["scarce_client_fraction"]),
            synthetic_ratio=float(raw["synthetic_ratio"]),
            byzantine_fraction=float(raw["byzantine_fraction"]),
            sign_flip_scale=float(raw["sign_flip_scale"]),
            trim_count=int(raw["trim_count"]),
            topology=str(raw["topology"]),
            seeds=tuple(int(value) for value in raw["seeds"]),
            aggregators=tuple(str(value) for value in raw["aggregators"]),
            scenarios=tuple(str(value) for value in raw["scenarios"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in {"adult", "fixture"}:
            raise ValueError("dataset must be adult or fixture")
        if self.clients < 3:
            raise ValueError("At least three peers are required")
        if self.rounds < 1 or self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("rounds, local_epochs, and batch_size must be positive")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two")
        for name, value in {
            "learning_rate": self.learning_rate,
            "partition_alpha": self.partition_alpha,
            "scarcity_fraction": self.scarcity_fraction,
            "scarce_client_fraction": self.scarce_client_fraction,
            "synthetic_ratio": self.synthetic_ratio,
            "byzantine_fraction": self.byzantine_fraction,
            "sign_flip_scale": self.sign_flip_scale,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.scarcity_fraction > 1 or self.scarce_client_fraction > 1:
            raise ValueError("scarcity fractions cannot exceed one")
        if self.synthetic_ratio > 2:
            raise ValueError("synthetic_ratio cannot exceed two")
        if not 0 < self.byzantine_fraction < 0.5:
            raise ValueError("byzantine_fraction must be between zero and one half")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"Unknown topology: {self.topology}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("seeds must fit an unsigned 32-bit integer")
        if not self.aggregators or any(value not in AGGREGATORS for value in self.aggregators):
            raise ValueError("Unknown or empty aggregator list")
        if not self.scenarios or any(value not in SCENARIOS for value in self.scenarios):
            raise ValueError("Unknown or empty scenario list")
        if self.trim_count < 0:
            raise ValueError("trim_count cannot be negative")
        participants = self.byzantine_peer_count
        if (
            "trimmed_mean" in self.aggregators
            and self.minimum_receiver_message_count <= 2 * self.trim_count
        ):
            raise ValueError(
                "trimmed_mean requires each receiver to aggregate more than twice "
                f"trim_count; topology {self.topology!r} has a minimum receiver "
                f"message count of {self.minimum_receiver_message_count}"
            )
        if self.trim_count < participants and "trimmed_mean" in self.aggregators:
            raise ValueError("trim_count must cover the configured Byzantine peer count")
        if (
            "noniid_synthetic_poison" in self.scenarios
            and self.scarce_peer_count < participants
        ):
            raise ValueError(
                "noniid_synthetic_poison requires at least as many scarce peers as "
                f"Byzantine peers; configured counts are {self.scarce_peer_count} "
                f"and {participants}"
            )


def load_config(path: Path) -> BenchmarkConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Benchmark configuration must be a JSON object")
    return BenchmarkConfig.from_dict(raw)
