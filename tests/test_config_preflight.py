from __future__ import annotations

import unittest

from decentralized_stress.config import BenchmarkConfig


def config_dict() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": "fixture",
        "clients": 8,
        "rounds": 1,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.04,
        "hidden_dim": 8,
        "partition_alpha": 0.5,
        "scarcity_fraction": 0.5,
        "scarce_client_fraction": 0.25,
        "synthetic_ratio": 0.5,
        "byzantine_fraction": 0.125,
        "sign_flip_scale": 5.0,
        "trim_count": 2,
        "topology": "all_to_all",
        "seeds": [17],
        "aggregators": ["trimmed_mean"],
        "scenarios": ["noniid_scarce_synthetic"],
    }


class ConfigPreflightTests(unittest.TestCase):
    def test_all_to_all_trimmed_mean_uses_full_receiver_set(self) -> None:
        config = BenchmarkConfig.from_dict(config_dict())
        self.assertEqual(config.minimum_receiver_message_count, 8)

    def test_ring_rejects_trim_count_that_exhausts_local_receiver_set(self) -> None:
        raw = config_dict()
        raw["topology"] = "ring"
        with self.assertRaisesRegex(ValueError, "minimum receiver message count of 3"):
            BenchmarkConfig.from_dict(raw)

    def test_synthetic_poison_rejects_too_few_scarce_peers(self) -> None:
        raw = config_dict()
        raw["byzantine_fraction"] = 0.25
        raw["scarce_client_fraction"] = 0.125
        raw["scenarios"] = ["noniid_synthetic_poison"]
        with self.assertRaisesRegex(ValueError, "at least as many scarce peers"):
            BenchmarkConfig.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
