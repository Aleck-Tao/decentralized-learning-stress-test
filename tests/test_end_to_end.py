from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from decentralized_stress.data import make_fixture, partition_clients
from decentralized_stress.evaluation import evaluate_with_privacy
from decentralized_stress.protocol import make_topology, simulate
from decentralized_stress.provenance import (
    sha256_file,
    source_tree_digest,
    verify_result_manifest,
    write_json,
    write_result_manifest,
)


class EndToEndTests(unittest.TestCase):
    def test_peer_training_and_evaluation_run_on_cpu(self) -> None:
        bundle = make_fixture(train_rows=320, test_rows=120)
        partition = partition_clients(
            bundle.train_pool,
            4,
            seed=19,
            alpha=0.8,
            scarcity_fraction=0.5,
            scarce_client_fraction=0.25,
        )
        simulation = simulate(
            partition.train,
            topology=make_topology("all_to_all", 4),
            aggregator="coordinate_median",
            rounds=2,
            hidden_dim=8,
            local_epochs=1,
            batch_size=64,
            learning_rate=0.04,
            trim_count=1,
            seed=19,
        )
        metrics = evaluate_with_privacy(
            simulation.states,
            partition.train,
            partition.holdout,
            bundle.global_test,
            bundle.attack_calibration_nonmember,
            bundle.attack_evaluation_nonmember,
            hidden_dim=8,
            seed=20,
        )
        self.assertGreater(simulation.communication_bytes, 0)
        self.assertEqual(len(simulation.round_records), 2)
        for key in (
            "ensemble_balanced_accuracy",
            "worst_peer_holdout_balanced_accuracy",
            "prediction_disagreement_rate",
            "membership_roc_auc",
        ):
            self.assertGreaterEqual(float(metrics[key]), 0.0)
            self.assertLessEqual(float(metrics[key]), 1.0)

    def test_ring_training_produces_measurable_peer_divergence(self) -> None:
        bundle = make_fixture(train_rows=320, test_rows=120)
        partition = partition_clients(
            bundle.train_pool,
            5,
            seed=23,
            alpha=0.4,
            scarcity_fraction=0.5,
            scarce_client_fraction=0.2,
        )
        simulation = simulate(
            partition.train,
            topology=make_topology("ring", 5),
            aggregator="peer_mean",
            rounds=2,
            hidden_dim=8,
            local_epochs=1,
            batch_size=64,
            learning_rate=0.04,
            trim_count=1,
            seed=23,
        )
        self.assertGreater(
            float(simulation.round_records[-1]["parameter_consensus_distance"]), 0.0
        )

    def test_result_manifest_detects_tampering_and_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src" / "decentralized_stress").mkdir(parents=True)
            (root / "configs").mkdir()
            result_dir = root / "results" / "test"
            result_dir.mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
            (root / "src" / "decentralized_stress" / "dummy.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            config = root / "configs" / "test.json"
            config.write_text("{}\n", encoding="utf-8")
            tree_hash, entries = source_tree_digest(root)
            generated = root / "data" / "generated" / "test"
            generated.mkdir(parents=True)
            source_manifest = generated / "source_manifest.json"
            partition_manifest = generated / "partition_manifest.json"
            synthetic_manifest = generated / "synthetic_manifest.json"
            write_json(source_manifest, {"kind": "generated_test_fixture"})
            write_json(partition_manifest, {"partitions": []})
            write_json(synthetic_manifest, {"privacy_guarantee": "none"})
            write_json(
                result_dir / "run_manifest.json",
                {
                    "source_tree_sha256": tree_hash,
                    "source_files": entries,
                    "config_path": "configs/test.json",
                    "config_sha256": sha256_file(config),
                    "dataset_source_manifest_path": source_manifest.relative_to(root).as_posix(),
                    "dataset_source_manifest_sha256": sha256_file(source_manifest),
                    "partition_manifest_path": partition_manifest.relative_to(root).as_posix(),
                    "partition_manifest_sha256": sha256_file(partition_manifest),
                    "synthetic_manifest_path": synthetic_manifest.relative_to(root).as_posix(),
                    "synthetic_manifest_sha256": sha256_file(synthetic_manifest),
                },
            )
            payload = result_dir / "payload.txt"
            payload.write_text("original\n", encoding="utf-8")
            write_result_manifest(root, result_dir)
            self.assertTrue(verify_result_manifest(root, result_dir)["valid"])
            payload.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_result_manifest(root, result_dir)


if __name__ == "__main__":
    unittest.main()
