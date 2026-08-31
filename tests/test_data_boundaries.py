from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from decentralized_stress.data import (
    assert_disjoint,
    augment_local_non_private,
    load_adult,
    make_fixture,
    partition_clients,
)


ROOT = Path(__file__).resolve().parents[1]


class DataBoundaryTests(unittest.TestCase):
    def test_pinned_adult_archive_loads_with_strict_splits(self) -> None:
        archive = ROOT / "data" / "raw" / "adult.zip"
        if not archive.is_file():
            self.skipTest("Pinned Adult archive has not been placed in data/raw yet")
        bundle = load_adult(ROOT)
        self.assertEqual(len(bundle.train_pool), 32_561)
        self.assertEqual(
            len(bundle.global_test)
            + len(bundle.attack_calibration_nonmember)
            + len(bundle.attack_evaluation_nonmember),
            16_281,
        )
        self.assertEqual(bundle.train_pool.feature_count, 62)
        self.assertNotIn("sex", bundle.feature_names)
        self.assertEqual(bundle.provenance["kind"], "external_real_dataset")

    def test_fixture_splits_and_client_partitions_do_not_overlap(self) -> None:
        bundle = make_fixture()
        partition = partition_clients(
            bundle.train_pool,
            8,
            seed=101,
            alpha=0.5,
            scarcity_fraction=0.25,
            scarce_client_fraction=0.25,
        )
        named = {f"train-{peer}": records for peer, records in partition.train.items()}
        named.update({f"holdout-{peer}": records for peer, records in partition.holdout.items()})
        named.update(
            {
                "global": bundle.global_test,
                "attack-calibration": bundle.attack_calibration_nonmember,
                "attack-evaluation": bundle.attack_evaluation_nonmember,
            }
        )
        assert_disjoint(named)
        self.assertEqual(len(partition.scarce_peer_ids), 2)
        self.assertIn("simulated peer partitions", str(partition.partition_manifest["boundary"]))
        manifest_by_peer = {
            int(entry["peer_id"]): entry
            for entry in partition.partition_manifest["clients"]
        }
        for peer_id in sorted(partition.train):
            entry = manifest_by_peer[peer_id]
            expected_train_counts = {
                "0": int(np.count_nonzero(partition.train[peer_id].y == 0.0)),
                "1": int(np.count_nonzero(partition.train[peer_id].y == 1.0)),
            }
            expected_holdout_counts = {
                "0": int(np.count_nonzero(partition.holdout[peer_id].y == 0.0)),
                "1": int(np.count_nonzero(partition.holdout[peer_id].y == 1.0)),
            }
            self.assertEqual(entry["train_label_counts"], expected_train_counts)
            self.assertEqual(entry["holdout_label_counts"], expected_holdout_counts)
            self.assertEqual(sum(expected_train_counts.values()), entry["train_rows"])
            self.assertEqual(sum(expected_holdout_counts.values()), entry["holdout_rows"])
            expected_holdout_hash = hashlib.sha256(
                "\n".join(partition.holdout[peer_id].source_ids).encode("utf-8")
            ).hexdigest()
            self.assertEqual(entry["holdout_source_ids_sha256"], expected_holdout_hash)

    def test_non_private_augmentation_is_local_and_poisoning_is_explicit(self) -> None:
        peer = make_fixture(train_rows=100, test_rows=60).train_pool
        clean, clean_manifest = augment_local_non_private(
            peer, peer_id=2, ratio=0.5, seed=44, poison_labels=False
        )
        poisoned, poison_manifest = augment_local_non_private(
            peer, peer_id=2, ratio=0.5, seed=44, poison_labels=True
        )
        generated = len(clean) - len(peer)
        self.assertEqual(generated, 50)
        np.testing.assert_array_equal(clean.y[-generated:], 1.0 - poisoned.y[-generated:])
        np.testing.assert_array_equal(peer.synthetic, np.zeros(len(peer), dtype=bool))
        self.assertTrue(clean.synthetic[-generated:].all())
        self.assertTrue(clean_manifest["non_dp"])
        self.assertFalse(clean_manifest["poisoned"])
        self.assertTrue(poison_manifest["poisoned"])
        self.assertEqual(poison_manifest["privacy_claim"], "none")

    def test_duplicate_source_id_is_rejected(self) -> None:
        bundle = make_fixture()
        with self.assertRaises(ValueError):
            assert_disjoint({"left": bundle.train_pool, "right": bundle.train_pool})


if __name__ == "__main__":
    unittest.main()
