from __future__ import annotations

from collections import OrderedDict
import unittest

import torch

from decentralized_stress.aggregation import aggregate_deltas
from decentralized_stress.attacks import sign_flip
from decentralized_stress.protocol import PeerMessage, make_topology


def delta(value: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(weight=torch.tensor([value], dtype=torch.float32))


class ProtocolAndAggregationTests(unittest.TestCase):
    def test_all_to_all_and_ring_have_only_peer_nodes(self) -> None:
        all_to_all = make_topology("all_to_all", 8)
        ring = make_topology("ring", 8)
        self.assertEqual(all_to_all.peer_ids, tuple(range(8)))
        self.assertTrue(all(len(values) == 7 for values in all_to_all.neighbours.values()))
        self.assertTrue(all(len(values) == 2 for values in ring.neighbours.values()))
        self.assertNotIn("server", str(all_to_all.neighbours).lower())

    def test_message_schema_contains_no_raw_examples(self) -> None:
        fields = set(PeerMessage.__dataclass_fields__)
        self.assertEqual(fields, {"sender_id", "round_index", "sample_count", "delta", "attacked"})
        self.assertFalse(fields.intersection({"x", "y", "features", "labels", "records"}))
        message = PeerMessage(0, 0, 5, delta(1.0), False)
        self.assertEqual(message.payload_bytes, 4)

    def test_mean_median_and_trimmed_mean_on_known_updates(self) -> None:
        values = [delta(value) for value in (1.0, 2.0, 3.0, 100.0)]
        mean = aggregate_deltas(values, "peer_mean", trim_count=1)
        median = aggregate_deltas(values, "coordinate_median", trim_count=1)
        trimmed = aggregate_deltas(values, "trimmed_mean", trim_count=1)
        self.assertAlmostEqual(float(mean["weight"]), 26.5)
        self.assertAlmostEqual(float(median["weight"]), 2.5)
        self.assertAlmostEqual(float(trimmed["weight"]), 2.5)

    def test_sign_flip_changes_only_declared_delta(self) -> None:
        original = delta(2.0)
        attacked = sign_flip(original, 5.0)
        self.assertEqual(float(attacked["weight"]), -10.0)
        self.assertEqual(float(original["weight"]), 2.0)

    def test_non_finite_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PeerMessage(0, 0, 1, delta(float("nan")), True)


if __name__ == "__main__":
    unittest.main()

