from __future__ import annotations

import unittest

import numpy as np

from decentralized_stress.data import Records
from decentralized_stress.evaluation import binary_metrics
from decentralized_stress.model import make_initial_state, train_local


class LocalTrainingTests(unittest.TestCase):
    def test_single_class_balanced_accuracy_equals_present_class_recall(self) -> None:
        labels = np.zeros(4, dtype=np.float32)
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.3], dtype=np.float32)
        groups = np.asarray([0, 1, 0, 1], dtype=np.int8)
        metrics = binary_metrics(labels, probabilities, groups)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], metrics["accuracy"])

    def test_class_imbalanced_local_training_stays_finite(self) -> None:
        rng = np.random.default_rng(5)
        records = Records(
            x=rng.normal(size=(40, 5)).astype(np.float32),
            y=np.asarray([0.0] * 36 + [1.0] * 4, dtype=np.float32),
            groups=np.asarray([0, 1] * 20, dtype=np.int8),
            source_ids=tuple(f"row:{index}" for index in range(40)),
            synthetic=np.zeros(40, dtype=bool),
        )
        state = make_initial_state(5, 6, 3)
        delta, loss = train_local(
            state,
            records,
            hidden_dim=6,
            epochs=1,
            batch_size=20,
            learning_rate=0.02,
            seed=4,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(value.numpy()).all() for value in delta.values()))


if __name__ == "__main__":
    unittest.main()
