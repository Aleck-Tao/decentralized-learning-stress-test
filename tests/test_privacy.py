from __future__ import annotations

import unittest

import numpy as np

from decentralized_stress.privacy import membership_from_scores, roc_auc


class PrivacyMetricTests(unittest.TestCase):
    def test_auc_handles_ties_and_perfect_ordering(self) -> None:
        self.assertAlmostEqual(
            roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])), 1.0
        )
        self.assertAlmostEqual(
            roc_auc(np.array([1, 0]), np.array([0.5, 0.5])), 0.5
        )

    def test_membership_threshold_uses_separate_calibration_scores(self) -> None:
        result = membership_from_scores(
            np.array([0.9, 0.8, 0.7]),
            np.array([0.3, 0.2, 0.1]),
            np.array([0.95, 0.85]),
            np.array([0.25, 0.15]),
        )
        self.assertEqual(result.roc_auc, 1.0)
        self.assertEqual(result.advantage, 1.0)
        self.assertEqual(result.calibration_members, 3)
        self.assertEqual(result.evaluation_members, 2)
        self.assertEqual(result.evaluation_nonmembers, 2)


if __name__ == "__main__":
    unittest.main()
