from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.synthesis_eval import (
    extract_pair_features_by_size,
    rejection_filter_by_size,
)


class RejectionMultiSizeTests(unittest.TestCase):
    def test_extract_pair_features_by_size_age_employ(self) -> None:
        # Two households: sizes 2 and 3.
        individual_data = np.array(
            [
                [12, 1],  # HH1 ego: age 0-5, employ full-time
                [1, 2],   # HH1 alter: age 30-34, employ homemaker
                [7, 5],   # HH2 ego: age 15-19, employ student
                [18, 9],  # HH2 alter: age 80-84, employ unemployed
                [16, 3],  # HH2 alter: age 20-24, employ retired
            ],
            dtype=int,
        )
        group_id = np.array([1, 1, 2, 2, 2], dtype=int)

        feat, gid, sizes, invalid = extract_pair_features_by_size(
            individual_data=individual_data,
            individual_group_id=group_id,
            age_col=0,
            employ_col=1,
        )

        expected_feat = np.array([[7, 3], [14, 5]], dtype=int)
        expected_gid = np.array([1, 2], dtype=int)
        expected_sizes = np.array([2, 3], dtype=int)

        self.assertTrue(np.array_equal(feat, expected_feat))
        self.assertTrue(np.array_equal(gid, expected_gid))
        self.assertTrue(np.array_equal(sizes, expected_sizes))
        self.assertEqual(invalid.size, 0)

    def test_rejection_filter_accept_all_when_probs_equal(self) -> None:
        proposal_pairs = np.array([[1, 1], [2, 1]], dtype=int)
        proposal_prob = np.array([[0.5], [0.5]], dtype=float)
        target_prob = proposal_prob.copy()
        rng = np.random.default_rng(11)

        accept, stats = rejection_filter_by_size(
            proposal_pairs=proposal_pairs,
            target_prob=target_prob,
            proposal_prob=proposal_prob,
            rng=rng,
            verbose=False,
        )

        self.assertTrue(np.all(accept))
        self.assertAlmostEqual(stats["accept_rate"], 1.0)

    def test_rejection_filter_counts_f1_zero_bins(self) -> None:
        proposal_pairs = np.array([[1, 1]], dtype=int)
        proposal_prob = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
        target_prob = np.array([[0.5, 0.25], [0.0, 0.25]], dtype=float)
        rng = np.random.default_rng(123)

        _, stats = rejection_filter_by_size(
            proposal_pairs=proposal_pairs,
            target_prob=target_prob,
            proposal_prob=proposal_prob,
            rng=rng,
            verbose=False,
        )

        self.assertEqual(stats["f1_zero_bins"], 2)


if __name__ == "__main__":
    unittest.main()
