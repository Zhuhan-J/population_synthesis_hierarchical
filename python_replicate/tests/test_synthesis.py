from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.synthesis_eval import (
    generate_synthetic_population,
    generate_with_two_person_rejection,
)


class SynthesisTests(unittest.TestCase):
    def test_generate_population_shapes(self) -> None:
        rng = np.random.default_rng(7)

        # G=2 household classes, M=2 individual classes
        pi_g = np.array([0.5, 0.5])
        pi_m = np.array([[0.7, 0.3], [0.2, 0.8]])

        # 4 household attributes (last one is household size: 1 or 2)
        phi_g = [
            np.array([[0.6, 0.3], [0.4, 0.7]]),
            np.array([[0.5, 0.8], [0.5, 0.2]]),
            np.array([[0.9, 0.4], [0.1, 0.6]]),
            np.array([[0.2, 0.1], [0.8, 0.9]]),
        ]

        # 2 individual attributes, universal classes
        phi_m = [
            np.array([[0.4, 0.8], [0.6, 0.2]]),
            np.array([[0.7, 0.3], [0.3, 0.7]]),
        ]

        res = generate_synthetic_population(
            n_households=10,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=3,
            rng=rng,
        )

        self.assertEqual(res.household_data.shape[0], 10)
        self.assertEqual(res.household_data.shape[1], 4)
        self.assertEqual(res.individual_data.shape[1], 2)
        self.assertEqual(res.individual_group_id.shape[0], res.individual_data.shape[0])
        self.assertTrue(np.all(res.household_class >= 1))
        self.assertTrue(np.all(res.individual_class >= 1))

    def test_generate_population_shapes_larger_g5_m8(self) -> None:
        rng = np.random.default_rng(77)
        G = 5
        M = 8

        pi_g = rng.dirichlet(np.ones(G))
        pi_m = rng.dirichlet(np.ones(M), size=G)

        # 4 household attributes, final one used as household size category.
        phi_g = []
        for n_cat in [4, 3, 5, 4]:
            mat = rng.gamma(shape=1.0, scale=1.0, size=(n_cat, G))
            phi_g.append(mat / np.sum(mat, axis=0, keepdims=True))

        # 6 individual attributes, universal individual classes M=8.
        phi_m = []
        for n_cat in [4, 4, 3, 5, 2, 4]:
            mat = rng.gamma(shape=1.0, scale=1.0, size=(n_cat, M))
            phi_m.append(mat / np.sum(mat, axis=0, keepdims=True))

        res = generate_synthetic_population(
            n_households=25,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=3,
            rng=rng,
        )

        self.assertEqual(res.household_data.shape, (25, 4))
        self.assertEqual(res.individual_data.shape[1], 6)
        self.assertEqual(res.individual_group_id.shape[0], res.individual_data.shape[0])
        self.assertTrue(np.all(res.household_class >= 1))
        self.assertTrue(np.all(res.household_class <= G))
        self.assertTrue(np.all(res.individual_class >= 1))
        self.assertTrue(np.all(res.individual_class <= M))


def _build_two_person_fixture(rng: np.random.Generator):
    """Build a model + target dataset where the proposal under-represents
    opposite-sex two-person households relative to the target."""
    G = 2
    M = 2
    pi_g = np.array([0.5, 0.5])
    pi_m = np.array([[0.5, 0.5], [0.5, 0.5]])

    # 4 household attrs; size column is index 3 with 2 categories. Force size=2 dominant.
    phi_g = [
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        # size attribute: high mass on category 2 (i.e. 2-person households).
        np.array([[0.05, 0.05], [0.95, 0.95]]),
    ]

    # 2 individual attrs: age (5 cats), sex (2 cats). Sex is near-uniform per class so
    # the synthetic proposal produces near-independent sex pairs in 2-person households.
    age_cats = 5
    phi_m = [
        np.array(
            [
                [0.20, 0.20],
                [0.20, 0.20],
                [0.20, 0.20],
                [0.20, 0.20],
                [0.20, 0.20],
            ]
        ),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
    ]

    # Target: 200 two-person households with strict opposite-sex pairing and varied ages.
    n_target_hh = 200
    target_individual_data = np.zeros((n_target_hh * 2, 2), dtype=int)
    target_individual_group_id = np.zeros(n_target_hh * 2, dtype=int)
    for h in range(n_target_hh):
        a1 = int(rng.integers(1, age_cats + 1))
        a2 = int(rng.integers(1, age_cats + 1))
        target_individual_data[2 * h, :] = [a1, 1]      # sex=1
        target_individual_data[2 * h + 1, :] = [a2, 2]  # sex=2
        target_individual_group_id[2 * h] = h + 1
        target_individual_group_id[2 * h + 1] = h + 1

    return pi_g, pi_m, phi_g, phi_m, target_individual_data, target_individual_group_id


def _two_person_sex_cramers_v(synth_individual_data, synth_individual_group_id) -> float:
    """Compute Cramer's V for (sex_A, sex_B) symmetrized across two-person households."""
    gid = np.asarray(synth_individual_group_id, dtype=int)
    sex = np.asarray(synth_individual_data[:, 1], dtype=int)
    counts = np.bincount(gid)
    two_person_gids = np.where(counts == 2)[0]
    if two_person_gids.size == 0:
        return 0.0

    table = np.zeros((2, 2), dtype=float)
    for g in two_person_gids:
        idx = np.where(gid == g)[0]
        s1, s2 = int(sex[idx[0]]), int(sex[idx[1]])
        if not (1 <= s1 <= 2 and 1 <= s2 <= 2):
            continue
        # Symmetrize so the table is order-independent.
        table[s1 - 1, s2 - 1] += 1.0
        table[s2 - 1, s1 - 1] += 1.0

    total = table.sum()
    if total <= 0:
        return 0.0
    p = table / total
    row_sums = p.sum(axis=1, keepdims=True)
    col_sums = p.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums
    expected = np.maximum(expected, 1e-12)
    chi2_n = float(np.sum((p - expected) ** 2 / expected))
    return float(np.sqrt(chi2_n / max(min(p.shape[0] - 1, p.shape[1] - 1), 1)))


class TwoPersonRejectionHelperTests(unittest.TestCase):
    def test_returns_exact_household_count(self) -> None:
        rng = np.random.default_rng(123)
        pi_g, pi_m, phi_g, phi_m, tgt_ind, tgt_gid = _build_two_person_fixture(rng)

        n_target = 50
        out = generate_with_two_person_rejection(
            n_households=n_target,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            target_individual_data=tgt_ind,
            target_individual_group_id=tgt_gid,
            household_size_col=3,
            age_col=0,
            sex_col=1,
            rng=np.random.default_rng(456),
        )
        self.assertEqual(out.household_data.shape[0], n_target)
        # Group ids must be a contiguous 1..n_target permutation per household.
        self.assertEqual(set(np.unique(out.individual_group_id).tolist()), set(range(1, n_target + 1)))

    def test_post_rejection_sex_cramers_v_improves(self) -> None:
        rng = np.random.default_rng(321)
        pi_g, pi_m, phi_g, phi_m, tgt_ind, tgt_gid = _build_two_person_fixture(rng)

        # Raw (no rejection) baseline.
        raw = generate_synthetic_population(
            n_households=300,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=3,
            rng=np.random.default_rng(42),
        )
        raw_cv = _two_person_sex_cramers_v(raw.individual_data, raw.individual_group_id)

        rejected = generate_with_two_person_rejection(
            n_households=300,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            target_individual_data=tgt_ind,
            target_individual_group_id=tgt_gid,
            household_size_col=3,
            age_col=0,
            sex_col=1,
            rng=np.random.default_rng(42),
        )
        rej_cv = _two_person_sex_cramers_v(rejected.individual_data, rejected.individual_group_id)

        # Sanity: raw should be near 0 (independent sex pairs); rejection should push it up.
        self.assertLess(raw_cv, 0.2)
        self.assertGreater(rej_cv, raw_cv + 0.2)

    def test_reproducibility_same_seed(self) -> None:
        rng = np.random.default_rng(999)
        pi_g, pi_m, phi_g, phi_m, tgt_ind, tgt_gid = _build_two_person_fixture(rng)

        a = generate_with_two_person_rejection(
            n_households=40,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            target_individual_data=tgt_ind,
            target_individual_group_id=tgt_gid,
            household_size_col=3,
            age_col=0,
            sex_col=1,
            rng=np.random.default_rng(2024),
        )
        b = generate_with_two_person_rejection(
            n_households=40,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            target_individual_data=tgt_ind,
            target_individual_group_id=tgt_gid,
            household_size_col=3,
            age_col=0,
            sex_col=1,
            rng=np.random.default_rng(2024),
        )
        np.testing.assert_array_equal(a.household_data, b.household_data)
        np.testing.assert_array_equal(a.individual_data, b.individual_data)
        np.testing.assert_array_equal(a.individual_group_id, b.individual_group_id)


if __name__ == "__main__":
    unittest.main()
