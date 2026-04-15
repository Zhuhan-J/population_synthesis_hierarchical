"""Synthetic population generation and postprocessing. 
Given model parameters from EM, generate synthetic households and individuals, then apply rejection correction for two-person households.

This module is the first implementation block for the Synthesis-Eval Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class SynthesisResult:
    """Container for generated synthetic population.

    household_data:
        Shape (N_household, H), categories coded as 1..d_k.
    individual_data:
        Shape (N_individual, I), categories coded as 1..d_k.
    individual_group_id:
        Shape (N_individual,), household id in 1-based indexing.
    household_class:
        Shape (N_household,), sampled household latent classes in 1..G.
    individual_class:
        Shape (N_individual,), sampled individual latent classes in 1..M.
    """

    household_data: np.ndarray
    individual_data: np.ndarray
    individual_group_id: np.ndarray
    household_class: np.ndarray
    individual_class: np.ndarray


def _group_members(individual_group_id: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return sorted unique group ids and index lists for each group."""
    gid = np.asarray(individual_group_id, dtype=int).ravel()
    groups = np.unique(gid)
    members: list[np.ndarray] = []
    for g in groups:
        members.append(np.where(gid == g)[0])
    return groups, members


# paper_ref: Section 4.3, Eq. (3) and Eq. (2)
# matlab_ref: run_create_2.m:47-63, run_create_2.m:152-199
# intent: sample household attributes first, then individuals conditional on household class.
def generate_synthetic_population(
    n_households: int,
    pi_g: np.ndarray,
    pi_m: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m: Sequence[np.ndarray],
    household_size_col: int = 3,
    rng: np.random.Generator | None = None,
) -> SynthesisResult:
    """Generate synthetic households and individuals from model artifacts.

    Parameters
    ----------
    n_households:
        Number of households to generate.
    pi_g:
        Shape (G,), household class prior probabilities.
    pi_m:
        Shape (G, M), conditional individual class probabilities by household class.
    phi_g:
        List of household attribute conditional probability matrices.
        Each entry has shape (d_k, G).
    phi_m:
        List of individual attribute conditional probability matrices.
        Universal-class variant: each entry has shape (d_k, M).
    household_size_col:
        0-based column index for household size attribute in generated household_data.
    """
    rng = rng or np.random.default_rng() # allow deterministic sampling with provided Generator

    pi_g = np.asarray(pi_g, dtype=float)
    pi_g = pi_g / pi_g.sum()

    pi_m = np.asarray(pi_m, dtype=float)
    if pi_m.ndim != 2:
        raise ValueError("pi_m must have shape (G, M)")

    # sample household classes for all households at once, based on pi_g.
    g_classes = rng.choice(np.arange(pi_g.size), size=n_households, p=pi_g)

    h_attrs = len(phi_g)
    household_data = np.zeros((n_households, h_attrs), dtype=int)

    for i in range(n_households):
        g = g_classes[i]
        for att, prob in enumerate(phi_g):
            p = np.asarray(prob, dtype=float)[:, g]
            p = p / p.sum()
            household_data[i, att] = int(rng.choice(np.arange(p.size), p=p)) + 1

    # Npax is treated as categorical and is interpreted as count in this dataset.
    sizes = household_data[:, household_size_col].astype(int)
    sizes = np.maximum(sizes, 1)
    n_individuals = int(np.sum(sizes))

    i_attrs = len(phi_m)
    individual_data = np.zeros((n_individuals, i_attrs), dtype=int)
    individual_group_id = np.zeros(n_individuals, dtype=int)
    individual_class = np.zeros(n_individuals, dtype=int)

    cursor = 0
    for i in range(n_households):
        g = g_classes[i]
        w = np.asarray(pi_m[g, :], dtype=float)
        w = w / w.sum()
        for _ in range(sizes[i]):
            m = int(rng.choice(np.arange(w.size), p=w))
            for att, prob in enumerate(phi_m):
                p = np.asarray(prob, dtype=float)[:, m]
                p = p / p.sum()
                individual_data[cursor, att] = int(rng.choice(np.arange(p.size), p=p)) + 1
            individual_group_id[cursor] = i + 1  # MATLAB-compatible 1-based household id.
            individual_class[cursor] = m + 1
            cursor += 1

    return SynthesisResult(
        household_data=household_data,
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        household_class=g_classes + 1,
        individual_class=individual_class,
    )


# paper_ref: Section 4.4, Eq. (16) and rejection-sampling paragraph
# matlab_ref: run_two_within_the_same_4.m:112-120
# intent: apply acceptance-rejection correction for two-person households.
def rejection_filter_two_person_households(
    proposal_pairs: np.ndarray,
    target_prob: np.ndarray,
    proposal_prob: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return acceptance mask for two-person households.

    proposal_pairs columns:
    - col 0: transformed age difference bin index (1-based)
    - col 1: transformed sex difference bin index (1-based)
    """
    rng = rng or np.random.default_rng()

    if proposal_pairs.ndim != 2 or proposal_pairs.shape[1] != 2:
        raise ValueError("proposal_pairs must be shaped (n, 2)")

    y1 = proposal_pairs[:, 0].astype(int) - 1
    y2 = proposal_pairs[:, 1].astype(int) - 1

    if np.any(y1 < 0) or np.any(y2 < 0):
        raise ValueError("proposal_pairs must be 1-based indices")

    ratio = target_prob / np.maximum(proposal_prob, 1e-15)
    M = float(np.max(ratio))
    alpha = target_prob[y1, y2] / (M * np.maximum(proposal_prob[y1, y2], 1e-15))
    u = rng.random(alpha.shape[0])
    return u <= alpha


# paper_ref: Section 4.4 (construction of y1/y2 for rejection)
# matlab_ref: run_two_within_the_same_4.m:96-107
# intent: construct transformed two-person household features used in rejection.
def two_person_pair_features(
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    age_col: int = 0,
    sex_col: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (age-difference, sex-difference) features for two-person households.

    Returns
    -------
    features:
        Shape (n_two_person_households, 2), values are 1-based bins.
    group_ids:
        Household ids corresponding to each feature row.
    """
    ind = np.asarray(individual_data, dtype=int)
    groups, members = _group_members(individual_group_id)

    feats: list[list[int]] = []
    feat_gid: list[int] = []
    for g, idx in zip(groups, members):
        if idx.size != 2:
            continue
        a = ind[idx[0], :]
        b = ind[idx[1], :]
        y1 = abs(int(a[age_col]) - int(b[age_col])) + 1
        y2 = abs(int(a[sex_col]) - int(b[sex_col])) + 1
        feats.append([y1, y2])
        feat_gid.append(int(g))

    if not feats:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)
    return np.asarray(feats, dtype=int), np.asarray(feat_gid, dtype=int)


def _feature_distribution(features: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    if features.size == 0:
        if shape is None:
            return np.zeros((1, 1), dtype=float)
        return np.zeros(shape, dtype=float)

    if shape is None:
        shape = (int(np.max(features[:, 0])), int(np.max(features[:, 1])))

    out = np.zeros(shape, dtype=float)
    for y1, y2 in features:
        if y1 < 1 or y2 < 1 or y1 > shape[0] or y2 > shape[1]:
            continue
        out[y1 - 1, y2 - 1] += 1.0
    total = out.sum()
    if total <= 0:
        return out
    return out / total


# paper_ref: Section 4.4, Eq. (16) and rejection-sampling step
# matlab_ref: run_two_within_the_same_4.m:112-120 and 175-241
# intent: postprocess synthetic population to correct two-person household relations.
def apply_two_person_rejection(
    synthetic: SynthesisResult,
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    age_col: int = 0,
    sex_col: int = 1,
    household_size_col: int | None = 3,
    rng: np.random.Generator | None = None,
) -> SynthesisResult:
    """Apply rejection-sampling correction to synthetic two-person households.

    Non-two-person households are retained unchanged.
    """
    rng = rng or np.random.default_rng()

    prop_feat, prop_gid = two_person_pair_features(
        synthetic.individual_data, synthetic.individual_group_id, age_col=age_col, sex_col=sex_col
    )
    tgt_feat, _ = two_person_pair_features(
        np.asarray(target_individual_data, dtype=int),
        np.asarray(target_individual_group_id, dtype=int),
        age_col=age_col,
        sex_col=sex_col,
    )

    if prop_feat.size == 0 or tgt_feat.size == 0:
        return synthetic

    shape = (
        max(int(np.max(prop_feat[:, 0])), int(np.max(tgt_feat[:, 0]))),
        max(int(np.max(prop_feat[:, 1])), int(np.max(tgt_feat[:, 1]))),
    )

    proposal_prob = _feature_distribution(prop_feat, shape=shape)
    target_prob = _feature_distribution(tgt_feat, shape=shape)
    accept_two = rejection_filter_two_person_households(prop_feat, target_prob, proposal_prob, rng=rng)

    keep_two_groups = set(prop_gid[accept_two].tolist())

    # Keep all households with size != 2 and accepted two-person households.
    all_groups = np.arange(1, synthetic.household_data.shape[0] + 1)

    sizes: np.ndarray | None = None
    if household_size_col is not None and 0 <= household_size_col < synthetic.household_data.shape[1]:
        sizes = synthetic.household_data[:, household_size_col].astype(int)
    else:
        # Fallback: infer household sizes from sampled individuals when size column is unavailable.
        sizes = np.bincount(synthetic.individual_group_id, minlength=synthetic.household_data.shape[0] + 1)[1:].astype(int)

    keep_groups: list[int] = []
    for g in all_groups:
        if int(sizes[g - 1]) != 2:
            keep_groups.append(int(g))
        elif int(g) in keep_two_groups:
            keep_groups.append(int(g))

    keep_groups_arr = np.asarray(keep_groups, dtype=int)
    if keep_groups_arr.size == 0:
        return synthetic

    # Re-index kept households to 1..N_new.
    old_to_new = {int(g): i + 1 for i, g in enumerate(keep_groups_arr.tolist())}

    keep_house_mask = np.isin(all_groups, keep_groups_arr)
    new_household_data = synthetic.household_data[keep_house_mask, :]
    new_household_class = synthetic.household_class[keep_house_mask]

    keep_ind_mask = np.isin(synthetic.individual_group_id, keep_groups_arr)
    new_ind_data = synthetic.individual_data[keep_ind_mask, :]
    new_ind_class = synthetic.individual_class[keep_ind_mask]
    new_ind_gid_old = synthetic.individual_group_id[keep_ind_mask]
    new_ind_gid = np.array([old_to_new[int(g)] for g in new_ind_gid_old], dtype=int)

    return SynthesisResult(
        household_data=new_household_data,
        individual_data=new_ind_data,
        individual_group_id=new_ind_gid,
        household_class=new_household_class,
        individual_class=new_ind_class,
    )
