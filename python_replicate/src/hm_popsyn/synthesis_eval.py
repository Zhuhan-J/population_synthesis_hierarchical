"""Synthetic population generation and postprocessing. 
Given model parameters from EM, generate synthetic households and individuals, then apply rejection correction for two-person households.

This module is the first implementation block for the Synthesis-Eval Agent.
"""

from __future__ import annotations

import math
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
    g_classes = rng.choice(np.arange(pi_g.size), size=n_households, p=pi_g) # classes for n_households, 0-based

    h_attrs = len(phi_g)
    household_data = np.zeros((n_households, h_attrs), dtype=int)

    for i in range(n_households):
        g = g_classes[i]
        for att, prob in enumerate(phi_g):
            p = np.asarray(prob, dtype=float)[:, g]
            p = p / p.sum() # probabilities for this attribute conditioned on household class g
            household_data[i, att] = int(rng.choice(np.arange(p.size), p=p)) + 1 # +1 because categories are 1-based; categories are assumed to be contiguous and start from 1, consistent with EM input/output conventions.

    # Npax is treated as categorical and is interpreted as count in this dataset --> extract sizes for individual generation. 
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
    verbose: bool = False,
) -> np.ndarray:
    """Return acceptance mask for two-person households.

    proposal_pairs columns:
    - col 0: transformed age difference bin index (1-based)
    - col 1: transformed sex difference bin index (1-based)
    
    If verbose=True, prints alpha calculation details for each pair.
    """
    rng = rng or np.random.default_rng()

    if proposal_pairs.ndim != 2 or proposal_pairs.shape[1] != 2:
        raise ValueError("proposal_pairs must be shaped (n, 2)")

    y1 = proposal_pairs[:, 0].astype(int) - 1
    y2 = proposal_pairs[:, 1].astype(int) - 1

    if np.any(y1 < 0) or np.any(y2 < 0):
        raise ValueError("proposal_pairs must be 1-based indices")

    ratio = target_prob / np.maximum(proposal_prob, 1e-15)
    M_raw = float(np.max(ratio))
    M = min(M_raw, 3.0)  # Cap M at 3.0 to prevent pathological rejection from extreme scaling
    alpha = target_prob[y1, y2] / (M * np.maximum(proposal_prob[y1, y2], 1e-15))
    alpha = np.clip(alpha, 0.0, 1.0)
    
    if verbose and len(alpha) > 0:
        accepted_count = np.sum(alpha > 0)
        print(f"      [ALPHA] Scaling factor M: raw={M_raw:.2e} → capped={M:.1f}")
        print(f"      [ALPHA] Total size=2 pairs: {len(alpha)}, accept-ready (α>0): {accepted_count} ({100*accepted_count/len(alpha):.1f}%)")
        shown = min(10, len(alpha))
        for i in range(shown):
            y1_orig, y2_orig = proposal_pairs[i]
            f1_val = proposal_prob[int(y1_orig)-1, int(y2_orig)-1]
            f2_val = target_prob[int(y1_orig)-1, int(y2_orig)-1]
            ratio_val = f2_val / max(f1_val, 1e-15)
            print(f"        Pair {i+1}: (y1={int(y1_orig)}, y2={int(y2_orig)}) f1={f1_val:.6f} f2={f2_val:.6f} f2/f1={ratio_val:.4f} α={alpha[i]:.6f}")
        if len(alpha) > shown:
            print(f"        ... ({len(alpha)-shown} more pairs)")
    
    u = rng.random(alpha.shape[0])
    return u <= alpha


# paper_ref: Section 4.4 (construction of y1/y2 for rejection)
# matlab_ref: run_two_within_the_same_4.m:96-107
# intent: construct transformed two-person household features used in rejection.
def two_person_pair_features(
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    age_col: int,
    sex_col: int,
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


def _rejection_diagnostics_for_batch(
    stage: str,
    batch_num: int,
    before_household_data: np.ndarray,
    after_household_data: np.ndarray,
    prop_feat: np.ndarray,
    alpha_vals: np.ndarray,
    accept_mask: np.ndarray,
    proposal_prob: np.ndarray,
    target_prob: np.ndarray,
    household_size_col: int = 3,
) -> None:
    """Print detailed diagnostics for rejection sampling.
    
    Verifies:
    1. Only size=2 households are affected
    2. Alpha values and acceptance rates per (y1, y2) pair
    3. Paper equation is correctly implemented
    """
    print(f"\n{'='*100}")
    print(f"[DIAG-REJECT] {stage} {batch_num}: REJECTION SAMPLING VERIFICATION")
    print(f"{'='*100}")
    
    # Get size distributions
    before_sizes = before_household_data[:, household_size_col].astype(int)
    after_sizes = after_household_data[:, household_size_col].astype(int)
    
    before_counts = {}
    after_counts = {}
    for s in np.unique(np.concatenate([before_sizes, after_sizes])):
        before_counts[int(s)] = int(np.sum(before_sizes == s))
        after_counts[int(s)] = int(np.sum(after_sizes == s))
    
    print(f"\n1. HOUSEHOLD SIZE DISTRIBUTION (before vs after rejection):")
    print(f"{'Size':>6} {'Before':>12} {'After':>12} {'Dropped':>12} {'Drop%':>10} Status")
    print("-" * 80)
    
    for size in sorted(set(before_counts.keys()) | set(after_counts.keys())):
        before = before_counts.get(size, 0)
        after = after_counts.get(size, 0)
        dropped = max(before - after, 0)
        drop_pct = 100.0 * dropped / before if before > 0 else 0.0
        
        if size == 2:
            status = "← REJECTION APPLIED"
        else:
            status = "(no rejection - should not change)"
            if dropped > 0:
                status = "⚠ WARNING: UNEXPECTED DROP!"
        
        print(f"{size:6d} {before:12d} {after:12d} {dropped:12d} {drop_pct:9.1f}% {status}")
    
    # Verify only size=2 affected
    print(f"\n2. VALIDATION: Only size=2 should be rejected")
    non_size2_issues = []
    for size in before_counts:
        if size != 2 and before_counts[size] > after_counts.get(size, 0):
            non_size2_issues.append(f"Size {size}: {before_counts[size]} → {after_counts.get(size, 0)}")
    
    if non_size2_issues:
        print(f"   ✗ FAIL: Non-size=2 households were dropped:")
        for issue in non_size2_issues:
            print(f"     - {issue}")
    else:
        print(f"   ✓ PASS: Only size=2 rejection was applied")
    
    # Alpha pair analysis
    print(f"\n3. SIZE=2 PAIR-LEVEL REJECTION ANALYSIS")
    print(f"   Paper equation Sec 4.4: α = f2(y1,y2) / (M × f1(y1,y2))")
    print(f"   where f1 = synthetic (proposal) distribution")
    print(f"         f2 = target (real data) distribution")
    print(f"         M = max(f2/f1) = scaling factor for normalization\n")
    
    if len(prop_feat) > 0:
        # Compute M with same capping logic as rejection_filter_two_person_households
        ratio = target_prob / np.maximum(proposal_prob, 1e-15)
        M_raw = float(np.max(ratio))
        M = min(M_raw, 3.0)  # Cap M at 3.0 to prevent pathological rejection
        print(f"   Computed M (raw={M_raw:.2e} → capped={M:.1f})\n")
        
        # Aggregate by (y1, y2) pair
        pair_stats = {}
        for i, (feat, alpha, accepted) in enumerate(zip(prop_feat, alpha_vals, accept_mask)):
            y1, y2 = int(feat[0]), int(feat[1])
            f1 = proposal_prob[y1-1, y2-1]
            f2 = target_prob[y1-1, y2-1]
            key = (y1, y2)
            if key not in pair_stats:
                pair_stats[key] = {'alphas': [], 'accepted': [], 'f1': f1, 'f2': f2}
            pair_stats[key]['alphas'].append(alpha)
            pair_stats[key]['accepted'].append(int(accepted))
        
        print(f"   Pair Statistics (showing first 15 pairs):")
        print(f"   {'(y1,y2)':>10} {'f1':>10} {'f2':>10} {'Avg α':>10} {'Accept%':>10} {'Pairs':>8}")
        print("-" * 80)
        
        for (y1, y2), stats in sorted(list(pair_stats.items())[:15]):
            avg_alpha = np.mean(stats['alphas'])
            accept_pct = 100.0 * np.mean(stats['accepted'])
            num_pairs = len(stats['alphas'])
            f1 = stats['f1']
            f2 = stats['f2']
            
            # Verify manual calculation
            manual_alpha = f2 / (M * max(f1, 1e-15))
            manual_alpha = min(manual_alpha, 1.0)
            
            print(f"  ({y1:2d},{y2:2d})    {f1:10.6f} {f2:10.6f} {avg_alpha:10.6f} {accept_pct:9.1f}% {num_pairs:8d}")
        
        if len(pair_stats) > 15:
            print(f"   ... and {len(pair_stats)-15} more (y1,y2) pairs")
    else:
        print(f"   (No size=2 pairs available)")
    
    print(f"\n{'='*100}\n")


def _feature_distribution(features: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Build an empirical 2D distribution from 1-based feature bins.

    Notes
    -----
    `features` follows the MATLAB-style 1-based bins (e.g., y1=1,2,...).
    The output array is NumPy (0-based), so counts are written to
    ``out[y1 - 1, y2 - 1]``.
    """
    # If there are no observations, return an all-zero table with a safe shape.
    if features.size == 0:
        if shape is None:
            return np.zeros((1, 1), dtype=float)
        return np.zeros(shape, dtype=float)

    # Infer the contingency-table size from observed max 1-based bin ids.
    if shape is None:
        shape = (int(np.max(features[:, 0])), int(np.max(features[:, 1])))

    # Accumulate counts into a 2D table.
    out = np.zeros(shape, dtype=float)
    for y1, y2 in features:
        # Skip malformed/out-of-range bins rather than raising.
        if y1 < 1 or y2 < 1 or y1 > shape[0] or y2 > shape[1]:
            continue
        # Map 1-based bins (paper/MATLAB convention) to 0-based NumPy indices.
        out[y1 - 1, y2 - 1] += 1.0

    # Normalize counts into probabilities; keep all-zero table if no valid counts.
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
    verbose: bool = False,
    stage: str = "Rejection",
    batch_num: int = 0,
) -> SynthesisResult:
    """Apply rejection-sampling correction to synthetic two-person households.

    Non-two-person households are retained unchanged.
    
    Parameters
    ----------
    verbose : bool
        If True, print detailed diagnostics about rejection process.
    stage : str
        Label for diagnostics output (e.g., 'First-pass', 'Top-up batch 1')
    batch_num : int
        Batch number for diagnostics
    """
    rng = rng or np.random.default_rng()

    prop_feat, prop_gid = two_person_pair_features( # synthetic pop feature pairs and their household ids
        synthetic.individual_data, synthetic.individual_group_id, age_col=age_col, sex_col=sex_col
    )
    tgt_feat, _ = two_person_pair_features( # target pop feature pairs and their household ids (not used in rejection but could be useful for diagnostics)
        np.asarray(target_individual_data, dtype=int),
        np.asarray(target_individual_group_id, dtype=int),
        age_col=age_col,
        sex_col=sex_col,
    )

    if prop_feat.size == 0 or tgt_feat.size == 0:
        print("No two-person features pair available for rejection correction; skipping.")
        return synthetic

    shape = (
        max(int(np.max(prop_feat[:, 0])), int(np.max(tgt_feat[:, 0]))),
        max(int(np.max(prop_feat[:, 1])), int(np.max(tgt_feat[:, 1]))),
    )

    proposal_prob = _feature_distribution(prop_feat, shape=shape) 
    target_prob = _feature_distribution(tgt_feat, shape=shape)
    
    if verbose:
        print(f"\n[REJECT-{stage} {batch_num}] Applying rejection filter to size=2 households...")
    
    accept_two = rejection_filter_two_person_households(
        prop_feat, target_prob, proposal_prob, rng=rng, verbose=verbose
    )

    keep_two_groups = set(prop_gid[accept_two].tolist())

    # Keep all households with size != 2 and accepted two-person households.
    all_groups = np.arange(1, synthetic.household_data.shape[0] + 1)

    sizes: np.ndarray | None = None # if household_size_col is provided and valid, use it to determine which households are two-person; otherwise, infer sizes from individual group ids.
    if household_size_col is not None and 0 <= household_size_col < synthetic.household_data.shape[1]:
        sizes = synthetic.household_data[:, household_size_col].astype(int)
    else:
        # Fallback: infer household sizes from sampled individuals when size column is unavailable.
        sizes = np.bincount(synthetic.individual_group_id, minlength=synthetic.household_data.shape[0] + 1)[1:].astype(int)

    keep_groups: list[int] = []
    for g in all_groups:
        if int(sizes[g - 1]) != 2: # if size column is available and indicates not two-person, keep the household without rejection; otherwise, rely on inferred sizes.
            keep_groups.append(int(g))
        elif int(g) in keep_two_groups:
            keep_groups.append(int(g))

    keep_groups_arr = np.asarray(keep_groups, dtype=int)
    if keep_groups_arr.size == 0:
        return synthetic
    
    # Compute diagnostics BEFORE re-indexing
    if verbose:
        # Create placeholder result to compute after-rejection sizes
        after_mask = np.isin(all_groups, keep_groups_arr)
        after_household = synthetic.household_data[after_mask, :]
        
        # Recalculate alphas for diagnostics
        alpha_for_diag = rejection_filter_two_person_households(
            prop_feat, target_prob, proposal_prob, rng=rng, verbose=False
        )
        
        _rejection_diagnostics_for_batch(
            stage=stage,
            batch_num=batch_num,
            before_household_data=synthetic.household_data,
            after_household_data=after_household,
            prop_feat=prop_feat,
            alpha_vals=alpha_for_diag,
            accept_mask=accept_two,
            proposal_prob=proposal_prob,
            target_prob=target_prob,
            household_size_col=household_size_col or 3,
        )

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


def _slice_first_households(synth: SynthesisResult, n_keep: int, hh_offset: int) -> SynthesisResult:
    """Take the first n_keep households from a SynthesisResult, re-indexing gids by hh_offset."""
    keep_h = min(int(n_keep), int(synth.household_data.shape[0]))
    if keep_h <= 0:
        return SynthesisResult(
            household_data=np.zeros((0, synth.household_data.shape[1]), dtype=int),
            individual_data=np.zeros((0, synth.individual_data.shape[1]), dtype=int),
            individual_group_id=np.zeros((0,), dtype=int),
            household_class=np.zeros((0,), dtype=int),
            individual_class=np.zeros((0,), dtype=int),
        )
    hh = synth.household_data[:keep_h, :]
    hh_class = synth.household_class[:keep_h]
    ind_mask = synth.individual_group_id <= keep_h
    return SynthesisResult(
        household_data=hh,
        individual_data=synth.individual_data[ind_mask, :],
        individual_group_id=synth.individual_group_id[ind_mask] + int(hh_offset),
        household_class=hh_class,
        individual_class=synth.individual_class[ind_mask],
    )


# paper_ref: Section 4.3 + Section 4.4 (Eq. 16) combined with target-count guarantee
# matlab_ref: run_create_2.m + run_two_within_the_same_4.m (oversample-then-reject pattern)
# intent: produce exactly n_households after Eq.16 rejection by redrawing fresh batches
#         instead of MATLAB's row-duplication hack, so the output has no duplicate rows.
def generate_with_two_person_rejection(
    n_households: int,
    pi_g: np.ndarray,
    pi_m: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m: Sequence[np.ndarray],
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    *,
    household_size_col: int = 3,
    age_col: int = 0,
    sex_col: int = 1,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 10,
    rng: np.random.Generator | None = None,
) -> SynthesisResult:
    """Generate exactly n_households synthetic households with two-person rejection applied.

    Single-pass rejection (`apply_two_person_rejection` alone) drops two-person households
    whose (|age-diff|, |sex-diff|) features are over-represented in the proposal vs PUMS,
    which can leave the size-2 stratum severely depleted. This helper oversamples up front
    and, if the post-rejection count is still short of the target, draws fresh batches and
    re-applies rejection until the target is reached.

    Returns a SynthesisResult with exactly ``n_households`` rows. Raises if the target
    cannot be reached within ``max_topup_iters``.
    """
    if n_households <= 0:
        raise ValueError("n_households must be positive")
    if initial_oversample < 1.0:
        raise ValueError("initial_oversample must be >= 1.0")
    if topup_factor < 1.0:
        raise ValueError("topup_factor must be >= 1.0")
    if max_topup_iters < 0:
        raise ValueError("max_topup_iters must be non-negative")

    rng = rng or np.random.default_rng()

    first_n = max(int(math.ceil(n_households * initial_oversample)), n_households)
    raw = generate_synthetic_population(
        n_households=first_n,
        pi_g=pi_g,
        pi_m=pi_m,
        phi_g=phi_g,
        phi_m=phi_m,
        household_size_col=household_size_col,
        rng=rng,
    )
    kept = apply_two_person_rejection(
        synthetic=raw,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        age_col=age_col,
        sex_col=sex_col,
        household_size_col=household_size_col,
        rng=rng,
        verbose=True,
        stage="First-pass",
        batch_num=0,
    )

    parts: list[SynthesisResult] = []
    hh_offset = 0
    take = _slice_first_households(kept, n_households, hh_offset)
    parts.append(take)
    hh_offset += take.household_data.shape[0]
    remaining = n_households - take.household_data.shape[0]

    iters = 0
    while remaining > 0:
        if iters >= max_topup_iters:
            raise RuntimeError(
                f"generate_with_two_person_rejection: could not reach target "
                f"{n_households} households after {max_topup_iters} top-up iterations "
                f"(short by {remaining}). Try increasing max_topup_iters or topup_factor."
            )
        iters += 1
        batch_n = max(int(math.ceil(remaining * topup_factor)), 200)
        batch_raw = generate_synthetic_population(
            n_households=batch_n,
            pi_g=pi_g,
            pi_m=pi_m,
            phi_g=phi_g,
            phi_m=phi_m,
            household_size_col=household_size_col,
            rng=rng,
        )
        batch_kept = apply_two_person_rejection(
            synthetic=batch_raw,
            target_individual_data=target_individual_data,
            target_individual_group_id=target_individual_group_id,
            age_col=age_col,
            sex_col=sex_col,
            household_size_col=household_size_col,
            rng=rng,
            verbose=True,
            stage=f"Top-up batch",
            batch_num=iters,
        )
        batch_take = _slice_first_households(batch_kept, remaining, hh_offset)
        if batch_take.household_data.shape[0] == 0:
            continue
        parts.append(batch_take)
        hh_offset += batch_take.household_data.shape[0]
        remaining -= batch_take.household_data.shape[0]

    household_data = np.vstack([p.household_data for p in parts])
    household_class = np.concatenate([p.household_class for p in parts])
    individual_data = np.vstack([p.individual_data for p in parts])
    individual_group_id = np.concatenate([p.individual_group_id for p in parts])
    individual_class = np.concatenate([p.individual_class for p in parts])

    return SynthesisResult(
        household_data=household_data,
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        household_class=household_class,
        individual_class=individual_class,
    )
