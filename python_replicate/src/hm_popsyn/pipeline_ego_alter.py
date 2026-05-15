"""End-to-end pipeline for the ego-alter hierarchical mixture model (Alternative A).

Generates households where Ind_id=1 is always the ego (sampled from ego
distributions) and Ind_id>=2 are alters (sampled from alter distributions).
All existing pipeline.py behaviour is preserved — this module adds alongside it.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np

from .em_ego_alter import (
    EgoAlterEMResult,
    fit_em_with_restarts_ego_alter,
)
from .pipeline import PipelineResult
from .synthesis_eval import (
    SynthesisResult,
    _size_counts_from_households,
    _scale_size_counts,
    _take_households_by_size,
    apply_rejection_by_size,
)


def generate_synthetic_population_ego_alter(
    n_households: int,
    pi_g: np.ndarray,
    pi_m_ego: np.ndarray,
    pi_m_alt: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m_ego: Sequence[np.ndarray],
    phi_m_alt: Sequence[np.ndarray],
    household_size_col: int,
    rng: np.random.Generator,
    ego_rel_code: int | None = None,
) -> SynthesisResult:
    """Generate households where Ind_id=1 is sampled from ego distributions.

    For each household:
    - ONE ego individual is sampled from (pi_m_ego, phi_m_ego).
    - n_size - 1 alter individuals are sampled from (pi_m_alt, phi_m_alt).
    The resulting individual_group_id assigns Ind_id=1 to the ego and
    Ind_id=2,3,... to alters (via the order in which they are stored).

    ego_rel_code: when set, Ind_RELATIONSHIP was removed from ego model features
        (phi_m_ego covers [AGE, EMPLOY, {GENDER, EDU}] only). This fixed integer
        is written to output col 2 for every ego row so the output layout matches
        alter rows: [AGE, EMPLOY, RELATIONSHIP, ...].
    """
    pi_g = np.asarray(pi_g, dtype=float)
    pi_g = pi_g / pi_g.sum()
    pi_m_ego = np.asarray(pi_m_ego, dtype=float)
    pi_m_alt = np.asarray(pi_m_alt, dtype=float)

    g_classes = rng.choice(np.arange(pi_g.size), size=n_households, p=pi_g)

    h_attrs = len(phi_g)
    household_data = np.zeros((n_households, h_attrs), dtype=int)
    for i in range(n_households):
        g = g_classes[i]
        for att, prob in enumerate(phi_g):
            p = np.asarray(prob, dtype=float)[:, g]
            p = p / p.sum()
            household_data[i, att] = int(rng.choice(np.arange(p.size), p=p)) + 1

    sizes = household_data[:, household_size_col].astype(int)
    sizes = np.maximum(sizes, 1)
    n_individuals = int(np.sum(sizes))

    k_ego = len(phi_m_ego)
    k_alt = len(phi_m_alt)
    # When ego_rel_code is set, phi_m_ego excludes RELATIONSHIP (removed at training time).
    # Output col 2 is reserved for the fixed RELATIONSHIP value; remaining ego attrs shift by +1.
    i_attrs = max(k_ego + (1 if ego_rel_code is not None else 0), k_alt)
    individual_data = np.zeros((n_individuals, i_attrs), dtype=int)
    individual_group_id = np.zeros(n_individuals, dtype=int)
    individual_class = np.zeros(n_individuals, dtype=int)

    cursor = 0
    for i in range(n_households):
        g = g_classes[i]

        # Sample ONE ego from ego distributions
        w_ego = np.asarray(pi_m_ego[g, :], dtype=float)
        w_ego = w_ego / w_ego.sum()
        m_ego = int(rng.choice(np.arange(w_ego.size), p=w_ego))
        if ego_rel_code is not None:
            # phi_m_ego: [AGE(0), EMPLOY(1), {GENDER(2), EDU(3)}]
            # Output:    [AGE(0), EMPLOY(1), RELATIONSHIP_fixed(2), {GENDER(3), EDU(4)}]
            for att in range(2):  # AGE and EMPLOY map directly
                p = np.asarray(phi_m_ego[att], dtype=float)[:, m_ego]
                p = p / p.sum()
                individual_data[cursor, att] = int(rng.choice(np.arange(p.size), p=p)) + 1
            individual_data[cursor, 2] = ego_rel_code  # fixed RELATIONSHIP at col 2
            for att in range(2, k_ego):  # GENDER→col 3, EDU→col 4, etc.
                p = np.asarray(phi_m_ego[att], dtype=float)[:, m_ego]
                p = p / p.sum()
                individual_data[cursor, att + 1] = int(rng.choice(np.arange(p.size), p=p)) + 1
        else:
            for att, prob in enumerate(phi_m_ego):
                p = np.asarray(prob, dtype=float)[:, m_ego]
                p = p / p.sum()
                individual_data[cursor, att] = int(rng.choice(np.arange(p.size), p=p)) + 1
        individual_group_id[cursor] = i + 1
        individual_class[cursor] = m_ego + 1
        cursor += 1

        # Sample n_size-1 alters from alter distributions
        if sizes[i] > 1:
            w_alt = np.asarray(pi_m_alt[g, :], dtype=float)
            w_alt = w_alt / w_alt.sum()
            for _ in range(sizes[i] - 1):
                m_alt = int(rng.choice(np.arange(w_alt.size), p=w_alt))
                for att, prob in enumerate(phi_m_alt):
                    p = np.asarray(prob, dtype=float)[:, m_alt]
                    p = p / p.sum()
                    individual_data[cursor, att] = int(rng.choice(np.arange(p.size), p=p)) + 1
                individual_group_id[cursor] = i + 1
                individual_class[cursor] = m_alt + 1
                cursor += 1

    return SynthesisResult(
        household_data=household_data,
        individual_data=individual_data,
        individual_group_id=individual_group_id,
        household_class=g_classes + 1,
        individual_class=individual_class,
    )


def generate_with_multi_size_rejection_ego_alter(
    n_households: int,
    pi_g: np.ndarray,
    pi_m_ego: np.ndarray,
    pi_m_alt: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m_ego: Sequence[np.ndarray],
    phi_m_alt: Sequence[np.ndarray],
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    *,
    household_size_col: int,
    age_col: int,
    employ_col: int,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 20,
    rng: np.random.Generator,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    min_target_samples: int = 30,
    ego_rel_code: int | None = None,
) -> SynthesisResult:
    """Ego-alter version of generate_with_multi_size_rejection."""
    if n_households <= 0:
        raise ValueError("n_households must be positive")

    first_n = max(int(math.ceil(n_households * initial_oversample)), n_households)
    raw = generate_synthetic_population_ego_alter(
        n_households=first_n,
        pi_g=pi_g, pi_m_ego=pi_m_ego, pi_m_alt=pi_m_alt,
        phi_g=phi_g, phi_m_ego=phi_m_ego, phi_m_alt=phi_m_alt,
        household_size_col=household_size_col, rng=rng,
        ego_rel_code=ego_rel_code,
    )

    if household_size_col < 0 or household_size_col >= raw.household_data.shape[1]:
        raise ValueError("household_size_col out of range for household_data")

    raw_counts = _size_counts_from_households(raw.household_data, household_size_col)
    target_counts = _scale_size_counts(raw_counts, n_households)

    kept = apply_rejection_by_size(
        synthetic=raw,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        age_col=age_col,
        employ_col=employ_col,
        household_size_col=household_size_col,
        rng=rng, verbose=True, stage="First-pass", batch_num=0,
        age_code_to_order=age_code_to_order,
        employ_code_to_group=employ_code_to_group,
        min_target_samples=min_target_samples,
    )

    parts: list[SynthesisResult] = []
    hh_offset = 0
    take, target_counts = _take_households_by_size(
        synth=kept,
        need_by_size=target_counts,
        household_size_col=household_size_col,
        hh_offset=hh_offset,
    )
    parts.append(take)
    hh_offset += take.household_data.shape[0]
    remaining = int(sum(target_counts.values()))

    iters = 0
    while remaining > 0:
        if iters >= max_topup_iters:
            raise RuntimeError(
                f"generate_with_multi_size_rejection_ego_alter: could not reach "
                f"target {n_households} after {max_topup_iters} top-up iterations "
                f"(short by {remaining})."
            )
        iters += 1
        batch_n = max(int(math.ceil(remaining * topup_factor)), 200)
        batch_raw = generate_synthetic_population_ego_alter(
            n_households=batch_n,
            pi_g=pi_g, pi_m_ego=pi_m_ego, pi_m_alt=pi_m_alt,
            phi_g=phi_g, phi_m_ego=phi_m_ego, phi_m_alt=phi_m_alt,
            household_size_col=household_size_col, rng=rng,
            ego_rel_code=ego_rel_code,
        )
        batch_kept = apply_rejection_by_size(
            synthetic=batch_raw,
            target_individual_data=target_individual_data,
            target_individual_group_id=target_individual_group_id,
            age_col=age_col,
            employ_col=employ_col,
            household_size_col=household_size_col,
            rng=rng, verbose=True, stage="Top-up batch", batch_num=iters,
            age_code_to_order=age_code_to_order,
            employ_code_to_group=employ_code_to_group,
            min_target_samples=min_target_samples,
        )
        batch_take, target_counts = _take_households_by_size(
            synth=batch_kept,
            need_by_size=target_counts,
            household_size_col=household_size_col,
            hh_offset=hh_offset,
        )
        if batch_take.household_data.shape[0] == 0:
            remaining = int(sum(target_counts.values()))
            continue
        parts.append(batch_take)
        hh_offset += batch_take.household_data.shape[0]
        remaining = int(sum(target_counts.values()))

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


def generate_with_two_person_rejection_ego_alter(
    n_households: int,
    pi_g: np.ndarray,
    pi_m_ego: np.ndarray,
    pi_m_alt: np.ndarray,
    phi_g: Sequence[np.ndarray],
    phi_m_ego: Sequence[np.ndarray],
    phi_m_alt: Sequence[np.ndarray],
    target_individual_data: np.ndarray,
    target_individual_group_id: np.ndarray,
    *,
    household_size_col: int,
    age_col: int,
    sex_col: int,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 10,
    rng: np.random.Generator,
) -> SynthesisResult:
    warnings.warn(
        "generate_with_two_person_rejection_ego_alter is deprecated; use "
        "generate_with_multi_size_rejection_ego_alter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_with_multi_size_rejection_ego_alter(
        n_households=n_households,
        pi_g=pi_g,
        pi_m_ego=pi_m_ego,
        pi_m_alt=pi_m_alt,
        phi_g=phi_g,
        phi_m_ego=phi_m_ego,
        phi_m_alt=phi_m_alt,
        target_individual_data=target_individual_data,
        target_individual_group_id=target_individual_group_id,
        household_size_col=household_size_col,
        age_col=age_col,
        employ_col=sex_col,
        initial_oversample=initial_oversample,
        topup_factor=topup_factor,
        max_topup_iters=max_topup_iters,
        rng=rng,
    )


def fit_and_generate_ego_alter(
    grpdata: np.ndarray,
    ego_indgid: np.ndarray,
    ego_inddata: np.ndarray,
    alt_indgid: np.ndarray,
    alt_inddata: np.ndarray,
    G: int,
    M: int,
    n_households: int,
    n_restarts: int = 5,
    max_iter: int = 1000,
    concentration: float = 1.0,
    concentration_ego: float = 1.5,
    tol: float = 1e-8,
    seed: int | None = None,
    apply_rejection: bool = True,
    age_col: int = 0,
    employ_col: int = 1,
    sex_col: int | None = None,
    household_size_col: int = 2,
    initial_oversample: float = 1.5,
    topup_factor: float = 1.7,
    max_topup_iters: int = 20,
    age_code_to_order: np.ndarray | None = None,
    employ_code_to_group: np.ndarray | None = None,
    min_target_samples: int = 30,
    ego_rel_code: int | None = None,
) -> PipelineResult:
    """Fit ego-alter EM then generate a synthetic population.

    ego_rel_code: fixed Ind_RELATIONSHIP code for ego rows. Set this when
        Ind_RELATIONSHIP has been dropped from ego_inddata before calling
        (it is constant for all egos — always "Respondent"). The generation
        writes this value at output col 2 for every ego row so the layout
        matches alter rows: [AGE, EMPLOY, RELATIONSHIP, ...].
    """
    if sex_col is not None:
        warnings.warn(
            "sex_col is deprecated; use employ_col for age x employ rejection.",
            DeprecationWarning,
            stacklevel=2,
        )
        employ_col = sex_col
    em_result, _ = fit_em_with_restarts_ego_alter(
        grpdata=grpdata,
        ego_indgid=ego_indgid,
        ego_inddata=ego_inddata,
        alt_indgid=alt_indgid,
        alt_inddata=alt_inddata,
        G=G,
        M=M,
        n_restarts=n_restarts,
        max_iter=max_iter,
        concentration=concentration,
        concentration_ego=concentration_ego,
        tol=tol,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    phi_g_lin = [np.exp(x) for x in em_result.log_phi_g]
    phi_m_ego_lin = [np.exp(x) for x in em_result.log_phi_m_ego]
    phi_m_alt_lin = [np.exp(x) for x in em_result.log_phi_m_alt]

    raw = generate_synthetic_population_ego_alter(
        n_households=n_households,
        pi_g=em_result.pi_g,
        pi_m_ego=em_result.pi_m_ego,
        pi_m_alt=em_result.pi_m_alt,
        phi_g=phi_g_lin,
        phi_m_ego=phi_m_ego_lin,
        phi_m_alt=phi_m_alt_lin,
        household_size_col=household_size_col,
        rng=rng,
        ego_rel_code=ego_rel_code,
    )

    if apply_rejection:
        # Rejection uses only age_col and employ_col (cols 0 and 1).
        # Use exactly those columns to avoid misalignment when ego_inddata
        # has fewer cols than alt_inddata (RELATIONSHIP removed from ego).
        k_rej = max(age_col, employ_col) + 1
        if alt_inddata.shape[0] > 0:
            combined_inddata = np.vstack([ego_inddata[:, :k_rej], alt_inddata[:, :k_rej]])
            combined_indgid = np.concatenate([ego_indgid, alt_indgid])
        else:
            combined_inddata = ego_inddata[:, :k_rej]
            combined_indgid = ego_indgid

        rej_rng = np.random.default_rng(seed + 1 if seed is not None else None)
        final = generate_with_multi_size_rejection_ego_alter(
            n_households=n_households,
            pi_g=em_result.pi_g,
            pi_m_ego=em_result.pi_m_ego,
            pi_m_alt=em_result.pi_m_alt,
            phi_g=phi_g_lin,
            phi_m_ego=phi_m_ego_lin,
            phi_m_alt=phi_m_alt_lin,
            target_individual_data=combined_inddata,
            target_individual_group_id=combined_indgid,
            household_size_col=household_size_col,
            age_col=age_col,
            employ_col=employ_col,
            initial_oversample=initial_oversample,
            topup_factor=topup_factor,
            max_topup_iters=max_topup_iters,
            rng=rej_rng,
            age_code_to_order=age_code_to_order,
            employ_code_to_group=employ_code_to_group,
            min_target_samples=min_target_samples,
            ego_rel_code=ego_rel_code,
        )
    else:
        final = raw

    # Wrap in PipelineResult using an EMResult-compatible view for diagnostics
    from .em import EMResult
    em_compat = EMResult(
        pi_g=em_result.pi_g,
        pi_m=em_result.pi_m_ego,  # expose ego weights as primary for logging
        log_phi_g=em_result.log_phi_g,
        log_phi_m=em_result.log_phi_m_ego,
        wj=em_result.wj,
        x_jik=em_result.x_jik_ego,
        likelihood=em_result.likelihood,
        converged=em_result.converged,
    )
    return PipelineResult(em_result=em_compat, synthetic_raw=raw, synthetic_final=final)
