"""Ego-Alter EM for the Alternative A hierarchical mixture extension.

Separate latent class distributions are learned for ego (Ind_id=1) and alter
(Ind_id>=2) individuals. Both use the same feature set (current dataset), but
the missing-data framing naturally extends to asymmetric feature sets without
architectural changes.

See docs/alt_a_ego_alter_em_derivation.md for the full mathematical derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .em import (
    ModelSelectionResult,
    _as_category_matrix,
    _init_log_phi,
    _normalize_rows,
    _prepare_group_index,
    bic_score,
    count_parameters_universal,
)
from .kernels import logsumexp, sum_eik


@dataclass(slots=True)
class EgoAlterEMResult:
    """Fitted parameters from the ego-alter EM.

    Attributes
    ----------
    pi_g:
        Household class weights, shape (G,).
    pi_m_ego:
        Conditional ego class weights per household class, shape (G, M).
    pi_m_alt:
        Conditional alter class weights per household class, shape (G, M).
    log_phi_g:
        Household attribute distributions, each shape (d_k, G).
    log_phi_m_ego:
        Ego individual attribute distributions, each shape (d_k, M).
    log_phi_m_alt:
        Alter individual attribute distributions, each shape (d_k, M).
    wj:
        Household responsibilities, shape (N_HH, G).
    x_jik_ego:
        Ego individual responsibilities, shape (N_HH, G, M).
    x_jik_alt:
        Alter individual responsibilities, shape (N_alt, G, M).
    likelihood:
        Log-likelihood trace.
    converged:
        Whether the algorithm reached tolerance.
    """

    pi_g: np.ndarray
    pi_m_ego: np.ndarray
    pi_m_alt: np.ndarray
    log_phi_g: list[np.ndarray]
    log_phi_m_ego: list[np.ndarray]
    log_phi_m_alt: list[np.ndarray]
    wj: np.ndarray
    x_jik_ego: np.ndarray
    x_jik_alt: np.ndarray
    likelihood: np.ndarray
    converged: bool


def em_ego_alter_universal(
    grpdata: np.ndarray,
    ego_indgid: np.ndarray,
    ego_inddata: np.ndarray,
    alt_indgid: np.ndarray,
    alt_inddata: np.ndarray,
    G: int,
    M: int,
    max_iter: int = 1000,
    concentration: float = 1.0,
    concentration_ego: float = 1.5,
    tol: float = 1e-7,
    rng: np.random.Generator | None = None,
    eps: float = 1e-12,
) -> EgoAlterEMResult:
    """Fit the ego-alter hierarchical mixture model via EM (Alternative A).

    Parameters
    ----------
    grpdata : (N_HH, H) — household-level categorical attributes.
    ego_indgid : (N_HH,) — household ID for each ego row (one ego per household).
    ego_inddata : (N_HH, K_ego) — individual attributes for Ind_id=1 rows.
    alt_indgid : (N_alt,) — household IDs for alter rows (Ind_id>=2).
    alt_inddata : (N_alt, K_alt) — individual attributes for alter rows.
    G : number of household latent classes.
    M : number of individual latent classes (shared between ego and alter).
    concentration : Dirichlet concentration for alter and household parameters.
    concentration_ego : Dirichlet concentration for ego parameters (default 1.5,
        higher than alter because ego observations per class are fewer).
    """
    if G <= 0 or M <= 0:
        raise ValueError("G and M must be positive")
    if max_iter <= 1:
        raise ValueError("max_iter must be greater than 1")

    rng = rng or np.random.default_rng()

    grpdata_c = _as_category_matrix("grpdata", grpdata)
    ego_c = _as_category_matrix("ego_inddata", ego_inddata)
    alt_c = _as_category_matrix("alt_inddata", alt_inddata) if alt_inddata.shape[0] > 0 else np.zeros((0, ego_inddata.shape[1]), dtype=int)

    ngrp = grpdata_c.shape[0]
    n_ego = ego_c.shape[0]
    n_alt = alt_c.shape[0]

    if n_ego != ngrp:
        raise ValueError(f"Expected one ego per household: n_ego={n_ego} != ngrp={ngrp}")

    ego_gid0 = _prepare_group_index(ego_indgid, ngrp)
    alt_gid0 = _prepare_group_index(alt_indgid, ngrp) if n_alt > 0 else np.zeros(0, dtype=int)

    grpcate = np.max(grpdata_c, axis=0)
    egocate = np.max(ego_c, axis=0)                               # shape (K_ego,)
    altcate = np.max(alt_c, axis=0) if n_alt > 0 else egocate    # shape (K_alt,)

    numgat    = grpdata_c.shape[1]
    numatt_ego = ego_c.shape[1]   # K_ego (may be > K_alt when ego-only features present)
    numatt_alt = alt_c.shape[1]   # K_alt

    # Initialize parameters — ego gets stronger Dirichlet smoothing
    log_phi_g     = _init_log_phi(grpcate, G, concentration, rng)
    log_phi_m_ego = _init_log_phi(egocate, M, concentration_ego, rng)  # (K_ego) entries
    log_phi_m_alt = _init_log_phi(altcate, M, concentration, rng)      # (K_alt) entries

    pi_g = rng.gamma(shape=1.0, scale=concentration, size=G)
    pi_g /= np.maximum(pi_g.sum(), eps)

    pi_m_ego = rng.gamma(shape=1.0, scale=concentration_ego, size=(G, M))
    pi_m_ego = _normalize_rows(pi_m_ego, eps)

    pi_m_alt = rng.gamma(shape=1.0, scale=concentration, size=(G, M))
    pi_m_alt = _normalize_rows(pi_m_alt, eps)

    lik = np.zeros(max_iter, dtype=float)
    converged = False
    n_iter_done = 0

    wj = np.zeros((ngrp, G), dtype=float)
    x_jik_ego = np.zeros((n_ego, G, M), dtype=float)
    x_jik_alt = np.zeros((n_alt, G, M), dtype=float)

    for it in range(max_iter):
        # ------- E-step: ego -------
        f_ego = np.zeros((n_ego, M), dtype=float)
        for att in range(numatt_ego):
            f_ego += log_phi_m_ego[att][ego_c[:, att] - 1, :]

        h_ego = np.exp(f_ego)[:, None, :] * pi_m_ego[None, :, :]  # (n_ego, G, M)
        denom_ego = np.maximum(np.sum(h_ego, axis=2, keepdims=True), eps)
        x_jik_ego = h_ego / denom_ego
        log_g_ego = np.log(np.maximum(np.sum(h_ego, axis=2), eps))  # (n_ego, G)

        # Ego contribution to household log-likelihood (one term per HH)
        tt0 = np.zeros((ngrp, G), dtype=float)
        for g in range(G):
            np.add.at(tt0[:, g], ego_gid0, log_g_ego[:, g])

        # ------- E-step: alters -------
        if n_alt > 0:
            f_alt = np.zeros((n_alt, M), dtype=float)
            for att in range(numatt_alt):
                f_alt += log_phi_m_alt[att][alt_c[:, att] - 1, :]

            h_alt = np.exp(f_alt)[:, None, :] * pi_m_alt[None, :, :]  # (n_alt, G, M)
            denom_alt = np.maximum(np.sum(h_alt, axis=2, keepdims=True), eps)
            x_jik_alt = h_alt / denom_alt
            log_g_alt = np.log(np.maximum(np.sum(h_alt, axis=2), eps))  # (n_alt, G)

            # Alter contributions accumulated into same tt0
            for g in range(G):
                np.add.at(tt0[:, g], alt_gid0, log_g_alt[:, g])

        # ------- E-step: household responsibilities -------
        temp_g = np.zeros((ngrp, G), dtype=float)
        for att in range(numgat):
            temp_g += log_phi_g[att][grpdata_c[:, att] - 1, :]

        temp = temp_g + tt0 + np.log(np.maximum(pi_g[None, :], eps))
        temp_max = np.max(temp, axis=1, keepdims=True)
        wj0 = np.exp(temp - temp_max)
        wj = wj0 / np.maximum(np.sum(wj0, axis=1, keepdims=True), eps)

        # ------- M-step: pi_g -------
        pi_g = np.sum(wj, axis=0) + eps
        pi_g /= np.maximum(pi_g.sum(), eps)

        # ------- M-step: ego class weights and feature distributions -------
        second_ego = x_jik_ego * wj[ego_gid0, :, None]  # (n_ego, G, M)
        pi_m_ego = np.sum(second_ego, axis=0) + eps
        pi_m_ego = _normalize_rows(pi_m_ego, eps)

        second2_ego = np.sum(second_ego, axis=1)  # (n_ego, M)
        for att in range(numatt_ego):
            tt = sum_eik(second2_ego, ego_c[:, att], int(egocate[att])) + eps
            log_phi_m_ego[att] = np.log(tt / np.maximum(np.sum(tt, axis=0, keepdims=True), eps))

        # ------- M-step: alter class weights and feature distributions -------
        if n_alt > 0:
            second_alt = x_jik_alt * wj[alt_gid0, :, None]  # (n_alt, G, M)
            pi_m_alt = np.sum(second_alt, axis=0) + eps
            pi_m_alt = _normalize_rows(pi_m_alt, eps)

            second2_alt = np.sum(second_alt, axis=1)  # (n_alt, M)
            for att in range(numatt_alt):
                tt = sum_eik(second2_alt, alt_c[:, att], int(altcate[att])) + eps
                log_phi_m_alt[att] = np.log(tt / np.maximum(np.sum(tt, axis=0, keepdims=True), eps))

        # ------- M-step: household attribute distributions -------
        for att in range(numgat):
            tt = sum_eik(wj, grpdata_c[:, att], int(grpcate[att])) + eps
            log_phi_g[att] = np.log(tt / np.maximum(np.sum(tt, axis=0, keepdims=True), eps))

        # ------- Log-likelihood -------
        temp_lik = temp_g + tt0 + np.log(np.maximum(pi_g[None, :], eps))
        lik[it] = float(np.sum(logsumexp(temp_lik, axis=1)))
        n_iter_done = it + 1

        if it >= 1:
            rel = abs((lik[it] - lik[it - 1]) / max(abs(lik[it - 1]), eps))
            if rel <= tol:
                converged = True
                break

    return EgoAlterEMResult(
        pi_g=pi_g,
        pi_m_ego=pi_m_ego,
        pi_m_alt=pi_m_alt,
        log_phi_g=log_phi_g,
        log_phi_m_ego=log_phi_m_ego,
        log_phi_m_alt=log_phi_m_alt,
        wj=wj,
        x_jik_ego=x_jik_ego,
        x_jik_alt=x_jik_alt,
        likelihood=lik[:n_iter_done],
        converged=converged,
    )


def fit_em_with_restarts_ego_alter(
    grpdata: np.ndarray,
    ego_indgid: np.ndarray,
    ego_inddata: np.ndarray,
    alt_indgid: np.ndarray,
    alt_inddata: np.ndarray,
    G: int,
    M: int,
    n_restarts: int = 5,
    max_iter: int = 1000,
    concentration: float = 1.0,
    concentration_ego: float = 1.5,
    tol: float = 1e-7,
    seed: int | None = None,
) -> tuple[EgoAlterEMResult, ModelSelectionResult]:
    """Run multiple EM restarts and select best model by BIC."""
    if n_restarts <= 0:
        raise ValueError("n_restarts must be positive")

    base_rng = np.random.default_rng(seed)

    grpdata_c = _as_category_matrix("grpdata", grpdata)
    ego_c = _as_category_matrix("ego_inddata", ego_inddata)
    grpcate = np.max(grpdata_c, axis=0)
    egocate = np.max(ego_c, axis=0)
    altcate = (
        np.max(_as_category_matrix("alt_inddata", alt_inddata), axis=0)
        if alt_inddata.shape[0] > 0
        else egocate
    )
    # BIC parameter count: ego params (over K_ego attrs) + alter params (over K_alt attrs)
    # Use separate counts to handle asymmetric feature sets (K_ego may differ from K_alt)
    n_params_ego = count_parameters_universal(G, M, grpcate, egocate)
    n_params_alt_ind = int(np.sum(altcate - 1) * M)  # alter individual attrs only
    n_params = n_params_ego + n_params_alt_ind

    # BIC uses combined individual count (ego + alter) for consistency with original
    n_ind = ego_inddata.shape[0] + alt_inddata.shape[0]

    best_res: EgoAlterEMResult | None = None
    best_bic = np.inf
    best_ll = -np.inf

    for _ in range(n_restarts):
        child_rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        res = em_ego_alter_universal(
            grpdata=grpdata_c,
            ego_indgid=ego_indgid,
            ego_inddata=ego_c,
            alt_indgid=alt_indgid,
            alt_inddata=alt_inddata,
            G=G,
            M=M,
            max_iter=max_iter,
            concentration=concentration,
            concentration_ego=concentration_ego,
            tol=tol,
            rng=child_rng,
        )
        ll = float(np.max(res.likelihood))
        bic = bic_score(ll, n_params, n_ind)
        if bic < best_bic:
            best_bic = bic
            best_ll = ll
            best_res = res

    assert best_res is not None
    return best_res, ModelSelectionResult(G=G, M=M, best_log_likelihood=best_ll, bic=float(best_bic))


def select_model_grid_ego_alter(
    grpdata: np.ndarray,
    ego_indgid: np.ndarray,
    ego_inddata: np.ndarray,
    alt_indgid: np.ndarray,
    alt_inddata: np.ndarray,
    g_values: Sequence[int],
    m_values: Sequence[int],
    n_restarts: int = 2,
    max_iter: int = 250,
    concentration: float = 1.0,
    concentration_ego: float = 1.5,
    tol: float = 1e-7,
    seed: int | None = None,
) -> tuple[EgoAlterEMResult, ModelSelectionResult, list[ModelSelectionResult]]:
    """BIC grid search over (G, M) for the ego-alter model."""
    results: list[ModelSelectionResult] = []
    best_model: EgoAlterEMResult | None = None
    best_summary: ModelSelectionResult | None = None

    base_rng = np.random.default_rng(seed)
    for G in g_values:
        for M in m_values:
            child_seed = int(base_rng.integers(0, 2**32 - 1))
            em_res, summary = fit_em_with_restarts_ego_alter(
                grpdata=grpdata,
                ego_indgid=ego_indgid,
                ego_inddata=ego_inddata,
                alt_indgid=alt_indgid,
                alt_inddata=alt_inddata,
                G=int(G),
                M=int(M),
                n_restarts=n_restarts,
                max_iter=max_iter,
                concentration=concentration,
                concentration_ego=concentration_ego,
                tol=tol,
                seed=child_seed,
            )
            results.append(summary)
            if best_summary is None or summary.bic < best_summary.bic:
                best_summary = summary
                best_model = em_res

    assert best_model is not None and best_summary is not None
    return best_model, best_summary, results
