"""End-to-end pipeline: fit EM, synthesize population, and apply rejection stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .em import EMResult, fit_em_with_restarts
from .synthesis_eval import (
    SynthesisResult,
    apply_two_person_rejection,
    generate_synthetic_population,
)


@dataclass(slots=True)
class PipelineResult:
    """Main end-to-end output container."""

    em_result: EMResult
    synthetic_raw: SynthesisResult
    synthetic_final: SynthesisResult


# paper_ref: Section 4.2 -> 4.4, Eq. (14), Eq. (3), Eq. (16)
# matlab_ref: run_multilevel_1.m, run_create_2.m, run_two_within_the_same_4.m
# intent: complete runnable workflow from observed data to final synthetic population.
def fit_and_generate(
    grpdata: np.ndarray,
    indgid: np.ndarray,
    inddata: np.ndarray,
    G: int,
    M: int,
    n_households: int,
    n_restarts: int = 5,
    max_iter: int = 1000,
    concentration: float = 1.0,
    tol: float = 1e-8,
    seed: int | None = None,
    apply_rejection: bool = True,
    age_col: int = 0,
    sex_col: int = 1,
    household_size_col: int = 3,
) -> PipelineResult:
    """Run fit -> synthetic generation -> optional rejection correction."""
    em_result, _ = fit_em_with_restarts(
        grpdata=grpdata,
        indgid=indgid,
        inddata=inddata,
        G=G,
        M=M,
        n_restarts=n_restarts,
        max_iter=max_iter,
        concentration=concentration,
        tol=tol,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    raw = generate_synthetic_population(
        n_households=n_households,
        pi_g=em_result.pi_g,
        pi_m=em_result.pi_m,
        phi_g=[np.exp(x) for x in em_result.log_phi_g],
        phi_m=[np.exp(x) for x in em_result.log_phi_m],
        household_size_col=household_size_col,
        rng=rng,
    )

    if apply_rejection:
        final = apply_two_person_rejection(
            synthetic=raw,
            target_individual_data=inddata,
            target_individual_group_id=indgid,
            age_col=age_col,
            sex_col=sex_col,
            household_size_col=household_size_col,
            rng=rng,
        )
    else:
        final = raw

    return PipelineResult(em_result=em_result, synthetic_raw=raw, synthetic_final=final)
