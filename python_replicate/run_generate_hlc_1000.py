from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.io import prepare_from_merged_table
from hm_popsyn.em import select_model_grid
from hm_popsyn.pipeline import fit_and_generate
from hm_popsyn.synthesis_eval import apply_two_person_rejection, generate_synthetic_population


def _parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("candidate list cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("all candidate values must be positive")
    return sorted(set(vals))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _copy_input_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def _slice_first_households(
    household_data: np.ndarray,
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
    n_keep: int,
    household_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_h = min(n_keep, household_data.shape[0])
    if keep_h <= 0:
        return np.zeros((0, household_data.shape[1]), dtype=int), np.zeros((0, individual_data.shape[1]), dtype=int), np.zeros((0,), dtype=int)

    hh_keep = household_data[:keep_h, :]
    ind_mask = individual_group_id <= keep_h
    ind_keep = individual_data[ind_mask, :]
    gid_keep = individual_group_id[ind_mask] + household_offset
    return hh_keep, ind_keep, gid_keep


def _merged_from_synthetic(
    household_data: np.ndarray,
    individual_data: np.ndarray,
    individual_group_id: np.ndarray,
) -> np.ndarray:
    rows: list[list[int]] = []
    for hh in range(1, household_data.shape[0] + 1):
        hh_attrs = household_data[hh - 1, :].astype(int).tolist()
        idx = np.where(individual_group_id == hh)[0]
        for i, ind_idx in enumerate(idx, start=1):
            ind_attrs = individual_data[ind_idx, :].astype(int).tolist()
            rows.append([hh, i, *hh_attrs, *ind_attrs])
    if not rows:
        return np.zeros((0, 2 + household_data.shape[1] + individual_data.shape[1]), dtype=int)
    return np.asarray(rows, dtype=int)


# paper_ref: Section 4.2 -> 4.4 flow
# matlab_ref: run_multilevel_1.m + run_create_2.m + run_two_within_the_same_4.m
# intent: load processed input, fit model, generate and reject, then export person-level table.
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 1000 synthetic households from processed_HLC_1000.npy")
    parser.add_argument(
        "--source-npy",
        default="/Users/zhuhan/Documents/HTS_Codes_Syn_POP/exported/processed_HLC_1000.npy",
        help="Absolute path to processed source numpy file",
    )
    parser.add_argument(
        "--data-copy",
        default="data/processed_HLC_1000.npy",
        help="Path to copy cleaned input inside python_replicate",
    )
    parser.add_argument(
        "--output-npy",
        default="data/generated_HLC_1000.npy",
        help="Output person-level synthetic table in same format",
    )
    parser.add_argument("--target-households", type=int, default=1000)
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--disable-gm-search", action="store_true", help="Skip BIC grid-search and use --G/--M directly")
    parser.add_argument("--g-candidates", default="3,4,5,6,7", help="Comma-separated G candidates for BIC search")
    parser.add_argument("--m-candidates", default="5,6,7,8,9", help="Comma-separated M candidates for BIC search")
    parser.add_argument("--selection-restarts", type=int, default=2, help="Restarts per (G,M) in model selection")
    parser.add_argument("--selection-max-iter", type=int, default=250, help="Max EM iterations per (G,M) in model selection")
    args = parser.parse_args()

    src = Path(args.source_npy)
    data_copy = Path(args.data_copy)
    out_npy = Path(args.output_npy)

    if not src.exists():
        raise FileNotFoundError(f"source npy not found: {src}")

    _copy_input_if_needed(src, data_copy)
    _log(f"[INFO] Copied/using input file: {data_copy}")

    raw = np.load(data_copy, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"Expected shape (N,7), got {raw.shape}")
    _log(f"[INFO] Raw input shape: {raw.shape}")

    # Input format:
    # col0 HH_id, col1 Ind_id, col2-4 HH attrs (col4 HH_TOTALPAX), col5-6 individual attrs.
    prepared = prepare_from_merged_table(
        raw,
        household_id_col=0,
        individual_id_col=1,
        household_attr_cols=(2, 3, 4),
        individual_attr_cols=(5, 6),
    )
    _log(
        "[INFO] Prepared arrays: "
        f"grpdata={prepared.grpdata.shape}, inddata={prepared.inddata.shape}, "
        f"households={prepared.grpdata.shape[0]}, individuals={prepared.inddata.shape[0]}"
    )

    selected_G = int(args.G)
    selected_M = int(args.M)

    if args.disable_gm_search:
        _log(f"[INFO] G/M search disabled. Using G={selected_G}, M={selected_M}")
    else:
        g_candidates = _parse_int_list(args.g_candidates)
        m_candidates = _parse_int_list(args.m_candidates)

        _log(
            "[INFO] Running BIC model selection for latent classes: "
            f"G in {g_candidates}, M in {m_candidates}, "
            f"restarts={args.selection_restarts}, max_iter={args.selection_max_iter}"
        )

        best_model, best_summary, summaries = select_model_grid(
            grpdata=prepared.grpdata,
            indgid=prepared.indgid,
            inddata=prepared.inddata,
            g_values=g_candidates,
            m_values=m_candidates,
            n_restarts=args.selection_restarts,
            max_iter=args.selection_max_iter,
            concentration=1.0,
            tol=1e-7,
            seed=args.seed,
        )

        for s in sorted(summaries, key=lambda x: x.bic):
            _log(
                "[BIC] "
                f"G={s.G:2d} M={s.M:2d} "
                f"BIC={s.bic:12.4f} "
                f"best_logL={s.best_log_likelihood:12.4f}"
            )

        selected_G = int(best_summary.G)
        selected_M = int(best_summary.M)
        _log(
            "[INFO] Optimal latent classes selected: "
            f"G={selected_G}, M={selected_M}, BIC={best_summary.bic:.4f}"
        )
        _log(
            "[INFO] Selection-step iteration count for best model: "
            f"{best_model.likelihood.size} (converged={best_model.converged})"
        )

    # First pass: fit and generate with rejection.
    first_n = max(args.target_households, int(args.target_households * 1.5))
    _log(f"[INFO] First-pass synthesis target (pre-trim): {first_n} households")
    first = fit_and_generate(
        grpdata=prepared.grpdata,
        indgid=prepared.indgid,
        inddata=prepared.inddata,
        G=selected_G,
        M=selected_M,
        n_households=first_n,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        tol=1e-7,
        seed=args.seed,
        apply_rejection=True,
        age_col=0,   # Ind_AGE
        sex_col=1,   # Ind_EMPLOY used as second rejection attribute
        household_size_col=2,
    )
    _log(
        "[INFO] Main fit diagnostics: " 
        f"iterations={first.em_result.likelihood.size}, "
        f"converged={first.em_result.converged}, "
        f"final_logL={first.em_result.likelihood[-1]:.4f}"
    )
    _log(
        "[INFO] First-pass output after rejection: "
        f"households={first.synthetic_final.household_data.shape[0]}, "
        f"individuals={first.synthetic_final.individual_data.shape[0]}"
    )

    hh_parts: list[np.ndarray] = []
    ind_parts: list[np.ndarray] = []
    gid_parts: list[np.ndarray] = []

    hh_offset = 0
    remaining = args.target_households

    # Take the first households until we reach the target number after rejection.
    hh_take, ind_take, gid_take = _slice_first_households(
        first.synthetic_final.household_data,
        first.synthetic_final.individual_data,
        first.synthetic_final.individual_group_id,
        remaining,
        hh_offset,
    )
    hh_parts.append(hh_take)
    ind_parts.append(ind_take)
    gid_parts.append(gid_take)

    hh_offset += hh_take.shape[0]
    remaining -= hh_take.shape[0]

    # Top-up if rejection removed too many households.
    rng = np.random.default_rng(args.seed + 1)
    n_topup_batches = 0
    while remaining > 0:
        n_topup_batches += 1
        batch_n = max(int(remaining * 1.7), 200)
        _log(f"[INFO] Top-up batch {n_topup_batches}: generating {batch_n} households (remaining target={remaining})")
        raw_syn = generate_synthetic_population(
            n_households=batch_n,
            pi_g=first.em_result.pi_g,
            pi_m=first.em_result.pi_m,
            phi_g=[np.exp(x) for x in first.em_result.log_phi_g],
            phi_m=[np.exp(x) for x in first.em_result.log_phi_m],
            household_size_col=2,
            rng=rng,
        )
        final_syn = apply_two_person_rejection(
            synthetic=raw_syn,
            target_individual_data=prepared.inddata,
            target_individual_group_id=prepared.indgid,
            age_col=0,
            sex_col=1,
            household_size_col=2,
            rng=rng,
        )

        hh_take, ind_take, gid_take = _slice_first_households(
            final_syn.household_data,
            final_syn.individual_data,
            final_syn.individual_group_id,
            remaining,
            hh_offset,
        )

        if hh_take.shape[0] == 0:
            continue

        hh_parts.append(hh_take)
        ind_parts.append(ind_take)
        gid_parts.append(gid_take)

        hh_offset += hh_take.shape[0]
        remaining -= hh_take.shape[0]
        _log(
            f"[INFO] Top-up batch {n_topup_batches} accepted {hh_take.shape[0]} households; "
            f"still remaining={remaining}"
        )

    household_data = np.vstack(hh_parts)
    individual_data = np.vstack(ind_parts)
    individual_group_id = np.concatenate(gid_parts)

    merged_out = _merged_from_synthetic(household_data, individual_data, individual_group_id)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, merged_out)

    out_csv = out_npy.with_suffix(".csv")
    header = "HH_id,Ind_id,HH_HOUSETYPE,HH_INCOME,HH_TOTALPAX,Ind_AGE,Ind_EMPLOY" 
    np.savetxt(out_csv, merged_out, fmt="%d", delimiter=",", header=header, comments="")

    _log(f"[INFO] Saved synthetic output npy: {out_npy}")
    _log(f"[INFO] Saved synthetic output csv: {out_csv}")
    _log(f"[INFO] Output shape (rows, cols): {merged_out.shape}")
    _log(f"[INFO] Unique households: {np.unique(merged_out[:, 0]).size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
