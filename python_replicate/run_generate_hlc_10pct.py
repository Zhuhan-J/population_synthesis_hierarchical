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
from hm_popsyn.metrics import evaluate_synthetic

import time
from tqdm import tqdm

# Input columns for processed_HLC_10pct_seed4556.npy (shape N x 11):
#   0  HH_id           identifier
#   1  Ind_id          identifier
#   2  HH_HOUSETYPE    HH attr
#   3  HH_INCOME       HH attr
#   4  HH_TOTALPAX     HH attr  (household size — used for 2-person rejection filter)
#   5  HH_CAR          HH attr
#   6  HH_BIKE         HH attr
#   7  HH_PLN_Area     HH attr  (omitted in "without_PLN" variant)
#   8  Ind_AGE         Ind attr  → age_col=0 in prepared inddata
#   9  Ind_EMPLOY      Ind attr  → index 1 in prepared inddata
#  10  Ind_RELATIONSHIP Ind attr  → sex_col=2 in prepared inddata

# Rejection sampling uses:
#   age_col=0            (Ind_AGE, 0-based index in prepared inddata)
#   sex_col=2            (Ind_RELATIONSHIP, 0-based index in prepared inddata)
#   household_size_col=2 (HH_TOTALPAX, 0-based index in prepared grpdata; same in both PLN variants)

# Column names for the original 11-column input (used by evaluate_synthetic).
_ORIG_COL_NAMES = [
    "HH_id", "Ind_id",
    "HH_HOUSETYPE", "HH_INCOME", "HH_TOTALPAX", "HH_CAR", "HH_BIKE", "HH_PLN_Area",
    "Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP",
]

_VARIANTS = [
    {
        "pln_label": "with_PLN",
        "household_attr_cols": (2, 3, 4, 5, 6, 7),
        "csv_header": (
            "HH_id,Ind_id,"
            "HH_HOUSETYPE,HH_INCOME,HH_TOTALPAX,HH_CAR,HH_BIKE,HH_PLN_Area,"
            "Ind_AGE,Ind_EMPLOY,Ind_RELATIONSHIP"
        ),
        "col_names": [
            "HH_id", "Ind_id",
            "HH_HOUSETYPE", "HH_INCOME", "HH_TOTALPAX", "HH_CAR", "HH_BIKE", "HH_PLN_Area",
            "Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP",
        ],
    },
    {
        "pln_label": "without_PLN",
        "household_attr_cols": (2, 3, 4, 5, 6),
        "csv_header": (
            "HH_id,Ind_id,"
            "HH_HOUSETYPE,HH_INCOME,HH_TOTALPAX,HH_CAR,HH_BIKE,"
            "Ind_AGE,Ind_EMPLOY,Ind_RELATIONSHIP"
        ),
        "col_names": [
            "HH_id", "Ind_id",
            "HH_HOUSETYPE", "HH_INCOME", "HH_TOTALPAX", "HH_CAR", "HH_BIKE",
            "Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP",
        ],
    },
]


def _save_synthetic_output(
    merged_data: np.ndarray,
    output_stem: Path,
    csv_header: str,
) -> None:
    npy_path = output_stem.with_suffix(".npy")
    csv_path = output_stem.with_suffix(".csv")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, merged_data)
    np.savetxt(csv_path, merged_data, fmt="%d", delimiter=",", header=csv_header, comments="")
    _log(f"[INFO] Saved npy: {npy_path}")
    _log(f"[INFO] Saved csv: {csv_path}")
    _log(f"[INFO] Output shape: {merged_data.shape}")
    _log(f"[INFO] Unique households: {np.unique(merged_data[:, 0]).size}")


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


def main() -> int:
    # timing the entire process
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Generate synthetic households from processed_HLC_10pct_seed4556.npy"
    )
    parser.add_argument(
        "--source-npy",
        default="/Users/zhuhan/Documents/HTS_Codes_Syn_POP/exported/processed_HLC_10pct_seed4556.npy",
    )
    parser.add_argument("--data-copy", default="data/processed_HLC_10pct_seed4556.npy")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--target-households", type=int, default=20000)
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--disable-gm-search", action="store_true")
    parser.add_argument("--g-candidates", default="3,4,5,6,7")
    parser.add_argument("--m-candidates", default="5,6,7,8,9")
    parser.add_argument("--selection-restarts", type=int, default=2)
    parser.add_argument("--selection-max-iter", type=int, default=600)
    args = parser.parse_args()

    src = Path(args.source_npy)
    data_copy = Path(args.data_copy)
    out_dir = Path(args.output_dir)

    if not src.exists():
        raise FileNotFoundError(f"source npy not found: {src}")

    _copy_input_if_needed(src, data_copy)
    _log(f"[INFO] Copied/using input: {data_copy}")

    raw = np.load(data_copy, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 11:
        raise ValueError(f"Expected shape (N, 11), got {raw.shape}")
    _log(f"[INFO] Raw input shape: {raw.shape}")

    # io.prepare_from_merged_table shifts every column so its minimum becomes 1
    # (matching MATLAB 1-based category encoding: arr = arr - arr.min(axis=0) + 1).
    # For columns whose raw minimum was already 1 the shift is 0; for 0-based binary
    # columns (HH_CAR, HH_BIKE) the shift is +1, making 0/1 → 1/2 in the output.
    # We record each column's raw minimum here so we can reverse the shift after
    # generation, restoring the original encoding in the saved files.
    raw_col_min = raw.min(axis=0)  # shape (11,)

    # ------------------------------------------------------------------
    # Phase 1: Prepare data + BIC model selection (once per PLN variant)
    # ------------------------------------------------------------------
    prepared_configs: list[dict] = []
    for variant in _VARIANTS:
        pln_label: str = variant["pln_label"]
        hh_attr_cols: tuple = variant["household_attr_cols"]

        _log(f"\n{'='*100}")
        _log(f"[INFO] Phase 1 — Input variant: {pln_label}  (household_attr_cols={hh_attr_cols})")
        _log(f"{'='*100}")

        prepared = prepare_from_merged_table(
            raw,
            household_id_col=0,
            individual_id_col=1,
            household_attr_cols=hh_attr_cols,
            individual_attr_cols=(8, 9, 10),
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
                f"[INFO] BIC model selection: G in {g_candidates}, M in {m_candidates}, "
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
                    f"[BIC] G={s.G:2d} M={s.M:2d} "
                    f"BIC={s.bic:12.4f} best_logL={s.best_log_likelihood:12.4f}"
                )
            selected_G = int(best_summary.G)
            selected_M = int(best_summary.M)
            _log(
                f"[INFO] Optimal: G={selected_G}, M={selected_M}, BIC={best_summary.bic:.4f}, "
                f"converged={best_model.converged}"
            )

        prepared_configs.append({
            "variant": variant,
            "prepared": prepared,
            "G": selected_G,
            "M": selected_M,
        })

    # ------------------------------------------------------------------
    # Phase 2: Generate + Evaluate (progress bar covers all 4 runs)
    # ------------------------------------------------------------------
    generation_tasks = [
        (cfg, apply_rej, rej_label)
        for cfg in prepared_configs
        for apply_rej, rej_label in [(True, "with_rejection"), (False, "without_rejection")]
    ]

    _log(f"\n{'='*100}")
    _log(f"[INFO] Phase 2 — Generating {len(generation_tasks)} variants (progress bar below)")
    _log(f"{'='*100}")

    for cfg, apply_rejection, rejection_label in tqdm(
        generation_tasks,
        desc="Generating",
        unit="run",
        ncols=90,
    ):
        variant = cfg["variant"]
        prepared = cfg["prepared"]
        selected_G = cfg["G"]
        selected_M = cfg["M"]

        pln_label = variant["pln_label"]
        hh_attr_cols = variant["household_attr_cols"]
        csv_header = variant["csv_header"]
        syn_col_names = variant["col_names"]

        _log(f"\n{'-'*80}")
        _log(f"[INFO] Generating: {pln_label} / {rejection_label}  (G={selected_G}, M={selected_M})")
        _log(f"{'-'*80}")

        result = fit_and_generate(
            grpdata=prepared.grpdata,
            indgid=prepared.indgid,
            inddata=prepared.inddata,
            G=selected_G,
            M=selected_M,
            n_households=args.target_households,
            n_restarts=args.n_restarts,
            max_iter=args.max_iter,
            tol=1e-7,
            seed=args.seed,
            apply_rejection=apply_rejection,
            age_col=0,            # Ind_AGE  (index 0 in prepared inddata)
            sex_col=2,            # Ind_RELATIONSHIP (index 2 in prepared inddata)
            household_size_col=2, # HH_TOTALPAX (index 2 in prepared grpdata)
        )

        _log(
            "[INFO] EM diagnostics: "
            f"iterations={result.em_result.likelihood.size}, "
            f"converged={result.em_result.converged}, "
            f"final_logL={result.em_result.likelihood[-1]:.4f}"
        )
        _log(
            "[INFO] Generated: "
            f"households={result.synthetic_final.household_data.shape[0]}, "
            f"individuals={result.synthetic_final.individual_data.shape[0]}"
        )

        merged_out = _merged_from_synthetic(
            result.synthetic_final.household_data,
            result.synthetic_final.individual_data,
            result.synthetic_final.individual_group_id,
        )

        # Reverse the min-shift for attribute columns: correction = raw_min - 1.
        # Cols 0,1 are freshly assigned HH_id/Ind_id — no correction needed.
        # Cols 2+ correspond to hh_attr_cols then individual_attr_cols in order.
        all_attr_raw_cols = list(hh_attr_cols) + [8, 9, 10]
        for out_offset, raw_col in enumerate(all_attr_raw_cols):
            correction = int(raw_col_min[raw_col]) - 1
            if correction != 0:
                merged_out[:, 2 + out_offset] += correction

        output_stem = out_dir / f"generated_HLC_10pct_{pln_label}_{rejection_label}"
        _save_synthetic_output(merged_out, output_stem, csv_header)
        _log(f"[INFO] Elapsed: {time.time() - start_time:.1f}s")

        # Evaluate synthetic quality vs original
        evaluate_synthetic(
            original_arr=raw,
            synthetic_arr=merged_out,
            original_col_names=_ORIG_COL_NAMES,
            synthetic_col_names=syn_col_names,
            label=f"{pln_label} / {rejection_label}",
        )

    _log(f"\n[INFO] All done. Total elapsed: {time.time() - start_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
