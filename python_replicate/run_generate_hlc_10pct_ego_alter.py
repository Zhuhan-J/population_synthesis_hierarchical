from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_popsyn.io import prepare_from_merged_table
from hm_popsyn.em_ego_alter import select_model_grid_ego_alter, fit_em_with_restarts_ego_alter
from hm_popsyn.pipeline_ego_alter import (
    generate_synthetic_population_ego_alter,
    generate_with_multi_size_rejection_ego_alter,
)
from hm_popsyn.metrics import evaluate_synthetic

import time
from tqdm import tqdm

# Input columns for processed_HLC_10pct_seed4556.npy (shape N x 11):
#   0  HH_id           identifier
#   1  Ind_id          identifier  (1 = ego/survey respondent, >=2 = alters)
#   2  HH_HOUSETYPE    HH attr
#   3  HH_INCOME       HH attr
#   4  HH_TOTALPAX     HH attr
#   5  HH_CAR          HH attr
#   6  HH_BIKE         HH attr
#   7  HH_PLN_Area     HH attr  (omitted in "without_PLN" variant)
#   8  Ind_AGE         Ind attr  → age_col=0 in prepared inddata
#   9  Ind_EMPLOY      Ind attr  → index 1 in prepared inddata
#  10  Ind_RELATIONSHIP Ind attr  → sex_col=2 in prepared inddata

# Rejection sampling uses:
#   age_col=0            (Ind_AGE, 0-based index in prepared inddata)
#   employ_col=1         (Ind_EMPLOY, 0-based index in prepared inddata)
#   household_size_col=2 (HH_TOTALPAX, 0-based index in prepared grpdata)

_COL_NAMES_11 = [
    "HH_id", "Ind_id",
    "HH_HOUSETYPE", "HH_INCOME", "HH_TOTALPAX", "HH_CAR", "HH_BIKE", "HH_PLN_Area",
    "Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP",
]
_COL_NAMES_13 = _COL_NAMES_11 + ["Ego_GENDER", "Ego_EDU"]
_ORIG_COL_NAMES = _COL_NAMES_11  # backward-compat alias

_VARIANTS = [
    # {
    #     "pln_label": "with_PLN",
    #     "household_attr_cols": (2, 3, 4, 5, 6, 7),
    #     "csv_header": (
    #         "HH_id,Ind_id,"
    #         "HH_HOUSETYPE,HH_INCOME,HH_TOTALPAX,HH_CAR,HH_BIKE,HH_PLN_Area,"
    #         "Ind_AGE,Ind_EMPLOY,Ind_RELATIONSHIP"
    #     ),
    #     "col_names": [
    #         "HH_id", "Ind_id",
    #         "HH_HOUSETYPE", "HH_INCOME", "HH_TOTALPAX", "HH_CAR", "HH_BIKE", "HH_PLN_Area",
    #         "Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP",
    #     ],
    # },
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
        # merges synthetic household + individual arrays back into the original merged format
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


def _average_metrics(metrics_list: list) -> dict | None:
    """Recursively average numeric leaves in a list of nested metric dicts."""
    non_none = [m for m in metrics_list if m is not None]
    if not non_none:
        return None
    if isinstance(non_none[0], dict):
        result = {}
        for k in non_none[0].keys():
            sub_vals = [m[k] for m in non_none if k in m]
            result[k] = _average_metrics(sub_vals)
        return result
    elif isinstance(non_none[0], float):
        return float(np.mean(non_none))
    elif isinstance(non_none[0], int):
        return int(round(float(np.mean(non_none))))
    else:
        return non_none[0]


def _format_metrics_row(m: dict | None) -> str:
    """Format a single metrics dict as a compact one-liner."""
    if m is None:
        return "N/A"
    parts = []
    if "level1" in m:
        l1 = m["level1"]
        s1 = l1.get("srmse", {})
        p1 = l1.get("prf", {})
        parts.append(
            f"L1 marg={s1.get('marginal', float('nan')):.4f} "
            f"biv={s1.get('bivariate', float('nan')):.4f} "
            f"P={p1.get('precision', float('nan')):.4f} "
            f"R={p1.get('recall', float('nan')):.4f} "
            f"F1={p1.get('f1', float('nan')):.4f}"
        )
        l2 = m.get("level2")
        if l2 is not None:
            s2 = l2.get("srmse", {})
            p2 = l2.get("prf", {})
            parts.append(
                f"L2 marg={s2.get('marginal', float('nan')):.4f} "
                f"biv={s2.get('bivariate', float('nan')):.4f} "
                f"P={p2.get('precision', float('nan')):.4f} "
                f"R={p2.get('recall', float('nan')):.4f} "
                f"F1={p2.get('f1', float('nan')):.4f}"
            )
        else:
            parts.append("L2 N/A")
    elif "hh_ind" in m:
        hi = m["hh_ind"]
        s = hi.get("srmse", {})
        p = hi.get("prf", {})
        parts.append(
            f"HH-IND marg={s.get('marginal', float('nan')):.4f} "
            f"biv={s.get('bivariate', float('nan')):.4f} "
            f"P={p.get('precision', float('nan')):.4f} "
            f"R={p.get('recall', float('nan')):.4f} "
            f"F1={p.get('f1', float('nan')):.4f}"
        )
    return " | ".join(parts) if parts else "N/A"


def _print_metrics_summary(run_metrics: list[dict], label: str, n_runs: int) -> None:
    """Print per-run metrics rows and a final average row."""
    avg = _average_metrics(run_metrics)
    _log(f"\n{'='*100}")
    _log(f"[METRICS SUMMARY] {label}  ({n_runs} runs)")
    _log(f"{'='*100}")
    for i, m in enumerate(run_metrics, 1):
        _log(f"  Run {i:2d}: {_format_metrics_row(m)}")
    _log(f"  {'Average':>6}: {_format_metrics_row(avg)}")
    _log(f"{'='*100}")


def main() -> int:
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Ego-alter HLC: generate synthetic households from exported/processed_HLC_10pct_HH_IND_EGO_seed505_20260505.npy"
    )
    parser.add_argument(
        "--source-npy",
        default="/Users/zhuhan/Documents/HTS_Codes_Syn_POP/exported/processed_HLC_10pct_HH_IND_EGO_seed505_20260505.npy", # default using hh-ind-ego
    )
    parser.add_argument("--data-copy", default="data/processed_HLC_10pct_HH_IND_EGO_seed505_20260505.npy")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--target-households", type=int, default=20000)
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=5) # increase for more stable BIC selection, especially with larger candidate sets
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-runs", type=int, default=5,
        help="Number of generation runs (EM is fitted once; generation repeats n-runs times with seeds seed+0, seed+1, ...).")
    parser.add_argument("--disable-gm-search", action="store_true")
    parser.add_argument("--g-candidates", default="3,4,5,6,7")
    parser.add_argument("--m-candidates", default="5,6,7,8,9")
    parser.add_argument("--selection-restarts", type=int, default=2)
    parser.add_argument("--selection-max-iter", type=int, default=600)
    parser.add_argument("--concentration-ego", type=float, default=1.5)
    parser.add_argument(
        "--target-npy",
        default="/Users/zhuhan/Documents/HTS_Codes_Syn_POP/exported/processed_HLC_hPOP_HH_IND_EGO_seed505_20260505.npy",
        help="Full-population npy used as evaluation target (not training data).",
    )
    args = parser.parse_args()

    src = Path(args.source_npy)
    data_copy = Path(args.data_copy)
    out_dir = Path(args.output_dir)

    if not src.exists():
        raise FileNotFoundError(f"source npy not found: {src}")

    _copy_input_if_needed(src, data_copy) # copy to a local data/ directory for consistent loading, but avoid unnecessary copying if already present
    _log(f"[INFO] Copied/using input: {data_copy}")

    # Try allow_pickle=False first (works for int64 and float64 files).
    # Fall back to allow_pickle=True for legacy object-dtype files (<NA> export),
    # converting pandas <NA> to np.nan automatically via pd.array conversion.
    try:
        raw = np.load(data_copy, allow_pickle=False)
    except ValueError: # fallback for legacy files with object dtype and <NA> values
        import pandas as pd
        raw_obj = np.load(data_copy, allow_pickle=True)
        raw = np.where(pd.isna(raw_obj), np.nan, raw_obj).astype(float)
        _log(f"[WARN] Loaded as object dtype and converted to float64. "
             f"Re-export with the fixed notebook cell for cleaner loading.")
    n_cols = raw.shape[1]
    if raw.ndim != 2 or n_cols not in (11, 13):
        raise ValueError(f"Expected shape (N, 11) or (N, 13), got {raw.shape}")
    _log(f"[INFO] Raw input shape: {raw.shape}  dtype={raw.dtype}")

    orig_col_names = _COL_NAMES_13 if n_cols == 13 else _COL_NAMES_11

    # Shared integer portion (cols 0-10): always finite, safe to cast to int
    raw_int = raw[:, :11].astype(int)         # shape (N, 11)
    raw_col_min = raw_int.min(axis=0)          # shape (11,), used for output correction

    # Ego-only portion (cols 11-12): float with np.nan for alter rows; None for 11-col files
    raw_ego_only = raw[:, 11:13] if n_cols == 13 else None

    # Load full-population target for evaluation metrics.
    target_npy_path = Path(args.target_npy)
    if not target_npy_path.exists():
        _log(f"[WARN] Target npy not found: {target_npy_path}. Falling back to training data for metrics.")
        target_arr = raw
        target_col_names = orig_col_names
    else:
        try:
            target_arr = np.load(target_npy_path, allow_pickle=False)
        except ValueError:
            import pandas as pd
            target_obj = np.load(target_npy_path, allow_pickle=True)
            target_arr = np.where(pd.isna(target_obj), np.nan, target_obj).astype(float)
        if target_arr.ndim != 2 or target_arr.shape[1] not in (11, 13):
            raise ValueError(f"Target npy: expected (N, 11) or (N, 13), got {target_arr.shape}")
        target_col_names = _COL_NAMES_13 if target_arr.shape[1] == 13 else _COL_NAMES_11
        _log(f"[INFO] Target population shape: {target_arr.shape}  (used for evaluation metrics)")

    # ------------------------------------------------------------------
    # Phase 1: Prepare data + BIC model selection + final EM fit (once per PLN variant)
    # ------------------------------------------------------------------
    prepared_configs: list[dict] = []
    for variant in _VARIANTS:
        pln_label: str = variant["pln_label"]
        hh_attr_cols: tuple = variant["household_attr_cols"]

        _log(f"\n{'='*100}")
        _log(f"[INFO] Phase 1 — Input variant: {pln_label}  (household_attr_cols={hh_attr_cols})")
        _log(f"{'='*100}")

        prepared = prepare_from_merged_table(
            raw_int,                     # always the 11-col int portion
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

        # Split into ego (Ind_id=1) and alter (Ind_id>=2) rows.
        # After prepare_from_merged_table sorts by (HH_id, Ind_id), the first
        # occurrence of each household is always the ego.
        first_idx = np.unique(prepared.indgid, return_index=True)[1]
        ego_mask = np.zeros(len(prepared.indgid), dtype=bool)
        ego_mask[first_idx] = True
        alt_mask = ~ego_mask

        ego_indgid = prepared.indgid[ego_mask]
        ego_inddata = prepared.inddata[ego_mask]    # (N_HH, 3) — shared attrs: AGE, EMPLOY, RELATIONSHIP
        alt_indgid = prepared.indgid[alt_mask]
        alt_inddata = prepared.inddata[alt_mask]    # (N_alt, 3)

        # Ind_RELATIONSHIP (col 2) is always "Respondent" for egos — remove from model features.
        # The fixed code is reinserted at col 2 during generation so output layout is preserved.
        _ego_rel_vals = ego_inddata[:, 2]
        _unique_rel = np.unique(_ego_rel_vals)
        if _unique_rel.size == 1:
            ego_rel_code = int(_unique_rel[0])
            _log(f"[INFO] Ego RELATIONSHIP is constant (prepared code={ego_rel_code}); "
                 f"removed from ego model features.")
        else:
            ego_rel_code = int(np.bincount(_ego_rel_vals).argmax())
            _log(f"[WARN] Ego RELATIONSHIP has {_unique_rel.size} unique values {_unique_rel}; "
                 f"using mode={ego_rel_code} as fixed output value.")
        ego_inddata = np.delete(ego_inddata, 2, axis=1)  # (N_HH, 2): [AGE, EMPLOY]

        ego_only_min = None
        if raw_ego_only is not None:
            # Align ego-only cols with prepare_from_merged_table's sort (lexsort HH_id, Ind_id)
            sort_order = np.lexsort((raw_int[:, 1], raw_int[:, 0]))
            ego_only_sorted = raw_ego_only[sort_order, :]        # (N, 2), NaN for alters
            ego_only_vals   = ego_only_sorted[ego_mask, :]       # (N_HH, 2), all finite

            # Apply same 1-based min-shift as prepare_from_merged_table
            ego_only_min     = np.nanmin(ego_only_vals, axis=0)  # [1, 1] for current data
            ego_only_shifted = (ego_only_vals - ego_only_min + 1).astype(int)

            ego_inddata = np.hstack([ego_inddata, ego_only_shifted])  # (N_HH, 4): [AGE, EMPLOY, GENDER, EDU]
            _log(
                f"[INFO] Ego-alter split: "
                f"ego rows={ego_inddata.shape[0]} (K_ego={ego_inddata.shape[1]}, excl. RELATIONSHIP), "
                f"alter rows={alt_inddata.shape[0]} (K_alt={alt_inddata.shape[1]})"
            )
        else:
            _log(
                f"[INFO] Ego-alter split: "
                f"ego rows={ego_inddata.shape[0]} (K_ego={ego_inddata.shape[1]}, excl. RELATIONSHIP), "
                f"alter rows={alt_inddata.shape[0]}"
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
            best_model, best_summary, summaries = select_model_grid_ego_alter(
                grpdata=prepared.grpdata,
                ego_indgid=ego_indgid,
                ego_inddata=ego_inddata,
                alt_indgid=alt_indgid,
                alt_inddata=alt_inddata,
                g_values=g_candidates,
                m_values=m_candidates,
                n_restarts=args.selection_restarts,
                max_iter=args.selection_max_iter,
                concentration=1.0,
                concentration_ego=args.concentration_ego,
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

        # Final EM fit with the selected G, M — fitted once, shared across all generation runs.
        _log(
            f"\n[INFO] Fitting final EM model: G={selected_G}, M={selected_M}, "
            f"n_restarts={args.n_restarts}, max_iter={args.max_iter}"
        )
        em_result, _ = fit_em_with_restarts_ego_alter(
            grpdata=prepared.grpdata,
            ego_indgid=ego_indgid,
            ego_inddata=ego_inddata,
            alt_indgid=alt_indgid,
            alt_inddata=alt_inddata,
            G=selected_G,
            M=selected_M,
            n_restarts=args.n_restarts,
            max_iter=args.max_iter,
            concentration=1.0,
            concentration_ego=args.concentration_ego,
            tol=1e-7,
            seed=args.seed,
        )
        _log(
            f"[INFO] EM fit complete: iterations={em_result.likelihood.size}, "
            f"converged={em_result.converged}, final_logL={em_result.likelihood[-1]:.4f}"
        )

        prepared_configs.append({
            "variant": variant,
            "prepared": prepared,
            "ego_indgid": ego_indgid,
            "ego_inddata": ego_inddata,
            "alt_indgid": alt_indgid,
            "alt_inddata": alt_inddata,
            "G": selected_G,
            "M": selected_M,
            "ego_only_min": ego_only_min,
            "ego_rel_code": ego_rel_code,
            "em_result": em_result,
        })

    # ------------------------------------------------------------------
    # Phase 2: Generate + Evaluate — n_runs times per (variant, rejection) combo
    # ------------------------------------------------------------------
    generation_tasks = [
        (cfg, apply_rej, rej_label)
        for cfg in prepared_configs
        for apply_rej, rej_label in [(True, "with_rejection"), (False, "without_rejection")]
    ]

    _log(f"\n{'='*100}")
    _log(
        f"[INFO] Phase 2 — Generating {len(generation_tasks)} variant(s) × {args.n_runs} run(s) "
        f"= {len(generation_tasks) * args.n_runs} total synthetic populations"
    )
    _log(f"{'='*100}")

    for cfg, apply_rejection, rejection_label in tqdm(
        generation_tasks,
        desc="Generating",
        unit="variant",
        ncols=90,
    ):
        variant = cfg["variant"]
        prepared = cfg["prepared"]
        ego_indgid = cfg["ego_indgid"]
        ego_inddata = cfg["ego_inddata"]
        alt_indgid = cfg["alt_indgid"]
        alt_inddata = cfg["alt_inddata"]
        selected_G = cfg["G"]
        selected_M = cfg["M"]
        em_result = cfg["em_result"]

        pln_label = variant["pln_label"]
        hh_attr_cols = variant["household_attr_cols"]
        ego_only_min = cfg["ego_only_min"]   # None for 11-col files
        ego_rel_code = cfg["ego_rel_code"]

        # Build col_names and csv_header dynamically based on whether ego-only cols exist
        _hh_names = [_COL_NAMES_11[c] for c in sorted(hh_attr_cols)]
        _ind_shared = ["Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP"]
        _ego_extra = ["Ego_GENDER", "Ego_EDU"] if ego_only_min is not None else []
        syn_col_names = ["HH_id", "Ind_id"] + _hh_names + _ind_shared + _ego_extra
        csv_header = ",".join(syn_col_names)

        _log(f"\n{'-'*80}")
        _log(
            f"[INFO] Variant: {pln_label} / {rejection_label}  "
            f"(G={selected_G}, M={selected_M}, concentration_ego={args.concentration_ego}, "
            f"n_runs={args.n_runs})"
        )
        _log(f"{'-'*80}")

        # Pre-compute linearised EM parameters (shared across all runs)
        phi_g_lin     = [np.exp(x) for x in em_result.log_phi_g]
        phi_m_ego_lin = [np.exp(x) for x in em_result.log_phi_m_ego]
        phi_m_alt_lin = [np.exp(x) for x in em_result.log_phi_m_alt]

        # Combined ego+alter inddata for rejection sampling reference target
        age_col = 0
        employ_col = 1
        household_size_col = 2
        k_rej = max(age_col, employ_col) + 1
        if alt_inddata.shape[0] > 0:
            combined_inddata = np.vstack([ego_inddata[:, :k_rej], alt_inddata[:, :k_rej]])
            combined_indgid  = np.concatenate([ego_indgid, alt_indgid])
        else:
            combined_inddata = ego_inddata[:, :k_rej]
            combined_indgid  = ego_indgid

        run_metrics: list[dict] = []

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            _log(f"\n[INFO] Run {run_idx + 1}/{args.n_runs}  (seed={run_seed})")

            if apply_rejection:
                rej_rng = np.random.default_rng(run_seed + 1)
                synth_final = generate_with_multi_size_rejection_ego_alter(
                    n_households=args.target_households,
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
                    max_topup_iters=40,
                    rng=rej_rng,
                    ego_rel_code=ego_rel_code,
                )
            else:
                rng = np.random.default_rng(run_seed)
                synth_final = generate_synthetic_population_ego_alter(
                    n_households=args.target_households,
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

            _log(
                f"[INFO] Generated: "
                f"households={synth_final.household_data.shape[0]}, "
                f"individuals={synth_final.individual_data.shape[0]}"
            )

            merged_out = _merged_from_synthetic(
                synth_final.household_data,
                synth_final.individual_data,
                synth_final.individual_group_id,
            )

            # Reverse the min-shift for shared attribute columns (HH + shared ind, cols 0-10).
            all_attr_raw_cols = list(hh_attr_cols) + [8, 9, 10]
            for out_offset, raw_col in enumerate(all_attr_raw_cols):
                correction = int(raw_col_min[raw_col]) - 1
                if correction != 0:
                    merged_out[:, 2 + out_offset] += correction

            # Reverse the min-shift for ego-only cols (Ego_GENDER, Ego_EDU) for ego rows only.
            # Alter rows have 0 in these positions (uninitialized) — do not correct them.
            if ego_only_min is not None:
                ego_only_start = 2 + len(hh_attr_cols) + 3  # first ego-only col in output
                ego_rows_mask = merged_out[:, 1] == 1        # Ind_id == 1
                for k, min_val in enumerate(ego_only_min):
                    correction = int(min_val) - 1
                    if correction != 0:
                        merged_out[ego_rows_mask, ego_only_start + k] += correction

            output_stem = out_dir / (
                f"generated_HLC_10pct_ego_alter_{pln_label}_{rejection_label}_run{run_idx + 1}_attr_{n_cols}"
            )
            _save_synthetic_output(merged_out, output_stem, csv_header)
            _log(f"[INFO] Elapsed: {time.time() - start_time:.1f}s")

            metrics = evaluate_synthetic(
                original_arr=target_arr,
                synthetic_arr=merged_out,
                original_col_names=target_col_names,
                synthetic_col_names=syn_col_names,
                label=f"ego_alter / {pln_label} / {rejection_label} / run {run_idx + 1}",
            )
            run_metrics.append(metrics)

        _print_metrics_summary(
            run_metrics,
            label=f"ego_alter / {pln_label} / {rejection_label}",
            n_runs=args.n_runs,
        )

    _log(f"\n[INFO] All done. Total elapsed: {time.time() - start_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
