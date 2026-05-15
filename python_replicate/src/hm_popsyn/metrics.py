"""SRMSE and Precision/Recall/F1 metrics for synthetic population evaluation.

Adapted from LLM-BN-HH_withoutLocation/evaluator.py for the HLC numpy array format.

Two evaluation levels:

  Level 1 — Ego-HH:
    One row per household = HH attributes + ego's individual attributes (Ind_id=1).
    Ego-only columns (e.g. Ind_GEN, Ind_EDU) are included if present and non-NaN
    for ego rows. Common columns between target and synthetic are used automatically,
    so with-PLN / without-PLN variants are handled.

  Level 2 — Ego-HH-Alter pairs:
    One row per ego-alter pair. Each row = (Level-1 ego+HH vars) + (alter individual
    vars with 'alt_' prefix). Size-1 households produce 0 pairs.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions (unchanged — used by both levels)
# ---------------------------------------------------------------------------

def srmse(
    pop_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    category_sets: dict[str, list] | None = None,
) -> dict[str, float]:
    """Standardized RMSE over marginal and bivariate distributions.

    For K variables the marginal concatenates K distributions (each summing to 1).
    The bivariate concatenates K*(K-1)/2 joint distributions (each summing to 1).
    SRMSE = RMSE / mean(pop_proportions).

    category_sets: optional {col: sorted_value_list} anchoring each column's bin set
        to the full-population codebook rather than the union of observed values.
        Extra zero-zero bins expand N_b and scale SRMSE by sqrt((N+B)/N) — see
        SRMSE_Example.md §4.

    Ref: SRMSE_Example.md in LLM-BN-HH_withoutLocation.
    """
    common_cols = sorted(set(pop_df.columns) & set(gen_df.columns))
    pop = pop_df[common_cols]
    gen = gen_df[common_cols]

    # --- Marginal ---
    pop_props, gen_props = [], []
    for col in common_cols:
        pop_counts = pop[col].dropna().value_counts()
        gen_counts = gen[col].dropna().value_counts()
        if category_sets is not None and col in category_sets:
            all_vals = category_sets[col]
        else:
            all_vals = sorted(set(pop_counts.index) | set(gen_counts.index))
        p = np.array([pop_counts.get(v, 0) for v in all_vals], dtype=float)
        g = np.array([gen_counts.get(v, 0) for v in all_vals], dtype=float)
        if p.sum() > 0:
            p /= p.sum()
        if g.sum() > 0:
            g /= g.sum()
        pop_props.append(p)
        gen_props.append(g)

    pop_props_arr = np.concatenate(pop_props)
    gen_props_arr = np.concatenate(gen_props)
    mean_pop = pop_props_arr.mean()
    srmse_mar = float(np.sqrt(np.mean((pop_props_arr - gen_props_arr) ** 2)) / mean_pop) if mean_pop > 0 else float("nan")

    # --- Bivariate ---
    pop_bi, gen_bi = [], []
    for col1, col2 in combinations(common_cols, 2):
        pop_pair = pop[[col1, col2]].dropna()
        gen_pair = gen[[col1, col2]].dropna()
        if len(pop_pair) == 0:
            continue
        pop_ct = pd.crosstab(pop_pair[col1], pop_pair[col2])
        gen_ct = pd.crosstab(gen_pair[col1], gen_pair[col2])
        if category_sets is not None:
            row_cats = category_sets.get(col1, sorted(set(pop_ct.index)   | set(gen_ct.index)))
            col_cats = category_sets.get(col2, sorted(set(pop_ct.columns) | set(gen_ct.columns)))
        else:
            row_cats = sorted(set(pop_ct.index)   | set(gen_ct.index))
            col_cats = sorted(set(pop_ct.columns) | set(gen_ct.columns))
        pop_ct = pop_ct.reindex(index=row_cats, columns=col_cats, fill_value=0)
        gen_ct = gen_ct.reindex(index=row_cats, columns=col_cats, fill_value=0)
        p = pop_ct.values.flatten().astype(float)
        g = gen_ct.values.flatten().astype(float)
        if p.sum() > 0:
            p /= p.sum()
        if g.sum() > 0:
            g /= g.sum()
        pop_bi.append(p)
        gen_bi.append(g)

    if pop_bi:
        pop_bi_arr = np.concatenate(pop_bi)
        gen_bi_arr = np.concatenate(gen_bi)
        mean_bi = pop_bi_arr.mean()
        srmse_bi = float(np.sqrt(np.mean((pop_bi_arr - gen_bi_arr) ** 2)) / mean_bi) if mean_bi > 0 else float("nan")
    else:
        srmse_bi = float("nan")

    return {"marginal": srmse_mar, "bivariate": srmse_bi}


def precision_recall_f1(pop_df: pd.DataFrame, gen_df: pd.DataFrame) -> dict[str, float | int]:
    """Precision/Recall/F1 via exact row-tuple matching.

    A generated row is a true positive if the same attribute tuple exists
    anywhere in the original population.
    Precision = TP / len(gen)   (fraction of synthetic tuples that are realistic)
    Recall    = TP / len(pop)   (fraction of real tuples reproduced)
    """
    common_cols = sorted(set(pop_df.columns) & set(gen_df.columns))
    pop_clean = pop_df[common_cols].dropna()
    gen_clean = gen_df[common_cols].dropna()

    pop_tuples = pop_clean.apply(tuple, axis=1)
    gen_tuples = gen_clean.apply(tuple, axis=1)

    pop_set = set(pop_tuples)
    gen_set = set(gen_tuples)

    precision = sum(1 for t in gen_tuples if t in pop_set) / len(gen_tuples) if len(gen_tuples) > 0 else 0.0
    recall = sum(1 for t in pop_tuples if t in gen_set) / len(pop_tuples) if len(pop_tuples) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "unique_pop": len(pop_set),
        "unique_gen": len(gen_set),
        "matching_types": len(pop_set & gen_set),
    }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_level1(
    arr: np.ndarray,
    col_names: list[str],
    id_cols: tuple[str, str] = ("HH_id", "Ind_id"),
    ind_id_col: str = "Ind_id",
    ego_id_val: int = 1,
) -> pd.DataFrame:
    """Extract ego (Ind_id=1) rows for Level-1 Ego-HH evaluation.

    Returns one row per household with HH attributes + ego individual attributes.
    Columns that are all-NaN after filtering (e.g. ego-only cols accidentally in
    alter rows) are dropped. Ego-only cols that ARE non-NaN for egos are kept.
    """
    df = pd.DataFrame(arr, columns=col_names)
    # Handle float arrays where Ind_id may be float due to NaN in other cols
    ind_id = df[ind_id_col]
    ego_mask = ind_id.astype(float).astype(int) == ego_id_val
    ego = df[ego_mask].copy()
    ego = ego.drop(columns=list(id_cols), errors="ignore")
    # Drop columns that are entirely NaN (safety — shouldn't happen for ego rows)
    ego = ego.dropna(axis=1, how="all")
    # add checkpoint to verify that ego rows count matches unique HH_id count (if HH_id exists)
    hh_id_col = id_cols[0]
    if hh_id_col in ego.columns:
        unique_hh_count = ego[hh_id_col].nunique()
        ego_row_count = len(ego)
        if unique_hh_count != ego_row_count:
            print(f"Warning: Level-1 extraction found {ego_row_count} ego rows but only {unique_hh_count} unique households based on '{hh_id_col}'.")
    return ego.reset_index(drop=True)


def _extract_level2(
    arr: np.ndarray,
    col_names: list[str],
    id_cols: tuple[str, str] = ("HH_id", "Ind_id"),
    ind_id_col: str = "Ind_id",
    hh_id_col: str = "HH_id",
    ego_id_val: int = 1,
    alter_ind_cols: tuple = ("Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP"),
) -> pd.DataFrame:
    """Extract ego-alter pairs for Level-2 Ego-HH-Alter evaluation.

    Each output row = (ego HH+individual attributes) + (one alter's individual
    attributes, renamed with 'alt_' prefix). Size-1 households produce 0 pairs.
    Alter individual columns not present in the data are silently skipped.
    """
    df = pd.DataFrame(arr, columns=col_names)
    ind_id = df[ind_id_col].astype(float).astype(int)
    ego_mask = ind_id == ego_id_val

    # Ego: all attributes except id cols, indexed by HH_id for the merge
    ego_df = df[ego_mask].copy()
    ego_df = ego_df.drop(columns=[ind_id_col], errors="ignore")
    ego_df = ego_df.dropna(axis=1, how="all")   # drop all-NaN ego-only cols if any
    ego_df = ego_df.set_index(hh_id_col)

    alter_df = df[~ego_mask].copy()
    avail_alter_cols = [c for c in alter_ind_cols if c in alter_df.columns]

    if alter_df.empty or not avail_alter_cols:
        return pd.DataFrame()

    # Keep HH_id + alter individual cols; rename alter cols with 'alt_' prefix
    alter_select = alter_df[[hh_id_col] + avail_alter_cols].copy()
    alter_select = alter_select.rename(columns={c: f"alt_{c}" for c in avail_alter_cols})

    # Merge: each alter row gets its ego's HH+individual attributes
    pairs = alter_select.merge(ego_df.reset_index(), on=hh_id_col, how="inner")
    pairs = pairs.drop(columns=[hh_id_col], errors="ignore")
    pairs = pairs.dropna(axis=1, how="all")
    return pairs.reset_index(drop=True)


def _extract_hh_ind(
    arr: np.ndarray,
    col_names: list[str],
    id_cols: tuple[str, str] = ("HH_id", "Ind_id"),
) -> pd.DataFrame:
    """Extract one row per individual (all members, no ego/alter distinction).

    Used when no ego-only attributes are present (11-col input). HH attributes
    are repeated for each member of the same household.
    """
    df = pd.DataFrame(arr, columns=col_names)
    df = df.drop(columns=list(id_cols), errors="ignore")
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def _full_cat_sets(
    common_cols: list[str],
    orig_full_df: pd.DataFrame,
) -> dict[str, list]:
    """Build {col: sorted_int_list} from the full original array (all rows, pre-filtering).

    For 'alt_X' columns, looks up source column 'X' in orig_full_df so that alter
    columns are anchored to the same full-population codebook as their shared variable.
    Columns absent from orig_full_df are omitted; callers fall back to observed union.
    """
    result: dict[str, list] = {}
    for col in common_cols:
        src = col[4:] if col.startswith("alt_") else col
        if src in orig_full_df.columns:
            result[col] = sorted(int(v) for v in orig_full_df[src].dropna().unique())
    return result


# ---------------------------------------------------------------------------
# Printing helper
# ---------------------------------------------------------------------------

def _print_metric_block(
    label: str,
    n_orig: int,
    n_syn: int,
    row_label: str,
    common_cols: list[str],
    cat_counts: dict[str, int],
    srmse_res: dict[str, float],
    prf_res: dict[str, float | int],
    alter_ind_cols: tuple,
    w: int = 74,
) -> None:
    n_bins_mar = sum(cat_counts.values())
    K = len(common_cols)
    n_variable_pairs = K * (K - 1) // 2
    n_bins_bi = sum(
        cat_counts[c1] * cat_counts[c2]
        for i, c1 in enumerate(common_cols)
        for c2 in common_cols[i + 1:]
    )

    print(f"\n{'='*w}")
    print(f"  {label}")
    print(f"{'='*w}")
    print(f"  {'Target ' + row_label + ':':<30} {n_orig:>10,}")
    print(f"  {'Synthetic ' + row_label + ':':<30} {n_syn:>10,}")
    print(f"  {'Variables evaluated (K):':<30} {K:>10}")
    print(f"{'-'*w}")
    print(f"  Variable breakdown (categories from full population — anchored codebook):")
    for col in common_cols:
        tag = "  alt →" if col.startswith("alt_") else "  ego+HH →" if col in [f"alt_{c}" for c in alter_ind_cols] else "       →"
        print(f"    {col:<32} {cat_counts[col]:>4} categories")
    print(f"{'-'*w}")
    print(f"  SRMSE — Marginal")
    print(f"    {'Total bins (Nb):':<30} {n_bins_mar:>8,}  (sum of category counts)")
    print(f"    {'SRMSE (marginal):':<30} {srmse_res['marginal']:>8.4f}  (lower is better)")
    print(f"  SRMSE — Bivariate")
    print(f"    {'Variable pairs K(K-1)/2:':<30} {n_variable_pairs:>8,}")
    print(f"    {'Total bins (Nb):':<30} {n_bins_bi:>8,}  (sum of pair cross-tab sizes)")
    print(f"    {'SRMSE (bivariate):':<30} {srmse_res['bivariate']:>8.4f}  (lower is better)")
    print(f"{'-'*w}")
    print(f"  Precision/Recall/F1  (exact {K}-variable tuple matching)")
    print(f"    {'Tuple size:':<30} {K:>8} variables per tuple")
    print(f"    {'Precision:':<30} {prf_res['precision']:>8.4f}  (synthetic tuples in target)")
    print(f"    {'Recall:':<30} {prf_res['recall']:>8.4f}  (target tuples reproduced)")
    print(f"    {'F1-score:':<30} {prf_res['f1']:>8.4f}")
    print(f"    {'Unique target tuples:':<30} {prf_res['unique_pop']:>8,}")
    print(f"    {'Unique syn tuples:':<30} {prf_res['unique_gen']:>8,}")
    print(f"    {'Matching tuple types:':<30} {prf_res['matching_types']:>8,}")
    print(f"{'='*w}")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_synthetic(
    original_arr: np.ndarray,
    synthetic_arr: np.ndarray,
    original_col_names: list[str],
    synthetic_col_names: list[str],
    label: str,
    id_cols: tuple[str, str] = ("HH_id", "Ind_id"),
    alter_ind_cols: tuple = ("Ind_AGE", "Ind_EMPLOY", "Ind_RELATIONSHIP"),
) -> dict:
    """Evaluate synthetic population quality against a target population.

    Evaluation mode is determined automatically from synthetic_col_names:

    - **Ego-only mode** (any column starts with ``"Ego_"``): ego-only attributes are
      present (13-col input). Runs Level-1 (Ego-HH) and Level-2 (Ego-HH-Alter pairs).
    - **HH-IND mode** (no ``"Ego_"`` columns): 11-col input with no ego/alter distinction.
      Runs a single HH-IND evaluation over all household members.

    In both modes, bin sets are anchored to the **full population** (all rows of
    original_arr before any filtering) so that SRMSE is consistent across runs and
    not affected by which categories happen to appear in filtered subsets.

    Parameters
    ----------
    original_arr : target (reference) population — shape (N_orig, n_cols).
        Recommended: full HTS population, not the training sample.
    synthetic_arr : generated synthetic population — shape (N_syn, n_cols).
    original_col_names : column names for original_arr.
    synthetic_col_names : column names for synthetic_arr.
    label : descriptive label printed in result headers.
    id_cols : identifier columns dropped before comparison.
    alter_ind_cols : individual-level columns used for alter side of Level-2 pairs
        (renamed with ``alt_`` prefix in the pairs DataFrame).
    """
    # Full-population DataFrame for category anchoring — all rows, before any filtering.
    orig_full_df = pd.DataFrame(original_arr, columns=original_col_names)

    # Detect mode from synthetic column names.
    has_ego_only = any(c.startswith("Ego_") for c in synthetic_col_names)

    results: dict = {}

    if has_ego_only:
        # ----------------------------------------------------------------
        # Level-1: Ego-HH (one row per household, ego's individual attrs)
        # ----------------------------------------------------------------
        orig_l1 = _extract_level1(original_arr, original_col_names, id_cols)
        syn_l1  = _extract_level1(synthetic_arr, synthetic_col_names, id_cols)
        common_l1 = sorted(set(orig_l1.columns) & set(syn_l1.columns))
        common_l1 = [c for c in common_l1 if c != "Ind_RELATIONSHIP"]  # constant for ego

        full_cats_l1 = _full_cat_sets(common_l1, orig_full_df)
        cat_counts_l1 = {
            col: len(full_cats_l1.get(
                col,
                sorted(set(orig_l1[col].dropna().unique()) | set(syn_l1[col].dropna().unique())),
            ))
            for col in common_l1
        }
        srmse_l1 = srmse(orig_l1[common_l1], syn_l1[common_l1], category_sets=full_cats_l1)
        prf_l1   = precision_recall_f1(orig_l1[common_l1], syn_l1[common_l1])

        _print_metric_block(
            label=f"LEVEL-1 (Ego-HH)  |  {label}",
            n_orig=len(orig_l1), n_syn=len(syn_l1),
            row_label="households",
            common_cols=common_l1, cat_counts=cat_counts_l1,
            srmse_res=srmse_l1, prf_res=prf_l1,
            alter_ind_cols=alter_ind_cols,
        )
        results["level1"] = {"srmse": srmse_l1, "prf": prf_l1, "cat_counts": cat_counts_l1}

        # ----------------------------------------------------------------
        # Level-2: Ego-HH-Alter pairs (one row per ego-alter pair)
        # ----------------------------------------------------------------
        orig_l2 = _extract_level2(original_arr, original_col_names, id_cols,
                                   alter_ind_cols=alter_ind_cols)
        syn_l2  = _extract_level2(synthetic_arr, synthetic_col_names, id_cols,
                                   alter_ind_cols=alter_ind_cols)

        if orig_l2.empty or syn_l2.empty:
            print(f"\n  [LEVEL-2 SKIPPED] No ego-alter pairs found (all size-1 households?)\n")
            results["level2"] = None
        else:
            common_l2 = sorted(set(orig_l2.columns) & set(syn_l2.columns))
            common_l2 = [c for c in common_l2 if c != "Ind_RELATIONSHIP"]  # constant for ego; alt_Ind_RELATIONSHIP kept

            # alt_* columns: anchor to alter rows only.
            # Ego's relationship code ("Respondent", verified code 9 in seed505 data) is
            # structurally impossible for alters — ego and alter code sets are disjoint.
            # Using all rows would inflate the alter bin set by 1 spurious zero-zero bin.
            if "Ind_id" in orig_full_df.columns:
                _ind_id      = orig_full_df["Ind_id"].astype(float).astype(int)
                _orig_alt_df = orig_full_df[_ind_id != 1]
            else:
                _orig_alt_df = orig_full_df
            _alt_cols    = [c for c in common_l2 if     c.startswith("alt_")]
            _nonalt_cols = [c for c in common_l2 if not c.startswith("alt_")]
            full_cats_l2 = _full_cat_sets(_nonalt_cols, orig_full_df)
            full_cats_l2.update(_full_cat_sets(_alt_cols, _orig_alt_df))
            cat_counts_l2 = {
                col: len(full_cats_l2.get(
                    col,
                    sorted(set(orig_l2[col].dropna().unique()) | set(syn_l2[col].dropna().unique())),
                ))
                for col in common_l2
            }
            srmse_l2 = srmse(orig_l2[common_l2], syn_l2[common_l2], category_sets=full_cats_l2)
            prf_l2   = precision_recall_f1(orig_l2[common_l2], syn_l2[common_l2])

            _print_metric_block(
                label=f"LEVEL-2 (Ego-HH-Alter pairs)  |  {label}",
                n_orig=len(orig_l2), n_syn=len(syn_l2),
                row_label="pairs",
                common_cols=common_l2, cat_counts=cat_counts_l2,
                srmse_res=srmse_l2, prf_res=prf_l2,
                alter_ind_cols=alter_ind_cols,
            )
            results["level2"] = {"srmse": srmse_l2, "prf": prf_l2, "cat_counts": cat_counts_l2}

            # Rejection-sampling diagnosis: SRMSE restricted to the 4 ego-alter
            # age × employment columns that rejection sampling directly targets.
            # The aggregate bivariate SRMSE dilutes this signal across 78 pairs;
            # this targeted block shows the correction isolated to the 4 relevant
            # pairs (6 bivariate combinations among 4 columns).
            _rej_cols = [c for c in ("Ind_AGE", "Ind_EMPLOY", "alt_Ind_AGE", "alt_Ind_EMPLOY")
                         if c in common_l2]
            if len(_rej_cols) >= 2:
                _cats_rej = {c: full_cats_l2[c] for c in _rej_cols if c in full_cats_l2}
                _srmse_rej = srmse(orig_l2[_rej_cols], syn_l2[_rej_cols], category_sets=_cats_rej)
                _nbins_mar = sum(cat_counts_l2.get(c, 0) for c in _rej_cols)
                _npairs    = len(_rej_cols) * (len(_rej_cols) - 1) // 2
                _nbins_bi  = sum(
                    cat_counts_l2.get(c1, 0) * cat_counts_l2.get(c2, 0)
                    for i, c1 in enumerate(_rej_cols)
                    for c2 in _rej_cols[i + 1:]
                )
                print(f"\n{'-'*74}")
                print(f"  [Rejection diagnosis — ego-alter age × employment]")
                print(f"    Columns : {_rej_cols}")
                print(f"    SRMSE marginal  ({_nbins_mar:3d} bins):            {_srmse_rej['marginal']:>8.4f}")
                print(f"    SRMSE bivariate ({_npairs} pairs, {_nbins_bi:4d} bins):   {_srmse_rej['bivariate']:>8.4f}")
                print(f"{'-'*74}")

    else:
        # ----------------------------------------------------------------
        # HH-IND: all household members, no ego/alter distinction
        # ----------------------------------------------------------------
        orig_hhi = _extract_hh_ind(original_arr, original_col_names, id_cols)
        syn_hhi  = _extract_hh_ind(synthetic_arr, synthetic_col_names, id_cols)
        common_hhi = sorted(set(orig_hhi.columns) & set(syn_hhi.columns))

        full_cats_hhi = _full_cat_sets(common_hhi, orig_full_df)
        cat_counts_hhi = {
            col: len(full_cats_hhi.get(
                col,
                sorted(set(orig_hhi[col].dropna().unique()) | set(syn_hhi[col].dropna().unique())),
            ))
            for col in common_hhi
        }
        srmse_hhi = srmse(orig_hhi[common_hhi], syn_hhi[common_hhi], category_sets=full_cats_hhi)
        prf_hhi   = precision_recall_f1(orig_hhi[common_hhi], syn_hhi[common_hhi])

        _print_metric_block(
            label=f"HH-IND (All members)  |  {label}",
            n_orig=len(orig_hhi), n_syn=len(syn_hhi),
            row_label="individuals",
            common_cols=common_hhi, cat_counts=cat_counts_hhi,
            srmse_res=srmse_hhi, prf_res=prf_hhi,
            alter_ind_cols=alter_ind_cols,
        )
        results["hh_ind"] = {"srmse": srmse_hhi, "prf": prf_hhi, "cat_counts": cat_counts_hhi}

    return results
