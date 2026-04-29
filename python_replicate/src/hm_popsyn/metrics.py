"""SRMSE and Precision/Recall/F1 metrics for synthetic population evaluation.

Adapted from LLM-BN-HH_withoutLocation/evaluator.py for the HLC numpy array format.

Both metrics operate on person-level data (one row per individual).
HH_id and Ind_id columns are dropped before comparison.
Common columns between original and synthetic are used automatically,
so with-PLN (11 attr cols) and without-PLN (10 attr cols) variants are handled.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def srmse(pop_df: pd.DataFrame, gen_df: pd.DataFrame) -> dict[str, float]:
    """Standardized RMSE over marginal and bivariate distributions.

    For K variables the marginal concatenates K distributions (each summing to 1).
    The bivariate concatenates K*(K-1)/2 joint distributions (each summing to 1).
    SRMSE = RMSE / mean(pop_proportions).

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
        all_idx = sorted(set(pop_ct.index) | set(gen_ct.index))
        all_col = sorted(set(pop_ct.columns) | set(gen_ct.columns))
        pop_ct = pop_ct.reindex(index=all_idx, columns=all_col, fill_value=0)
        gen_ct = gen_ct.reindex(index=all_idx, columns=all_col, fill_value=0)
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


def evaluate_synthetic(
    original_arr: np.ndarray,
    synthetic_arr: np.ndarray,
    original_col_names: list[str],
    synthetic_col_names: list[str],
    label: str,
    id_cols: tuple[str, str] = ("HH_id", "Ind_id"),
) -> dict:
    """Compare original and synthetic person-level arrays using SRMSE and P/R/F1.

    Parameters
    ----------
    original_arr : shape (N_orig, n_orig_cols) — the raw input population
    synthetic_arr : shape (N_syn, n_syn_cols) — the generated output
    original_col_names : column names for original_arr (length = n_orig_cols)
    synthetic_col_names : column names for synthetic_arr (length = n_syn_cols)
    label : descriptive name printed in the results header
    id_cols : identifier columns to drop before comparison
    """
    orig_df = (
        pd.DataFrame(original_arr, columns=original_col_names)
        .drop(columns=list(id_cols), errors="ignore")
    )
    syn_df = (
        pd.DataFrame(synthetic_arr, columns=synthetic_col_names)
        .drop(columns=list(id_cols), errors="ignore")
    )

    common_cols = sorted(set(orig_df.columns) & set(syn_df.columns))
    K = len(common_cols)

    # Per-variable category counts (union of original + synthetic values)
    cat_counts = {}
    for col in common_cols:
        all_vals = set(orig_df[col].dropna().unique()) | set(syn_df[col].dropna().unique())
        cat_counts[col] = len(all_vals)

    n_bins_marginal = sum(cat_counts.values())
    n_bins_bivariate = sum(
        cat_counts[c1] * cat_counts[c2]
        for i, c1 in enumerate(common_cols)
        for c2 in common_cols[i + 1:]
    )
    n_pairs = K * (K - 1) // 2

    srmse_res = srmse(orig_df, syn_df)
    prf_res = precision_recall_f1(orig_df, syn_df)

    w = 74
    print(f"\n{'='*w}")
    print(f"  EVALUATION  |  {label}")
    print(f"{'='*w}")
    print(f"  {'Original persons:':<30} {len(orig_df):>10,}")
    print(f"  {'Synthetic persons:':<30} {len(syn_df):>10,}")
    print(f"  {'Variables evaluated (K):':<30} {K:>10}")
    print(f"{'-'*w}")
    print(f"  Variable breakdown (categories in original ∪ synthetic):")
    for col in common_cols:
        print(f"    {col:<28} {cat_counts[col]:>4} categories")
    print(f"{'-'*w}")
    print(f"  SRMSE — Marginal")
    print(f"    {'Total bins (Nb):':<28} {n_bins_marginal:>10,}  (sum of all category counts)")
    print(f"    {'SRMSE (marginal):':<28} {srmse_res['marginal']:>10.4f}  (lower is better)")
    print(f"  SRMSE — Bivariate")
    print(f"    {'Variable pairs K(K-1)/2:':<28} {n_pairs:>10,}")
    print(f"    {'Total bins (Nb):':<28} {n_bins_bivariate:>10,}  (sum of all pair cross-tab sizes)")
    print(f"    {'SRMSE (bivariate):':<28} {srmse_res['bivariate']:>10.4f}  (lower is better)")
    print(f"{'-'*w}")
    print(f"  Precision/Recall/F1  (exact row-tuple matching on {K} variables)")
    print(f"    {'Precision:':<28} {prf_res['precision']:>10.4f}  (synthetic tuples in real data)")
    print(f"    {'Recall:':<28} {prf_res['recall']:>10.4f}  (real tuples reproduced)")
    print(f"    {'F1-score:':<28} {prf_res['f1']:>10.4f}")
    print(f"    {'Unique pop tuples:':<28} {prf_res['unique_pop']:>10,}")
    print(f"    {'Unique syn tuples:':<28} {prf_res['unique_gen']:>10,}")
    print(f"    {'Matching tuple types:':<28} {prf_res['matching_types']:>10,}")
    print(f"{'='*w}\n")

    return {"srmse": srmse_res, "prf": prf_res, "cat_counts": cat_counts,
            "n_bins_marginal": n_bins_marginal, "n_bins_bivariate": n_bins_bivariate}
