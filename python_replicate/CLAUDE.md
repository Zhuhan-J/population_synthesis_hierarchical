# Project Context: HLC Population Synthesis

## Purpose

Python replication and extension of the Hierarchical Latent Class (HLC) population synthesis
model (Sun, Earth & Cai 2018). The key extension adds ego-alter social network attributes —
separate EM distributions for ego (survey respondent) vs alters (other household members).

---

## Two Data Variants

| Variant | Columns | Ego-only attrs | Run script |
|---------|---------|---------------|------------|
| `HH_IND` (11-col) | HH attrs + shared ind attrs | No | `run_generate_hlc_10pct.py` |
| `HH_IND_EGO` (13-col) | HH attrs + shared ind attrs + `Ego_GENDER`, `Ego_EDU` | Yes | `run_generate_hlc_10pct_ego_alter.py` |

Individual attribute columns in prepared data (0-based after `prepare_from_merged_table`):
- `0` = `Ind_AGE` (18 categories, natural age-bin order)
- `1` = `Ind_EMPLOY` (11 categories, alphabetical)
- `2` = `Ind_RELATIONSHIP` (12 categories) — **not modelled for ego** (always "Respondent")

Codebook constants live in `src/hm_popsyn/synthesis_eval.py`. Two sets exist:
- Default: seed505 (canonical, natural-order age, alphabetical employment)
- Legacy `_*_LEGACY`: seed4556 (random encoding — pass explicitly when loading those files)

---

## Source Layout

```
src/hm_popsyn/
  em.py                  # EM for standard (non-ego-alter) model
  em_ego_alter.py        # EM with separate ego / alter distributions
  synthesis_eval.py      # Generation + rejection sampling; codebook constants
  pipeline.py            # fit → generate → rejection (standard)
  pipeline_ego_alter.py  # fit → generate → rejection (ego-alter)
  metrics.py             # SRMSE + P/R/F1 evaluation
  io.py                  # prepare_from_merged_table (min-shifts to 1-based)

docs/
  rejection_multi_size.md   # Full design rationale for rejection sampling
  
LLM-BN-HH_withoutLocation/SRMSE_Example.md   # SRMSE formula + bin-count effect (§4)
```

---

## Key Design Decisions

### Rejection sampling features
`y1 = round(mean |age_ordinal(ego) − age_ordinal(alter)|)` coarsened to 5 bins  
`y2 = round(mean |employ_group(ego) − employ_group(alter)|)` (6 groups)  
Feature space: 5 × 6 = 30 cells. M capped at 3.0. Sparsity guard: skip if < 30 training
households of that size. See `docs/rejection_multi_size.md` for full rationale.

### Ego RELATIONSHIP removal
`Ind_RELATIONSHIP` is always "Respondent" for egos. It is dropped from `ego_inddata`
before EM and generation. The fixed code is passed as `ego_rel_code` and reinserted at
output column 2 so the file format is preserved. Only applies to the ego-alter pipeline.

### Evaluation modes (metrics.py)
`evaluate_synthetic()` auto-detects mode from `synthetic_col_names`:
- Any `Ego_*` column → **ego-only mode**: Level-1 (Ego-HH) + Level-2 (Ego-HH-Alter pairs)
- No `Ego_*` columns → **HH-IND mode**: flat evaluation over all household members

SRMSE bins are anchored to the full-population target (all rows) — not just observed
categories — for consistent comparability across runs. `Ind_RELATIONSHIP` is excluded
from Level-1 and Level-2 metrics in ego-only mode (constant, zero signal).

A **rejection diagnosis block** prints after Level-2, computing SRMSE restricted to the
4 ego-alter age × employment columns that rejection sampling directly targets.

---

## Common Commands

```bash
# Full ego-alter run (13-col HH_IND_EGO, BIC model selection)
python run_generate_hlc_10pct_ego_alter.py \
  --target-households 22524 --n-restarts 3 --max-iter 2000 --seed 255

# Quick smoke test (disable BIC search, small N)
python run_generate_hlc_10pct_ego_alter.py \
  --disable-gm-search --G 4 --M 8 \
  --target-households 200 --n-restarts 1 --max-iter 50 --seed 42

# Standard (non-ego-alter) run with seed4556 data
python run_generate_hlc_10pct.py \
  --disable-gm-search --G 5 --M 8 \
  --target-households 500 --n-restarts 1 --max-iter 50 --seed 42
```

---

## Ongoing Work / Gotchas

- The training source (`10pct`) and full-population evaluation target (`hPOP`) must use the
  **same integer encoding** — built in `hierarchical_LC_sampleV1_data_generate.ipynb` from
  the full population and applied to both files.
- Rejection sampling references the **training data** (not the full population) as its
  composition target. This is a known limitation: small differences between 10% sample and
  full population can prevent rejection from clearly improving full-population SRMSE.
- Aggregate Level-2 SRMSE dilutes rejection sampling's signal across 78 pairs; use the
  **rejection diagnosis block** (4-column targeted SRMSE) to properly assess the correction.
