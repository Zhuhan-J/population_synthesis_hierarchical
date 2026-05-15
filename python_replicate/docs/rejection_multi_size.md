# Rejection Sampling for Multi-Size Households

## 1. Paper Baseline (Sun, Earth & Cai 2018, Section 4.4)

The HLC model assumes **conditional independence** of household members given the household
latent class (Eq. 16):

```
p(x_i1, ..., x_ini | z_i) = ∏_j p(x_ij | z_i)
```

This ignores inter-person correlations within households. As a post-processing correction,
the paper applies **rejection sampling** to two-person households.

Each synthetic two-person household is represented as a 2D point:
- **y1** = |age_A − age_B|   (absolute age difference, age in 14 bins)
- **y2** = |sex_A − sex_B|   (binary: 0 = same sex, 1 = different sex)

The acceptance ratio is:

```
α_i = f_target(y1, y2) / (M × f_proposal(y1, y2))
```

where:
- `f_proposal` = empirical joint distribution from synthetic population
- `f_target`   = empirical joint distribution from real HTS data
- `M` = max over all (y1, y2) of f_target / f_proposal ≈ 2.97 in the paper's case study

A uniform random u ~ U[0,1] is drawn; the household is **accepted** if u ≤ α_i.

**Paper limitation:** Only applied to 2-person households.

---

## 2. Extension to Multi-Size Households

This codebase extends the paper's approach to **all household sizes ≥ 2**, using a
per-size strategy:

- For each household size s ∈ {2, 3, 4, ...}, build **separate** empirical distributions
  f_proposal(s) and f_target(s) from synthetic and training households of that size.
- Apply rejection sampling independently to each size group.
- Size-1 households are always accepted (no inter-person correlation to correct).

Per-size distributions are the right approach because the joint composition distributions
differ substantially across sizes: a 4-person household has a fundamentally different
age-composition structure than a 2-person household.

---

## 3. Feature y2: Sex → Employment Group

The paper uses `y2 = |sex_A − sex_B|` (binary) because same-sex vs. opposite-sex couple
composition is the dominant correction signal in Singapore travel survey households.

This implementation uses **employment group difference** instead, which captures social
homophily more relevant to the ego-alter network context:

| New code | Employment Type          | Group | Old code (seed4556) |
|----------|--------------------------|-------|---------------------|
| 1        | Employed Full-Time       | 0     | 1                   |
| 2        | Employed Part-Time       | 0     | 6                   |
| 9        | Self Employed            | 0     | 4                   |
| 6        | National Service         | 0     | 8                   |
| 3        | Full Time Student        | 1     | 5                   |
| 4        | Homemaker                | 2     | 2                   |
| 8        | Retired                  | 3     | 3                   |
| 7        | Non-schooling Child      | 4     | 7                   |
| 10       | Unemployed               | 5     | 9                   |
| 5        | Long-term Medical Leave  | 5     | 10                  |
| 11       | Voluntary Worker         | 5     | 11                  |

The 6 semantic groups reduce noise while preserving the meaningful distinctions
(employed / student / homemaker / retired / child / not-working).

All recoding is for rejection computation only. The original Ind_EMPLOY codes are
unchanged in the final synthetic output.

---

## 4. Aggregation: MAX → MEAN for Size ≥ 3

### The MAX problem

For a household with one ego and multiple alters, the earlier implementation used:

```
y1 = max over all alters( |age_ordinal(ego) − age_ordinal(alter)| )
```

This collapses structurally distinct households to identical feature bins:

**Household A:** ego=35yr, alters=[37yr, 60yr]
- Per-alter age diffs (ordinal): [2, 25]
- MAX = **25** → bin 26

**Household B:** ego=35yr, alters=[10yr, 12yr]
- Per-alter age diffs (ordinal): [25, 23]
- MAX = **25** → bin 26

Both map to the **same bin** despite completely different structures
(adult couple + elderly parent vs. parent + two young children).
Rejection cannot distinguish them.

### MEAN as replacement

Using MEAN (rounded to nearest integer bin):

```
y1 = round( mean over all alters( |age_ordinal(ego) − age_ordinal(alter)| ) )
```

Revisiting the examples:

**Household A:** ego=35yr, alters=[37yr, 60yr]
- Diffs: [2, 25] → MEAN = **13.5** → rounded → 14

**Household B:** ego=35yr, alters=[10yr, 12yr]
- Diffs: [25, 23] → MEAN = **24.0** → rounded → 24

Now distinct. ✓

### For size = 2 (paper-equivalent case)

When there is exactly one alter, MEAN == MAX. The paper's behaviour is exactly preserved
for two-person households — MEAN introduces no change there.

---

## 5. Age Bin Coarsening (age_diff_n_bins=5)

### Problem with raw ordinal differences

With 18 age groups, ordinal differences range 0–17 (e.g. diff=17 means '0-5' years vs
'85+' years). Employment diffs range 0–5 (6 groups). The raw feature space is 18×6=108 cells.

With 48–87 synthetic households per size in a typical first pass, the average occupancy
is **0.44–0.81 cells per household** — the vast majority of cells are empty. Two problems
follow:

1. M_raw becomes inflated: any cell with target mass but zero proposal mass produces a
   ratio of f_target / ε ≈ 1e13, dominating M even after limiting to both-positive cells.
2. f1_zero_bins (cells with target > 0 but proposal = 0) is large: 60–70 out of 108,
   meaning households landing in those cells always get α=1 (accepted regardless).

### Coarser bins reduce sparsity

Mapping 18 ordinal positions → 5 coarser bins:

| Age diff (ordinal) | Coarse bin | Approximate years gap |
|--------------------|------------|-----------------------|
| 0 – 3              | 1          | 0 – 15 years          |
| 4 – 7              | 2          | 20 – 35 years         |
| 8 – 10             | 3          | 40 – 50 years         |
| 11 – 14            | 4          | 55 – 70 years         |
| 15 – 17            | 5          | 75+ years             |

Formula: `bin = min(diff × 5 // 18 + 1, 5)` where diff ∈ [0, 17].

Feature space: **5 × 6 = 30 cells** instead of 108.
Average occupancy with 48 households: **1.6 per cell** — much better populated.
Typical f1_zero_bins: **3 out of 30** instead of 68 out of 108.

The parameter `age_diff_n_bins` in `apply_rejection_by_size` controls this. Set to 18
to disable coarsening (original full-resolution behaviour).

### Effect on M_raw

With fewer empty cells, the both-positive mask covers most of the feature space.
M_raw reflects actual distributional differences rather than estimation noise.
For size-2 with ~48 synthetic and ~500 training households, M_raw typically falls in
the range 1–6, with the cap at 3.0 engaging only when genuine over/under-representation
is more than 3×.

---

## 6. M Cap at 3.0

```python
M = min(M_raw, 3.0)  where  M_raw = max(target_prob / proposal_prob)
                                      over cells where both are > 0
```

The paper's case study reports M_raw ≈ 2.97, so 3.0 is a natural round-up that matches
the empirical value. Capping M prevents extreme acceptance variance: a very high M_raw
(e.g. 15) would make most households reject, requiring many top-up batches, while adding
little additional benefit compared to M=3.

The cap means: for cells where target/proposal > 3, we only correct up to 3× (those cells
still get α=1, so their households are always accepted). We do NOT fully eliminate the
over-representation; we reduce it by at most 3×.

For sparse distributions (small first-pass batch), M_raw > 3 is common and expected.
This is why the cap is a design choice rather than an error.

---

## 7. Sparsity Guard (min_target_samples)

For rare household sizes (6, 7, 8+), the training data may have very few examples.
With e.g. 10 training size-7 households, the 2D target distribution (even with 5×6=30 cells)
has only 10 observations spread over 30 bins. The distribution is too noisy to be useful.

**Guard:** If a household size has fewer than `min_target_samples` (default = 30) training
examples, the rejection step is skipped — all households of that size are accepted.

Before the per-size loop, a summary is printed:
```
[INFO-First-pass 0] Sparsity guard (< 30 target examples): sizes [8, 9] — rejection bypassed, all accepted.
```

---

## 8. Oversample-then-Redraw Loop (Top-up Logic)

### The problem

Rejection removes some synthetic households. To guarantee exactly N households in the
final output, we cannot just generate N and apply rejection (we'd end up short).

### Phase 1 — First-pass oversample

Generate `N × initial_oversample` households (default 1.5×) so that after rejection,
enough remain per size to fill the target quota.

**Target counts** (how many per size the final output needs) are computed from the
first-pass size distribution, scaled to N:

```python
raw_counts  = {2: 90, 3: 60, 4: 40, 5: 25, 6: 8, 7: 2}   # from first pass of 225 households
target_counts = scale(raw_counts, N=150)
            = {2: 60, 3: 40, 4: 27, 5: 17, 6: 5, 7: 1}   # proportional rescaling
```

Apply rejection → get some subset of households per size.

**Example after first-pass rejection** (assuming 60% acceptance on average):
```
accepted = {2: 54, 3: 38, 4: 27, 5: 17, 6: 5, 7: 0}
need     = {2:  6, 3:  2, 4:  0, 5:  0, 6: 0, 7: 1}   # deficit per size
remaining = 6 + 2 + 0 + 0 + 0 + 1 = 9 households short
```

### Phase 2 — Top-up batches

The loop generates fresh batches until all size deficits are filled:

```python
while remaining > 0:
    batch_n = max(ceil(remaining × topup_factor), 200)   # topup_factor=1.7, min batch 200
    batch_raw = generate_synthetic_population(batch_n)
    batch_kept = apply_rejection_by_size(batch_raw, ...)
    take from batch_kept by size to fill remaining deficits
    update remaining
```

**Example iteration 1** (9 remaining, generates 200 fresh households):
- 200 new households generated
- Rejection applied → 120 pass (60% acceptance)
- Take what's needed: +6 size-2, +2 size-3, +1 size-7 (if available)
- Remaining = 0 → done

If iteration 1 doesn't fully fill all sizes (e.g. no size-7 households generated in this batch), the loop continues with another batch until either all sizes are filled or `max_topup_iters` (default 10) is reached.

### Why generate all sizes during top-up?

A top-up batch generates households of ALL sizes according to the model's distribution,
not just the missing sizes. This is correct because:
1. Generating conditionally on a specific household size would require conditioning the
   EM model's household-level generation, which is not straightforward.
2. The minimum batch size of 200 ensures there's a reasonable chance of producing rare sizes.
3. The `topup_factor=1.7` oversamples relative to the remaining deficit to account for
   rejection losses within each batch.

### Size distribution stability

The `target_counts` are derived from the model's first random draw, not from the training
data's empirical size distribution. This introduces small sampling variance: if the first
draw generates 38% size-2 instead of the model's true expected 35%, the entire output
locks in at 38%. With N=20,000, this variance is typically ±0.5% per size — negligible in
practice. If exact reproducibility of size distribution is needed, the target_counts can
be set from the training data's empirical distribution instead.

---

## 9. Legacy Codebook Constants

Two codebook encodings exist because data was re-exported with a unified natural-order
scheme (seed505) after the original random-order encoding (seed4556):

| Constant | Codebook | Ind_AGE '0-5' | Ind_EMPLOY 'Homemaker' |
|----------|----------|---------------|------------------------|
| `AGE_CODE_TO_ORDER` | seed505 (natural) | code 1 → ordinal 0 | — |
| `_AGE_CODE_TO_ORDER_LEGACY` | seed4556 (random) | code 12 → ordinal 0 | — |
| `EMPLOY_CODE_TO_GROUP` | seed505 (alphabetical) | — | code 4 → group 2 |
| `_EMPLOY_CODE_TO_GROUP_LEGACY` | seed4556 (original) | — | code 2 → group 2 |

Pass the legacy constants from `run_generate_hlc_10pct.py` (seed4556 data):

```python
from hm_popsyn.synthesis_eval import _AGE_CODE_TO_ORDER_LEGACY, _EMPLOY_CODE_TO_GROUP_LEGACY

fit_and_generate(
    ...,
    employ_col=1,
    age_code_to_order=_AGE_CODE_TO_ORDER_LEGACY,
    employ_code_to_group=_EMPLOY_CODE_TO_GROUP_LEGACY,
)
```

For `run_generate_hlc_10pct_ego_alter.py` (seed505 data), the defaults apply — no custom
constants needed.
