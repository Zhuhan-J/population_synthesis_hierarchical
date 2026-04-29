# Hierarchical Mixture Model for Population Synthesis (Python Replication)

This project is organized so you can learn the implementation from the command line before installing as a package.

## 1. Test-First Learning Order
Start with tests. They are designed to move from smallest building blocks to full integration.

1. tests/test_kernels.py
- Purpose: validates MATLAB/MEX parity helpers in src/hm_popsyn/kernels.py.
- What it checks:
	- int_hist: 1-based histogram behavior.
	- sum_eik: grouped aggregation by 1-based mask.
	- mnrnd_new: one-hot categorical row sampling.
	- sumdims: MATLAB-style dimension summation.
- Why first: these kernels are used by EM updates and probability operations.

2. tests/test_em.py
- Purpose: checks that EM training in src/hm_popsyn/em.py runs and returns coherent probabilistic outputs.
- What it checks:
	- output shapes for pi_g, pi_m, wj, x_jik.
	- normalization constraints (probabilities sum to 1).
	- non-empty likelihood trace.
- Why second: this is the core estimation stage used by synthesis.

3. tests/test_synthesis.py
- Purpose: validates synthetic generation in src/hm_popsyn/synthesis_eval.py.
- What it checks:
	- household and individual matrix shapes.
	- class labels are 1-based.
	- individual-group indexing consistency.
- Why third: this confirms generation logic after parameters are available.

4. tests/test_gibbs.py
- Purpose: checks Bayesian alternative inference in src/hm_popsyn/gibbs.py.
- What it checks:
	- posterior mean parameter shapes.
	- trace dimensions.
	- probability normalization.
- Why fourth: Gibbs is a second inference path, separate from EM-first pipeline.

5. tests/test_pipeline.py
- Purpose: end-to-end integration check across io -> em -> synthesis -> rejection.
- What it checks:
	- merged table preparation.
	- EM fitting + synthetic generation pipeline.
	- final indexing and shape validity after rejection stage.
- Why last: this is the closest test to real usage.

## 2. Source Folder Structure and How It Fits Together

src/hm_popsyn/kernels.py
- Low-level numerical helpers translated from MATLAB/MEX behavior.
- Used by EM and probability computations.

src/hm_popsyn/em.py
- Universal-class EM implementation.
- Contains model fitting, multi-restart selection, and BIC-based model comparison.

src/hm_popsyn/gibbs.py
- Universal-class Gibbs sampler (Bayesian inference option).
- Produces posterior mean parameters and latent traces.

src/hm_popsyn/synthesis_eval.py
- Synthetic population generation from fitted parameters.
- Two-person rejection correction utilities and feature extraction.

src/hm_popsyn/evaluation.py
- Evaluation helpers, including Cramer's V and marginal distributions.

src/hm_popsyn/io.py
- Input preparation from merged household-individual table.
- Sort/remap/category handling to match MATLAB coding assumptions.

src/hm_popsyn/pipeline.py
- Orchestrates fit -> generate -> optional rejection in one function.

src/hm_popsyn/runner.py
- Command-line entry point for full workflow execution.
- Reads CSV, prepares data, runs pipeline, writes npz outputs.

src/hm_popsyn/__init__.py
- Public export surface that ties modules together for import convenience.

## 3. Explicit Command-Line Learning Path (No Installation Required)

Run from python_replicate.

Step A: Kernel-only tests

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest tests.test_kernels -v

Step B: EM-only test

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest tests.test_em -v

Step C: Synthesis-only test

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest tests.test_synthesis -v

Step D: Gibbs-only test

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest tests.test_gibbs -v

Step E: Full integration test

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest tests.test_pipeline -v

Step F: Full test suite

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v

## 4. Full Objective Run from Command Line

After tests pass, run the full objective (fit parameters and generate synthetic population):

PYTHONPATH=src ../.venv/bin/python -m hm_popsyn.runner --input data/merged_household_individual.csv --output outputs/synth_result.npz --G 10 --M 10 --n-households 5000 --n-restarts 5 --max-iter 1000 --seed 12345

Output in outputs/synth_result.npz includes:
- fitted parameters: pi_g, pi_m, log_phi_g, log_phi_m
- generated data: household_data, individual_data, individual_group_id
- latent classes: household_class, individual_class
- diagnostics: likelihood, converged

## 5. Theory-to-Code Mapping Summary

- Inference equations (paper Sec. 3.3-4.2) -> src/hm_popsyn/em.py and src/hm_popsyn/gibbs.py
- Generation model (paper Sec. 4.3) -> src/hm_popsyn/synthesis_eval.py: generate_synthetic_population
- Rejection correction (paper Sec. 4.4) -> src/hm_popsyn/synthesis_eval.py: apply_two_person_rejection
- Evaluation (paper Sec. 4.3/4.4 and Eq. 15 usage) -> src/hm_popsyn/evaluation.py

If you want a deeper function-by-function shape trace, start with tests in the above order and inspect called functions in each module.

## 6. Two-Person Household Rejection Sampling

The rejection stage applies acceptance-rejection sampling to two-person households to correct the synthesis toward empirical target joint distributions of member attributes (age difference, sex difference, etc.).

### Paper Equation (Section 4.4, Eq. 16)

For a pair (i, j) in a proposed size-2 household, let:
- y₁ = age-difference bin, y₂ = sex-difference bin
- f₁(y₁, y₂) = synthetic (proposal) distribution from generated sample
- f₂(y₁, y₂) = target (real data) distribution from input data
- M = max_{y₁,y₂} (f₂/f₁) = normalization factor

The acceptance probability is:
```
α(y₁, y₂) = f₂(y₁, y₂) / (M × f₁(y₁, y₂))
```

Each pair is accepted with probability α, rejected with probability 1−α. Non-size-2 households are never modified.

### Implementation: M Capping to Prevent Pathological Rejection

In theory, M = max(f₂/f₁) ensures α ∈ [0, 1] after clipping. However, with finite samples:
- If any (y₁, y₂) bin appears in f₁ with very low frequency (e.g., 1e-15 due to rounding), then M can become astronomically large (tens of trillions)
- This causes **all** α values to collapse to near-zero, leading to 100% rejection and wasting computational effort

**Solution: Cap M at 3.0** (tunable if needed). This:
- Prevents extreme scaling from noise in small bins
- Maintains meaningful rejection (still penalizes frequent synthetic pairs with zero target frequency)
- Yields realistic acceptance rates (~80%–90% rejection for empirical datasets)

### Example: M-Capping in Action

**Without capping:**
```
M_raw = 3.85e+13 → α = 0.000 for all pairs → 100% rejection
```

**With M capped at 3.0:**
```
M_capped = min(M_raw, 3.0) = 3.0 → α ∈ [0, 1] → ~87.8% rejection (realistic)
Pair (5,1): α = 0.5 → 50% pass/fail per random draw → distribution of pair outcomes meaningful
```

### Diagnostics Output

When running synthesis with rejection (e.g., `run_generate_hlc_1000.py`), you will see:

```
[DIAG-REJECT] First-pass 0: REJECTION SAMPLING VERIFICATION
====================================================================================================

1. HOUSEHOLD SIZE DISTRIBUTION (before vs after rejection):
  Size       Before   After  Dropped  Drop%
     1          13      13        0    0.0% (no rejection - should not change)
     2          41       5       36   87.8% ← REJECTION APPLIED
     3          51      51        0    0.0% (no rejection - should not change)
  ...

2. VALIDATION: Only size=2 should be rejected
   ✓ PASS: Only size=2 rejection was applied

3. SIZE=2 PAIR-LEVEL REJECTION ANALYSIS
   Computed M (raw=3.85e+13 → capped=3.0)
   
   Pair Statistics:
      (y1,y2)   f1      f2    Avg α  Accept%
  ( 5, 1)  0.048780  0.057692  0.500  100.0%  ← high acceptance
  ( 2, 1)  0.097561  0.057692  0.500    0.0%  ← random draws
  (11, 2)  0.024390  0.000000  0.000    0.0%  ← no target → always reject
```

**Key columns:**
- **f1**: Frequency of (y₁, y₂) in synthetic sample
- **f2**: Frequency of (y₁, y₂) in target data
- **Avg α**: Average acceptance probability across all pair instances
- **Accept%**: Actual fraction accepted (depends on random draws)

### Tuning the Capping Threshold

If you need different rejection behavior:
- **Increase cap** (e.g., `M = min(M_raw, 5.0)`): softer rejection, more households retained
- **Decrease cap** (e.g., `M = min(M_raw, 1.5)`): stricter rejection, fewer households retained

Edit `src/hm_popsyn/synthesis_eval.py` line 167:
```python
M = min(M_raw, 3.0)  # ← Change 3.0 to your threshold
```

Then re-run tests and pipeline. The diagnostic output will show the updated M and acceptance rates.
