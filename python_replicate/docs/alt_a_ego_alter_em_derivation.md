# Alternative A: Ego-Alter Hierarchical Mixture — EM Derivation

---

## 1. Motivation

The base HLC model (Sun, Earth & Cai 2018, Eq. 16) assumes **exchangeability** — all individuals within a household are treated identically:

$$p(\mathbf{x}_{i1}, \ldots, \mathbf{x}_{in_i} \mid z_i = g) = \prod_{j=1}^{n_i} p(\mathbf{x}_{ij} \mid z_i = g)$$

Every member is drawn i.i.d. from the same conditional distribution. This means:
- `Ind_id=1` (the survey respondent, the *ego*) and `Ind_id≥2` (the *alters*) are treated identically.
- After generation, `Ind_id=1` is meaningless — it is just the first randomly sampled member, not the ego.
- The model cannot reproduce structural patterns like "the survey respondent tends to be a working-age adult" or "the ego's employment status is correlated with their spouse's relationship type."

**What Alternative A changes:** The model learns *separate* latent class distributions for the ego and for alters. Both θ and μ are kept different for ego vs alter:

- $\boldsymbol{\mu}^\text{ego} \neq \boldsymbol{\mu}^\text{alt}$: the probability of belonging to each individual latent class *m* differs between egos and alters, given the same household class *g*.
- $\boldsymbol{\theta}^\text{ego} \neq \boldsymbol{\theta}^\text{alt}$: the conditional feature distributions also differ — within the same individual class *m*, egos and alters may exhibit different observable patterns.

The **missing data framing** is the principled reason this works cleanly in EM, and it also provides a natural path for adding future ego-specific features (e.g., gender, education) without any structural code change.

---

## 2. The Missing Data Framing — What It Means Concretely

### 2.1 Feature sets

Define:

- $\mathcal{K}_\text{alt}$: features observed for **every** individual (both ego and alters). Currently: `{Ind_AGE, Ind_EMPLOY, Ind_RELATIONSHIP}`.
- $\mathcal{K}_\text{ego}$: features observed for the **ego only**. In the future this would include `{ego_gender, ego_edu}` from the HTS survey's PAX table. Currently, $\mathcal{K}_\text{ego} = \mathcal{K}_\text{alt}$ (no ego-extra features yet).
- $\mathcal{K}_\text{ego\,only} = \mathcal{K}_\text{ego} \setminus \mathcal{K}_\text{alt}$: the ego-only features.

### 2.2 Data table view

The raw person-level data, as seen by the model:

| Row | Person | Ind_AGE | Ind_EMPLOY | Ind_REL | ego_gender | ego_edu | Source |
|-----|--------|---------|------------|---------|------------|---------|--------|
| HH1, Ind_id=1 | ego | 3 | 2 | 1 (Respondent) | **F** | **Degree** | *observed* |
| HH1, Ind_id=2 | alter | 4 | 3 | 2 (Spouse) | **?** | **?** | *missing* |
| HH1, Ind_id=3 | alter | 1 | 1 | 3 (Child) | **?** | **?** | *missing* |
| HH2, Ind_id=1 | ego | 5 | 4 | 1 (Respondent) | **M** | **O-level** | *observed* |
| HH2, Ind_id=2 | alter | 2 | 1 | 3 (Child) | **?** | **?** | *missing* |

**Current dataset:** The `processed_HLC_10pct_seed4556.npy` file was built from the HTS HH survey table only — the ego-extra features (gender, edu) were not included. So right now $\mathcal{K}_\text{ego\,only} = \emptyset$. The missing data framing still applies because it explains *why* ego and alter parameters are estimated from separate row sets.

### 2.3 Shared individual latent class space

Both ego and alters are indexed by the **same** $M$ individual latent classes, $m \in \{1, \ldots, M\}$. This is a deliberate architectural choice — not a consequence of the missing data framing.

**What "same class space" means concretely:**

The $M$ classes represent latent *person types* — e.g., class $m=1$ might correspond to "young employed adult", $m=2$ to "school-age child", $m=3$ to "elderly retired". Both ego and alter are drawn from this same vocabulary of $M$ types. What differs between ego and alter is:

1. **How likely each type is** — controlled by $\mu_{gm}^\text{ego}$ vs $\mu_{gm}^\text{alt}$. The survey respondent (ego) is more likely to be drawn from the "working-age adult" class than a random household member (alter) who might often be a child. The same class index $m$ is used, but its probability differs.

2. **What each type looks like** — controlled by $\theta_{c,m}^{\text{ego},(k)}$ vs $\theta_{c,m}^{\text{alt},(k)}$. Within class $m=1$ ("young employed adult"), the ego's age/employment distribution might be slightly different from an alter's, because survey design biases which adult in the household fills the questionnaire.

**Why not use separate class spaces $M_\text{ego} \neq M_\text{alt}$?**

If ego and alter had entirely separate class indices, there would be no shared vocabulary — "ego class $m=2$" and "alter class $m=2$" would mean completely unrelated things. The model would lose the ability to capture patterns like "when the household is class $g=4$ (large family), both the ego AND alters are more likely to be class $m=2$ (working-age adult)." The shared $M$ preserves this coupling: the household class $g$ governs the distribution over the same set of $M$ types for both ego and alter, making the model coherent.

**Visual summary of the shared structure:**

```
Household class g (shared for the whole household)
         │
         ├──► HH attributes:  φ_g            (one set, shared)
         │
         ├──► Ego individual class m  ~  μ^ego_{g,m}   ──► features via θ^ego_{c,m}
         │
         └──► Each alter individual class m  ~  μ^alt_{g,m}  ──► features via θ^alt_{c,m}
```

The *same* $g$ and *same* $m$ index appear in both ego and alter branches — only the distribution tables $\mu$ and $\theta$ differ.

**Consequence for parameter count:**

The Alt A model has exactly double the individual-level parameters of the original HLC:

| Parameter | Original HLC | Alt A |
|---|---|---|
| Individual class weights | $\mu_{gm}$: $(G \times M)$ | $\mu_{gm}^\text{ego}$, $\mu_{gm}^\text{alt}$: $2 \times (G \times M)$ |
| Feature distributions | $\theta_{c,m}^{(k)}$: $(d_k \times M)$ per feature | $\theta^{\text{ego}}$, $\theta^{\text{alt}}$: $2 \times (d_k \times M)$ per feature |
| Household weights | $\phi_{c,g}^{(k)}$, $\lambda_g$: unchanged | same |

The household-level parameters are completely unchanged. Only the individual-level parameters double — and the ego half is estimated from 2,651 rows while the alter half is estimated from 6,918 rows.

### 2.4 Why this is "missing at random"

The ego-extra features are absent from alter rows not because of the alter's own value on those features — a child in the household certainly has a gender and an education level. They are absent because the HTS survey simply did not ask the survey respondent to report those details for each household member. The absence is determined by survey design (which person is the respondent), not by the alter's own characteristics. This is the textbook definition of **Missing At Random (MAR)**.

Under MAR, EM can ignore the missing data in the likelihood without introducing bias: we simply marginalize over the unknown values.

---

## 3. The Likelihood Under the Missing Data View

### 3.1 The key identity: marginalizing missing features gives 1

Suppose individual class $m$ has a valid probability distribution over feature $k$:

$$\sum_{c=1}^{d_k} \theta_{c, m}^{(k)} = 1$$

If an alter row is *missing* feature $k$ (because $k \in \mathcal{K}_\text{ego\,only}$), the likelihood contribution for that feature is the sum over all possible values:

$$\sum_{c=1}^{d_k} \theta_{c, m}^{(k)} = 1$$

**The missing feature's contribution drops out of the likelihood.** This means alter rows contribute to the likelihood using only $\mathcal{K}_\text{alt}$ features, exactly as if those ego-extra features did not exist for alters. No special handling is needed in the code — alter rows simply do not have those feature columns.

### 3.2 Full household likelihood

For household $i$ with one ego and $n_i - 1$ alters:

$$p(\mathbf{x}_i) = \sum_{g=1}^G \lambda_g \underbrace{\prod_{k=1}^H \phi_{x_i^k, g}^{(k)}}_{\substack{\text{household-level} \\ \text{attribute terms}}} \;\cdot\; \underbrace{\left[\sum_{m=1}^M \mu_{gm}^\text{ego} \prod_{k \in \mathcal{K}_\text{ego}} \theta_{x_{i,\text{ego}}^k,\; m}^{\text{ego},(k)}\right]}_{\text{one ego — uses all } K_\text{ego} \text{ features}} \;\cdot\; \prod_{j=2}^{n_i} \underbrace{\left[\sum_{m=1}^M \mu_{gm}^\text{alt} \prod_{k \in \mathcal{K}_\text{alt}} \theta_{x_{ij}^k,\; m}^{\text{alt},(k)}\right]}_{\text{each alter — uses only } K_\text{alt} \text{ features}}$$

**Intuition for each term:**

- $\lambda_g$: how probable is household class $g$?
- $\prod_k \phi_{x_i^k, g}^{(k)}$: given class $g$, how well does household $i$'s dwelling type, income, size, etc. fit?
- $\sum_m \mu_{gm}^\text{ego} \prod_k \theta^{\text{ego}}$: given class $g$, what is the probability of observing this ego's age, employment, relationship, and any extra ego features?
- $\prod_j \sum_m \mu_{gm}^\text{alt} \prod_k \theta^{\text{alt}}$: given class $g$, what is the probability of observing each alter's shared features?

The whole expression is summed over $g$ because we do not directly observe the household class — it is a latent variable that EM will infer.

**Parameters:**

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $\lambda_g$ | $(G,)$ | household class weights |
| $\phi_{c,g}^{(k)}$ | $(d_k, G)$ | HH attribute distribution per class |
| $\mu_{gm}^\text{ego}$ | $(G, M)$ | ego's individual class weights given HH class |
| $\mu_{gm}^\text{alt}$ | $(G, M)$ | alter's individual class weights given HH class |
| $\theta_{c,m}^{\text{ego},(k)}$ | $(d_k, M)$ | ego feature distributions per individual class |
| $\theta_{c,m}^{\text{alt},(k)}$ | $(d_k, M)$ | alter feature distributions per individual class |

Both $\boldsymbol{\theta}^\text{ego}$ and $\boldsymbol{\theta}^\text{alt}$ are estimated separately, even for the shared features $\mathcal{K}_\text{alt}$. This allows the model to learn that, say, ego's employment distribution within class $m=3$ differs from an alter's employment distribution within the same class.


The log-likelihood of the entire dataset is then:
$$\mathcal{L} = \sum_{i=1}^{N_\text{HH}} \log p(\mathbf{x}_i) = \sum_{i=1}^{N_\text{HH}} \log \left[ \sum_{g=1}^G \lambda_g \prod_{k=1}^H \phi_{x_i^k, g}^{(k)} \cdot \left(\sum_{m=1}^M \mu_{gm}^\text{ego} \prod_{k \in \mathcal{K}_\text{ego}} \theta_{x_{i,\text{ego}}^k, m}^{\text{ego},(k)}\right) \cdot \prod_{j=2}^{n_i} \left(\sum_{m=1}^M \mu_{gm}^\text{alt} \prod_{k \in \mathcal{K}_\text{alt}} \theta_{x_{ij}^k, m}^{\text{alt},(k)}\right) \right]$$
---

## 4. E-step: Computing Responsibilities

The E-step answers: *given the current parameter estimates, what is the posterior probability that each individual belongs to each latent class?*

### 4.1 Why ego and alter responsibilities decouple

Because the likelihood factorises into a separate ego term and separate alter terms (see Section 3.2), the posterior responsibilities also factorize. There is no need to compute a joint responsibility across all members of a household — each ego and each alter is handled independently, using its own parameter set.

Importantly, even though ego and alter have **separate** $\mu$ and $\theta$ tables, the individual class index $m$ still runs over the same range $\{1, \ldots, M\}$ for both. The E-step therefore produces two responsibility tensors of the same shape:
- `x_jik_ego`: shape $(N_\text{HH},\, G,\, M)$ — one slice per ego
- `x_jik_alt`: shape $(N_\text{alt},\, G,\, M)$ — one slice per alter

Each entry `x_jik_ego[i, g, m]` answers: *"given the current parameters, what fraction of ego $i$'s identity should be attributed to household class $g$ and individual class $m$?"* The answer uses $\mu^\text{ego}_{gm}$ and $\theta^\text{ego}$ — the ego-specific tables — not the alter tables.

### 4.2 Ego individual class responsibility

For ego $i$ (one per household):

$$r_{i,m}^\text{ego} \;=\; p(z_{i,\text{ego}} = m \mid \mathbf{x}_{i,\text{ego}}, z_i = g) \;\propto\; \mu_{gm}^\text{ego} \prod_{k \in \mathcal{K}_\text{ego}} \theta_{x_{i,\text{ego}}^k,\; m}^{\text{ego},(k)}$$

In practice, the ego responsibilities are computed in log-space for numerical stability:

```
# f_ego[i, m] = log-probability of ego i's features under individual class m
f_ego[i, m] = Σ_{k ∈ K_ego}  log_phi_m_ego[k][ x_ego[i, k] - 1,  m ]

# h_ego[i, g, m] = unnormalized responsibility, incorporating class prior μ^ego
h_ego[i, g, m] = exp(f_ego[i, m]) × μ^ego[g, m]

# x_jik_ego[i, g, m] = normalized responsibility (posterior probability)
x_jik_ego[i, g, m] = h_ego[i, g, m] / Σ_{m'} h_ego[i, g, m']
```

The denominator normalizes over all $M$ individual classes, so that $\sum_m x_{jik\_ego}[i, g, m] = 1$ for each $(i, g)$.

### 4.3 Alter individual class responsibility

For alter row $j$ (belonging to some household $i_j$, using only $\mathcal{K}_\text{alt}$ features):

$$r_{j,m}^\text{alt} = p(z_{j,\text{alt}} = m \mid \mathbf{x}_{j,\text{alt}}, z_{i_j} = g) = \;\propto\; \mu_{gm}^\text{alt} \prod_{k \in \mathcal{K}_\text{alt}} \theta_{x_{ij}^k,\; m}^{\text{alt},(k)}$$

Notice: the ego-extra features $k \in \mathcal{K}_\text{ego\,only}$ do **not** appear. As derived in Section 3.1, those features marginalize to 1 and drop out. Concretely:

```
# f_alt[j, m] = log-probability of alter j's features under individual class m
#               only K_alt features are used; ego-extra features are absent
f_alt[j, m] = Σ_{k ∈ K_alt}  log_phi_m_alt[k][ x_alt[j, k] - 1,  m ]

# Same normalization as ego
h_alt[j, g, m] = exp(f_alt[j, m]) × μ^alt[g, m]
x_jik_alt[j, g, m] = h_alt[j, g, m] / Σ_{m'} h_alt[j, g, m']
```

**The crucial point:** because the alter loop uses `log_phi_m_alt` (not `log_phi_m_ego`), and because alter rows do not carry ego-extra feature columns, alter responsibilities never influence the ego parameters — and vice versa. The two sets of parameters are estimated from completely separate row sets.

### 4.4 Household class responsibility

Having computed individual responsibilities, the E-step then determines which household class $g$ each household most likely belongs to. The household responsibility $w_{ig} = p(z_i = g \mid \mathbf{x}_i)$ integrates information from:
1. The household attributes (dwelling type, income, size, …)
2. The ego's marginal likelihood under class $g$
3. Every alter's marginal likelihood under class $g$

**Step 1** — compute the *marginal* log-probability of each ego given $g$ (summing out the individual class $m$):

$$\ell^\text{ego}_{i,g} \;=\; \log \sum_{m=1}^M h_\text{ego}[i, g, m]$$

```
log_g_ego[i, g] = log( Σ_m  h_ego[i, g, m] )
```

**Step 2** — compute the *marginal* log-probability of each alter given $g$:

$$\ell^\text{alt}_{j,g} \;=\; \log \sum_{m=1}^M h_\text{alt}[j, g, m]$$

```
log_g_alt[j, g] = log( Σ_m  h_alt[j, g, m] )
```

**Step 3** — accumulate all individual contributions into one household-level sum `tt0[i, g]`:

$$\texttt{tt0}[i, g] \;=\; \ell^\text{ego}_{i,g} \;+\; \sum_{j \in \text{alters of HH } i} \ell^\text{alt}_{j,g}$$

```
# One ego per household: direct assignment
tt0[i, g] = log_g_ego[i, g]

# Each alter j adds its log-marginal to the household it belongs to
for each alter j with household id hh_j:
    tt0[hh_j, g] += log_g_alt[j, g]
```

Note that for **size-1 households** (ego only, no alters), the sum over alters is empty, so `tt0[i, g] = log_g_ego[i, g]` alone. These households still contribute to fitting the ego parameters, but contribute nothing to the alter parameters.

**Step 4** — combine household attribute log-probabilities and the prior:

$$\texttt{temp}[i, g] \;=\; \log \lambda_g + \sum_{k=1}^H \log \phi_{x_i^k, g}^{(k)} + \texttt{tt0}[i, g]$$

$$w_{ig} \;=\; \propto \lambda_g \prod_{k=1}^H \phi_{x_i^k, g}^{(k)} \cdot [\sum_{m=1}^M \mu_{gm}^\text{ego} \prod_{k \in \mathcal{K}_\text{ego}} \theta_{x_i^k, m}^{\text{ego},(k)}] \cdot \prod_{j=2}^{n_i}[\sum_{m=1}^M \mu_{gm}^\text{alt} \prod_{k \in \mathcal{K}_\text{alt}} \theta_{x_{ij}^k, m}^{\text{alt},(k)}]$$

<!-- $$w_{ig} = \frac{\exp(\texttt{temp}[i,g] - \max_g \texttt{temp}[i,\cdot])}{\sum_{g'} \exp(\texttt{temp}[i,g'] - \max_g \texttt{temp}[i,\cdot])}$$ -->

The subtraction of the row maximum is a standard log-sum-exp stabilisation trick to prevent numerical overflow.

---

## 5. M-step: Updating Parameters

The M-step maximises the expected complete-data log-likelihood with respect to each parameter, treating the responsibilities computed in the E-step as fixed weights.

### 5.1 Intuition

Think of each responsibility $x_{jik\_ego}[i, g, m]$ as a *fractional count*: individual $i$ (the ego of household $i$) is assigned fraction $x_{jik\_ego}[i, g, m]$ to household class $g$ and individual class $m$. Similarly for alters. The M-step then updates each parameter by recomputing a weighted average over these fractional counts.

**The separation of ego and alter in the M-step is exactly where the "missing data" framing becomes concrete:**

- Parameters involving ego-only features ($k \in \mathcal{K}_\text{ego\,only}$) are updated exclusively from ego fractional counts — alter rows have no column for these features, so they contribute zero to those parameters.
- Parameters for shared features ($k \in \mathcal{K}_\text{alt}$) are updated from ego and alter separately, producing different $\theta^\text{ego}$ and $\theta^\text{alt}$ estimates.

Specifically, the loglikelihood is:
$$\mathcal{L} = \sum_{i=1}^{N_\text{HH}} \log \left[ \sum_{g=1}^G \lambda_g \prod_{k=1}^H \phi_{x_i^k, g}^{(k)} \cdot \left(\sum_{m=1}^M \mu_{gm}^\text{ego} \prod_{k \in \mathcal{K}_\text{ego}} \theta_{x_{i,\text{ego}}^k, m}^{\text{ego},(k)}\right) \cdot \prod_{j=2}^{n_i} \left(\sum_{m=1}^M \mu_{gm}^\text{alt} \prod_{k \in \mathcal{K}_\text{alt}} \theta_{x_{ij}^k, m}^{\text{alt},(k)}\right) \right]$$

### 5.2 Joint responsibility weights

First, combine individual and household responsibilities into a single joint weight:

**For ego row $i$:**
$$\texttt{second\_ego}[i, g, m] \;=\; x_{jik\_ego}[i, g, m] \;\times\; w_{ig}$$

In words: "ego $i$ contributes fractional count $\texttt{second\_ego}[i,g,m]$ to the joint cell $(g, m)$ for ego parameters."

**For alter row $j$ (from household $h_j$):**
$$\texttt{second\_alt}[j, g, m] \;=\; x_{jik\_alt}[j, g, m] \;\times\; w_{h_j, g}$$

In words: "alter $j$ contributes fractional count $\texttt{second\_alt}[j,g,m]$ to the joint cell $(g, m)$ for alter parameters."

Note that alter rows use the household responsibility $w_{h_j, g}$ of *their own household* $h_j$, not the ego's responsibility. This correctly propagates household-level uncertainty to alter parameters.

### 5.3 Household class weights $\lambda_g$

$$\hat\lambda_g \;\propto\; \sum_{i=1}^{N_\text{HH}} w_{ig}$$

Unchanged from the original model — household class weights are always estimated from household-level responsibilities.

### 5.4 Household attribute distributions $\phi_{c,g}^{(k)}$

$$\hat\phi_{c,g}^{(k)} \;\propto\; \sum_{i=1}^{N_\text{HH}} w_{ig} \cdot \mathbf{1}[x_i^k = c]$$

Also unchanged. Household attribute parameters are never touched by individual responsibilities.

### 5.5 Ego individual class weights $\mu_{gm}^\text{ego}$

Marginalise `second_ego` over the individual dimension to get the total fractional count assigned to cell $(g, m)$:

$$\hat\mu_{gm}^\text{ego} \;\propto\; \sum_{i=1}^{N_\text{HH}} \texttt{second\_ego}[i, g, m]$$

Only ego rows appear in this sum. Alter rows do not contribute to $\mu^\text{ego}$.

### 5.6 Alter individual class weights $\mu_{gm}^\text{alt}$

$$\hat\mu_{gm}^\text{alt} \;\propto\; \sum_{j=1}^{N_\text{alt}} \texttt{second\_alt}[j, g, m]$$

Only alter rows appear. Size-1 households (no alters) contribute nothing to $\mu^\text{alt}$.

### 5.7 Ego feature distributions $\theta_{c,m}^{\text{ego},(k)}$

First collapse the joint weights over the household dimension:

$$\texttt{second2\_ego}[i, m] \;=\; \sum_{g=1}^G \texttt{second\_ego}[i, g, m]$$

Then the feature update is a weighted frequency count:

$$\hat\theta_{c,m}^{\text{ego},(k)} \;\propto\; \sum_{i=1}^{N_\text{HH}} \texttt{second2\_ego}[i, m] \cdot \mathbf{1}[x_{i,\text{ego}}^k = c]$$

**For shared features** ($k \in \mathcal{K}_\text{alt}$): this sum runs over 2,651 ego rows, producing $\theta^\text{ego}$ estimated from ego data only.

**For ego-only features** ($k \in \mathcal{K}_\text{ego\,only}$, future extension): the same formula applies — only ego rows have this feature column, so they automatically receive all the weight. Alter rows have no column for this feature and contribute nothing. **This is the missing data property in action: no special code is needed for ego-extra features — they just have fewer rows contributing to the sum.**

### 5.8 Alter feature distributions $\theta_{c,m}^{\text{alt},(k)}$

$$\texttt{second2\_alt}[j, m] \;=\; \sum_{g=1}^G \texttt{second\_alt}[j, g, m]$$

$$\hat\theta_{c,m}^{\text{alt},(k)} \;\propto\; \sum_{j=1}^{N_\text{alt}} \texttt{second2\_alt}[j, m] \cdot \mathbf{1}[x_{ij}^k = c], \quad k \in \mathcal{K}_\text{alt}$$

Estimated from 6,918 alter rows. Since alters never have ego-only features, $\theta^\text{alt}$ is only defined over $\mathcal{K}_\text{alt}$.

---

## 6. Why the Missing-Data Framing Is More Than a Framing

It might seem like Alt A is simply "two separate models glued together." The missing-data framing reveals why it is genuinely one joint model:

1. **Shared household class.** Both the ego and all alters in household $i$ share the *same* household class $z_i = g$. Their individual likelihoods are multiplied together in the household likelihood (Section 3.2), so information from the alter rows informs the household class posterior $w_{ig}$, which in turn flows back into the ego parameters in the M-step (Sections 5.5, 5.7). The ego and alter estimations are **jointly optimised**, not run independently.

2. **Natural extension path.** When ego-extra features are added to ego rows in the future, the only change is passing a wider `ego_inddata` matrix to `em_ego_alter_universal`. The M-step already computes `second2_ego` from ego rows only — those new feature columns are automatically learned from ego data, and alters' missing values for those features continue to marginalize to 1 without any code change.

3. **Contrast with a "two separate models" approach.** An alternative would be to run one EM on egos and another on alters independently. But then the household class estimated in Model 1 (from egos only) would not be informed by alter data, and household coherence between ego and alter would be lost.

---

## 7. Overfitting and Regularization

### 7.1 The sparse-cell problem

Ego parameters $\theta_{c,m}^{\text{ego},(k)}$ are estimated from 2,651 ego rows (vs 9,569 in the original pooled model). With $G=5$, $M=8$ classes, each $(g, m)$ cell draws on average:

$$\frac{2{,}651}{5 \times 8} \approx 66 \text{ ego observations per cell (soft-weighted)}$$

For a rare age category that appears in only ~40 egos total, the expected soft-count in a specific $(g, m)$ cell is approximately $40/40 = 1$ ego. This is very small. Without regularization:
- The MLE estimate for that category in that cell could be exactly 0.
- The model would then **never generate** an ego of that age in that class.
- This is a form of overfitting: the model memorises that a particular combination was absent in the training sample, rather than learning that it is merely rare.

### 7.2 Dirichlet regularization

The `concentration_ego` parameter adds a Dirichlet prior: each category in each cell starts with `concentration_ego` pseudo-observations. This ensures no category has probability zero.

**Why `concentration_ego > concentration_alt`:**

| Parameter set | Observations per cell | Risk of zero counts |
|---|---|---|
| $\theta^\text{ego}$ | ~66 | High |
| $\theta^\text{alt}$ | ~173 | Low |

Setting `concentration_ego = 1.5` (vs the default `concentration_alt = 1.0`) adds proportionally stronger smoothing to the ego parameters, compensating for the smaller ego dataset.

---

## 8. Connection to the Current Code

### 8.1 Mapping from derivation to `em.py` lines (original model)

| Derivation step | `em.py` lines | Description |
|---|---|---|
| Individual log-prob $f_\text{ind}$ | 194–196 | `f_ind += log_phi_m[att][...]` for all N individuals |
| Responsibilities $x_{jik}$ | 198–200 | `h = exp(f_ind) * pi_m; x_jik = h / sum(h)` |
| Individual marginal $\ell_{ji}$ | 201 | `log_g_ji = log(sum_m h)` |
| Accumulate `tt0` | 203–205 | `np.add.at(tt0[:,g], gid0, log_g_ji[:,g])` — all N individuals |
| Household log-prob `temp` | 207–211 | HH attrs + tt0 + log(pi_g) |
| Household responsibilities $w_{ig}$ | 212–214 | softmax over $g$ |
| Update $\pi_m$ | 216–219 | `second = x_jik * wj[gid0]; pi_m = sum(second)` |
| Update $\phi_g$ | 225–228 | `sum_eik(wj, ...)` |
| Update $\phi_m$ | 230–234 | `second2 = sum_g(second); sum_eik(second2, ...)` |

### 8.2 The delta for `em_ego_alter.py`

Every step above is replicated twice — once for ego rows, once for alter rows — with separate parameter arrays:

| Original | Alt A ego | Alt A alter |
|---|---|---|
| `f_ind` (N rows) | `f_ego` (N_HH rows, K_ego features) | `f_alt` (N_alt rows, K_alt features) |
| `x_jik` (N,G,M) | `x_jik_ego` (N_HH,G,M) | `x_jik_alt` (N_alt,G,M) |
| single `tt0` accumulation | `tt0 += log_g_ego` (one per HH) | `tt0 += log_g_alt` (one per alter) |
| `pi_m` (G,M) | `pi_m_ego` from `second_ego` | `pi_m_alt` from `second_alt` |
| `log_phi_m[k]` | `log_phi_m_ego[k]` from ego soft-counts | `log_phi_m_alt[k]` from alter soft-counts |

`log_phi_g`, `pi_g`, and the household responsibility `wj` computation are unchanged — they operate at the household level and are unaffected by the ego/alter split.
