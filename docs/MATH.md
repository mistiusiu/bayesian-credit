# Mathematics → code map

Source of truth: proposal §3.3 (PDF pp. 36–40, printed pp. 26–30) as transcribed in
[SPEC.md](SPEC.md) §2. Paths are relative to `core/bayesian_credit/`. Notation: `g` = borrower
group (segmenter key), `a`/`k` = arm (lending policy) index, `d`/`n` = observed defaults /
non-defaults, `C, D, R, L` = Current, Delinquent, Recovered, Absorbed Loss.

## 1. Proposal §3.3, equation by equation

| # | Equation (proposal page) | Implemented by | Tested in |
|---|---|---|---|
| 1 | `A_i = (L_i, r_i)` lending policy (p. 26) | `data/arms.py::LendingPolicy(amount, rate, term)` (term from Table 7) | — (contract) |
| 2 | `p_{g,a} = P(D = 1 \| X_g, A_a)` (p. 26) | `models/hierarchical.py::HierarchicalBetaBernoulli.posterior(g, a)`, `.p_default(g, a)`; `models/two_stage.py::TwoStageBetaBeta.risk(g, a).p_default` | `test_hierarchical.py` |
| 3 | `p_0 ~ Beta(α_0, β_0)` (p. 26) | `HierarchicalBetaBernoulli.global_prior` (`HierarchyConfig.alpha0/beta0`); posterior `.global_posterior()` = `Beta(α_0 + D, β_0 + N − D)` | `test_priors_are_built_top_down…` |
| 4 | `p_g ~ Beta(α_g, β_g)` (p. 26) | `.group_prior(g)` = `BetaPosterior.from_mean_concentration(m_0, κ_group)`, `m_0` = global posterior mean; `.group_posterior(g)` | same |
| 5 | `p_{g,a} ~ Beta(α_{g,a}, β_{g,a})` (p. 26) | `.leaf_prior(g, a)` = `from_mean_concentration(σ(logit(m_g) + t(g, a)), κ_leaf)`, `m_g` = group posterior mean, `t` = policy-effect tilt (§2.6; `t ≡ 0` without it) | same; `test_unseen_leaf_equals_its_shrunk_group_posterior`, `test_tilt_moves_the_leaf_prior_mean_on_the_logit_scale` |
| 6 | `p_{g,a} \| D ~ Beta(α_{g,a} + d_{g,a}, β_{g,a} + n_{g,a})` (p. 26) | `models/beta.py::BetaPosterior.update(d, n)`; `HierarchicalBetaBernoulli.fit / update / posterior`; counts `(d, n)` = `.counts(g, a)` | `test_beta.py::test_conjugate_update…`, `test_counts_propagate_to_all_levels`, `test_fit_equals_sequential_updates_after_refresh` |
| 7 | `E[p_{g,a}] = (α_{g,a} + d_{g,a}) / (α_{g,a} + β_{g,a} + d_{g,a} + n_{g,a})` (p. 26) | `BetaPosterior.mean`; `HierarchicalBetaBernoulli.p_default(g, a)` | `test_conjugate_update_matches_proposal_posterior_mean`, `test_priors_are_built_top_down_and_posterior_matches_proposal` |
| 8 | `R_{g,a} \| D = 1 ~ Beta(α^R_{g,a}, β^R_{g,a})` (p. 26) | `models/recovery.py::BetaMoMRecovery.posterior(g, a)` / `.resolve(g, a)` (fallback leaf → (group, term) → group → global → Beta(2, 5), §2.9); `TwoStageBetaBeta.risk(g, a).recovery` | `test_recovery.py` |
| 9 | `α^R = μ(μ(1 − μ)/σ² − 1)` (p. 27) | `BetaPosterior.from_moments(μ, σ²)` called by `BetaMoMRecovery._try_level`; μ, σ², N from `RunningMoments` (Welford) | `test_method_of_moments_formula`, `test_leaf_beta_is_the_method_of_moments_of_its_data` |
| 10 | `β^R = (1 − μ)(μ(1 − μ)/σ² − 1)` (p. 27) | same | same |
| 11 | `S = {C, D, R, L}` (p. 27) | `types.py::State` (re-exported by `models/mctm.py`) | — |
| 12 | `P = [[p_CC, p_CD, 0, 0], [p_DC, p_DD, p_DR, p_DL], [0,0,1,0], [0,0,0,1]]` (p. 27) | `models/mctm.py::MarkovChainTransitionMatrix.matrix` (Dirichlet posterior mean, `fit_counts`), structural zeros `allowed_transitions(TransitionStructure.PROPOSAL)`, checks `validate_transition_matrix` | `test_mctm.py::test_structural_zeros…`, `test_rows_sum_to_one…` |
| 13 | `P_RR = 1, P_LL = 1` (p. 27) | absorbing rows fixed in `fit_counts`, enforced by `validate_transition_matrix` | `test_rows_sum_to_one_and_absorbing_rows_are_fixed`, `test_invalid_matrices_raise` |
| 14 | `lim_{k→∞} π_k = [0, 0, π_R^∞, π_L^∞]` (p. 27) | `.stationary(pi0=START_CURRENT)` via `.absorption_probabilities()` (`B = N·R_block`, `N = (I − Q)⁻¹`) | `test_absorption_probabilities_equal_limit_of_matrix_power` |
| 15 | `π_t = [π_C(t), π_D(t), π_R(t), π_L(t)]`, `π_t = π_0 Pᵗ` (p. 27) | `.propagate(pi0, t)`, `.trajectory(pi0, horizon)`; `START_CURRENT`, `START_DELINQUENT` | `test_propagate_and_trajectory` |
| 16 | `P(Default) = π_D(t) + π_L(t)` (p. 28) | `.p_default(t, pi0)`, `.p_default_curve(horizon)` | `test_p_default_is_non_decreasing_from_current` |
| 17 | "delinquents collapse" at maturity; `π_D(t) → 0`, `π_L → E[p_d]`, `π_R → E[R]` (p. 28) | `.maturity_collapse(pi)` (C → R, D split `p_DR : p_DL`); `.p_loss_given_state(state, m)` is the MCTM counterpart of `E[p_d]` — see assumption M10 | `test_maturity_collapse`, `test_p_loss_given_state_includes_maturity_collapse` |
| 18 | `P(R \| D, t)`, `P(L \| D, t)` (p. 28) | `.p_recover_given_delinquent(t)`, `.p_loss_given_delinquent(t)` (`t=None` → limit `B[D, ·]`) | `test_propagate_and_trajectory`, `test_absorption_probabilities…` |
| 19 | `A_k = (L_k, r_k)`, `𝒜 = {A_1, …, A_K}` (p. 28) | `data/arms.py::ArmSpace` (arm 0 = null/decline) | — (contract) |
| 20 | `θ_k ~ N(μ_k, σ_k²)`, draw `θ̃_k`, `A_t = argmax_k θ̃_k` (p. 29) | `bandits/arm_state.py::GaussianArmPosterior.sample`, `bandits/thompson.py::GaussianThompsonSampling.select` (decision agent) | decision agent |
| 21 | `R_t = Revenue_t − Expected Loss_t − Funding Cost_t` (p. 29) | `rewards/compound.py::CompoundRewardFunction.expected / sample / terminal / interim` (closed form in §2.4 below) | `test_compound_reward.py` |
| 22 | `E[L_t] = L_t · P(D \| X, A_t) · (1 − E[R \| X, A_t])` (p. 29) | `RewardDistribution.components["expected_loss"] = c·E[p]·(1 − E[R])`, `c = L` for `exposure_basis="full"`; `RiskEstimate.expected_loss_fraction = E[p](1 − E[R])` | `test_certain_default_without_recovery…`, `test_components_add_up…` |
| 23 | `μ_k^{(t+1)} = μ_k^{(t)} + (r_t − μ_k^{(t)})/(n_k + 1)`, `n_k ← n_k + 1` (p. 29) | `GaussianArmPosterior.update` (decision agent) | decision agent |
| 24 | `σ_k^{2(t+1)} = ((n_k − 1)σ_k^{2(t)} + (r_t − μ^{(t)})(r_t − μ^{(t+1)}))/n_k` (p. 30) | `GaussianArmPosterior.update` (decision agent) | decision agent |
| 25 | `lim μ_k = E[r_k]`, `lim σ_k² = Var(r_k)` (p. 30) | consequence of 23–24 | decision agent |
| 26 | PSI `= Σ (a_i − e_i) ln(a_i / e_i)` (G.5, Table 9) | `monitoring/psi.py::PSICalculator.psi` (decision agent) | decision agent |

## 2. Implementation equations (SPEC §2) not spelled out in the proposal

### 2.1 Beta value object and HDI — `models/beta.py`, `models/hierarchical.py`
| Equation | Code |
|---|---|
| `Var[p] = αβ/((α+β)²(α+β+1))`, `E[p²] = Var + E[p]²` | `BetaPosterior.var`, `.second_moment` |
| HDI = narrowest `[F⁻¹(q), F⁻¹(q + mass)]`, `q ∈ [0, 1 − mass]` | `BetaPosterior.hdi` (one Beta); `hierarchical.py::beta_hdi(alpha, beta, mass)` (vectorised golden-section search, used by `summary()` and `predict_table()`) |
| child prior `Beta(κm, κ(1 − m))` | `BetaPosterior.from_mean_concentration(m, κ)` |
| shrinkage weight `w = n/(n + κ)`; `E[p_child] = w·(d/n) + (1 − w)·m` | `HierarchicalBetaBernoulli.shrinkage_weight`, `summary()["shrinkage_weight"]` (`test_posterior_mean_is_shrinkage_weighted_average`) |
| `κ = "auto"`: `r_c = d_c/n_c` (`n_c ≥ 5`), `m = Σ n_c r_c / Σ n_c`, `v = Σ n_c (r_c − m)² / Σ n_c − m(1 − m)/mean(n_c)` floored at `1e-6`, `κ = m(1 − m)/v − 1` clipped to `[kappa_min, kappa_max]` | `hierarchical.py::beta_binomial_kappa`, used by `HierarchicalBetaBernoulli.refresh_priors` |

### 2.2 Recovery running statistics — `models/recovery.py`
| Equation | Code |
|---|---|
| Welford: `n ← n+1; δ = x − x̄; x̄ ← x̄ + δ/n; M2 ← M2 + δ(x − x̄)`; `σ² = M2/(n − 1)` | `RunningMoments.push`, `.var`; batch fit uses the two-pass `Σ(x − x̄)²` (equal to 1e-9, `test_online_update_matches_batch_fit`) |
| MoM guards: μ clipped to `[eps, 1 − eps]`; `σ² ≥ μ(1−μ)` → `NumericalInstabilityError` → next level; `σ² ≤ 1e-12` → `α + β = max_concentration` | `BetaPosterior.from_moments`, `BetaMoMRecovery._try_level` |

### 2.3 Markov chain — `models/mctm.py`
| Equation | Code |
|---|---|
| Dirichlet row posterior `P_i· ~ Dir(prior_strength·1_allowed + n_i·)`, mean `(s + n_ij)/Σ_{allowed}(s + n_ik)` | `fit_counts`, `sample_matrix(rng)` (`test_dirichlet_posterior_mean_and_mle`, `test_sample_matrix…`) |
| `Q = P[{C,D},{C,D}]`, `N = (I − Q)⁻¹`, `B = N·P[{C,D},{R,L}]`, `t = N·1` | `fundamental_matrix`, `absorption_probabilities`, `expected_time_to_absorption`; singular `I − Q` (`|det| < 1e-12`) → `InvalidTransitionMatrixError` |
| collapse: `π_R' = π_R + π_C + π_D·p_DR/(p_DR + p_DL)`, `π_L' = π_L + π_D·p_DL/(p_DR + p_DL)` (50/50 if both 0) | `maturity_collapse` |
| `F(m) = π_L(m)/π_L'(T)` for `m < T`, `F(T) = 1` (uniform `m/T` if `π_L'(T) = 0`); `f(m) = F(m) − F(m − 1)` | `default_timing_cdf(T, start)`, `default_timing_pmf(T, start)` |
| `q(s, m) = [collapse(e_s Pᵐ)]_L` | `p_loss_given_state(state, remaining_months)` |
| `ℓ = Σ_{n_ij > 0} n_ij log P_ij` (−∞ if some `P_ij = 0`) | `log_likelihood(counts)` |
| `e_i = Σ_j |P_ij − n_ij/n_i|` | `row_prediction_error(counts)` |

### 2.4 Rewards — `rewards/amortization.py`, `rewards/compound.py`
Level-payment schedule for `(L, r, T)`, `i = r/12`:

| Equation | Code |
|---|---|
| `M = L·i(1+i)^T/((1+i)^T − 1)` (`L/T` if `r = 0`) | `installment` (= `data/arms.py::annuity_installment`) |
| `B_m = L(1+i)^m − M((1+i)^m − 1)/i`, `I_m = i·B_{m−1}`, `P_m = B_{m−1} − B_m` | `schedule`, `schedule_arrays`, `outstanding_balances` (`[B_0 … B_T]`) |
| `WAL = Σ m·P_m/(12L) = Σ B_{m−1}/(12L) = total_interest/(r·L)` | `weighted_average_life_years`, `total_interest` |
| `Σ_{m=1}^{k} B_{m−1} = L·S − (M/i)(S − k)`, `S = ((1+i)^k − 1)/i` | `cumulative_opening_balance` |

Closed form of `CompoundRewardFunction.expected(policy, risk, mctm)` (also in the module docstring):

```
gross_interest = Σ_m I_m
b              = Σ_m I_m · F(m)              F = mctm.default_timing_cdf(T), else m/T
funding_cost   = funding_rate · Σ_m B_{m−1} / 12  (= funding_rate · L · WAL)
ε              = 1 ("full")  |  Σ_m f(m) B_{m−1} / L ("outstanding")
c = ε·L,  a = gross_interest − funding_cost
Reward(p, R)   = a − b·p − c·p·(1 − R) = a − p·h(R),   h(R) = (b + c) − c·R
E[Reward]      = a − (b + c)E[p] + c·E[p]E[R]
E[h] = (b + c) − c·E[R],   E[h²] = (b + c)² − 2(b + c)c·E[R] + c²E[R²]
Var[Reward]    = E[p²]E[h²] − (E[p]E[h])²            (p ⫫ R)
```

| Equation | Code |
|---|---|
| constants `(gross_interest, funding_cost, b, c, ε)` | `CompoundRewardFunction.terms(policy, mctm)` → `RewardTerms` |
| prepayment: `(a, b) → (ρ·a, ρ·b)` (§2.8) | `expected / sample / interim(..., realization=ρ)` |
| `(E, Var)` vectorised | `compound.py::reward_moments(a, b, c, E[p], E[p²], E[R], E[R²])` |
| Monte Carlo `a − p((b + c) − cR)` | `CompoundRewardFunction.sample(policy, p_draws, R_draws, mctm)` (matches the closed form within 1% at 200k draws) |
| risk-adjusted score `mean − λ·std` | `RewardDistribution.score(λ)`, `CompoundRewardFunction.score(dist)` (`λ = RewardConfig.risk_aversion`) |
| realised `revenue − loss − funding_rate·Σ_{m≤k} B_{m−1}/12`, `k = min(months_on_book, T)` | `terminal(row)`, `terminal_many(table)` |
| interim (state `s`, age `month`, `months_paid`): `realized = Σ_{m≤months_paid} I_m`, `q = mctm.p_loss_given_state(s, T − month)`, `future = Σ_{m>month} I_m(1 − q·F_rem(m − month))`, `loss = B_{months_paid}·q·(1 − E[R])`, `mean = realized + future − loss − funding_cost` | `interim(policy, state, month, months_paid, risk, mctm)` |

### 2.5 Predictive metrics — `evaluation/predictive.py`
| Metric | Formula | Function |
|---|---|---|
| Brier | `mean((p − y)²)` | `brier` |
| log loss | `−mean(y log p + (1 − y) log(1 − p))`, `p ∈ [ε, 1 − ε]` | `log_loss` |
| AUC | `(Σ rank(positives) − n₊(n₊+1)/2)/(n₊n₋)`, average ranks for ties | `roc_auc`; curve `roc_curve` |
| ECE | `Σ_b (n_b/N)·|frac_pos_b − mean_pred_b|`, equal-width bins | `expected_calibration_error` (+ `reliability_table`) |
| MAE / RMSE | `mean|y − ŷ|`, `√mean(y − ŷ)²` | `mae`, `rmse` |
| transition log-likelihood | `Σ_{n_ij>0} n_ij log P_ij` (optionally `/Σ n_ij`) | `transition_log_likelihood` |
| transition row error | `Σ_j |P_ij − n_ij/n_i|` | `transition_row_error` |

### 2.6 Structured leaf prior (policy effect) — `models/policy_effect.py`, `models/hierarchical.py`

**Why.** The proposal specifies the leaf `p_{g,a} ~ Beta(α_{g,a}, β_{g,a})` but leaves
`α_{g,a}, β_{g,a}` open. Shrinking every leaf to its group mean treats the cells of the arm
grid as exchangeable: an arm nobody in group `g` ever received (a grade-A borrower at 28 %)
gets exactly the group's default probability, so `Revenue − L·p·(1 − R) − Funding` grows
with amount and rate without any risk penalty and every optimiser runs to "largest
affordable loan at the highest rate" (shadow uplift ≈ +1,500 %, lendable 100 %, before this
change). The literature review (§2.2.1, Scott 2010) motivates a *structured* model across
policy factors instead of independent cells; the leaf prior mean is therefore a function of
the arm's rate, term and size.

**Evidence** (Lending Club, resolved loans issued < 2015 whose term had elapsed by
2019-03, 406k rows): across the sub-grades of a grade, `d logit(default)/d rate` is 8–37
per unit rate (median ≈ 16; grade C: C1 → C5 default 14.9 % → 20.3 % as the mean rate goes
13.6 % → 16.1 %); within one sub-grade, LC's repricing over time carries little signal
(median slope ≈ 1.5). A 60-month term adds 3–7 points at the same sub-grade (C1: 18.8 % vs
14.4 %). Payment-to-income (`12·installment/annual_inc`) above 15 % vs at most 5 % roughly
doubles default (A1 9.4 % vs 2.1 %, C3 21.1 % vs 15.5 %, F3 43.8 % vs 24.1 %); the raw amount
adds little once PTI is accounted for. Fitted within-group coefficients (logit per unit) on
the pipeline runs of 2026-10-16: Lending Club 300k sample (53,858 elapsed training loans)
`β = (int_rate 4.0 ± 0.9, term60 0.49, pti 5.9, log_amount −0.19)`; synthetic 200k
`β = (12.6, 0.44, 5.0, −0.21)`. The within-group rate slope is smaller than the
across-sub-grade one because the groups pool vintages and part of the price signal is
carried by PTI.

**Model** (offset logistic regression, fitted once on the training window):

```
logit P(D_i = 1) = logit(m_{g(i)}) + β·(x_i − x̄_{g(i)})
x_i  = [int_rate, 1[term = 60], pti_i, log(loan_amnt)],   pti_i = clip(12·installment_i / annual_inc_i, 0, 1.5)
m_g  = (d_g + 1)/(n_g + 2)                                  (shrunk group default rate of the fit sample)
```

* Rows: resolved training loans with a 0/1 outcome, a group, a mapped arm and valid features
  (rows with non-finite / non-positive income are dropped when PTI is used). The regression
  uses only the loans whose term had elapsed by the snapshot when at least 20,000 exist
  (no resolution bias towards early defaults), else all of them.
* Every feature is centred **within its group** on the fit sample, so `β` is a
  within-group effect, free of the grade → rate confounding between groups (adding a
  group-level constant to the rates leaves `β` unchanged — tested).
* `β` maximises the mean log-likelihood minus `ridge/2·‖β_s‖²` on standardised centred
  features (`ridge = 1e-3`) by Newton–Raphson with step halving (≤ 25 iterations, fully
  vectorised; ≈ 0.1 s for 50k rows). A feature without within-group variation is dropped;
  no feature left, a Hessian with condition number > 1e12, or no convergence raise
  `NumericalInstabilityError`, and the fit falls back to `β = 0` (no tilt) with the reason
  in `diagnostics`. Fewer than `min_rows = 5000` usable rows also gives `β = 0`.

**Tilt.** For arm `a = (L_a, r_a, T_a)` of group `g`:

```
x_{g,a} = [r_a, 1[T_a = 60], clip(12·installment(a)/median_income_g, 0, 1.5), log L_a]
t(g, a) = clip(β·(x_{g,a} − x̄_g), −3, 3)                    (max_abs_logit_shift; t = 0 for the null arm)
m_{g,a} = σ(logit(m_g) + t(g, a)),   α_{g,a} = κ_leaf·m_{g,a},   β_{g,a} = κ_leaf·(1 − m_{g,a})
```

`x̄_g` and `median_income_g` are taken over all resolved rows of the group (the rows behind
`m_g`); an unseen group uses the global means / median and its arms get
`Beta(κ_group·σ(logit(m_0) + t), κ_group·(1 − σ(logit(m_0) + t)))`. The clip keeps an
extrapolated arm within a factor ≈ e³ of the group odds.

**What changes and what does not.** Only the leaf prior *mean*: κ_leaf, the group and
global levels, the conjugate update `Beta(α_{g,a} + d, β_{g,a} + n)`, the posterior mean
formula and the online `update()` are identical, so a leaf's own data dominate as they
accumulate (shrinkage weight `n/(n + κ_leaf)`, tested). `t ≡ 0` reproduces the untilted
model bit for bit. The tilt is frozen with the model (like κ, H2); `refresh_priors()`
re-applies it to the new anchors.

| Equation | Code |
|---|---|
| `x_i`, validity, `pti` clip | `PolicyEffectModel._row_features` |
| `m_g`, centring, Newton–Raphson, fallback | `PolicyEffectModel.fit` / `fit_arrays` / `_newton` |
| `x_{g,a}`, `t(g, a)` | `PolicyEffectModel.arm_features`, `.raw_effect`, `.effect` |
| `(g, a) ↦ effect(g, arm_space[a])` | `PolicyEffectTilt` (installed by `TwoStageBetaBeta.fit` when `HierarchyConfig.policy_effect` and an arm space are given) |
| `m_{g,a} = σ(logit(m_g) + t)` | `hierarchical.py::tilted_mean`, `HierarchicalBetaBernoulli.leaf_prior`, `.unseen_group_prior`, `.prior_tilt`; `summary()["prior_tilt_logit"]` |

### 2.7 Lending guardrails — `data/arms.py`, `data/reference_rates.py`

An arm is eligible for a borrower when every guardrail whose input is known holds (G1–G5
below); the shadow engine, every simulated policy and the hindsight oracle choose among the
same eligible arms (the historical policy additionally keeps its own arm):

```
installment_a ≤ 0.20 · annual_inc / 12                 (max_payment_to_income)
L_a           ≤ 0.50 · annual_inc                      (max_amount_to_income)
dti/100 + 12·installment_a / annual_inc ≤ 0.43         (max_total_dti; LC dti is in percent)
|r_a − ref(sub_grade)| ≤ 0.05                          (rate_corridor)
ref(sub_grade) = mean training int_rate of the sub-grade → grade → global
```

The defaults follow US consumer-underwriting conventions. The 43 % total debt-to-income
ceiling is the General Qualified Mortgage limit of the CFPB Ability-to-Repay/QM rule,
12 CFR 1026.43(e)(2)(vi) (2013). The 2020 General QM amendment replaced it with a price
test, but 43 % is still the standard DTI benchmark. Lending Club's `dti` excludes mortgage
payments, so the test is a lower bound of the full back-end ratio. The 20 % payment-to-income
cap and the 50 % amount-to-income cap are lender conventions for unsecured instalment credit
(personal and auto lending policies cap PTI at 15–20 %), not regulations.

| Rule | Code |
|---|---|
| guardrails, null arm, fallback arm | `ArmSpace.eligible`, `.eligible_many`, `.fallback_arms` |
| reference rate | `ReferenceRates.fit / lookup / for_context / for_frame` (saved as `fit/reference_rates.json`) |

### 2.8 Prepayment: realised interest — `rewards/realization.py`, `rewards/compound.py`

**Why.** The closed form of §2.4 values a non-defaulting loan's full contractual schedule.
Most Lending Club borrowers repay early. In the training loans whose term had elapsed, fully
paid 36-month loans realised ≈ 82 % of their scheduled interest and 60-month loans ≈ 72 %
(LC 300k sample; synthetic: 81 % / 79 %). Ignoring this overvalues long terms and large
balances, so the optimisers preferred 60-month loans almost always.

**Estimate** (`InterestRealization.fit`, on the training window):

```
ρ_i     = clip(total_rec_int_i / scheduled_interest(funded_amnt_i, int_rate_i, term_i), 0, 1)     fully-paid loans
ρ(g, T) = mean of ρ_i over grade g and term T            (cells with n ≥ 30; else the next level)
fallback  (grade, term) → term → global → 1;   every table entry clipped to [0.3, 1.0]
```

Only loans whose term elapsed by the snapshot are used when at least 1,000 exist, because
recent vintages' fully-paid loans are the early payers. A per-loan ratio above 1 comes from
late interest or rounding and is capped at 1.

**Reward.** With ρ = ρ(grade(g), T_a), the expected-interest term and the funding cost are
multiplied by ρ (prepayment shortens both the interest stream and the balance-time that
has to be funded):

```
Revenue = ρ · Σ_m I_m (1 − p·F(m)),   Funding = ρ · funding_rate · Σ_m B_{m−1}/12
Reward(p, R) = ρ·a − p·((ρ·b + c) − c·R)          i.e. (a, b) → (ρ·a, ρ·b), c unchanged
E[Reward] = ρ·a − (ρ·b + c)·E[p] + c·E[p]·E[R];   Var as in §2.4 with b → ρ·b
interim: the expected *future* interest and the funding cost are multiplied by ρ, the
         interest already received is not (so interim(C, month 0) = expected for every ρ)
```

`RewardConfig.prepayment_adjustment = "empirical"` (default) switches it on and `None`
values the contract (ρ = 1, bit-identical to the unadjusted formulas). The realised
`terminal()` reward is never adjusted. ρ depends only on the group's grade and the arm's
term, so the simulation's `(group, arm)` caches stay exact (tested bit for bit). The
learning environment, the hindsight yardstick, the bandits' warm start and the shadow engine
all use the reference fit's table (`fit/interest_realization.json`), i.e. the same formula.

| Equation | Code |
|---|---|
| ρ table, fallback, clip | `InterestRealization.fit / factor / source / table` |
| grade of a group key | `realization.py::group_grade(group, segmenter)` |
| `(g, a) ↦ ρ` | `RealizationLookup` (`FittedModels.realization_lookup()`, `.realization(g, a)`) |
| `(a, b) → (ρ·a, ρ·b)` | `CompoundRewardFunction.expected / sample / interim(..., realization=ρ)` |

### 2.9 Recovery by term — `models/recovery.py`

On the principal basis (§1.2) a defaulted 60-month loan has amortised less principal. In
the LC training data, 60-month defaults recover 0.38 of principal against 0.485 for
36-month ones. An unobserved 60-month leaf should therefore not borrow a group mean
dominated by 36-month loans. Stage 2's fallback chain gains a level:

```
leaf (g, a) → (g, term(a)) → g → global → Beta(2, 5)           (each level: n ≥ min_obs and a valid MoM)
```

The (g, term) node keeps its own Welford moments (batch `fit` and online `update`). The arm
→ term map comes from the arm space (`TwoStageBetaBeta` passes it, like the policy-effect
tilt); without it, or with `RecoveryConfig.term_level = False`, the chain is the SPEC's.

| Rule | Code |
|---|---|
| chain, `group_term` level, `term_of` | `BetaMoMRecovery._chain / resolve / group_term_posterior / term_of / set_term_of_arm` |
| summary / diagnostics | `summary()` rows with `level = group_term` and a `term` column; `diagnostics["source_group_term"]`, `n_group_terms` |

## 3. Assumptions made beyond SPEC (risk layer)

Every **[decision]** below is implemented as described and covered by a test.

**Stage 1 — `HierarchicalBetaBernoulli`**
- H1. `counts(g, a)` returns `(d, n)` = (defaults, non-defaults), the proposal's notation; `n_obs = d + n` is what the shrinkage weight `n/(n + κ)` uses.
- H2. Child priors (the anchor means `m_0`, `m_g` and the κ's) are frozen at `fit()` and only re-derived by `refresh_priors()`. `update()` changes counts at all three levels, so every posterior reflects all data, but one arm's outcome never moves another arm's prior mid-run. That keeps the arm posteriors that Thompson Sampling and the shadow engine read during a run stable and reproducible. A group first seen online anchors its leaves at `m_0` (its prior mean) until the next refresh.
- H3. Unseen group → the group prior (concentration `κ_group`); unseen leaf of a known group → its leaf prior (concentration `κ_leaf`).
- H4. `"auto"` κ uses `n`-weighted moments over children with `n ≥ 5`. `κ_leaf` is estimated per group; a group whose leaves cannot identify κ (fewer than 2 qualifying leaves) uses the median of the identified groups' values, else the numeric config default (20 / 10, clipped to `[kappa_min, kappa_max]`). The same default is used for `κ_group` when fewer than 2 groups qualify. A children mean `m ∈ {0, 1}` gives `kappa_max` (no measurable heterogeneity).
- H5. `fit()` replaces any previous state. Rows with a missing group, `arm < 0`/missing arm, or a missing default are skipped (`diagnostics["skipped"]`). Non-0/1 defaults or mismatched lengths raise `ValueError`. `update()` with `arm < 0` or a missing flag is a no-op.
- H6. `summary()` adds `key` (`"global"`, `g`, `"g#a"`), `prior_mean` (for a leaf: the tilted leaf prior mean), `empirical_rate`, `kappa` and `prior_tilt_logit` (the leaf's `t(g, a)`, 0 for global/group rows) to the required columns. The global row has `shrinkage_weight = 0` and `kappa = α_0 + β_0`. HDIs come from the vectorised `beta_hdi`, which agrees with `BetaPosterior.hdi` to ≤ 1e-5 and is about 30× faster for thousands of leaves. It lives in `hierarchical.py` because `beta.py` is a contract file.

**Structured leaf prior (policy effect)** — §2.6
- S1. The tilt is a fixed function of `(g, a)` estimated once per fit (the reference fit,
  the simulation's as-of environment, which only uses outcomes observed before the stream
  start and "elapsed" relative to it, the hindsight yardstick, and every breaker
  re-segmentation re-fit it on their own rows). It is not updated online.
- S2. `β` is an associational within-group effect (it also absorbs within-group risk
  differences that track the policy, e.g. sub-grade within a grade); it is used to rank
  counterfactual arms, not as a causal price elasticity. The clip (±3 logit) bounds the
  extrapolation.
- S3. An arm's PTI uses the group's median income, not the borrower's: the leaf prior is per
  `(g, a)`; borrower-level affordability is the job of the guardrails (G1–G3).
- S4. A group without income data uses the group's mean PTI (zero PTI contribution); an
  unseen group uses the global means and median income.
- S5. The hierarchy stores the tilt twice in `to_dict()`: the serialised tilt
  (`PolicyEffectTilt` = model + arm space) and the table of values for every seen group ×
  arm (`leaf_tilt_table`). A non-serialisable tilt function is restored from that table
  (`TiltTable`; unknown pairs → 0).
- S6. With `HierarchyConfig.policy_effect = False`, without an arm space or without the
  policy columns, no tilt is installed and Stage 1 is exactly the untilted SPEC model
  (`TwoStageBetaBeta.policy_effect_info["status"]` says which).

**Lending guardrails** — §2.7
- G1. Affordability `installment ≤ max_payment_to_income · annual_inc / 12` (0.20; unsecured
  instalment-credit convention).
- G2. Exposure `amount ≤ max_amount_to_income · annual_inc` (0.50; convention).
- G3. Total debt `dti/100 + 12·installment/annual_inc ≤ max_total_dti` (0.43, the QM rule's
  ceiling, 12 CFR 1026.43(e)(2)(vi)). `dti` is LC's percent and excludes mortgage payments;
  a missing or negative `dti` skips the rule.
- G4. Pricing `|rate − reference_rate| ≤ rate_corridor` (0.05; comparisons with a 1e-9
  tolerance). The reference rate is the training mean `int_rate` of the borrower's
  sub-grade (fallback grade, then global) — a pure function of pre-decision context. Fits
  saved before reference rates existed load with `reference_rates = None`: no corridor.
- G5. A guardrail whose input is unknown (income missing or ≤ 0, no reference rate) or
  that is configured as `None` is skipped. The null arm is always eligible. If no lending
  arm survives, the one lending arm with the rate closest to the reference (any rate
  without one) and the smallest installment is kept, so a minimal counter-offer exists
  next to "decline" (and the old "keep the cheapest arm" rule is the special case without
  a null arm or a reference).

**Prepayment (interest realization)** — §2.8
- W7. ρ is estimated once per fit, like the MCTM, and not updated online; the simulation's
  as-of environment uses it as well, the same compromise as for the MCTM.
- W8. ρ multiplies the whole expected-interest term (also the interest a defaulter pays
  before defaulting) and the full funding cost; prepayment is not modelled per amount or
  rate, only per grade and term.
- W9. A fit without a table (saved before it existed, or without `total_rec_int`) and
  `prepayment_adjustment = None` both give ρ = 1.

**Stage 2 — `BetaMoMRecovery`**
- R1. σ² is the sample variance `M2/(n − 1)`; `n = 1` has undefined variance and so falls back.
- R2. Recovery rates outside `[0, 1]` are clipped (counted in `diagnostics["clipped"]`); missing rates and `arm < 0` rows are skipped.
- R3. `diagnostics` is recomputed from the current state over all known nodes. It counts nodes, not queries: `source_{leaf,group,global,prior}` (where each known leaf's Beta comes from), `insufficient_<level>` (`0 < n < min_obs`), `invalid_<level>` (MoM raised), `global_source`.
- R4. Ultimate fallback Beta(2, 5) with source `"prior"`. `summary()` reports the node's observed `mean`/`var` (the MoM inputs) next to the `alpha`/`beta` actually used after fallback, plus `expected_recovery`.
- R5. `RecoveryConfig.basis` is not used by the model: the data layer computes `y_recovery_rate` on the chosen basis.
- R6. The (group, term) level (§2.9) uses the arm's grid term. Changing the arm → term map
  needs a re-`fit` to rebuild the level's statistics. A v1 dict, i.e. a model saved before
  the level existed, restores without it.

**Facade — `TwoStageBetaBeta`**
- T1. The constructor also accepts a `FrameworkConfig` as its first argument; `from_config(cfg)` is equivalent. `from_config(cfg, arm_space)` / `arm_space=` enables the structured leaf prior (§2.6); `fit(table, snapshot_date=)` defines its elapsed-term rows.
- T2. `RiskEstimate.n_default_obs` / `n_recovery_obs` are leaf-level counts. An extra field `recovery_source` records the fallback level.
- T3. `predict_table` returns NaN for rows with a missing group or `arm < 0`, adds `expected_loss_fraction`, and accepts `group_col` / `arm_col` (e.g. to score shadow arms).
- T4. `update()` ignores a recovery rate unless the loan defaulted. A table missing derived columns raises `DataSourceError`.

**MCTM**
- M1. `EXTENDED` adds only C → R; C → L stays a structural zero (as in `types.TransitionStructure`).
- M2. Counts in structurally-zero cells, and off-diagonal counts in absorbing rows, are ignored and reported in `diagnostics`. An unfitted chain is the prior mean (uniform over allowed cells, or "stay put" under the MLE).
- M3. `from_matrix` chains carry no Dirichlet parameters, so `sample_matrix` returns the matrix unchanged.
- M4. `P(R|D,t)`, `P(L|D,t)` are pure chain propagation (no collapse); `t=None` gives the limit. `p_loss_given_state` *does* include the collapse at the end.
- M5. `default_timing_cdf` / `_pmf` take an optional `start` state (used for `F_rem` in the interim reward). The timing is absorption into L, i.e. charge-off, which lags the last payment by the charge-off lag, so the interest lost by defaulters is slightly understated. The pmf includes the maturity-collapse lump at `m = T`.
- M6. `π_L(t)` is non-decreasing (L is absorbing). `π_D(t) + π_L(t)` is monotone for realistic credit matrices but not in general, since it can fall when delinquents cure fast; the monotonicity test uses realistic counts.
- M7. `log_likelihood` of an all-zero count matrix is 0.0 (an empty sum), whereas `predictive.transition_log_likelihood` returns NaN for "no transitions" to follow the metric convention.
- M8. `summary()` uses `State.label` for `from_state`/`to_state` and adds an `allowed` column. `prior` equals `prior_strength` on allowed transient cells and 0 elsewhere.
- M9. `fingerprint` (a hash of matrix + structure) keys the reward caches, so re-fitting a chain invalidates them.
- M10. The proposal's `lim π_R(t) = E[R]` mixes two different quantities. π_R is the probability of *absorbing in R* (repayment), whereas the Stage-2 `E[R]` is the *recovery rate given default*. The like-for-like convergence is `π_L'(T)` (`p_loss_given_state(C, T)`) vs the Stage-1 `E[p]`. Both are exposed, so evaluation can report the gap; it is not enforced.

**Rewards**
- W1. The reward variance is over posterior (parameter) uncertainty of `(p, R)`, not the Bernoulli outcome variance. That is the uncertainty Thompson Sampling and `mean − λ·std` need.
- W2. `expected()` charges funding on the full contractual schedule; `terminal()` charges it on the balance-months actually on book. `terminal()` uses `funded_amnt` (falling back to `loan_amnt`), floors `y_months_on_book` and caps it at the term, and returns NaN when a required field is missing. `terminal_many` is its vectorised closed-form twin.
- W3. In `interim()` the exposure is `B_{months_paid}`, the principal still outstanding. This equals `B_month` whenever the borrower is up to date and is the correct exposure for a delinquent borrower; it ignores `exposure_basis`. Interest missed between `months_paid` and `month` is not counted (conservative). Funding is the full-life cost in every state, including RECOVERED / LOSS.
- W4. Interim std: the closed form with `b' = Σ_{m>month} I_m F_rem`, `c' = B_{months_paid}` and `p' ~ Beta(qκ, (1 − q)κ)`, where `κ = α + β` of the Stage-1 posterior (that posterior re-centred at `q`). LOSS: std `= B·std(R)`; RECOVERED: std 0. Without an MCTM, `q = E[p]` in C and `max(E[p], 0.5)` in D. At `month = 0` in C without an MCTM, `interim == expected` exactly (tested).
- W5. With uniform timing, a certain defaulter pays `Σ I_m(1 − m/T)`. That is ≈ 0.64 of the gross interest for 16 %/36 m, because amortised interest is front-loaded; it is exactly `(T − 1)/2T ≈ ½` only for level interest. The "≈ half" test therefore asserts the exact formula plus a 0.45–0.70 band.
- W6. `RewardDistribution.__float__` returns the mean (compatibility with SPEC's float-valued `interim`). Schedules are `lru_cache`d (16 384 entries); `RewardTerms` and interim future-interest terms are cached per `(L, r, T, mctm.fingerprint, cfg)`.

**Predictive metrics**
- P1. Pairs with a missing or non-finite value are dropped. Empty or undefined metrics return NaN (e.g. AUC with one class). Mismatched lengths, non-binary `y`, or `p ∉ [0, 1]` raise `ValueError`.
- P2. Added `roc_curve` (same points as `sklearn.metrics.roc_curve(drop_intermediate=False)`) for the calibration/ROC endpoint.
- P3. ECE uses equal-width bins (`p = 1` goes in the last bin) and lists every bin in the reliability table; empty bins have NaN means.
- P4. `transition_row_error` reports every row (NaN without transitions, e.g. absorbing rows), whereas `MarkovChainTransitionMatrix.row_prediction_error` reports the transient rows only. `transition_log_likelihood(normalize=True)` gives the mean per transition.
