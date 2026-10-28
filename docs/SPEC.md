# Bayesian Credit — Build Specification

**Source of truth:** `Data Science Reasearch Proposal on Bayesian Contextual Bandits.pdf` (Misati Randy Nyambane, USIU-Africa, DSA4900). Section references below (§3.3, Appendix F/G, Tables 3–11) point at that document. Everything in this spec is derived from it; where the proposal leaves an implementation detail open, the chosen default is marked **[decision]** and must be configurable.

The product is a **shadow lending engine**: a closed-loop Bayesian sequential decision system that replays historical Lending Club loans as a stream of borrowers, estimates risk hierarchically, selects lending policies with contextual Thompson Sampling, evaluates counterfactual policies, and is governed by a PSI circuit breaker.

```
Borrower Context → Bayesian Risk Estimation → Credit-State Prediction → Risk-Adjusted Reward
      → Policy Selection → Observed Outcomes → Posterior Update → Policy Adaptation
PSI Sentinel runs across the loop; the shadow engine runs alongside the historical policy.
```

---

## 0. Repository layout

```
bayesian-credit/
├── core/                      # pure-Python modelling library, package `bayesian_credit` (NO Django imports)
│   ├── pyproject.toml
│   ├── bayesian_credit/
│   │   ├── config.py          # dataclass configs            (written by planner — shared contract)
│   │   ├── exceptions.py      # exception hierarchy           (planner — shared contract)
│   │   ├── data/
│   │   │   ├── schema.py      # Stage enum, RETAINED_VARIABLES (Table 3), LeakageGuard   (planner)
│   │   │   ├── arms.py        # LendingPolicy, ArmSpace        (planner — shared contract)
│   │   │   ├── segmentation.py# Segmenter                      (planner — shared contract)
│   │   │   ├── loader.py      # LendingClubLoader (DuckDB)     [agent: data]
│   │   │   ├── synthetic.py   # SyntheticLendingClub           [agent: data]
│   │   │   ├── outcomes.py    # default flag / recovery / realized revenue derivation [agent: data]
│   │   │   └── panel.py       # LoanStatePanel (MCTM panel reconstruction)            [agent: data]
│   │   ├── models/
│   │   │   ├── beta.py        # BetaPosterior value object     (planner)
│   │   │   ├── hierarchical.py# HierarchicalBetaBernoulli      [agent: risk]
│   │   │   ├── recovery.py    # BetaMoMRecovery                [agent: risk]
│   │   │   ├── two_stage.py   # TwoStageBetaBeta facade        [agent: risk]
│   │   │   └── mctm.py        # MarkovChainTransitionMatrix    [agent: risk]
│   │   ├── rewards/
│   │   │   ├── amortization.py# installment, schedule, WAL     [agent: risk]
│   │   │   └── compound.py    # CompoundRewardFunction         [agent: risk]
│   │   ├── bandits/
│   │   │   ├── arm_state.py   # GaussianArmPosterior (Welford) [agent: decision]
│   │   │   ├── base.py        # Policy protocol                [agent: decision]
│   │   │   ├── thompson.py    # GaussianThompsonSampling       [agent: decision]
│   │   │   ├── contextual.py  # ContextualThompsonSampling     [agent: decision]
│   │   │   └── baselines.py   # EpsilonGreedy, UCB1, Greedy, RandomPolicy, HistoricalPolicy [agent: decision]
│   │   ├── monitoring/
│   │   │   ├── psi.py         # PSICalculator                  [agent: decision]
│   │   │   └── sentinel.py    # PSISentinel + CircuitBreaker   [agent: decision]
│   │   ├── shadow/
│   │   │   └── engine.py      # ShadowLendingEngine            [agent: simulation]
│   │   ├── simulation/
│   │   │   ├── events.py      # event dataclasses              [agent: simulation]
│   │   │   ├── store.py       # EventStore (append-only JSONL), Checkpoint [agent: simulation]
│   │   │   ├── feedback.py    # DelayedFeedbackQueue           [agent: simulation]
│   │   │   └── runner.py      # SimulationRunner (resumable)   [agent: simulation]
│   │   └── evaluation/
│   │       ├── predictive.py  # Brier, AUC, ECE, MAE/RMSE, transition log-lik [agent: risk]
│   │       ├── policy.py      # reward/regret/selection/profitability/inclusion [agent: decision]
│   │       └── report.py      # EvaluationReport               [agent: simulation]
│   └── tests/
├── api/                       # Django 6 + DRF project (`config` settings pkg, `apps/*`)
├── frontend/                  # React 19 + Vite + TypeScript + Tailwind + Recharts
├── deploy/                    # docker/, k8s/, chaos/
├── docs/                      # SPEC.md (this), MATH.md, DATA.md, API.md, DEPLOY.md
├── scripts/                   # download_data.sh, make_synthetic.py, run_pipeline.py
└── data/                      # raw/ processed/ (gitignored); artifacts/ runs/ (gitignored)
```

Python: **3.14** venv at `.venv` (3.15 has no duckdb wheel). Installed: numpy 2.5, scipy 1.18, pandas 3.0 (copy-on-write + string dtype are ON by default — never rely on chained assignment), duckdb 1.5, pyarrow 26, Django 6.1, DRF 3.18, drf-spectacular, django-cors-headers, django-filter, pytest, pytest-django, hypothesis, scikit-learn 1.9 (metrics only), ruff. Node 24 + pnpm for the frontend.

Conventions: type hints everywhere, dataclasses for value objects, `numpy.random.Generator` passed explicitly (no global RNG), every public class has a docstring that cites the proposal equation/section it implements, custom exceptions from `bayesian_credit.exceptions` for numerical boundary conditions (§3.5 "comprehensive exception handling"). `ruff check` clean. Tests with pytest; property tests with hypothesis where cheap.

---

## 1. Data contract (Appendix F)

### 1.1 Stages and leakage rule (Table 3 / F.3)

Every retained variable has exactly one `Stage`:

| Stage | Columns |
|---|---|
| `PRE_DECISION` | annual_inc, annual_inc_joint, application_type, dti, dti_joint, emp_length, fico_range_low, fico_range_high, home_ownership, purpose, verification_status, verified_status_joint, delinq_2yrs, delinq_amnt, inq_last_6mths, inq_last_12m, open_acc, total_acc, revol_bal, revol_util, installment, mort_acc, pub_rec, pub_rec_bankruptcies, collections_12_mths_ex_med, num_accts_ever_120_pd, pct_tl_nvr_dlq, num_sats, tot_coll_amt, tot_cur_bal, tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit, bc_util, all_util, grade, sub_grade |
| `POLICY` | loan_amnt, int_rate, term |
| `DISBURSEMENT` | funded_amnt, issue_d |
| `INTERIM` | last_fico_range_low, last_fico_range_high, last_pymnt_amnt, last_pymnt_d, out_prncp, out_prncp_inv, num_tl_30dpd, num_tl_90g_dpd_24m, mths_since_last_delinq, next_pymnt_d |
| `TERMINAL` | loan_status, recoveries, collection_recovery_fee, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee |

Plus `loan_id` (LC `id`) kept **only** as a row key (never a feature). No free text, URLs, zip, state, emp_title, member_id (F.4 data minimization).

**Hard rule:** `BorrowerContext` (the thing a bandit sees at decision time) may contain only `PRE_DECISION` columns. `LeakageGuard.assert_stage(columns, allowed=...)` raises `LeakageError` otherwise. The runner constructs contexts through the guard — this is tested.

### 1.2 Canonical `LoanTable`

A pandas DataFrame with the columns above, canonical dtypes:
- money/ratios → float64; `int_rate` as a **fraction** (0.1299 not "12.99%"); `revol_util`, `bc_util`, `all_util`, `pct_tl_nvr_dlq` as fractions too.
- `term` → int (36/60); `emp_length` → float years (0–10, "< 1 year"→0, "10+ years"→10, NaN kept).
- `issue_d`, `last_pymnt_d`, `next_pymnt_d` → `datetime64[ns]` (LC format "Dec-2015").
- categoricals → pandas string dtype: grade, sub_grade, home_ownership, purpose, verification_status, verified_status_joint, application_type, loan_status.

Derived columns (added by `outcomes.derive_outcomes(df, snapshot_date, cfg)`), all prefixed `y_`:
- `y_resolved` (bool): terminal outcome known. loan_status ∈ {Fully Paid, Charged Off, Default, "Does not meet the credit policy. Status:Fully Paid", "...Charged Off"} → True; {Current, In Grace Period, Late (16-30 days), Late (31-120 days)} → False.
- `y_default` (float, NaN if unresolved): 1 for Charged Off / Default variants, 0 for Fully Paid variants.
- `y_recovery_rate` (float, NaN unless default): **[decision]** `clip((total_rec_prncp + recoveries − collection_recovery_fee) / funded_amnt, 0, 1)` — the fraction of principal recovered by any means given default. This makes the proposal's `E[L] = L·P(D)·(1−E[R])` exactly the expected principal loss. Alternative `recovery_basis="post_chargeoff"` = `(recoveries − fee) / (funded_amnt − total_rec_prncp)`.
- `y_realized_revenue` = total_rec_int + total_rec_late_fee.
- `y_realized_loss` = max(funded_amnt − total_rec_prncp − (recoveries − collection_recovery_fee), 0) (0 for fully paid).
- `y_months_on_book` = months(issue_d → last_pymnt_d or snapshot), capped at term.
- `y_resolution_date` (datetime, NaT if unresolved): fully paid → last_pymnt_d; default → last_pymnt_d + `chargeoff_lag_months` (**[decision]** 4; LC charges off at 120+ DPD) — if last_pymnt_d is NaT use issue_d + lag.
- `y_historical_arm` (int): nearest arm in the `ArmSpace` to (loan_amnt, int_rate, term).
- `y_group` (string): segmenter key.

### 1.3 Sources
- `LendingClubLoader(path, arm_space, segmenter)` — reads the Kaggle CSV/CSV.GZ with DuckDB, **projecting only retained columns** (data minimization), parsing percent strings and "Mon-YYYY" dates, filtering `policy_code == 1` if present, dropping rows with null loan_amnt/int_rate/term/issue_d/loan_status. Supports `max_rows`, `sample_fraction`, `seed`, `issue_from`, `issue_to`. Returns `LoanTable`. Also `to_parquet(path)` cache so the 1.6 GB CSV is parsed once.
- `SyntheticLendingClub(seed).generate(n, start="2012-01", end="2018-12", snapshot="2018-12-31")` — realistic LC-schema generator used for tests, CI, and when the Kaggle file is absent. Must reproduce the key dependencies: sub_grade → int_rate (≈ 5.3% for A1 → 30% for G5 with noise), FICO/DTI/grade → default probability (A ≈ 3% … G ≈ 35%), default → recovery Beta(≈1.2, 6) + partially repaid principal, term 36/60 mix (~75/25), amounts 1k–40k in 25s, issue month arrivals growing over time, loan status consistent with issue date + term + snapshot (recent loans "Current"), late fees on ~8% of fully paid loans, a configurable **distribution shift** (`shift_at="2016-06", shift={"fico_range_low": -25, "dti": +5}`) so the PSI Sentinel has something to catch. All retained columns present.

### 1.4 Segmentation (`Segmenter`) — "borrower group g"
`Segmenter(dimensions: list[str])` maps a context row → group key string `"B|debt_consolidation|RENT"`. Supported dimensions (each a pure function of PRE_DECISION columns): `grade`, `sub_grade`, `purpose_group` (debt_consolidation / credit_card / other), `home_ownership` (RENT/OWN/MORTGAGE/OTHER), `fico_band` (<660, 660–699, 700–739, 740–779, 780+), `dti_band` (<10, 10–20, 20–30, 30+), `income_band` (<40k, 40–75k, 75–120k, 120k+), `emp_band` (<2, 2–5, 6–9, 10+), `verification_status`, `application_type`. **[decision]** default `["grade", "purpose_group", "home_ownership"]`. `refine()` returns a new Segmenter with the next dimension from `refinement_order` appended (used by the breaker's "resegment" action). `coarsen()` is the inverse.

### 1.5 Arm space (`ArmSpace`) — "A_k = (L_k, r_k)" (§3.3) with term (Table 7)
`LendingPolicy(amount: float, rate: float, term: int)`; `ArmSpace.from_grid(amounts, rates, terms, include_null=True)`. **[decision]** default grid amounts `[5k,10k,15k,20k,25k,30k,35k,40k]`, rates `[0.07,0.10,0.13,0.16,0.19,0.22,0.25,0.28]`, terms `[36,60]` → 128 arms + arm 0 = **null arm** `(0, 0, 0)` = "decline", reward 0 (the system must be able to choose not to lend, §1.5). `nearest(amount, rate, term) -> int`, `eligible(context, cfg) -> ndarray[bool]` (affordability guard: `installment(arm) ≤ cfg.max_payment_to_income × annual_inc/12`; null arm always eligible), `to_frame()`. *Addendum (§7.2):* `eligible(annual_inc, *, dti=None, reference_rate=None)` / `eligible_many(...)` apply the full set of lending guardrails (affordability, amount-to-income, total DTI, rate corridor around the sub-grade reference rate).

---

## 2. Models (§3.3) — equation → code

### 2.1 `BetaPosterior(alpha, beta)` (planner-written value object)
mean, var, `sample(rng, size)`, `hdi(0.95)`, `update(successes, failures)`, `pdf_grid(n)`; raises `InvalidPosteriorError` for non-positive params.

### 2.2 Stage 1 — `HierarchicalBetaBernoulli` (§3.3 eqs. p₀ ~ Beta(α₀,β₀); p_g ~ Beta(α_g,β_g); p_{g,a} ~ Beta(α_{g,a},β_{g,a}); posterior Beta(α_{g,a}+d, β_{g,a}+n); mean (α+d)/(α+β+d+n))

Three levels: global → group g → leaf (g, a). **[decision]** priors are built top-down by empirical-Bayes shrinkage: the child's prior is the parent's posterior mean with concentration κ: `α_g = κ₁·m₀, β_g = κ₁·(1−m₀)`, `α_{g,a} = κ₂·m_g, β_{g,a} = κ₂·(1−m_g)` (*addendum §7.1:* the leaf mean is tilted to `σ(logit(m_g) + t(g,a))` by the policy-effect model). κ configurable (`HierarchyConfig.kappa_group=20, kappa_leaf=10`) or `"auto"` = Beta-binomial method-of-moments across children `κ = m(1−m)/Var(child rates) − 1` clipped to `[1, 1000]`.

```python
class HierarchicalBetaBernoulli:
    def __init__(self, cfg: HierarchyConfig): ...
    def fit(self, groups: Sequence[str], arms: Sequence[int], defaults: Sequence[int]) -> Self
    def update(self, group: str, arm: int, defaulted: bool) -> None      # online, all 3 levels
    def posterior(self, group: str, arm: int) -> BetaPosterior            # leaf; unseen leaf → its prior (shrunk group)
    def group_posterior(self, group) -> BetaPosterior; def global_posterior(self) -> BetaPosterior
    def p_default(self, group, arm) -> float                              # posterior mean
    def summary(self) -> pd.DataFrame   # level, key, alpha, beta, n_obs, defaults, mean, hdi_low, hdi_high, shrinkage_weight
    def counts(self, group, arm) -> tuple[int, int]
```
`shrinkage_weight = n/(n+κ)` — the fraction of the posterior mean coming from data vs parent. Exposed so the UI can visualise shrinkage.

### 2.3 Stage 2 — `BetaMoMRecovery` (R_{g,a}|D=1 ~ Beta(α^R, β^R); α^R = μ(μ(1−μ)/σ² − 1); β^R = (1−μ)(μ(1−μ)/σ² − 1))
Fit on defaulted loans' `y_recovery_rate` per leaf, falling back to group, then global, when `N < cfg.min_obs` (**[decision]** 5) or the MoM is invalid (`σ² ≥ μ(1−μ)` ⇒ raise/handle `NumericalInstabilityError` internally and fall back). Guards: clip μ into `[1e-4, 1−1e-4]`; if σ² < 1e-8 use κ = cfg.max_concentration (1000). Same `posterior / expected_recovery / sample / update / summary` shape as 2.2 (online `update` keeps running mean/variance per leaf via Welford).

### 2.4 `TwoStageBetaBeta` facade
```python
@dataclass
class RiskEstimate:
    group: str; arm: int
    p_default: BetaPosterior; recovery: BetaPosterior
    @property expected_loss_fraction -> float   # E[p]·(1−E[R])
class TwoStageBetaBeta:
    def fit(self, table: pd.DataFrame) -> Self            # uses y_resolved rows; y_default; y_recovery_rate; y_group; y_historical_arm
    def risk(self, group, arm) -> RiskEstimate
    def sample(self, group, arm, n, rng) -> tuple[np.ndarray, np.ndarray]   # (p draws, R draws)
    def update(self, group, arm, defaulted: bool, recovery_rate: float | None) -> None
    def predict_table(self, table) -> pd.DataFrame        # p_default mean/HDI, E[R] per row (for evaluation)
```

### 2.5 `MarkovChainTransitionMatrix` (S = {C, D, R, L}; P with absorbing R, L; π_t = π₀Pᵗ; P(Default,t) = π_D(t) + π_L(t); limit π_∞ = [0,0,π_R,π_L])
States `State.CURRENT=0, DELINQUENT=1, RECOVERED=2, LOSS=3`. Proposal matrix has **structural zeros** P[C,R]=P[C,L]=0 (a loan leaves Current only via Delinquent; "Recovered" = repayment following delinquency). `TransitionStructure.PROPOSAL` enforces those zeros (payoff of a never-delinquent loan is treated as right-censoring in estimation and as the "maturity collapse" in prediction). **[decision]** `TransitionStructure.EXTENDED` additionally allows C→R (prepayment/payoff). Default = `PROPOSAL`; both selectable in API/UI.

```python
class MarkovChainTransitionMatrix:
    def __init__(self, structure=TransitionStructure.PROPOSAL, prior_strength: float = 1.0): ...
    @classmethod from_matrix(cls, P, structure) ; def fit_counts(self, counts: np.ndarray) -> Self   # Dirichlet(prior+counts) rows; prior_strength=0 → MLE
    matrix: np.ndarray (posterior mean, validated: rows sum to 1, absorbing rows, structural zeros)
    def sample_matrix(self, rng) -> np.ndarray             # Dirichlet row draws (transition uncertainty)
    def propagate(self, pi0, t) -> np.ndarray; def trajectory(self, pi0, horizon) -> np.ndarray[(horizon+1,4)]
    def p_default(self, t, pi0=START_CURRENT) -> float     # π_D(t)+π_L(t)
    def absorption_probabilities(self) -> np.ndarray[(2,2)]  # B = N·R, N=(I−Q)⁻¹ ; rows C,D ; cols R,L
    def expected_time_to_absorption(self) -> np.ndarray[(2,)]
    def p_recover_given_delinquent(self, t) / p_loss_given_delinquent(self, t)   # P(R|D,t), P(L|D,t)
    def maturity_collapse(self, pi) -> np.ndarray          # at maturity: C→R; D splits by p_DR:p_DL  (proposal: "delinquents collapse")
    def default_timing_cdf(self, horizon) -> np.ndarray    # F(m) = P(absorbed into L by m | eventually L), used by the reward
    def log_likelihood(self, counts) -> float; def row_prediction_error(self, counts) -> np.ndarray
```
Raises `InvalidTransitionMatrixError` for singular (I−Q) or invalid rows.

### 2.6 `LoanStatePanel` — reconstructing monthly states from the LC snapshot (G.2)
LC is a snapshot, not a panel; the state path is reconstructed deterministically per loan (documented heuristic, **[decision]**):
- Fully Paid: C for each month to payoff; then R (EXTENDED) / censored (PROPOSAL). If `total_rec_late_fee > 0` insert one cured delinquency episode (C→D, D→C) — late fees are only charged when a payment was late.
- Charged Off / Default: C until the month of last payment, then D for `chargeoff_lag_months` (4), then L. No payments ever → D from month 1.
- Current: C→C for months on book (censored); cured episode if late fees > 0.
- In Grace Period / Late (16–30): …C→D, in D (censored). Late (31–120): C→D then D→D × (months since last payment − 1).
Outputs `counts: np.ndarray(4,4)` and optionally per-loan compressed paths (for holdout transition log-likelihood). Also provides `interim_state(loan_row, month)` so the simulation can emit interim observations month by month.

### 2.7 Rewards (R_t = Revenue_t − Expected Loss_t − Funding Cost_t;  E[L_t] = L_t·P(D|X,A_t)·(1 − E[R|X,A_t]))
`amortization.py`: `installment(L, r, T)`, `schedule(L, r, T) -> DataFrame(month, interest, principal, balance)`, `weighted_average_life_years(L, r, T)`, `total_interest(L, r, T)`.

`CompoundRewardFunction(cfg: RewardConfig)` with `funding_rate` (**[decision]** 0.04/yr), `risk_aversion λ` (0 default; risk-adjusted score = mean − λ·std, the "risk appetite" knob), `exposure_basis="full"` (PDF: L_t) | `"outstanding"`:
- `expected(policy, risk: RiskEstimate, mctm=None) -> RewardDistribution(mean, std, components)`; closed form:
  `Revenue = Σ_m I_m · (1 − E[p]·F(m))` with `F` = `mctm.default_timing_cdf(T)` (uniform `m/T` if no MCTM — defaults happen mid-life on average), `E[Loss] = L·E[p]·(1−E[R])`, `Funding = funding_rate · L · WAL`. Variance from Beta moments (p and R independent) — document the algebra in the docstring.
- `sample(policy, p_draws, R_draws, mctm) -> np.ndarray` Monte Carlo rewards (for predictive intervals / TS seeding).
- `terminal(row) -> float` realized historical reward: `y_realized_revenue − y_realized_loss − funding_rate·funded_amnt·min(y_months_on_book, term)/12 · 0.5`… **no** — use the exact schedule: funding cost on realized WAL = `funding_rate · Σ_m balance_{m−1}/12` over months actually on book. Document.
- `interim(policy, state: State, month, risk, mctm) -> float` compound signal: interest realized to date (schedule × months in C) + expected remainder given current state (via `mctm.propagate` from that state) − expected loss given state − funding. This is the "interim repayment behaviour" surrogate (§1.7.2, §2.2.4).
- Null arm → 0.

### 2.8 Bandit (θ_k ~ N(μ_k, σ_k²); A_t = argmax θ̃_k; μ ← μ + (r−μ)/(n+1); n ← n+1; σ² ← ((n−1)σ²_old + (r−μ_old)(r−μ_new))/n)
`GaussianArmPosterior(mu, sigma2, n=0)` — `update(r)` is **exactly** the proposal's incremental mean + online (Welford) variance; n=1 leaves σ² at the prior value (variance undefined). `sample(rng)` ~ N(μ, σ²) (`posterior_scale="proposal"`, default) or N(μ, σ²/n) (`"standard_error"`). `ucb(c)` helper for the UCB baseline.

`Policy` protocol: `select(context: BorrowerContext, eligible: np.ndarray[bool], rng) -> Decision(arm, group, scores)`, `update(decision, reward, kind: "interim"|"terminal")`, `state() -> dict` (serialisable), `freeze()/unfreeze()` (breaker), `snapshot()/restore()`.

`ContextualThompsonSampling(arm_space, segmenter, warm_start, cfg)` — one `GaussianThompsonSampling` per group (the hierarchy makes it *contextual*; context → g via the Segmenter). Cold-start for a new group: `warm_start="risk_model"` (**[decision]** default: μ_k⁰ = reward_fn.expected(arm, two_stage.risk(g,k)).mean, σ_k⁰ = that std — "the Two-Stage Beta-Beta is used as the sampling distribution for the automated decisions"), `"global"` (copy the pooled bandit's posteriors), `"flat"` (μ=0, σ=cfg.prior_sigma). Interim rewards update with weight `cfg.interim_weight` (default 0.5 — counts as a fractional observation; document the generalised Welford with weights) or are ignored when 0.

Baselines (same protocol): `EpsilonGreedy(ε=0.1)`, `UCB1(c=2)`, `Greedy`, `RandomPolicy`, `HistoricalPolicy` (always the loan's `y_historical_arm` — the reference policy, Table 8).

### 2.9 PSI Sentinel (G.5, Table 9: <0.10 normal; 0.10–0.25 watch; >0.25 breaker)
`PSICalculator(reference_df, features, n_bins=10, eps=1e-4)` — numeric bins from reference deciles, categoricals by level; `psi(current_df) -> PSIResult(per_feature: dict[str, float], bucket_tables, aggregate_max, aggregate_mean)`. PSI = Σ (a_i − e_i)·ln(a_i/e_i).

`CircuitBreaker` state machine: `CLOSED → OPEN` (aggregate ≥ 0.25) → `HALF_OPEN` (after human `acknowledge(action)`) → `CLOSED` after `cfg.recovery_readings` consecutive normal readings; `WATCH` is a flag, not a state. Actions: `resume` (keep policy, unfreeze), `resegment` (refine Segmenter, refit risk model on data to date, re-warm-start bandits), `rollback` (restore champion snapshot). While OPEN: policies are frozen (no `update`), `select` falls back to `cfg.fallback="champion_snapshot"|"decline"|"historical"`. `PSISentinel.evaluate(window_df, epoch) -> SentinelReading` and emits events. `breaker_mode` in simulation: `"pause"` (stop and wait for API acknowledge), `"auto_ack"` (acknowledge with `cfg.auto_action` after `cfg.auto_ack_epochs`), `"off"` (monitor only).

### 2.10 Shadow engine (G.4, Table 8)
`ShadowLendingEngine(two_stage, mctm, reward_fn, arm_space, segmenter, n_samples=2000)`:
- `evaluate(context, historical_policy=None, realized_reward=None, rng=None) -> ShadowReport` with per-arm rows (arm, amount, rate, term, p_default mean/hdi, expected_recovery, reward mean/std, q05/q50/q95, prob_positive, eligible), `recommended_arm` (argmax risk-adjusted mean among eligible), `historical_arm` + its expected & realized reward, `uplift_expected`, `uplift_vs_realized`, `lendable: bool` (∃ eligible arm with prob_positive ≥ cfg.lendable_threshold 0.6), `group`.
- `evaluate_portfolio(table, max_loans) -> (DataFrame per loan, PortfolioSummary)`: total expected profit historical vs shadow, by group/grade; counts of `rescued` (historical realized loss → shadow positive), `upsized` (recommended amount > historical with positive reward), `repriced_down/up`, `declined`; inclusion metrics (share lendable by grade; mean offered rate by grade vs historical).

### 2.11 Simulation (§3.6, G.3) — the replay environment
Epoch = issue month. `SimulationRunner(cfg: SimulationConfig, table, two_stage, mctm, reward_fn, policies: dict[str, Policy], sentinel, store)`:
1. Reference window = loans with `issue_d < cfg.start` (fit set). Stream = `cfg.start ≤ issue_d ≤ cfg.end`, optionally `max_loans` sampled with `seed`.
2. Per epoch: (a) sentinel evaluates the epoch's arrivals (window of `psi_window_months`) → may trip breaker; (b) release due feedback from `DelayedFeedbackQueue` — interim (monthly, from `LoanStatePanel.interim_state`, up to `interim_months`) and terminal (at `y_resolution_date`) — and apply `policy.update`; the two-stage risk model is also updated with **realized** historical outcomes as they resolve (`feedback_mode` below); (c) for each arriving borrower: context (LeakageGuard) → each policy selects an arm → schedule feedback.
3. **Feedback modes [decision]:** `"simulated"` (default) — reward for the chosen arm is drawn from the *environment model* (two-stage posterior predictive + MCTM) at resolution time: Bayesian simulation (§3.6); `"replay"` — classic rejection-sampling replay (Li et al. 2011): only when the chosen arm == `y_historical_arm` is the realized reward used, else no update; `"hybrid"` — realized if matching, simulated otherwise.
4. Metrics per epoch per policy: n_decisions, cumulative expected reward, cumulative realized-where-known, pseudo-regret vs oracle (oracle = per-group argmax arm under the **hindsight** model fitted on all resolved loans), rolling mean reward, arm selection counts, decline rate, breaker state, PSI per feature, mean offered amount/rate.
5. Event-sourced & resumable: append-only JSONL `EventStore` (`BorrowerArrived, PolicySelected, FeedbackReleased, PosteriorUpdated(sampled), SentinelEvaluated, BreakerTripped, ReviewAcknowledged, CheckpointWritten, EpochCompleted`); checkpoint every `checkpoint_every` epochs (policies' `state()`, two-stage/MCTM state, queue, rng bit-generator state, epoch). `SimulationRunner.resume(run_dir)` continues; **determinism test**: kill-and-resume equals uninterrupted for the same seed. This is the G.6 "state recovery" experiment in software.
6. `EventSink` interface with `JSONLSink` default; `KafkaSink` optional adapter (import guarded).

### 2.12 Evaluation (Table 11)
`predictive.py`: `brier`, `roc_auc`, `expected_calibration_error(n_bins=10)` + reliability table, `log_loss`, `mae/rmse`, `transition_log_likelihood`, `transition_row_error`. `policy.py`: cumulative reward/regret curves, convergence (rolling), selection frequency matrix (amount × rate per term), profitability vs historical, inclusion table. `report.py`: `EvaluationReport.build(...) -> dict` grouped by Table 11 dimensions + runtime (CPU seconds, peak RSS, wall time → "computational feasibility").

Temporal split **[decision]**: fit on `issue_d < 2015-01-01`, stream 2015-01 → 2018-12 by default (configurable). Predictive metrics only on resolved loans.

---

## 3. API contract (Django + DRF, base `/api/v1/`, JSON, CORS open in dev)

Background work runs as **subprocess jobs** (`python manage.py run_job <job_id>`); a `Job` row tracks `status (queued|running|succeeded|failed|cancelled)`, `progress (0–1)`, `message`, `log_tail`, `started_at`, `finished_at`, `pid`. In k8s the same command is a Job pod. Artifacts (pickled core objects, parquet, run dirs) live under `ARTIFACT_ROOT` (default `./data/artifacts`).

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /health/` | | `{status, version, python, core_version}` |
| `POST /datasets/` | `{name, source: "synthetic"\|"lending_club", path?, n?: int, seed?, max_rows?, sample_fraction?, issue_from?, issue_to?, shift?: {...}}` | Dataset (+ job) |
| `GET /datasets/`, `GET /datasets/{id}/` | | `{id, name, source, status, n_loans, issue_min, issue_max, resolved_share, default_rate, status_counts, created_at, job}` |
| `GET /datasets/{id}/variables/` | | Table 3 rows `{name, stage, primary_use, present, dtype, null_share}` |
| `GET /datasets/{id}/summary/` | `?features=` | histograms/category counts for context features + monthly arrivals |
| `GET /datasets/{id}/loans/` | `?limit&offset&grade&purpose&home_ownership&resolved&search` | paginated rows (context + policy + outcome fields) |
| `POST /fits/` | `{dataset, name, segmentation: {dimensions}, arm_space: {amounts, rates, terms}, hierarchy: {kappa_group, kappa_leaf, alpha0, beta0}, recovery: {min_obs}, mctm: {structure, prior_strength, chargeoff_lag_months}, reward: {funding_rate, risk_aversion}, train_end}` | Fit (+ job) |
| `GET /fits/`, `GET /fits/{id}/` | | config + status + `metrics` (Brier, AUC, ECE, log loss, recovery MAE/RMSE, transition log-lik, n_train, n_test, fit_seconds) |
| `GET /fits/{id}/groups/` | `?level=global\|group\|leaf&group=` | hierarchy summary rows (2.2 `summary()`) |
| `GET /fits/{id}/risk/` | `?group=&arm=` | `RiskEstimate` + pdf grids |
| `GET /fits/{id}/calibration/` | | reliability bins + ROC points |
| `GET /fits/{id}/mctm/` | `?structure=&horizon=60&start=current\|delinquent` | matrix, counts, absorption, expected steps, trajectory, p_default curve |
| `GET /fits/{id}/arms/` | | arm table with pooled expected reward per arm |
| `POST /fits/{id}/shadow/evaluate/` | `{loan_id}` or `{context: {...}, historical_policy?: {amount, rate, term}}` | ShadowReport |
| `POST /fits/{id}/shadow/portfolio/` | `{max_loans, seed, filters?}` | job → `GET /fits/{id}/shadow/portfolio/{job_id}/` summary + per-grade tables |
| `POST /simulations/` | `{fit, name, policies: [...], feedback_mode, start, end, max_loans, seed, interim_months, interim_weight, psi: {features, window_months, thresholds}, breaker_mode, checkpoint_every}` | Simulation (+ job) |
| `GET /simulations/`, `GET /simulations/{id}/` | | status, progress, current_epoch, breaker_state, headline metrics |
| `GET /simulations/{id}/timeseries/` | `?policy=` | per-epoch metric rows |
| `GET /simulations/{id}/arms/` | `?policy=&epoch=` | selection counts + posterior μ/σ/n per arm |
| `GET /simulations/{id}/events/` | `?type=&after=&limit=` | event rows |
| `GET /simulations/{id}/sentinel/` | | PSI per feature per epoch + breaker transitions |
| `GET /simulations/{id}/report/` | | EvaluationReport |
| `POST /simulations/{id}/{pause\|resume\|cancel\|chaos}/` | | `chaos` SIGKILLs the worker; `resume` restarts from the last checkpoint |
| `GET /sentinel/reviews/` `?simulation=` ; `POST /sentinel/reviews/{id}/acknowledge/` `{action, note}` | | human-in-the-loop review queue |
| `GET /jobs/{id}/` | | job status |

OpenAPI at `/api/schema/`, Swagger at `/api/docs/`.

---

## 4. Frontend (React 19 + Vite + TS + Tailwind v4 + Recharts + React Router + TanStack Query)
Pages: **Overview** (closed-loop diagram with live status, KPI tiles), **Data** (ingest; Table 3 stages; feature distributions; arrivals), **Risk Models** (fit form; hierarchy table with posterior mean + HDI bars + shrinkage; calibration & ROC; recovery; metrics), **MCTM** (4×4 heatmap, state-occupancy trajectory, P(default|t), absorption probs, structure toggle), **Bandit Lab** (simulation form; live cumulative reward/regret per policy; rolling reward; arm-selection heatmap amount×rate; posterior μ over time; breaker overlays; pause/resume/chaos), **Shadow Engine** (borrower table → context card → arm heatmap of expected reward → recommended vs historical → reward percentiles → portfolio summary), **Sentinel** (PSI timelines with 0.10/0.25 bands; breaker state; review queue with acknowledge form), **Evaluation** (Table 11 grid; JSON export). Dark/light, responsive, polling for running jobs. Read the `dataviz` skill before charting.

---

## 5. Deploy
`deploy/docker/Dockerfile.api` (python:3.14-slim; gunicorn), `Dockerfile.frontend` (node build → nginx w/ `/api` proxy), `docker-compose.yml` (api, frontend, optional postgres; volumes for data/artifacts). `deploy/k8s/`: namespace, ConfigMap, PVC, api Deployment/Service, frontend Deployment/Service, Traefik Ingress (k3s default), simulation Job template, resource requests sized for consumer nodes. `deploy/chaos/chaos-monkey.sh` deletes the simulation pod at random intervals; `README` shows resume-from-checkpoint recovery (G.6). Prometheus scrape annotations on the API; `/metrics` via a minimal endpoint.

---

## 6. Definition of done
- `pytest` green in `core/` and `api/`; `ruff check` clean; `pnpm build` + `tsc --noEmit` clean.
- `scripts/run_pipeline.py --source synthetic` runs ingest → fit → shadow portfolio → simulation (TS vs baselines) → report in < 3 min on a laptop, and the same with `--source lending_club --max-rows 200000` on the Kaggle file.
- Kill-and-resume determinism test passes.
- README documents: math→code map, data stages/leakage rule, how to get the Kaggle data, run locally, docker, k8s, and the documented **[decision]** assumptions.

---

## 7. Documented decisions (addenda after the first end-to-end runs)

The first pipeline runs (synthetic and Lending Club, 2026-10-16) showed a modelling
degeneracy: with every leaf shrunk to its group mean, an arm nobody in the group ever
received (grade A at 28 %) is exactly as risky as the group average, so the reward grows
with amount and rate without a risk penalty and the shadow engine and the bandits converge
on "largest affordable loan at the highest rate" (shadow uplift ≈ +1,500 %, lendable
100 %). The **[decision]**s below address it; each is configurable and documented in
[MATH.md](MATH.md) §2.6–2.9 (assumptions S1–S6, G1–G5, W7–W9, R6) and
[SIMULATION.md](SIMULATION.md) §3.

### 7.1 Structured leaf prior (policy effect) — `models/policy_effect.py`
The proposal leaves `α_{g,a}, β_{g,a}` open; following §2.2.1 (Scott 2010: a structured
model across policy factors instead of independent cells) the leaf prior mean is
`m_{g,a} = σ(logit(m_g) + t(g,a))`, `t(g,a) = clip(β·(x_{g,a} − x̄_g), ±3)`, with
`x = [int_rate, 1[term=60], pti, log(loan_amnt)]` and `β` from an offset logistic
regression `logit P(D=1) = logit(m_g) + β·(x_i − x̄_g)` on the resolved training loans
(elapsed-term loans when ≥ 20k), features centred within group, Newton–Raphson with a
1e-3 ridge, `β = 0` fallback on numerical failure. κ_leaf and the conjugate update are
unchanged. Config: `HierarchyConfig.policy_effect = True`, `HierarchyConfig.policy_effect_cfg
= PolicyEffectConfig(features, ridge=1e-3, max_abs_logit_shift=3.0, min_rows=5000,
min_elapsed_rows=20000, max_iter=25, pti_cap=1.5)`. `TwoStageBetaBeta.from_config(cfg,
arm_space)` fits and installs it; `HierarchicalBetaBernoulli(cfg, leaf_prior_tilt=fn)` /
`set_leaf_prior_tilt(fn)` is the hook.

### 7.2 Lending guardrails — `data/arms.py`, `data/reference_rates.py`
Eligible arms satisfy `installment ≤ 0.20·annual_inc/12`, `amount ≤ 0.50·annual_inc`,
`dti/100 + 12·installment/annual_inc ≤ 0.43` and `|rate − reference_rate| ≤ 0.05`, the
reference rate being the training mean `int_rate` of the borrower's sub-grade (→ grade →
global; `ReferenceRates`, saved as `fit/reference_rates.json`). The defaults follow US
consumer-underwriting conventions. 43 % is the QM rule's DTI ceiling (12 CFR
1026.43(e)(2)(vi)); 20 % payment-to-income and 50 % amount-to-income are unsecured-lending
conventions. A guardrail with an unknown input is skipped; the null arm is always eligible;
if no lending arm survives, the lending arm closest to the reference rate with the smallest
installment is kept. Config: `ArmSpaceConfig.max_payment_to_income = 0.20`,
`rate_corridor = 0.05`, `max_amount_to_income = 0.50`, `max_total_dti = 0.43` (`None`
switches one off). The shadow engine, every simulated policy and the hindsight oracle use
the same eligibility (the historical policy keeps its own arm); fits saved before reference
rates existed run without the corridor.

### 7.3 Prepayment-aware expected reward — `rewards/realization.py`, `rewards/compound.py`
The proposal's revenue term assumes the full contractual schedule; Lending Club's fully-paid
loans realise ≈ 82 % (36 m) / 72 % (60 m) of their scheduled interest. `InterestRealization`
(fitted in `fit_models` on the fully-paid training loans whose term elapsed, by grade ×
term → term → global, entries clipped to [0.3, 1.0], saved as
`fit/interest_realization.json`) gives ρ, and `CompoundRewardFunction.expected / sample /
interim(..., realization=ρ)` multiply the expected interest and the funding cost by ρ. Config:
`RewardConfig.prepayment_adjustment = "empirical"` (`None` = contract). The environment, the
hindsight oracle, the warm start and the shadow engine all use it; `terminal()` is unchanged.

### 7.4 Recovery by term — `models/recovery.py`
Stage 2's fallback chain is leaf (g, a) → **(g, term of a)** → g → global → prior, because a
60-month default has amortised less principal (LC: 0.38 vs 0.485 recovered). Config:
`RecoveryConfig.term_level = True`; the arm → term map comes from the arm space.

### 7.5 Unbiased hindsight yardstick — `simulation/runner.py`
The oracle / regret yardstick's risk model is fitted on the resolved loans whose term elapsed
by the snapshot (`SimulationConfig.hindsight_sample = "elapsed"`). Fitting it on every
resolved loan (`"resolved"`, the original protocol) inflates the 2015–2018 default rates
because only early defaulters have resolved: 26 % vs 15 % on elapsed loans. With prepayment
margins this bias turned every policy's expected reward negative. The hindsight MCTM still
uses the full panel.

### 7.6 Risk-appetite sweep — `shadow/engine.py`
`PortfolioSummary.by_risk_aversion` reports, for λ ∈ {0, 0.5, 1} (`risk_aversions=` of
`evaluate_portfolio` / `run_shadow_portfolio`), the shadow policy chosen by `mean − λ·std`
on the same arm tables and eligibility: expected profit, uplift, mean recommended amount and
rate, 36-month share, decline share. `RewardConfig.risk_aversion` stays 0.

