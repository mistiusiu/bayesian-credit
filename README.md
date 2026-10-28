# Bayesian Credit — a shadow lending engine

**Contextual Multi-Armed Bandits in Bayesian Sequential Decision-Making in Credit Risk Modeling**
Misati Randy Nyambane · BSc Data Science & Analytics · USIU-Africa · DSA4900

This repository is the research artifact described in the proposal
(`Data Science Reasearch Proposal on Bayesian Contextual Bandits.pdf`): an open-source
prototype that reframes credit risk from a static prediction problem into an
**adaptive Bayesian policy-optimisation problem**. Instead of asking only
*"will this borrower default?"* it asks *"given what we know now, how much should we lend,
at what price — and how should that policy change as evidence arrives?"*

```
Borrower Context → Bayesian Risk Estimation → Credit-State Prediction → Risk-Adjusted Reward
      → Policy Selection → Observed Outcomes → Posterior Update → Policy Adaptation
          ▲ PSI Sentinel (circuit breaker) runs across the loop
          ▲ Shadow lending engine runs alongside the historical policy
```

## What is inside

| Layer | Proposal § | Implementation |
|---|---|---|
| Hierarchical **Two-Stage Beta-Beta** risk model: default `p_{g,a}` (Beta-Bernoulli, global → group → group×policy shrinkage) and recovery `R_{g,a}\|D=1` (Beta, method of moments) | §3.3 | `core/bayesian_credit/models/{hierarchical,recovery,two_stage}.py` |
| **MCTM** — 4-state absorbing Markov chain {Current, Delinquent, Recovered, Absorbed Loss}; `π_t = π₀Pᵗ`, `P(Default,t)=π_D(t)+π_L(t)`, absorption via the fundamental matrix | §3.3, G.2 | `core/bayesian_credit/models/mctm.py`, panel reconstruction in `data/panel.py` |
| **Compound risk-adjusted reward** `R_t = Revenue_t − E[Loss]_t − Funding_t`, `E[L_t] = L_t·P(D\|X,A_t)·(1−E[R\|X,A_t])`, with MCTM default timing and interim (surrogate) rewards | §3.3, §1.7.2 | `core/bayesian_credit/rewards/` |
| **Contextual Thompson Sampling** over arms `A_k = (L_k, r_k, term)` with Gaussian posteriors `θ_k ~ N(μ_k, σ_k²)`, incremental mean and online (Welford) variance updates; ε-greedy / UCB1 / greedy / random / historical baselines | §3.3, G.3 | `core/bayesian_credit/bandits/` |
| **Shadow lending engine** — counterfactual evaluation of every arm for every borrower from posterior-predictive draws; uplift vs the historical policy; inclusion metrics | §3.6, G.4 | `core/bayesian_credit/shadow/` |
| **PSI Sentinel / Bayesian circuit breaker** — PSI < 0.10 normal, 0.10–0.25 watch, > 0.25 trip → human review (resume / re-segment / rollback) | §3.3, G.5 | `core/bayesian_credit/monitoring/` |
| **Replay simulation** — event-sourced, checkpointed, resumable stream of Lending Club borrowers with real delayed feedback | §3.6, G.6 | `core/bayesian_credit/simulation/` |
| Evaluation (Table 11): Brier, AUC, calibration, recovery MAE/RMSE, transition likelihood, regret, profitability, PSI, runtime | §3.7, G.7 | `core/bayesian_credit/evaluation/` |
| REST API (Django + DRF, OpenAPI) | §3.5 | `api/` |
| Research console (React + Vite + TypeScript) | — | `frontend/` |
| Containers, K3s/Talos manifests, chaos script | §3.5, G.6 | `deploy/` |

The equation → code map is in [`docs/MATH.md`](docs/MATH.md); the data contract (Table 3
stages, leakage rule, outcome definitions) in [`docs/DATA.md`](docs/DATA.md); the full
build specification in [`docs/SPEC.md`](docs/SPEC.md); the API in [`docs/API.md`](docs/API.md).

## Data

The empirical case study is the Lending Club accepted-loans file (Kaggle
`wordsforthewise/lending-club`, CC0; 2.26 M loans, 2007-06 → 2018-12). Only the variables
retained in Appendix F Table 3 are loaded (data minimisation), and each is tagged with its
lifecycle stage:

* **Pre-decision** — borrower context the bandit may see (income, DTI, FICO, grade, …)
* **Policy** — `loan_amnt`, `int_rate`, `term` (the arm)
* **Disbursement** — `funded_amnt`, `issue_d` (time index)
* **Interim** — repayment / delinquency signals as they would have become available
* **Terminal** — `loan_status`, recoveries, payments (only ever used to *update* posteriors)

`LeakageGuard` enforces that a `BorrowerContext` never contains interim or terminal
columns. A synthetic Lending-Club twin (`bayesian_credit.data.synthetic`) runs the whole
system when the Kaggle file is absent (tests, CI, demos).

```bash
scripts/download_data.sh            # needs a Kaggle token in ~/.kaggle/access_token
python scripts/make_synthetic.py --n 200000 --out data/processed/synthetic.parquet
```

## Quickstart

```bash
# backend
/usr/bin/python3.14 -m venv .venv && . .venv/bin/activate
pip install -e "core[dev]" && pip install -r api/requirements.txt
cd core && pytest -q && cd ..
python scripts/run_pipeline.py --source synthetic          # ingest → fit → shadow → simulate → report
cd api && python manage.py migrate && python manage.py runserver 8010   # http://localhost:8010/api/docs/

# frontend
cd frontend && pnpm install && pnpm dev                     # http://localhost:5173 with the in-browser mock API
cd frontend && VITE_API_PROXY_TARGET=http://localhost:8010 pnpm dev:live   # against the live API

# containers
docker compose -f deploy/docker/docker-compose.yml up --build
```

See `make help` for the common commands; `make verify` runs lint, both test suites, the
frontend typecheck/lint/build and a small synthetic pipeline (`scripts/verify.sh`).

## First end-to-end results (2026-10-16)

`python scripts/run_pipeline.py --source lending_club --max-rows 300000 --max-loans 50000`
— 300 k random Lending Club loans, models fitted on issues before 2015-01, 50 k borrowers
streamed 2015-01 → 2018-12 with real delayed feedback, six policies, one laptop core,
**77 s wall-clock, 2.1 GB peak RSS** (full summary and `report.json` in `data/runs/`).

| Table 11 dimension | Result |
|---|---|
| Default prediction (hold-out, term elapsed, n = 49 k) | Brier 0.125 (climatology 0.130), ROC-AUC 0.667, ECE 0.031 |
| Recovery prediction | MAE 0.194, RMSE 0.231 |
| Credit-state modelling | hold-out log-likelihood −0.069 per transition; data-driven `p_DR = 0.021`, `p_DL = 0.234` |
| Policy performance (cum. expected reward, hindsight yardstick) | Thompson $86 M · Thompson-SE $88 M · ε-greedy $84 M · UCB1 $81 M · **historical $30 M** · random $38 M |
| Bandit performance | Thompson avg regret $1.4 k/decision vs $2.6 k historical; top-3 arm share 7 %, 36-month share 47 % in the last six epochs |
| Counterfactual (shadow engine, 200 k loans) | uplift decomposition: repricing only +128 % → + resizing +282 % → + term choice +419 %; λ = 1 risk appetite: +343 %, 29 % 36-month, mean amount $21 k |
| Distributional monitoring | max PSI 0.256 (`revol_util`, 2017-12) → one breaker trip, 6 watch epochs, resumed after review |
| Computational feasibility | ingest 3 s · fit 0.9 s · shadow 8 s · simulation 49 s (6.6 k decisions/s) |
| Infrastructure resilience | worker killed after epoch 5 and resumed from checkpoint: decisions and time-series **bit-identical** (recovery 4.5 s) |

Read the counterfactual numbers as *model quantities*: the engine has no take-up or
adverse-selection term beyond the within-group policy effect, so most of the remaining
uplift comes from larger exposures on borrowers the model believes are safe — the
decomposition and the risk-appetite sweep are there precisely so that this is visible.
The synthetic twin (`--source synthetic`) reproduces the same qualitative picture, trips the
Sentinel on its built-in 2016-06 distribution shift, and runs in 73 s.

## Documented modelling decisions

The proposal leaves a handful of implementation details open; each choice below is a
configuration default, not a hard-coded assumption (see `core/bayesian_credit/config.py`):

* **Hierarchical priors** — child priors are the parent's posterior mean with a
  concentration κ (pseudo-count); κ can be fixed or estimated by the Beta-binomial method
  of moments ("auto").
* **Structured leaf prior (policy effect)** — a group × arm leaf is not assumed to be as
  safe as its group: its prior mean is `σ(logit(m_g) + t(g,a))`, where the tilt `t` comes
  from an offset logistic regression of default on rate, 60-month term, payment-to-income
  and log amount, centred within each group and clipped to ±3 logit (Scott 2010, §2.2.1).
  κ and the conjugate update are unchanged, so observed data still dominate
  (`HierarchyConfig.policy_effect`, `models/policy_effect.py`, docs/MATH.md §2.6).
* **Lending guardrails** — an arm is eligible only if the installment is ≤ 20 % of monthly
  income, the amount ≤ 50 % of annual income, total DTI ≤ 43 % (the Qualified-Mortgage
  ceiling, 12 CFR 1026.43(e)(2)(vi)) and the rate within ±5 points of the borrower's
  sub-grade reference rate (training mean `int_rate`); unknown inputs skip a rule, decline
  is always available and one minimal counter-offer is kept (`ArmSpaceConfig`,
  `data/reference_rates.py`, docs/MATH.md §2.7).
* **Prepayment-aware reward** — expected interest and funding cost are multiplied by the
  realised share of the scheduled interest of fully-paid training loans, by grade and term
  (≈ 0.82 for 36 and 0.72 for 60 months on Lending Club); `RewardConfig.prepayment_adjustment`
  (`rewards/realization.py`, docs/MATH.md §2.8).
* **Recovery by term** — Stage 2 falls back leaf → (group, term) → group → global, since
  60-month defaults recover less principal (`RecoveryConfig.term_level`, docs/MATH.md §2.9).
* **Unbiased yardstick** — the simulation's hindsight oracle is fitted on loans whose term
  had elapsed by the snapshot (`SimulationConfig.hindsight_sample`, docs/SIMULATION.md §3);
  recent vintages' resolved loans are mostly early defaults.
* **Recovery rate** — fraction of principal recovered by any means given default,
  `(total_rec_prncp + recoveries − collection_recovery_fee) / funded_amnt`, which makes
  `L·p·(1−E[R])` exactly the expected principal loss. A post-charge-off basis is available.
* **MCTM structure** — the proposal's matrix has structural zeros `P[C,R] = P[C,L] = 0`
  (`PROPOSAL`); an `EXTENDED` structure allows payoff `C → R`. Lending Club is a snapshot,
  so monthly state paths are reconstructed with a documented heuristic (`data/panel.py`).
* **Null arm** — arm 0 is "decline" with reward 0, so the bandit decides *whether* to
  lend, not only how much and at what price.
* **Reward timing** — default *level* comes from the Beta-Beta posterior, default *timing*
  from the MCTM, so "the MCTM and Two-Stage Beta-Beta converge and tell the same narrative".
* **Counterfactual feedback** — `simulated` (posterior-predictive environment),
  `replay` (rejection-sampling replay, unbiased but data-hungry) or `hybrid`.
* **Thompson posterior scale** — samples from `N(μ_k, σ_k²)` exactly as written in §3.3;
  `N(μ_k, σ_k²/n_k)` is available as `posterior_scale="standard_error"`.

## Licence

PolyForm Noncommercial 1.0.0 (dual-licensed; see `LICENSE.md`). The Lending Club data is
CC0 and is not redistributed in this repository.
