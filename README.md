<div align="center">

# Bayesian Credit

### A shadow lending engine — contextual Thompson Sampling for credit decisions that learn

[![CI](https://github.com/mistiusiu/bayesian-credit/actions/workflows/ci.yml/badge.svg)](https://github.com/mistiusiu/bayesian-credit/actions/workflows/ci.yml)
[![Python 3.12–3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](core/pyproject.toml)
[![Django 6](https://img.shields.io/badge/Django-6.1-092E20?logo=django&logoColor=white)](api/)
[![React 19](https://img.shields.io/badge/React-19-20232A?logo=react&logoColor=61DAFB)](frontend/)
[![Tests](https://img.shields.io/badge/tests-739%20passing-2EA043)](scripts/verify.sh)
[![Licence: PolyForm Noncommercial](https://img.shields.io/badge/licence-PolyForm%20NC%201.0.0-blue)](LICENSE.md)

**Contextual Multi-Armed Bandits in Bayesian Sequential Decision-Making in Credit Risk Modeling**
Misati Randy Nyambane · BSc Data Science & Analytics · United States International University – Africa · DSA4900

*Traditional credit models answer "will this borrower default?". This system answers the question a lender actually has:*
***"given what we know right now, how much should we lend, at what price — and how should that policy change as evidence arrives?"***

</div>

---

## The idea in one picture

Credit risk is reframed from a static prediction problem into an **adaptive Bayesian policy-optimisation problem**. Every borrower passes through a closed loop: risk is estimated hierarchically, borrower states are propagated through a Markov chain, a risk-adjusted reward is formed, a lending policy (amount × rate × term) is chosen by Thompson Sampling, and the delayed, partial feedback that loans produce over years is fed back into the posteriors. A **PSI Sentinel** watches the loop and trips a circuit breaker when the population drifts; a **shadow lending engine** runs alongside the historical policy and prices the roads not taken.

```mermaid
flowchart LR
    classDef risk fill:#1d3557,stroke:#1d3557,color:#fff
    classDef dec fill:#e63946,stroke:#e63946,color:#fff
    classDef gov fill:#f4a261,stroke:#f4a261,color:#111
    classDef obs fill:#2a9d8f,stroke:#2a9d8f,color:#fff

    X["Borrower context X_i<br/>(pre-decision variables only)"]:::obs
    B["Two-Stage Beta-Beta<br/>p_{g,a} · R_{g,a}"]:::risk
    M["MCTM<br/>C → D → R / L"]:::risk
    R["Compound reward<br/>Revenue − E[Loss] − Funding"]:::risk
    T["Contextual Thompson Sampling<br/>A_t = argmax θ̃_k"]:::dec
    O["Observed outcomes<br/>interim · terminal (delayed)"]:::obs
    U["Posterior update<br/>Beta(α+d, β+n) · Welford μ,σ²"]:::risk

    X --> B --> M --> R --> T --> O --> U --> T
    U -.-> B

    S{{"PSI Sentinel<br/>circuit breaker"}}:::gov
    S -. "PSI > 0.25 → freeze, review" .-> T
    X -. monitored .-> S

    SH[["Shadow lending engine<br/>every arm, every borrower"]]:::dec
    B -.-> SH
    M -.-> SH
```

---

## What it does on real data — in 77 seconds

`python scripts/run_pipeline.py --source lending_club --max-rows 300000 --max-loans 50000` — 300 k random Lending Club loans (2007–2018), models fitted on issues before 2015, **50 k borrowers replayed month by month 2015 → 2018 with real delayed feedback**, six policies, one laptop core, 2.1 GB peak RSS.

<table>
<tr>
<td align="center"><b>0.667</b><br/><sub>ROC-AUC, elapsed-term hold-out (n = 49 k)</sub></td>
<td align="center"><b>0.031</b><br/><sub>expected calibration error</sub></td>
<td align="center"><b>$86 M vs $30 M</b><br/><sub>Thompson vs historical policy, cumulative expected reward</sub></td>
<td align="center"><b>$1.4 k vs $2.6 k</b><br/><sub>average regret per decision</sub></td>
</tr>
<tr>
<td align="center"><b>p<sub>DL</sub> = 0.243</b><br/><sub>delinquent → loss, estimated from 1.66 M transitions</sub></td>
<td align="center"><b>1 trip</b><br/><sub>PSI 0.256 on <code>revol_util</code>, 2017-12 → human review</sub></td>
<td align="center"><b>bit-identical</b><br/><sub>worker killed mid-run, resumed from checkpoint</sub></td>
<td align="center"><b>6.6 k decisions/s</b><br/><sub>49 s for 300 k decisions</sub></td>
</tr>
</table>

### Policies learn; the historical policy does not

Cumulative expected reward under the hindsight yardstick, in USD millions — series in order: **Thompson** (red), **ε-greedy** (orange), **UCB1** (teal), random (grey), historical (navy). Thompson Sampling, ε-greedy and UCB1 separate from the reference policy within the first year and keep separating; random lending is worth more than the historical policy only because the model is risk-neutral and the historical book is conservative.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"xyChart": {"plotColorPalette": "#e63946, #f4a261, #2a9d8f, #8d99ae, #1d3557", "titleColor": "#333", "xAxisLabelColor": "#555", "yAxisLabelColor": "#555"}}}}%%
xychart-beta
    title "Cumulative expected reward (USD M)"
    x-axis ["2015-01","2015-09","2016-05","2017-01","2017-09","2018-05","2018-12"]
    y-axis "USD millions" 0 --> 90
    line [1.6, 12.1, 28.2, 42.0, 56.7, 71.8, 86.0]
    line [1.9, 13.3, 28.3, 41.1, 55.2, 70.0, 84.0]
    line [0.6, 9.9, 24.8, 37.9, 52.2, 67.1, 81.2]
    line [0.7, 6.0, 12.5, 18.1, 24.1, 31.7, 37.9]
    line [0.5, 4.5, 9.1, 13.3, 18.4, 23.7, 29.5]
```

Cumulative pseudo-regret against the per-group oracle — **Thompson** (red), **Thompson with σ²/n** (orange), historical (teal), random (grey). The bandits' curves bend; the historical and random policies' do not.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"xyChart": {"plotColorPalette": "#e63946, #f4a261, #2a9d8f, #8d99ae, #1d3557", "titleColor": "#333", "xAxisLabelColor": "#555", "yAxisLabelColor": "#555"}}}}%%
xychart-beta
    title "Cumulative regret vs the per-group oracle (USD M)"
    x-axis ["2015-01","2015-09","2016-05","2017-01","2017-09","2018-05","2018-12"]
    y-axis "USD millions" 0 --> 130
    line [1.4, 13.6, 26.7, 37.1, 48.5, 60.1, 71.5]
    line [1.4, 14.2, 27.9, 37.9, 48.2, 58.6, 69.1]
    line [2.5, 21.2, 45.8, 65.8, 86.9, 108.2, 128.0]
    line [2.3, 19.7, 42.3, 60.9, 81.1, 100.2, 119.7]
```

### The Sentinel catches a drift the model never saw

Aggregate PSI of the monitored pre-decision features against the pre-2015 reference window. It climbs through 2017 as Lending Club's borrower mix changes, crosses the 0.10 watch band in mid-2017 and trips the 0.25 breaker in December 2017; the policies freeze on their champion snapshot, a review is opened, and after acknowledgement the reference is re-based and the PSI falls back to zero.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"xyChart": {"plotColorPalette": "#e63946, #f4a261, #2a9d8f, #8d99ae, #1d3557", "titleColor": "#333", "xAxisLabelColor": "#555", "yAxisLabelColor": "#555"}}}}%%
xychart-beta
    title "Aggregate PSI per epoch (watch 0.10 · breaker 0.25)"
    x-axis ["2015-01","2015-07","2016-01","2016-07","2017-01","2017-07","2017-10","2017-11","2017-12","2018-01","2018-06","2018-12"]
    y-axis "PSI" 0 --> 0.3
    line [0.054, 0.059, 0.074, 0.085, 0.065, 0.120, 0.168, 0.206, 0.256, 0.003, 0.022, 0.058]
```

### Default risk is heterogeneous — and the hierarchy knows it

Posterior default probability by Lending Club grade (data-weighted mean over the groups of each grade, 76 groups, 1 657 leaves). The same borrowers, under the **shadow engine's** counterfactual policies, are repriced within ±5 points of their sub-grade reference rate — never blindly at the top of the grid.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"xyChart": {"plotColorPalette": "#e63946, #f4a261, #2a9d8f, #8d99ae, #1d3557", "titleColor": "#333", "xAxisLabelColor": "#555", "yAxisLabelColor": "#555"}}}}%%
xychart-beta
    title "Posterior P(default) by grade, pre-2015 training window"
    x-axis [A, B, C, D, E, F, G]
    y-axis "P(default)" 0 --> 0.45
    bar [0.059, 0.110, 0.186, 0.249, 0.307, 0.346, 0.379]
```

> **Read the counterfactual numbers as model quantities.** The shadow engine reports a +419 % expected-profit uplift over the historical book, decomposed as *repricing only* +128 % → *+ resizing* +282 % → *+ term choice* +419 %, and a risk-appetite sweep (λ = 1: +343 %, 29 % 36-month, mean amount $21 k). There is no take-up or adverse-selection term beyond the within-group policy effect, so most of the remaining uplift is larger exposures on borrowers the model believes are safe. The decomposition and the λ sweep exist precisely so that this is visible rather than hidden — see [`docs/SIMULATION.md` §10](docs/SIMULATION.md).

---

## The mathematics, and where it lives

Every equation in §3.3 of the proposal is implemented once, tested, and mapped in [`docs/MATH.md`](docs/MATH.md).

| Component | Model | Code |
|---|---|---|
| **Stage 1 — default** | $p_{g,a}\mid D \sim \mathrm{Beta}(\alpha_{g,a}+d,\ \beta_{g,a}+n)$ with priors built top-down $p_0 \to p_g \to p_{g,a}$; leaf prior mean tilted by a within-group policy effect $\sigma\!\big(\mathrm{logit}\,m_g + \beta\cdot(x_{g,a}-\bar x_g)\big)$ | `models/hierarchical.py`, `models/policy_effect.py` |
| **Stage 2 — recovery** | $R_{g,a}\mid D{=}1 \sim \mathrm{Beta}(\alpha^R,\beta^R)$, $\ \alpha^R=\mu\!\left(\tfrac{\mu(1-\mu)}{\sigma^2}-1\right)$, $\beta^R=(1-\mu)\!\left(\tfrac{\mu(1-\mu)}{\sigma^2}-1\right)$; fallback leaf → (group, term) → group → global | `models/recovery.py`, `models/beta.py` |
| **MCTM** | $S=\{C,D,R,L\}$, $\pi_t=\pi_0P^t$, $P(\text{Default},t)=\pi_D(t)+\pi_L(t)$, absorption $B=(I-Q)^{-1}R$ | `models/mctm.py`, `data/panel.py` |
| **Reward** | $R_t=\text{Revenue}_t-\mathbb{E}[L_t]-\text{Funding}_t$, $\ \mathbb{E}[L_t]=L_t\,P(D\mid X,A_t)\,(1-\mathbb{E}[R\mid X,A_t])$; default *level* from the Beta posterior, *timing* from the MCTM, interest scaled by the realised share of scheduled interest ($\rho_{36}=0.82$, $\rho_{60}=0.72$) | `rewards/compound.py`, `rewards/realization.py` |
| **Bandit** | $\theta_k\sim\mathcal N(\mu_k,\sigma_k^2)$, $A_t=\arg\max_k\tilde\theta_k$, $\ \mu_k\leftarrow\mu_k+\tfrac{r_t-\mu_k}{n_k+1}$, $\ \sigma_k^2\leftarrow\tfrac{(n_k-1)\sigma_k^2+(r_t-\mu_k^{(t)})(r_t-\mu_k^{(t+1)})}{n_k}$, one arm set per borrower group | `bandits/arm_state.py`, `bandits/contextual.py` |
| **Sentinel** | $\mathrm{PSI}=\sum_i (a_i-e_i)\ln\tfrac{a_i}{e_i}$; < 0.10 normal · 0.10–0.25 watch · > 0.25 breaker | `monitoring/psi.py`, `monitoring/sentinel.py` |

### The hierarchy shares strength without flattening heterogeneity

```mermaid
flowchart TD
    classDef lvl0 fill:#264653,color:#fff,stroke:#264653
    classDef lvl1 fill:#2a9d8f,color:#fff,stroke:#2a9d8f
    classDef lvl2 fill:#e9c46a,color:#111,stroke:#e9c46a
    P0["portfolio  p₀ ~ Beta(α₀, β₀)"]:::lvl0
    G1["group g = B | debt_consolidation | RENT<br/>prior mean = m₀, concentration κ_group"]:::lvl1
    G2["group g′ = E | other | MORTGAGE"]:::lvl1
    L1["arm a = $10k @ 13% / 36m<br/>n = 686 → data dominate"]:::lvl2
    L2["arm a′ = $10k @ 28% / 36m<br/>n = 0 → prior tilted by policy effect<br/>P(D) 0.057 → 0.115"]:::lvl2
    L3["arm a″ = $40k @ 28% / 60m<br/>n = 0 → P(D) 0.256"]:::lvl2
    P0 --> G1
    P0 --> G2
    G1 --> L1
    G1 --> L2
    G1 --> L3
```

### Borrowers move through credit states — the chain fitted on 1.66 M real transitions

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Current
    Current --> Current : 0.993
    Current --> Delinquent : 0.007
    Delinquent --> Current : 0.002
    Delinquent --> Delinquent : 0.729
    Delinquent --> Recovered : 0.026
    Delinquent --> AbsorbedLoss : 0.243
    Recovered --> [*]
    AbsorbedLoss --> [*]
    note right of Delinquent
        expected time to absorption 4.9 months
        P(loss | delinquent) = 0.90
    end note
```

The proposal's matrix has structural zeros `P[C,R] = P[C,L] = 0` — a loan leaves *Current* only through *Delinquent* — and that is the default (`PROPOSAL`); an `EXTENDED` structure that allows pay-off `C → R` is one flag away.

### Feedback arrives late, and the replay respects that

```mermaid
sequenceDiagram
    autonumber
    participant S as Stream (issue month)
    participant Z as PSI Sentinel
    participant P as Policy (Thompson)
    participant E as Environment (Beta-Beta + MCTM)
    participant Q as Delayed-feedback queue

    S->>Z: arrivals of epoch t
    Z-->>P: level normal / watch / breaker
    S->>P: borrower context X_i (leakage-guarded)
    P->>P: sample θ̃_k, pick arm A_t
    P->>Q: schedule interim (t+1 … t+12) and terminal (resolution month)
    Note over Q: unresolved loans never report — real censoring
    Q-->>E: due feedback at epoch t+k
    E-->>P: reward draw for the chosen arm
    P->>P: μ_k, σ_k² ← Welford update
    S->>E: realized historical outcome resolves → posterior update
```

### Governance is a state machine, not a dashboard widget

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open : PSI > 0.25 — policies frozen, champion snapshot serves
    Open --> HalfOpen : human acknowledges (resume · resegment · rollback)
    HalfOpen --> Closed : 3 consecutive normal readings
    HalfOpen --> Open : PSI > 0.25 again
```

---

## Architecture

```mermaid
flowchart TB
    subgraph core["core/  —  bayesian_credit  (pure Python, 592 tests)"]
        direction LR
        D["data<br/>schema · leakage guard · loader · synthetic twin · state panel"]
        MO["models<br/>Beta-Beta · policy effect · MCTM"]
        RW["rewards<br/>amortisation · realisation · compound"]
        BA["bandits<br/>Thompson · contextual · baselines"]
        MN["monitoring<br/>PSI · circuit breaker"]
        SH["shadow<br/>counterfactual engine"]
        SI["simulation<br/>environment · queue · runner · run-dir protocol"]
        EV["evaluation<br/>Table 11 report"]
        D --> MO --> RW --> BA --> SI
        MN --> SI
        SH --> SI --> EV
    end
    API["api/  —  Django 6 + DRF<br/>OpenAPI · subprocess jobs · chaos / resume · review queue<br/>147 tests"]
    WEB["frontend/  —  React 19 + Vite + TS<br/>Overview · Data · Risk · MCTM · Bandit Lab · Shadow · Sentinel · Evaluation"]
    DEP["deploy/  —  Docker · kustomize for K3s/Talos · chaos monkey · CI"]
    core --> API --> WEB
    core --> DEP
    API --> DEP
```

```
bayesian-credit/
├── core/bayesian_credit/      the modelling library (no Django inside)
├── api/                       Django REST API — run-directory file protocol, jobs, chaos/resume
├── frontend/                  the research console (mock mode by default, live mode via /api proxy)
├── deploy/                    Dockerfiles, docker-compose, kustomize manifests, chaos-monkey.sh
├── scripts/                   run_pipeline.py · verify.sh · download_data.sh · make_synthetic.py
└── docs/                      SPEC · MATH · DATA · SIMULATION · API · DEPLOY
```

### Data minimisation and leakage are enforced, not hoped for

Only the Appendix F Table 3 variables are loaded, and each carries its lifecycle stage. `LeakageGuard` makes it impossible to construct a `BorrowerContext` from interim or terminal columns — the API returns a `400 LeakageError` if you try.

```mermaid
flowchart LR
    classDef ok fill:#2a9d8f,color:#fff,stroke:#2a9d8f
    classDef pol fill:#e9c46a,color:#111,stroke:#e9c46a
    classDef late fill:#e76f51,color:#fff,stroke:#e76f51
    A["PRE-DECISION<br/>income · DTI · FICO · grade · utilisation …<br/>37 columns"]:::ok
    B["POLICY<br/>loan_amnt · int_rate · term<br/>= the arm"]:::pol
    C["DISBURSEMENT<br/>funded_amnt · issue_d"]:::pol
    D["INTERIM<br/>payments · delinquency · out_prncp"]:::late
    E["TERMINAL<br/>loan_status · recoveries · totals"]:::late
    A --> B --> C --> D --> E
    A -. "the only input the bandit sees" .-> X((decision))
    D -. "surrogate rewards, month by month" .-> U((posterior update))
    E -. "terminal reward at resolution" .-> U
```

---

## The research console

Eight pages that make the loop visible: the hierarchy explorer with posterior means, 95 % HDIs and shrinkage weights; the MCTM heatmap and state-occupancy trajectories; a **Bandit Lab** with live cumulative reward and regret, arm-selection heatmaps, a chaos button and resume-from-checkpoint; a **Shadow Engine** that lays every arm out as a heatmap with guardrail eligibility hatched, the recommendation against the historical decision, and the portfolio uplift decomposition with a risk-appetite selector; a **Sentinel** page with PSI timelines, breaker state and the human-in-the-loop review queue; and the Table 11 evaluation grid.

| Page | What you see |
|---|---|
| Overview | the closed loop with live status, KPI tiles, the latest events |
| Data | ingestion (synthetic twin or Kaggle file), Table 3 stage table, feature distributions, loan browser |
| Risk Models | hierarchy explorer, policy-effect coefficients, calibration & ROC, recovery Beta, ρ by grade × term |
| MCTM | transition heatmap with counts, occupancy trajectories, P(default \| t), absorption probabilities |
| Bandit Lab | per-policy reward/regret, arm-selection heatmap, posterior histories, Simpson's-paradox panel, pause / resume / cancel / chaos |
| Shadow Engine | per-borrower arm heatmap, eligibility strip, recommendation vs history, portfolio decomposition & λ sweep |
| Sentinel | PSI per feature with threshold bands, breaker timeline, bucket tables, review queue |
| Evaluation | the nine Table 11 dimensions, runtime and resilience, JSON export |

---

## Quickstart

```bash
# clone + data
git clone https://github.com/mistiusiu/bayesian-credit.git && cd bayesian-credit
scripts/download_data.sh                 # Kaggle token in ~/.kaggle/access_token; or skip and use the synthetic twin

# backend
python3.14 -m venv .venv && . .venv/bin/activate
pip install -e "core[dev]" -r api/requirements.txt
python scripts/run_pipeline.py --source synthetic               # the whole loop on the synthetic twin, ~70 s
python scripts/run_pipeline.py --source lending_club --max-rows 300000   # on real data, ~80 s
cd api && python manage.py migrate && python manage.py demo && python manage.py runserver 8010   # http://localhost:8010/api/docs/

# console
cd frontend && pnpm install && pnpm dev                        # mock mode, http://localhost:5173
VITE_API_PROXY_TARGET=http://localhost:8010 pnpm dev:live      # against the live API

# everything, verified
make verify
```

Containers and the cluster:

```bash
docker compose -f deploy/docker/docker-compose.yml up --build   # console :8080, API :8000
kubectl apply -k deploy/k8s                                     # K3s on Talos — see docs/DEPLOY.md
deploy/chaos/chaos-monkey.sh --api http://localhost:8010 --simulation 1 --rounds 3   # kill the worker, watch it resume
```

---

## Documented modelling decisions

The proposal leaves a handful of details open. Each choice is a configuration default, not a hard-coded assumption (`core/bayesian_credit/config.py`), and each is written up in [`docs/SPEC.md` §7](docs/SPEC.md) and [`docs/MATH.md`](docs/MATH.md).

| Decision | Why |
|---|---|
| **Hierarchical priors** — child prior = parent posterior mean at concentration κ (fixed or Beta-binomial "auto") | sparse group × policy cells shrink to the portfolio instead of exploding |
| **Structured leaf prior** — offset logistic policy effect on rate, 60-month term, payment-to-income, log amount, centred within group, clipped at ±3 logit | an arm nobody in the group ever received is *not* as safe as the group average (Scott 2010, §2.2.1) |
| **Lending guardrails** — PTI ≤ 20 %, amount ≤ 50 % of income, total DTI ≤ 43 % (the Qualified-Mortgage ceiling), rate within ±5 pts of the sub-grade reference | the counterfactual stays inside a credit policy a lender would recognise |
| **Prepayment-aware reward** — expected interest and funding scaled by the realised share of scheduled interest, by grade × term | fully-paid 60-month loans earn 72 % of their scheduled interest, not 100 % |
| **Recovery by term** — fallback leaf → (group, term) → group → global | 60-month defaults recover 0.38 of principal vs 0.49 for 36-month |
| **Recovery definition** — `(total_rec_prncp + recoveries − fee) / funded_amnt` | makes `L·p·(1−E[R])` exactly the expected principal loss |
| **Null arm** — arm 0 is *decline* with reward 0 | the bandit decides *whether* to lend, not only how much |
| **Unbiased yardstick** — the hindsight oracle is fitted on loans whose term had elapsed | recent vintages' resolved loans are mostly early defaults |
| **Thompson posterior scale** — `N(μ_k, σ_k²)` exactly as written, `N(μ_k, σ_k²/n_k)` one flag away | the proposal's σ² converges to Var(r_k), so exploration never fully vanishes |
| **MCTM structure** — the proposal's structural zeros by default, `EXTENDED` optional | Lending Club is a snapshot; monthly paths are reconstructed with a documented heuristic |

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/SPEC.md`](docs/SPEC.md) | the build specification: contracts, interfaces, API, decisions |
| [`docs/MATH.md`](docs/MATH.md) | all 26 equations of §3.3 → code, plus the assumptions S1–S6 / G1–G5 |
| [`docs/DATA.md`](docs/DATA.md) | Table 3 stages, parsing rules, outcome definitions, panel heuristic, synthetic twin calibration |
| [`docs/SIMULATION.md`](docs/SIMULATION.md) | epochs, feedback modes, oracle & regret, run-directory protocol, resume semantics, caveats |
| [`docs/API.md`](docs/API.md) | every endpoint with examples; `/api/docs/` serves the OpenAPI schema |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | images, compose, K3s/Talos, the Table 10 resilience experiments |

## Citing

```bibtex
@misc{nyambane2026bayesiancredit,
  author = {Nyambane, Misati Randy},
  title  = {Contextual Multi-Armed Bandits in Bayesian Sequential Decision-Making in Credit Risk Modeling},
  year   = {2026},
  school = {United States International University -- Africa},
  note   = {Research prototype: \url{https://github.com/mistiusiu/bayesian-credit}}
}
```

## Licence

PolyForm Noncommercial 1.0.0, dual-licensed — see [`LICENSE.md`](LICENSE.md) for commercial terms. The Lending Club data is CC0 and is not redistributed here.
