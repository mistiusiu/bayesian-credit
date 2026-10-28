# Bayesian Credit API — contract v1

This is the REST contract between the Django/DRF backend (`api/`) and the research console
(`frontend/`). It is the human-readable twin of
[`frontend/src/api/types.ts`](../frontend/src/api/types.ts), which is **authoritative**: every
JSON shape below is a TypeScript interface there, and the two files change together.

The console ships with an in-browser mock of this API (`VITE_API_MOCK=1`, see
[`frontend/src/mocks`](../frontend/src/mocks)) that returns realistic, internally consistent
fixtures. All example payloads in this document were produced by that mock and trimmed
(`// …` marks elisions).

Endpoints marked **★** are additions to SPEC §3 that the console needs. §11 lists every
addition and every decision made where SPEC §3 only gave a sketch.

---

## 1. Conventions

| Topic | Rule |
|---|---|
| Base path | `/api/v1/` (the console calls `/api/v1/...`; Vite and nginx proxy `/api` to Django) |
| Format | JSON in and out, `Content-Type: application/json`, UTF-8 |
| Keys | `snake_case` |
| Ids | integers (`"id": 3`), never strings or UUIDs |
| Timestamps | RFC 3339 UTC strings without milliseconds: `"2026-10-16T09:41:07Z"` |
| Epochs / months | `"YYYY-MM"` strings (`"2016-06"`): issue months, simulation epochs, `train_end`, `start`/`end` |
| Money | plain JSON numbers in **US dollars** (not cents, not strings) |
| Rates, shares, probabilities | **fractions** in [0, 1]: `0.1299`, never `12.99` or `"12.99%"` |
| `dti` | keeps Lending Club's percent-point scale (`18.5` means 18.5 %) — the only exception |
| Missing values | `null`, never `NaN`, `""` or an omitted key (optional *request* fields may be omitted) |
| Enums | lowercase snake_case strings (§2), except event types, which are PascalCase class names |
| Group keys | dimension values joined by `|` in the fit's `segmentation.dimensions` order: `"B|debt_consolidation|RENT"` |
| Arm ids | index into the fit's arm space; arm `0` is the null arm (decline) when `include_null` is true; then term-major, amount, rate (same order as `ArmSpace.from_grid`) |
| Ordering | collections are returned newest first (`id` descending) unless stated otherwise |
| Auth | none in development; CORS open in dev |

### 1.1 Pagination

Resource collections (`/datasets/`, `/fits/`, `/simulations/`, `/jobs/`,
`/sentinel/reviews/`, `/datasets/{id}/loans/`) use DRF `LimitOffsetPagination`:

```json
{ "count": 62039, "next": "/api/v1/datasets/2/loans/?limit=1&offset=1", "previous": null, "results": [ … ] }
```

`limit` defaults to 50 (max 500); `offset` defaults to 0. The client also accepts a bare
array for these endpoints (defensive), but the envelope is the contract. Computed
sub-resources (`/variables/`, `/groups/`, `/timeseries/`, …) return the documented object or
array directly, without the envelope.

### 1.2 Errors

| Status | When | Body |
|---|---|---|
| 400 | validation failed | DRF field errors `{ "name": ["This field is required."] }`; dotted keys for nested fields (`"hierarchy.kappa_group"`); `non_field_errors` for cross-field rules |
| 404 | unknown id / route | `{ "detail": "fit not found" }` |
| 409 | the resource exists but is not in a state that allows the request (dataset/fit not `ready`, invalid simulation transition, review already acknowledged) | `{ "detail": "fit 4 is not ready (status running)" }` |
| 5xx | bug / worker failure | `{ "detail": "…" }` |

The console shows `detail` verbatim, maps field errors onto form fields, and does not retry 4xx.

### 1.3 Background work

Ingest, fit, simulation and shadow-portfolio requests are long-running. They return
immediately with the created resource (status `queued`) and an embedded [Job](#41-job).
Work runs as `python manage.py run_job <job_id>` (a k8s Job in the cluster). The console polls
the resource (or the job) every **1.5 s** while it is `queued`/`running` and stops when it
settles. Lifecycle mapping:

| Job `status` | Dataset / Fit `status` |
|---|---|
| `queued` | `queued` |
| `running` | `running` |
| `succeeded` | `ready` |
| `failed` | `failed` |
| `cancelled` | `cancelled` |

Simulations have their own richer lifecycle (§8.3); their job stays `running` while the
simulation is `running`, `paused` or `awaiting_review`, becomes `failed` when the worker
dies (`interrupted`), and `succeeded` when the run completes.

### 1.4 Units used by report metrics (`MetricUnit`)

| Unit | Meaning | Console rendering |
|---|---|---|
| `usd` | US dollars | `$84.8M` / `$1,284` |
| `usd_per_decision` | US dollars per policy decision | `$2,383/decision` |
| `fraction` | share in [0, 1] | `13.0%` |
| `ratio` | unitless statistic (Brier, AUC, PSI, MAE of a rate) | `0.0865`, `0.679` |
| `count` | integer count | `1,284` |
| `seconds` | duration in seconds | `8m 51s` |
| `megabytes` | MiB | `1.6 GB` |
| `nats` | natural-log likelihood units | `−0.103` |
| `score` | unitless integer-like score (FICO points) | `705` |

---

## 2. Vocabulary

| Type | Values |
|---|---|
| `Stage` (Table 3) | `pre_decision` · `policy` · `disbursement` · `interim` · `terminal` |
| `VariableKind` | `numeric` · `categorical` · `date` |
| `DataSource` | `synthetic` · `lending_club` |
| `JobStatus` | `queued` · `running` · `succeeded` · `failed` · `cancelled` |
| `JobKind` | `ingest` · `fit` · `simulation` · `shadow_portfolio` |
| `BuildStatus` (datasets, fits) | `queued` · `running` · `ready` · `failed` · `cancelled` |
| `SimulationStatus` | `queued` · `running` · `paused` · `awaiting_review` · `interrupted` · `completed` · `failed` · `cancelled` |
| `TransitionStructure` | `proposal` · `extended` |
| `CreditState` (matrix order) | `current` · `delinquent` · `recovered` · `loss` |
| `FeedbackMode` | `simulated` · `replay` · `hybrid` |
| `BreakerMode` | `pause` · `auto_ack` · `off` |
| `BreakerState` | `closed` · `open` · `half_open` |
| `SentinelLevel` (Table 9) | `normal` (< watch) · `watch` (watch ≤ PSI ≤ breaker) · `breaker` (> breaker) |
| `ReviewAction` | `resume` · `resegment` · `rollback` |
| `ReviewStatus` | `pending` · `acknowledged` |
| `BreakerFallback` | `champion_snapshot` · `decline` · `historical` |
| `PolicyName` | `thompson` · `ucb1` · `epsilon_greedy` · `greedy` · `historical` · `random` |
| `SegmentationDimension` | `grade` · `sub_grade` · `purpose_group` · `home_ownership` · `fico_band` · `dti_band` · `income_band` · `emp_band` · `verification_status` · `application_type` |
| `HierarchyLevel` | `global` · `group` · `leaf` |
| `RecoveryBasis` | `principal` · `post_chargeoff` |
| `EventLevel` | `info` · `warning` · `critical` |
| `PortfolioOutcomeClass` | `rescued` · `upsized` · `repriced_down` · `repriced_up` · `declined` · `unchanged` |
| `FeatureUnit` (dataset summary) | `usd` · `fraction` · `pct` · `count` · `score` · `years` · `months` · `none` |

---

## 3. Endpoint index

| Method & path | Request | Response | |
|---|---|---|---|
| `GET /health/` | | `Health` | |
| `GET /jobs/` | `?status=queued,running&kind=` | `Paginated<Job>` | ★ |
| `GET /jobs/{id}/` | | `Job` | |
| `POST /datasets/` | `DatasetCreate` | 201 `Dataset` | |
| `GET /datasets/` | | `Paginated<Dataset>` | |
| `GET /datasets/{id}/` | | `Dataset` | |
| `GET /datasets/{id}/variables/` | | `VariableRow[]` | |
| `GET /datasets/{id}/summary/` | `?features=a,b` | `DatasetSummary` | |
| `GET /datasets/{id}/loans/` | `?limit&offset&grade&purpose&home_ownership&resolved&search` | `Paginated<LoanRecord>` | |
| `POST /fits/` | `FitCreate` | 201 `Fit` | |
| `GET /fits/` | `?dataset=` | `Paginated<Fit>` | `dataset` filter ★ |
| `GET /fits/{id}/` | | `Fit` | |
| `GET /fits/{id}/groups/` | `?level=global\|group\|leaf&group=` | `HierarchyRow[]` | |
| `GET /fits/{id}/risk/` | `?group=&arm=` | `RiskEstimate` | |
| `GET /fits/{id}/calibration/` | | `Calibration` | |
| `GET /fits/{id}/mctm/` | `?structure=&horizon=&start=` | `Mctm` | |
| `GET /fits/{id}/arms/` | | `FitArms` | |
| `POST /fits/{id}/shadow/evaluate/` | `ShadowEvaluateRequest` | `ShadowReport` | |
| `POST /fits/{id}/shadow/portfolio/` | `PortfolioRequest` | 202 `Job` | |
| `GET /fits/{id}/shadow/portfolio/` | | `PortfolioRun[]` | ★ |
| `GET /fits/{id}/shadow/portfolio/{job_id}/` | | `PortfolioResult` | |
| `POST /simulations/` | `SimulationCreate` | 201 `Simulation` | |
| `GET /simulations/` | `?fit=` | `Paginated<Simulation>` | `fit` filter ★ |
| `GET /simulations/{id}/` | | `Simulation` | |
| `GET /simulations/{id}/timeseries/` | `?policy=` | `Timeseries` | |
| `GET /simulations/{id}/arms/` | `?policy=&epoch=&group=` | `SimArms` | `group` ★ |
| `GET /simulations/{id}/posteriors/` | `?policy=&group=&arms=1,2` | `PosteriorHistory` | ★ |
| `GET /simulations/{id}/simpson/` | `?policy=&epoch=` | `SimpsonPanel` | ★ |
| `GET /simulations/{id}/events/` | `?type=A,B&after=&limit=&order=asc\|desc` | `EventPage` | `order` ★ |
| `GET /simulations/{id}/sentinel/` | | `SentinelTimeline` | |
| `GET /simulations/{id}/sentinel/buckets/` | `?feature=&epoch=` | `PsiBucketTable` | ★ |
| `GET /simulations/{id}/report/` | | `EvaluationReport` | |
| `POST /simulations/{id}/pause/` · `resume/` · `cancel/` · `chaos/` | empty body | `Simulation` | |
| `GET /sentinel/reviews/` | `?simulation=&status=` | `Paginated<Review>` | `status` ★ |
| `POST /sentinel/reviews/{id}/acknowledge/` | `AcknowledgeRequest` | `Review` | |

OpenAPI stays at `/api/schema/` and Swagger at `/api/docs/` (drf-spectacular).

---

## 4. Health and jobs

### `GET /health/`

```json
{ "status": "ok", "version": "0.1.0", "python": "3.14.0", "core_version": "0.1.0" }
```

`status` is `"ok"` or `"degraded"` (e.g. artifact root not writable).

### 4.1 Job

`GET /jobs/{id}/` returns a Job; `GET /jobs/?status=queued,running&kind=simulation` returns
`Paginated<Job>` (both filters accept comma lists). The console polls the active-job list every
3 s to light the navigation status dots.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `kind` | `JobKind` | |
| `status` | `JobStatus` | |
| `progress` | fraction | 0..1 |
| `message` | string | latest human-readable status line |
| `log_tail` | string[] | last ≤ 50 log lines, oldest first |
| `created_at` / `started_at` / `finished_at` | timestamp / null | |
| `target` | `{ type: "dataset" \| "fit" \| "simulation", id }` / null | resource the job builds or drives; shadow-portfolio jobs point at their **fit** |
| `pid` | int / null | worker pid while running |

```json
{
  "id": 101, "kind": "ingest", "status": "succeeded", "progress": 1, "message": "dataset ready",
  "log_tail": ["[12:18:42] resolving source and schema", "[12:18:48] projecting the 60 retained Table 3 columns (data minimisation, F.4)"],
  "created_at": "2026-10-13T12:18:42Z", "started_at": "2026-10-13T12:18:42Z", "finished_at": "2026-10-13T12:19:23Z",
  "target": { "type": "dataset", "id": 1 }, "pid": null
}
```

---

## 5. Datasets

### 5.1 `POST /datasets/` → 201 `Dataset`

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | string | — | required, non-empty |
| `source` | `synthetic` \| `lending_club` | — | required |
| `path` | string / null | server setting | `lending_club` only |
| `n` | int / null | 200000 | `synthetic` only; 1,000 – 2,500,000 |
| `seed` | int / null | 42 | |
| `max_rows` | int / null | null (whole file) | `lending_club` only, ≥ 1,000 |
| `sample_fraction` | number / null | null | `lending_club` only, (0, 1] |
| `issue_from` / `issue_to` | epoch / null | `2012-01` / `2018-12` | inclusive; within 2007-06 … 2018-12 |
| `shift` | `{ at: epoch, deltas: { feature: number } }` / null | null | `synthetic` only; additive shift in feature units applied to loans issued on/after `at`, e.g. `{ "at": "2016-06", "deltas": { "fico_range_low": -25, "dti": 5 } }` |

```json
{ "name": "Synthetic · 50k · seed 42", "source": "synthetic", "n": 50000, "seed": 42,
  "issue_from": "2012-01", "issue_to": "2018-12",
  "shift": { "at": "2016-06", "deltas": { "fico_range_low": -25, "dti": 5 } } }
```

### 5.2 `GET /datasets/` · `GET /datasets/{id}/` → `Dataset`

Statistics are `null` (and `status_counts` `{}`) until `status` is `ready`.

```jsonc
{
  "id": 2, "name": "Synthetic twin · 200k · shift 2016-06", "source": "synthetic", "status": "ready",
  "params": { "name": "Synthetic twin · 200k · shift 2016-06", "source": "synthetic", "path": null, "n": 200000, "seed": 42,
              "max_rows": null, "sample_fraction": null, "issue_from": "2012-01", "issue_to": "2018-12",
              "shift": { "at": "2016-06", "deltas": { "fico_range_low": -25, "dti": 5 } } },
  "n_loans": 200000, "n_resolved": 113145, "issue_min": "2012-01", "issue_max": "2018-12",
  "resolved_share": 0.5657,          // terminal outcome known (Fully Paid / Charged Off / Default variants)
  "default_rate": 0.1914,            // defaults / resolved
  "status_counts": { "Fully Paid": 91490, "Current": 84321, "Charged Off": 21650, "Late (31-120 days)": 1465,
                     "In Grace Period": 610, "Late (16-30 days)": 459, "Default": 5 },
  "created_at": "2026-10-13T16:18:42Z",
  "job": { "id": 102, "kind": "ingest", "status": "succeeded", "progress": 1 /* … Job */ }
}
```

### 5.3 `GET /datasets/{id}/variables/` → `VariableRow[]`

All 60 Table 3 variables in Table 3 order (`core/bayesian_credit/data/schema.py::RETAINED_VARIABLES`),
not paginated. 409 until the dataset is ready.

```json
[ { "name": "annual_inc", "stage": "pre_decision", "primary_use": "Borrower context / risk estimation",
    "kind": "numeric", "present": true, "dtype": "float64", "null_share": 0 },
  { "name": "loan_status", "stage": "terminal", "primary_use": "Default and credit-state outcome",
    "kind": "categorical", "present": true, "dtype": "string", "null_share": 0 } ]
```

`dtype` is the pandas dtype string (`float64`, `int64`, `string`, `datetime64[ns]`) or `null`
when absent; `null_share` is null when absent.

### 5.4 `GET /datasets/{id}/summary/?features=a,b` → `DatasetSummary`

Without `features` the default set is `fico_range_low, dti, annual_inc, revol_util,
inq_last_6mths, emp_length, loan_amnt, int_rate, grade, purpose, home_ownership,
verification_status, term`. Each feature carries its `stage` so the console can badge it.

* numeric: `unit` (`FeatureUnit`), `n`, `null_share`, `mean`, `std`, `p05`, `p50`, `p95`,
  `bins` = ~20 equal-width `{lo, hi, count}` between p01 and p99 (outliers clipped into the edge bins);
* categorical: `levels` sorted by count desc, at most 12 (tail folded into `"other"`).

`monthly_arrivals` has one row per month between `issue_min` and `issue_max` (zero-filled).

```jsonc
{
  "dataset": 2, "n_loans": 200000,
  "features": [
    { "feature": "fico_range_low", "kind": "numeric", "stage": "pre_decision", "unit": "score", "n": 200000,
      "null_share": 0, "mean": 692.0278, "std": 34.3366, "p05": 640, "p50": 690, "p95": 755,
      "bins": [ { "lo": 600, "hi": 610, "count": 260 }, { "lo": 610, "hi": 620, "count": 790 } /* … */ ] },
    { "feature": "grade", "kind": "categorical", "stage": "pre_decision", "n": 200000, "null_share": 0,
      "levels": [ { "level": "B", "count": 62039 }, { "level": "C", "count": 52950 } /* … */ ] }
  ],
  "monthly_arrivals": [ { "epoch": "2012-01", "n": 349, "n_resolved": 349, "n_default": 51 } /* … one per month */ ],
  "status_counts": { "Fully Paid": 91490, "Current": 84321 /* … */ },
  "grade_counts": { "A": 45045, "B": 62039 /* … */ },
  "term_counts": { "36": 154379, "60": 45621 }
}
```

### 5.5 `GET /datasets/{id}/loans/` → `Paginated<LoanRecord>`

Query: `limit`, `offset`, `grade=A,B`, `purpose=debt_consolidation,credit_card` (raw LC
purpose), `home_ownership=RENT,OWN`, `resolved=true|false`, `search=<loan_id substring>`.
Ordered by `loan_id` ascending.

A record separates the lifecycle stages so leakage is visible in the payload itself:
`context` contains **only** the 37 PRE_DECISION columns (all keys present, `null` when
missing), `policy` the POLICY columns, `outcome` the derived TERMINAL `y_*` values.

```jsonc
{
  "loan_id": 1000000, "issue_d": "2012-01", "funded_amnt": 23700,
  "group": "B|debt_consolidation|MORTGAGE",   // y_group under the default segmentation
  "historical_arm": 34,                        // y_historical_arm on the default arm grid
  "context": {
    "annual_inc": 78600, "annual_inc_joint": null, "application_type": "Individual", "dti": 10.98, "dti_joint": null,
    "emp_length": 10, "fico_range_low": 705, "fico_range_high": 709, "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation", "verification_status": "Not Verified", "verified_status_joint": null,
    "delinq_2yrs": 0, "delinq_amnt": 0, "inq_last_6mths": 0, "inq_last_12m": null, "open_acc": 12, "total_acc": 20,
    "revol_bal": 15161, "revol_util": 0.4533, "installment": 752, "mort_acc": 4, "pub_rec": 0,
    "pub_rec_bankruptcies": 0, "collections_12_mths_ex_med": 0, "num_accts_ever_120_pd": 3, "pct_tl_nvr_dlq": 0.8422,
    "num_sats": 12, "tot_coll_amt": 0, "tot_cur_bal": 331129, "tot_hi_cred_lim": 406346, "total_bal_ex_mort": 26084,
    "total_bc_limit": 48006, "bc_util": 0.2624, "all_util": null, "grade": "B", "sub_grade": "B1"
  },
  "policy": { "loan_amnt": 23700, "int_rate": 0.0885, "term": 36 },
  "outcome": {
    "loan_status": "Fully Paid", "resolved": true,
    "defaulted": false,          // y_default; null while unresolved
    "recovery_rate": null,       // y_recovery_rate (principal basis); null unless defaulted
    "realized_revenue": 846.17, "realized_loss": 0,
    "realized_reward": 470.55,   // revenue − loss − funding on the realized schedule (SPEC §2.7 terminal)
    "months_on_book": 5, "resolution_epoch": "2012-06"
  }
}
```

---

## 6. Fits (Two-Stage Beta-Beta + MCTM)

### 6.1 `POST /fits/` → 201 `Fit`

`dataset` must be `ready` (else 409). Every config block is optional and partial; omitted
keys take these defaults (the **[decision]** defaults of `core/bayesian_credit/config.py`):

| Field | Default | Validation |
|---|---|---|
| `dataset` | — | required id |
| `name` | — | required |
| `segmentation.dimensions` | `["grade", "purpose_group", "home_ownership"]` | ≥ 1 distinct `SegmentationDimension` |
| `arm_space.amounts` | `[5000, 10000, …, 40000]` (step 5000) | > 0 |
| `arm_space.rates` | `[0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28]` | fractions in (0, 1) |
| `arm_space.terms` | `[36, 60]` | positive integers (months) |
| `arm_space.include_null` | `true` | arm 0 = decline |
| `arm_space.max_payment_to_income` | `0.20` | guardrail: installment ≤ x · annual_inc / 12; `null` = off |
| `arm_space.max_amount_to_income` | `0.50` | guardrail: amount ≤ x · annual_inc; `null` = off |
| `arm_space.max_total_dti` | `0.43` | guardrail: dti/100 + 12·installment/annual_inc ≤ x (QM ceiling, 12 CFR 1026.43(e)(2)(vi)); `null` = off |
| `arm_space.rate_corridor` | `0.05` | guardrail: \|rate − sub-grade reference rate\| ≤ x; `null` = off |
| `hierarchy.kappa_group` / `kappa_leaf` | `20` / `10` | > 0 or `"auto"` (Beta-binomial method of moments) |
| `hierarchy.alpha0` / `beta0` | `1` / `1` | > 0 |
| `hierarchy.policy_effect` | `true` | structured leaf prior (policy-effect tilt, docs/MATH.md §2.6) |
| `hierarchy.policy_effect_cfg` | see `PolicyEffectConfig` | partial object: `features`, `ridge` (1e-3), `max_abs_logit_shift` (3.0), `min_rows` (5000), `min_elapsed_rows` (20000), `max_iter` (25), `pti_cap` (1.5) |
| `recovery.min_obs` | `5` | ≥ 1 |
| `recovery.basis` | `"principal"` | `principal` \| `post_chargeoff` |
| `recovery.term_level` | `true` | Stage-2 fallback leaf → (group, term) → group → global (docs/MATH.md §2.9) |
| `mctm.structure` | `"proposal"` | `proposal` \| `extended` |
| `mctm.prior_strength` | `1` | ≥ 0 (0 = MLE) |
| `mctm.chargeoff_lag_months` | `4` | ≥ 1 |
| `reward.funding_rate` | `0.04` | annual fraction in [0, 0.5] |
| `reward.risk_aversion` | `0` | λ ≥ 0, score = mean − λ·std |
| `reward.lendable_threshold` | `0.6` | in [0, 1] |
| `reward.prepayment_adjustment` | `"empirical"` | `"empirical"` (expected interest and funding × realised share ρ by grade × term, docs/MATH.md §2.8) \| `null` (contract) |
| `train_end` | `"2015-01"` | epoch inside the dataset range; fit on `issue_d < first day of train_end` |

```json
{ "dataset": 2, "name": "baseline · grade × purpose × home",
  "segmentation": { "dimensions": ["grade", "purpose_group", "home_ownership"] },
  "hierarchy": { "kappa_group": "auto", "kappa_leaf": 10 },
  "mctm": { "structure": "extended" },
  "reward": { "funding_rate": 0.04, "risk_aversion": 0.25 },
  "train_end": "2015-01" }
```

### 6.2 `GET /fits/?dataset=` · `GET /fits/{id}/` → `Fit`

`config` is the fully resolved `FitConfig` (defaults filled). `metrics`, `n_groups` are null
until `ready`. `n_arms` counts the null arm.

```jsonc
{
  "id": 3, "name": "baseline · grade × purpose × home", "dataset": 2, "dataset_name": "Synthetic twin · 200k · shift 2016-06",
  "status": "ready",
  "config": { "segmentation": { "dimensions": ["grade", "purpose_group", "home_ownership"] },
              "arm_space": { "amounts": [5000, 10000 /* … */], "rates": [0.07, 0.1 /* … */], "terms": [36, 60],
                             "include_null": true, "max_payment_to_income": 0.4 },
              "hierarchy": { "kappa_group": 20, "kappa_leaf": 10, "alpha0": 1, "beta0": 1 },
              "recovery": { "min_obs": 5, "basis": "principal" },
              "mctm": { "structure": "proposal", "prior_strength": 1, "chargeoff_lag_months": 4 },
              "reward": { "funding_rate": 0.04, "risk_aversion": 0, "lendable_threshold": 0.6 },
              "train_end": "2015-01" },
  "metrics": {
    "brier": 0.08653, "brier_baseline": 0.09181, "auc": 0.679, "ece": 0.009, "log_loss": 0.3076,
    "recovery_mae": 0.1578, "recovery_rmse": 0.1928,
    "transition_log_lik": -146865.5, "transition_log_lik_per_obs": -0.10254, "transition_prediction_error": 0.01299,
    "n_train": 46645, "n_test": 34065, "n_defaults_train": 6712, "base_rate_train": 0.1439, "base_rate_test": 0.0998,
    "fit_seconds": 38.4
  },
  "n_groups": 84, "n_arms": 129, "created_at": "2026-10-14T12:18:42Z", "job": { /* Job */ }
}
```

Metric definitions (Table 11, predictive dimensions):

| Field | Definition |
|---|---|
| `brier` | mean (p̂ − y)² on the holdout |
| `brier_baseline` | Brier of the constant training base rate (skill reference) |
| `auc` | ROC-AUC of p̂ on the holdout |
| `ece` | expected calibration error over the 10 reliability bins (prediction deciles, §6.5) |
| `log_loss` | mean binary cross-entropy (nats) |
| `recovery_mae` / `recovery_rmse` | error of E[R] vs realized `y_recovery_rate` on defaulted holdout loans |
| `transition_log_lik` | holdout Σ log P[i,j] over loan-month transitions; `_per_obs` divides by their count |
| `transition_prediction_error` | mean absolute error of the transient rows of P vs holdout empirical frequencies |
| `n_train` / `n_test` | resolved training loans / holdout loans |

Holdout = resolved loans with `issue_d ≥ train_end`. **Recommended** (and what the mock
does): restrict it to *matured* cohorts (issue + term ≤ snapshot) so the resolved-only sample
is not dominated by early defaults; report the size in `n_test` either way.

### 6.3 `GET /fits/{id}/groups/` → `HierarchyRow[]`

| Query | Returns |
|---|---|
| none | the global row followed by every group row |
| `level=global` | `[global]` |
| `level=group` | all groups, **including groups with no training data** (posterior = prior, `n_obs` 0) — the cartesian product of observed dimension values (84 for the default segmentation) |
| `level=leaf&group=<key>` | that group's observed leaves (arms with `n_obs > 0`); 400 without `group` |

```jsonc
{
  "level": "group", "key": "A|credit_card|OTHER", "parent": "global", "group": "A|credit_card|OTHER", "arm": null,
  "dims": { "grade": "A", "purpose_group": "credit_card", "home_ownership": "OTHER" },
  "alpha": 4.878, "beta": 21.122, "n_obs": 6, "defaults": 2,
  "mean": 0.18762,                     // (α + d) / (α + β + n), here α, β already include the data
  "hdi_low": 0.0534, "hdi_high": 0.33591,
  "prior_mean": 0.14391,               // the parent's posterior mean: the shrinkage target
  "kappa": 20, "shrinkage_weight": 0.2308,   // n / (n + κ)
  "empirical_rate": 0.33333,           // d / n, null when n = 0
  "recovery": { "mean": 0.3857, "alpha": 2.22, "beta": 3.537, "n_obs": 2, "source": "global" },
  "n_children": 5                      // observed leaves (groups: leaves; global: groups)
}
```

Key formats: global `"global"`, group = group key, leaf `"<group>#<arm>"`. Leaf rows also
carry `dims.arm` = the arm label (`"$5k @ 7% / 36m"`). `alpha`/`beta` are the posterior
parameters; `hdi_*` the 95 % highest-density interval.

### 6.4 `GET /fits/{id}/risk/?group=&arm=` → `RiskEstimate`

`group` is required (400 otherwise). Without `arm` the group-level posterior is returned
(`arm`, `policy`, `reward` null). `p_default.level` says which node produced the estimate
(`leaf` when observed, otherwise its prior); `recovery.level` follows the min-obs fallback.
Density grids have 101 points on [0, 1]; `p_default_prior` is the node's prior.

```jsonc
{
  "fit": 3, "group": "B|debt_consolidation|MORTGAGE", "arm": 27,
  "policy": { "amount": 20000, "rate": 0.13, "term": 36 }, "label": "$20k @ 13% / 36m",
  "p_default": { "alpha": 5.128, "beta": 100.872, "mean": 0.04838, "std": 0.02074, "hdi_low": 0.01259, "hdi_high": 0.08948,
                 "level": "leaf", "n_obs": 96, "defaults": 4, "shrinkage_weight": 0.9057 },
  "recovery": { "alpha": 2.15, "beta": 3.198, "mean": 0.402, "std": 0.1946, "hdi_low": 0.0528, "hdi_high": 0.7663,
                "level": "group", "n_obs": 4 },
  "expected_loss_fraction": 0.02893,                       // E[p]·(1 − E[R])
  "reward": { "mean": 2266.41, "std": 3146.03, "revenue": 4155.63, "expected_loss": 578.56,
              "funding_cost": 1310.66, "risk_adjusted": 2266.41 },
  "pdf": { "p_default": [ { "x": 0, "density": 0.00424 }, { "x": 0.01, "density": 1.45725 } /* … 101 */ ],
           "p_default_prior": [ /* 101 points */ ], "recovery": [ /* 101 points */ ] }
}
```

### 6.5 `GET /fits/{id}/calibration/` → `Calibration`

Reliability: 10 **equal-frequency** bins (prediction deciles) with `lo`/`hi` = min/max
prediction in the bin. ROC: ≤ 200 points, `fpr` ascending, from (0, 0) to (1, 1).
`recovery` = the global stage-2 Beta (method of moments on defaulted *training* loans) and
the holdout histogram of realized recovery rates (20 bins on [0, 1]).

```jsonc
{
  "fit": 3, "n_test": 34065, "base_rate": 0.0998, "brier": 0.08653, "auc": 0.679, "ece": 0.009, "log_loss": 0.3076,
  "bins": [ { "bin": 0, "lo": 0.0028, "hi": 0.0282, "n": 3406, "mean_predicted": 0.0193, "observed_rate": 0.0379 } /* … 10 */ ],
  "roc": [ { "fpr": 0, "tpr": 0, "threshold": 1 }, { "fpr": 0.0004, "tpr": 0.0021, "threshold": 0.5631 } /* … */ ],
  "recovery": {
    "alpha": 2.22, "beta": 3.537, "mean": 0.3857, "n_defaults": 6712,
    "histogram": [ { "lo": 0, "hi": 0.05, "count": 41 } /* … 20 */ ],
    "pdf": [ { "x": 0, "density": 0.01045 } /* … 101 */ ],
    "mae": 0.1578, "rmse": 0.1928,
    "by_grade": [ { "grade": "A", "mean": 0.4071, "n": 312 } /* … */ ]
  }
}
```

### 6.6 `GET /fits/{id}/mctm/?structure=&horizon=60&start=current` → `Mctm`

Defaults: `structure` = the fit's configured structure (the other one is estimated on
demand from the same panel), `horizon` 60 (1–120), `start` `current` | `delinquent`.
All matrices are indexed by `states` = `["current", "delinquent", "recovered", "loss"]`.

| Field | Meaning |
|---|---|
| `matrix` | 4×4 posterior-mean P (Dirichlet rows); rows sum to 1; absorbing rows are unit vectors |
| `counts` | 4×4 observed loan-month transition counts (training panel); absorbing rows 0 |
| `structural_zeros` | 4×4 booleans: forced zeros (C→R, C→L under `proposal`; C→L under `extended`) and the off-diagonal cells of absorbing rows |
| `matrix_low` / `matrix_high` | elementwise 95 % marginal credible bounds (Beta marginals of the Dirichlet rows) |
| `absorption` | B = N·R with N = (I − Q)⁻¹: rows [current, delinquent] × cols [recovered, loss] |
| `expected_steps` | expected months to absorption from [current, delinquent] (N·1) |
| `trajectory` | π_t = π₀Pᵗ for t = 0…horizon (`horizon + 1` rows) |
| `p_default_curve` | P(Default, t) = π_D(t) + π_L(t), with its two components |
| `maturity` | the maturity collapse at t = horizon: `before` = π_T, `after` = C mass → R, D mass split p_DR : p_DL; `p_default` = `after[3]` |
| `log_likelihood`, `n_transitions` | holdout transition log-likelihood (nats) and its loan-month count |
| `row_prediction_error` | per-row holdout mean |ΔP| (absorbing rows 0) |

```jsonc
{
  "fit": 3, "structure": "proposal", "states": ["current", "delinquent", "recovered", "loss"],
  "state_labels": ["Current", "Delinquent", "Recovered", "Absorbed Loss"],
  "matrix": [[0.984475, 0.015525, 0, 0], [0.346482, 0.40049, 0.05034, 0.202688], [0, 0, 1, 0], [0, 0, 0, 1]],
  "counts": [[3997559, 63039, 0, 0], [31839, 36802, 4625, 18625], [0, 0, 0, 0], [0, 0, 0, 0]],
  "structural_zeros": [[false, false, true, true], [false, false, false, false], [true, true, false, true], [true, true, true, false]],
  "matrix_low": [[0.984355, 0.015405, 0, 0] /* … */], "matrix_high": [[0.984595, 0.015645, 0, 0] /* … */],
  "absorption": [[0.198951, 0.801049], [0.198951, 0.801049]], "expected_steps": [156.569, 92.156],
  "start": "current", "horizon": 36,
  "trajectory": [ { "t": 0, "current": 1, "delinquent": 0, "recovered": 0, "loss": 0 } /* … 37 rows */ ],
  "p_default_curve": [ { "t": 1, "p_default": 0.015525, "p_loss": 0, "p_delinquent": 0.015525 } /* … */ ],
  "maturity": { "t": 36, "before": [0.780125, 0.020422, 0.039681, 0.159772], "after": [0, 0, 0.82387, 0.17613], "p_default": 0.17613 },
  "log_likelihood": -146865.5, "n_transitions": 1432297, "row_prediction_error": [0.014882, 0.011098, 0, 0], "prior_strength": 1
}
```

### 6.7 `GET /fits/{id}/arms/` → `FitArms`

Every arm in grid order with its pooled economics: `expected_reward` = group-share-weighted
expected reward of offering this arm to every borrower (USD per loan), `reward_std`,
`p_default`, `expected_recovery` (same weighting), `historical_count` / `historical_share`
= training loans whose historical policy snaps to the arm.

```jsonc
{
  "fit": 3, "amounts": [5000, 10000 /* … */], "rates": [0.07, 0.1 /* … */], "terms": [36, 60],
  "arms": [
    { "arm": 0, "amount": 0, "rate": 0, "term": 0, "is_null": true, "installment": 0, "label": "decline",
      "expected_reward": 0, "reward_std": 0, "p_default": 0, "expected_recovery": 0, "historical_count": 0, "historical_share": 0 },
    { "arm": 1, "amount": 5000, "rate": 0.07, "term": 36, "is_null": false, "installment": 154.39, "label": "$5k @ 7% / 36m",
      "expected_reward": 11.87, "reward_std": 874.67, "p_default": 0.06841, "expected_recovery": 0.3918,
      "historical_count": 1649, "historical_share": 0.03535 }
    /* … 129 arms */
  ]
}
```

`ArmRow` (shared by fit, shadow and simulation arm payloads): `arm`, `amount`, `rate`, `term`,
`is_null`, `installment` (monthly, USD), `label` (`"$15k @ 13% / 36m"` or `"decline"`).

---

## 7. Shadow lending engine

### 7.1 `POST /fits/{id}/shadow/evaluate/` → 200 `ShadowReport` (synchronous)

Two request forms:

```json
{ "loan_id": 1000021 }
```
```json
{ "context": { "grade": "C", "sub_grade": "C2", "annual_inc": 62000, "dti": 21.4, "fico_range_low": 690,
               "home_ownership": "RENT", "purpose": "credit_card" },
  "historical_policy": { "amount": 12000, "rate": 0.1399, "term": 36 } }
```

`context` may contain **PRE_DECISION columns only**. Any other key (e.g. `loan_status`,
`int_rate`, `out_prncp`) is rejected with **400** and a LeakageGuard message:
`{ "context": ["LeakageError: columns not available at stage 'pre_decision': loan_status"] }`.
Missing context keys are treated as unknown (`null`). 409 if the fit is not ready, 404 for an
unknown `loan_id`.

| Field | Meaning |
|---|---|
| `context` | the exact context the engine saw (all 37 keys, `null` when unknown) |
| `context_columns` | the non-null columns used — all PRE_DECISION |
| `leakage_safe` | `true` when the context passed `LeakageGuard.assert_allowed(…, PRE_DECISION)` |
| `n_samples` | posterior-predictive draws per arm |
| `arms[]` | one `ShadowArmRow` per arm, grid order (null arm first, all zeros, always eligible) |
| `arms[].eligible` | passes every lending guardrail (payment-to-income, amount-to-income, total DTI, rate corridor; a guardrail with an unknown input is skipped) — or is the historical arm |
| `arms[].realization` ★ | prepayment factor ρ(grade, term) applied to the arm's reward |
| `eligibility` ★ | `{reference_rate, n_eligible, n_eligible_lending, historical_admitted, guardrails, rate_corridor_applied}` (§13.1) |
| `arms[].p_default_mean`, `_hdi_low/high` | leaf posterior of (group, arm) |
| `arms[].expected_recovery` | E[R \| D] of the leaf (with fallback) |
| `arms[].reward_mean`, `reward_std` | closed-form reward moments (SPEC §2.7) |
| `arms[].reward_q05/q50/q95`, `prob_positive` | Monte-Carlo posterior predictive |
| `arms[].risk_adjusted` | `reward_mean − λ·reward_std`, the argmax criterion |
| `recommended_arm` | argmax `risk_adjusted` among eligible arms (null arm = 0 competes) |
| `historical_policy`, `historical_arm` | the issued (amount, rate, term) and its nearest arm |
| `historical_expected_reward` | model reward of the **exact** historical policy |
| `historical_realized_reward` | `outcome.realized_reward`, null when unresolved |
| `uplift_expected` / `uplift_vs_realized` | recommended `reward_mean` minus the two historical figures |
| `lendable` | ∃ eligible lending arm with `prob_positive ≥ lendable_threshold` |
| `outcome` | post-hoc outcome of the historical loan, evaluation only; null for context requests |

```jsonc
{
  "fit": 3, "loan_id": 1000021, "issue_d": "2012-01", "group": "B|debt_consolidation|MORTGAGE",
  "context": { "annual_inc": 94000, "dti": 14.2, "fico_range_low": 710, "grade": "B", "sub_grade": "B3" /* … all 37 keys */ },
  "context_columns": ["annual_inc", "application_type" /* … 32 */], "leakage_safe": true,
  "n_samples": 1000, "risk_aversion": 0, "lendable_threshold": 0.6, "max_payment_to_income": 0.4,
  "arms": [
    { "arm": 0, "amount": 0, "rate": 0, "term": 0, "is_null": true, "installment": 0, "label": "decline", "eligible": true,
      "p_default_mean": 0, "p_default_hdi_low": 0, "p_default_hdi_high": 0, "expected_recovery": 0, "reward_mean": 0,
      "reward_std": 0, "reward_q05": 0, "reward_q50": 0, "reward_q95": 0, "prob_positive": 0, "risk_adjusted": 0 },
    { "arm": 1, "amount": 5000, "rate": 0.07, "term": 36, "is_null": false, "installment": 154.39, "label": "$5k @ 7% / 36m",
      "eligible": true, "p_default_mean": 0.10522, "p_default_hdi_low": 0.00818, "p_default_hdi_high": 0.22524,
      "expected_recovery": 0.402, "reward_mean": -104.75, "reward_std": 1051.2, "reward_q05": -3226.81,
      "reward_q50": 239.09, "reward_q95": 239.09, "prob_positive": 0.891, "risk_adjusted": -104.75 }
    /* … 129 arms */
  ],
  "recommended_arm": 115,
  "historical_policy": { "amount": 14250, "rate": 0.1038, "term": 36 }, "historical_arm": 18,
  "historical_expected_reward": 846.83, "historical_realized_reward": -12551.38,
  "uplift_expected": 3626.69, "uplift_vs_realized": 17024.9, "lendable": true,
  "outcome": { "loan_status": "Charged Off", "resolved": true, "defaulted": true, "recovery_rate": 0.0987,
               "realized_revenue": 475.45, "realized_loss": 12843.65, "realized_reward": -12551.38,
               "months_on_book": 4, "resolution_epoch": "2012-09" }
}
```

### 7.2 Portfolio evaluation

`POST /fits/{id}/shadow/portfolio/` → **202** `Job` (kind `shadow_portfolio`, target = the fit)

```json
{ "max_loans": 5000, "seed": 7, "filters": { "grade": ["E", "F", "G"], "resolved_only": true } }
```

`max_loans` 100 – 50,000. Optional `filters`: `grade[]`, `purpose[]`, `home_ownership[]`,
`issue_from`, `issue_to` (default: `issue_d ≥ train_end`), `resolved_only`.

`GET /fits/{id}/shadow/portfolio/` ★ → `PortfolioRun[]` (newest first, not paginated):
`{ job, request, summary | null }` — lets the console show past runs after a reload.

`GET /fits/{id}/shadow/portfolio/{job_id}/` → `PortfolioResult`. While the job runs,
`summary` is null and the arrays are empty (poll it).

Each sampled borrower gets exactly one class, first match in this order:

| Class | Rule |
|---|---|
| `declined` | recommended arm is the null arm |
| `rescued` | historical loan realised a loss (`realized_reward < 0`) and the shadow arm (different from the historical one) has positive expected reward |
| `upsized` | recommended amount > historical amount, positive expected reward |
| `repriced_down` / `repriced_up` | recommended rate below / above the historical rate |
| `unchanged` | otherwise (same arm) |

```jsonc
{
  "job": { "id": 108, "kind": "shadow_portfolio", "status": "succeeded", "progress": 1 /* … */ },
  "fit": 3, "request": { "max_loans": 5000, "seed": 7, "filters": { "resolved_only": false } },
  "summary": {
    "n_loans": 5000, "n_lendable": 5000, "share_lendable": 1,
    "expected_profit_historical": 3247698.5, "expected_profit_shadow": 17660230.18,
    "uplift": 14412531.68, "uplift_pct": 4.4378,           // uplift / |expected_profit_historical|
    "realized_profit_historical": -2600010.76,            // resolved loans only; null if none resolved
    "counts": { "rescued": 502, "upsized": 3997, "repriced_down": 96, "repriced_up": 207, "declined": 0, "unchanged": 198 }
  },
  "by_grade": [ { "grade": "A", "n": 1202, "expected_profit_historical": 305262.07, "expected_profit_shadow": 3749858.16,
                  "realized_profit_historical": -303261.99, "counts": { "rescued": 43, "upsized": 1094 /* … */ } } /* … A–G */ ],
  "inclusion": [ { "grade": "A", "n": 1202, "share_lendable": 1, "decline_rate_shadow": 0, "mean_rate_historical": 0.0659,
                   "mean_rate_shadow": 0.1111, "mean_amount_historical": 15066, "mean_amount_shadow": 35649 } /* … */ ],
  "top_loans": [ { "loan_id": 2131300, "grade": "E", "group": "E|debt_consolidation|RENT", "historical_arm": 126,
                   "recommended_arm": 77, "historical_label": "$40k @ 22% / 60m", "recommended_label": "$10k @ 19% / 60m",
                   "historical_expected_reward": -17521.39, "shadow_expected_reward": 1972.07,
                   "realized_reward": -25156.66, "class": "rescued" } /* … 50, largest |shadow − historical| first */ ]
}
```

`mean_rate_shadow` / `mean_amount_shadow` are over non-declined borrowers (null if all declined).

---

## 8. Simulations

### 8.1 `POST /simulations/` → 201 `Simulation`

`fit` must be `ready` (409). Omitted fields take these defaults
(`SimulationConfig` / `PSIConfig` in `core/bayesian_credit/config.py`):

| Field | Default | Validation / meaning |
|---|---|---|
| `fit`, `name` | — | required |
| `policies` | — | required, ≥ 1 distinct `PolicyName` |
| `feedback_mode` | `"simulated"` | `simulated` (posterior-predictive environment) · `replay` (rejection sampling: update only when arm == historical) · `hybrid` |
| `start` / `end` | `"2015-01"` / `"2018-12"` | inclusive epochs inside the dataset range |
| `max_loans` | `50000` | borrowers sampled from the stream with `seed`; `null` = all |
| `seed` | `42` | |
| `interim_months` | `12` | months of interim feedback per loan |
| `interim_weight` | `0.5` | weight of an interim observation in [0, 1] (0 = ignore) |
| `psi.features` | `["fico_range_low", "dti", "annual_inc", "revol_util", "inq_last_6mths", "grade"]` | PRE_DECISION features |
| `psi.window_months` | `3` | arrivals window per reading |
| `psi.thresholds` | `{ "watch": 0.10, "breaker": 0.25 }` | `0 < watch < breaker` |
| `psi.aggregate` | `"max"` | `max` \| `mean` across features |
| `psi.recovery_readings` | `3` | consecutive normal readings that close a half-open breaker |
| `psi.fallback` | `"champion_snapshot"` | decisions while the breaker is open |
| `psi.auto_action` | `"resume"` | applied when `breaker_mode = "auto_ack"` |
| `psi.auto_ack_epochs` | `1` | epochs before the automatic acknowledgement |
| `breaker_mode` | `"auto_ack"` | `pause` (wait for a human) · `auto_ack` · `off` (monitor only) |
| `checkpoint_every` | `6` | epochs between checkpoints (≥ 1) |
| `hindsight_sample` ★ | `"elapsed"` | loans the oracle / regret yardstick is fitted on: resolved loans whose term elapsed by the snapshot (unbiased) · `resolved` (every resolved loan; biased towards early defaults) |

```json
{ "fit": 3, "name": "Thompson vs baselines · simulated feedback",
  "policies": ["thompson", "ucb1", "epsilon_greedy", "historical", "random"],
  "feedback_mode": "simulated", "start": "2015-01", "end": "2018-12", "max_loans": 50000, "seed": 42,
  "interim_months": 12, "interim_weight": 0.5,
  "psi": { "features": ["fico_range_low", "dti", "annual_inc", "revol_util", "inq_last_6mths", "grade"],
           "window_months": 3, "thresholds": { "watch": 0.1, "breaker": 0.25 }, "auto_action": "resegment" },
  "breaker_mode": "pause", "checkpoint_every": 6 }
```

### 8.2 `GET /simulations/?fit=` · `GET /simulations/{id}/` → `Simulation`

| Field | Meaning |
|---|---|
| `status`, `progress` | `SimulationStatus`; `progress = epochs_done / epochs_total` |
| `config` | fully resolved `SimulationConfig` |
| `current_epoch` | last **completed** epoch (null before the first) |
| `breaker_state`, `sentinel_level`, `psi_aggregate` | as of `current_epoch` |
| `pending_reviews` | pending sentinel reviews of this run |
| `last_checkpoint_epoch` | newest checkpoint on disk |
| `restarts` | resume-from-checkpoint restarts so far |
| `worker` | `{ pid, alive, last_heartbeat }` |
| `last_recovery` | the most recent kill → resume cycle (G.6 state recovery), see below; null if never killed |
| `headline[]` | one `PolicyHeadline` per configured policy through `current_epoch` (`[]` before the first epoch) |
| `uplift_vs_historical` | best adaptive policy's `cum_reward` − historical `cum_reward` (null without `historical`) |

`RecoveryInfo`: `killed_at`, `killed_at_epoch`, `resumed_at`, `resumed_from_epoch` (checkpoint
epoch, null = from the start), `replayed_epochs`, `deterministic_match` (null until the replay
passes the kill point; then whether the replayed `EpochCompleted` metrics equal the pre-kill
ones — the SPEC §2.11 determinism property), `recovery_seconds` (kill → replay back past the
kill point).

`PolicyHeadline`: `policy`, `n_decisions`, `cum_reward`, `cum_realized`, `cum_regret`,
`mean_reward` (= cum_reward / n_decisions), `decline_rate`, `mean_amount`, `mean_rate`
(decision-weighted; amount and rate over non-declined decisions).

```jsonc
{
  "id": 2, "name": "Thompson vs baselines · simulated feedback", "fit": 3, "fit_name": "baseline · grade × purpose × home",
  "dataset": 2, "status": "completed", "progress": 1, "config": { /* SimulationConfig */ },
  "epochs_total": 48, "epochs_done": 48, "current_epoch": "2018-12",
  "breaker_state": "closed", "sentinel_level": "watch", "psi_aggregate": 0.1122, "pending_reviews": 0,
  "last_checkpoint_epoch": "2018-12", "restarts": 1,
  "worker": { "pid": null, "alive": false, "last_heartbeat": "2026-10-14T16:27:36Z" },
  "last_recovery": { "killed_at": "2026-10-14T16:23:08Z", "killed_at_epoch": "2017-04", "resumed_at": "2026-10-14T16:23:49Z",
                     "resumed_from_epoch": "2016-12", "replayed_epochs": 3, "deterministic_match": true, "recovery_seconds": 69.8 },
  "headline": [ { "policy": "thompson", "n_decisions": 50002, "cum_reward": 119139216.01, "cum_realized": 68928145.93,
                  "cum_regret": 53421152.6, "mean_reward": 2382.69, "decline_rate": 0.0023, "mean_amount": 28190, "mean_rate": 0.1404 }
                /* … one per policy */ ],
  "uplift_vs_historical": 84775988.42,
  "created_at": "2026-10-14T16:18:42Z", "started_at": "2026-10-14T16:18:46Z", "finished_at": "2026-10-14T16:27:36Z",
  "job": { /* Job, kind "simulation" */ }
}
```

### 8.3 Lifecycle and controls

```
queued ─▶ running ─┬─▶ completed
                   ├─(pause)──────────▶ paused ─(resume)─▶ running
                   ├─(breaker trips, breaker_mode=pause)─▶ awaiting_review ─(review acknowledged)─▶ running
                   ├─(chaos / worker death)─▶ interrupted ─(resume: new worker from last checkpoint)─▶ running
                   └─▶ failed
any non-terminal ─(cancel)─▶ cancelled
```

`POST /simulations/{id}/{action}/` (empty body) → the updated `Simulation`; invalid
transitions → **409** with `detail`:

| Action | Allowed from | Effect |
|---|---|---|
| `pause` | `running` | finish the current epoch, keep the worker, `paused` |
| `resume` | `paused` → continue; `interrupted` → spawn a worker that restores the last checkpoint (policies, risk model, feedback queue, RNG bit-generator state) and replays the epochs after it; `restarts += 1`, `last_recovery` updated | 409 from `awaiting_review` ("acknowledge the review") |
| `cancel` | `queued`, `running`, `paused`, `awaiting_review`, `interrupted` | stop, `cancelled` |
| `chaos` | `running` | **SIGKILL** the worker (Table 10 "workload termination / state recovery"); status `interrupted`, `worker.alive = false`, `last_recovery.killed_*` set |

When a breaker trips with `breaker_mode = "pause"`, the trip epoch completes with policies
frozen (fallback decisions), a `Review` is created, the run waits in `awaiting_review`, and
acknowledging the review resumes it from the next epoch.

### 8.4 `GET /simulations/{id}/timeseries/?policy=` → `Timeseries`

Rows for completed epochs, ordered by epoch then by `config.policies` order. Sums are over the
epoch's decisions unless the field is `cum_*` / `rolling_*`; `breaker_state` and
`psi_aggregate` are global per epoch and repeated on every policy row.

| Field | Meaning |
|---|---|
| `n_decisions` | borrowers decided in the epoch (same for every policy) |
| `reward`, `cum_reward` | Σ expected risk-adjusted reward of the chosen arms (USD) |
| `realized`, `cum_realized` | Σ realized reward released this epoch (interim × `interim_weight` + terminal) |
| `regret`, `cum_regret` | Σ pseudo-regret vs the hindsight oracle (best arm per group on the model fitted to all resolved loans) |
| `rolling_reward` | mean reward **per decision** over the trailing 6 epochs |
| `decline_rate` | share of decisions on the null arm |
| `mean_amount`, `mean_rate` | over non-declined decisions (null if all declined) |

```jsonc
{
  "simulation": 2, "policies": ["thompson"], "epochs": ["2015-01", "2015-02" /* … 48 */],
  "rows": [
    { "epoch": "2015-01", "policy": "thompson", "n_decisions": 917, "reward": -1054818.53, "cum_reward": -1054818.53,
      "realized": 0, "cum_realized": 0, "regret": 4387287.73, "cum_regret": 4387287.73, "rolling_reward": -1150.29,
      "decline_rate": 0.0088, "mean_amount": 21886, "mean_rate": 0.1426, "breaker_state": "closed", "psi_aggregate": 0.0313 }
    /* … */
  ]
}
```

### 8.5 `GET /simulations/{id}/arms/?policy=&epoch=&group=` → `SimArms`

Defaults: `policy` = first configured, `epoch` = last completed (400 if not completed),
`group` ★ = pooled. 409 before the first epoch.

`SimArmRow` = `ArmRow` + `mu`, `sigma` (posterior N(μ, σ²) of the arm — the group's bandit
when `group` is given, otherwise the decision-weighted pooled posterior; `null` for policies
without one before any selection), `n` (effective observations, interim weighted),
`selections` / `cum_selections` (this epoch / start → epoch), `share` / `cum_share`.
`groups[]` lists every group with its decision count (desc) for the group picker.

```jsonc
{
  "simulation": 2, "policy": "thompson", "epoch": "2018-12", "group": null, "n_decisions": 1165,
  "amounts": [5000 /* … */], "rates": [0.07 /* … */], "terms": [36, 60],
  "arms": [ { "arm": 1, "amount": 5000, "rate": 0.07, "term": 36, "is_null": false, "installment": 154.39,
              "label": "$5k @ 7% / 36m", "mu": 17.17, "sigma": 886.73, "n": 61.5, "selections": 1,
              "cum_selections": 103, "share": 0.00126, "cum_share": 0.00206 } /* … all arms */ ],
  "groups": [ { "group": "B|debt_consolidation|MORTGAGE", "n": 4467 } /* … */ ]
}
```

### 8.6 `GET /simulations/{id}/posteriors/?policy=&group=&arms=1,2` ★ → `PosteriorHistory`

Bandit posterior history for one group (from the sampled `PosteriorUpdated` snapshots, every
`snapshot_every` epochs plus the last completed epoch). Default `arms`: the four most-selected
lending arms of the group; default `group`: the largest.

```jsonc
{
  "simulation": 2, "policy": "thompson", "group": "B|debt_consolidation|MORTGAGE", "snapshot_every": 3,
  "arms": [ { "arm": 115, "amount": 35000, "rate": 0.13, "term": 60, "is_null": false, "installment": 796.36, "label": "$35k @ 13% / 60m" } /* … */ ],
  "rows": [ { "epoch": "2015-03", "arm": 115, "mu": 5437.5, "sigma": 10228.83, "n": 48.3 } /* … epochs × arms */ ]
}
```

`sigma` is the proposal-scale σ (reward dispersion, SPEC §2.8); the console derives σ/√n.

### 8.7 `GET /simulations/{id}/simpson/?policy=&epoch=` ★ → `SimpsonPanel`

Aggregation-bias check (§3.1, Simpson's paradox): `pooled_best_arm` = the arm with the highest
decision-weighted posterior mean when all borrowers are pooled; per group, the group's best
arm (only arms with ≥ 3 effective observations qualify) vs the pooled arm *inside that group*.

```jsonc
{
  "simulation": 2, "policy": "thompson", "epoch": "2018-12",
  "pooled_best_arm": 99, "pooled_best_label": "$25k @ 13% / 60m", "pooled_best_mu": 1911.6,
  "n_groups": 84, "n_agree": 29,
  "forfeited_per_decision": 1589.72,                    // Σ_g w_g · gap_g
  "groups": [ { "group": "B|debt_consolidation|MORTGAGE", "dims": { "grade": "B", "purpose_group": "debt_consolidation", "home_ownership": "MORTGAGE" },
                "n": 4467, "group_best_arm": 60, "group_best_label": "$40k @ 16% / 36m", "group_best_mu": 9858.16,
                "pooled_arm_mu": 3482.62, "gap": 6375.54, "agrees": false } /* … ordered by n desc */ ]
}
```

### 8.8 `GET /simulations/{id}/events/` → `EventPage`

Query: `type=A,B` (comma list), `after=<seq>` (exclusive), `limit` (default 100, max 1000),
`order=asc|desc` ★ (`desc` = newest first, the log tail). Without `type`, the per-borrower
types (`BorrowerArrived`, `PolicySelected`) are excluded. `last_seq` is the highest seq in the
store (use it as the next `after` when tailing incrementally).

| `type` | Level | `payload` |
|---|---|---|
| `RunStarted` | info | `{ policies, seed }` |
| `BorrowerArrived` | info | `{ loan_id, group }` |
| `PolicySelected` | info | `{ loan_id, policy, arm }` |
| `FeedbackReleased` | info | `{ interim, terminal }` (counts) |
| `PosteriorUpdated` | info | `{ policy, group, arm, mu, sigma }` (sampled snapshot) |
| `SentinelEvaluated` | info / warning / critical by level | `{ aggregate, level, top_feature }` (also used for "breaker closed") |
| `BreakerTripped` | critical | `{ review, psi }` |
| `ReviewAcknowledged` | warning | `{ review, action }` |
| `CheckpointWritten` | info | `{ epoch, bytes }` |
| `EpochCompleted` | info | `{ n_decisions, reward: { policy: usd }, breaker_state }` |
| `RunPaused` · `RunResumed` · `WorkerKilled` · `RunCompleted` · `RunCancelled` · `RunFailed` | lifecycle | e.g. `RunResumed { from_checkpoint, replayed_epochs }`, `WorkerKilled { signal, pid }` |

```jsonc
{
  "simulation": 2,
  "results": [
    { "seq": 175, "type": "RunCompleted", "epoch": "2018-12", "ts": "2026-10-14T16:27:36Z", "level": "info",
      "message": "run completed; evaluation report written", "payload": {} },
    { "seq": 174, "type": "CheckpointWritten", "epoch": "2018-12", "ts": "2026-10-14T16:27:36Z", "level": "info",
      "message": "checkpoint written at 2018-12 (policies, risk model, feedback queue, RNG state)",
      "payload": { "epoch": "2018-12", "bytes": 7067000 } }
  ],
  "last_seq": 175, "has_more": true
}
```

### 8.9 `GET /simulations/{id}/sentinel/` → `SentinelTimeline`

Completed epochs only. `features_psi` has one row per feature × epoch; `aggregate` one row per
epoch (`aggregate` = `psi_max` or `psi_mean` per `aggregate_mode`). `reference.rebased_at`
lists epochs where a `resegment`/`rollback` acknowledgement rebased the reference window.
`shift_counts` = epochs at watch / breaker level (Table 11 "frequency of detected shifts").

```jsonc
{
  "simulation": 2, "features": ["fico_range_low", "dti", "annual_inc", "revol_util", "inq_last_6mths", "grade"],
  "thresholds": { "watch": 0.1, "breaker": 0.25 }, "aggregate_mode": "max", "window_months": 3,
  "reference": { "from": "2012-01", "to": "2014-12", "n": 46645, "rebased_at": ["2016-07"] },
  "features_psi": [ { "epoch": "2015-01", "feature": "fico_range_low", "psi": 0.0313, "level": "normal" } /* … */ ],
  "aggregate": [ { "epoch": "2015-01", "psi_max": 0.0313, "psi_mean": 0.021, "aggregate": 0.0313, "level": "normal",
                   "breaker_state": "closed", "top_feature": "fico_range_low" } /* … */ ],
  "transitions": [
    { "epoch": "2016-06", "ts": "2026-10-14T16:21:38Z", "from_state": "closed", "to_state": "open",
      "reason": "aggregate PSI 0.285 > 0.25", "psi_aggregate": 0.2851, "review": 2, "action": null },
    { "epoch": "2016-07", "ts": "2026-10-14T16:21:48Z", "from_state": "open", "to_state": "half_open",
      "reason": "review acknowledged: resegment", "psi_aggregate": 0.0561, "review": 2, "action": "resegment" },
    { "epoch": "2016-10", "from_state": "half_open", "to_state": "closed", "reason": "3 consecutive normal readings" /* … */ }
  ],
  "current": { "epoch": "2018-12", "breaker_state": "closed", "level": "watch", "psi_aggregate": 0.1122, "top_feature": "fico_range_low" },
  "shift_counts": { "watch": 8, "breaker": 1 }
}
```

Breaker semantics (SPEC §2.9): `closed → open` when the aggregate exceeds the breaker
threshold; `open → half_open` on acknowledgement; `half_open → closed` after
`recovery_readings` consecutive normal readings; `watch` is a level, not a state.

### 8.10 `GET /simulations/{id}/sentinel/buckets/?feature=&epoch=` ★ → `PsiBucketTable`

The `PSIResult.bucket_tables` entry for one feature and epoch: numeric features use the
reference deciles as bins, categoricals use levels. `contribution = (a − e)·ln(a / e)`; the
contributions sum to `psi`.

```jsonc
{
  "simulation": 2, "feature": "fico_range_low", "epoch": "2016-06", "kind": "numeric", "psi": 0.2851, "level": "breaker",
  "n_reference": 46645, "n_window": 3149,
  "buckets": [ { "bucket": "[660, 670)", "lo": 660, "hi": 670, "expected": 0.1, "actual": 0.2043, "contribution": 0.07456 } /* … 10 */ ]
}
```

### 8.11 `GET /simulations/{id}/report/` → `EvaluationReport`

`status` is `"partial"` until the run completes (409 before the first epoch). `dimensions` are
the nine Table 11 rows, each tagged with the G.7 evaluation axis:

| `key` | `axis` | Metric keys |
|---|---|---|
| `default_prediction` | predictive | `brier`, `auc`, `ece`, `log_loss` |
| `recovery_prediction` | predictive | `recovery_mae`, `recovery_rmse` |
| `credit_state_modeling` | predictive | `transition_log_lik` (per loan-month), `transition_prediction_error` |
| `policy_performance` | decision | `cum_reward`, `mean_reward`, `cum_realized` |
| `bandit_performance` | decision | `cum_regret`, `convergence_epochs`, `top5_share`, `decline_rate` |
| `counterfactual_evaluation` | decision | `uplift_vs_historical`, `uplift_per_decision`, `uplift_pct` |
| `distributional_monitoring` | adaptability | `psi_max`, `watch_epochs`, `breaker_epochs`, `breaker_trips`, `time_to_close` |
| `computational_feasibility` | feasibility | `cpu_seconds`, `cpu_utilization`, `peak_memory`, `wall_seconds` |
| `infrastructure_resilience` | feasibility | `recovery_seconds`, `failed_workloads`, `restarts` |

`ReportMetric`: `key`, `label`, `value`, `unit` (§1.4), `better` (`higher` / `lower` / null),
`baseline` + `baseline_label` (historical policy for decision metrics, base rate / random /
threshold for the others), `by_policy` (per-policy values for policy-level metrics), `note`.
The console colours Δ = value − baseline by `better`; the backend may add metrics (the grid
renders whatever arrives).

```jsonc
{
  "simulation": 2, "fit": 3, "dataset": 2, "generated_at": "2026-10-16T14:18:47Z", "status": "final",
  "dimensions": [
    { "key": "default_prediction", "label": "Default prediction", "axis": "predictive",
      "metrics": [ { "key": "brier", "label": "Brier score", "value": 0.08653, "unit": "ratio", "better": "lower",
                     "baseline": 0.09181, "baseline_label": "training base rate", "by_policy": null, "note": null } /* … */ ] },
    { "key": "policy_performance", "label": "Policy performance", "axis": "decision",
      "metrics": [ { "key": "cum_reward", "label": "Cumulative risk-adjusted reward", "value": 119139216.01, "unit": "usd",
                     "better": "higher", "baseline": 34363227.59, "baseline_label": "historical policy",
                     "by_policy": { "thompson": 119139216.01, "ucb1": 71865340.18, "historical": 34363227.59 /* … */ },
                     "note": null } /* … */ ] }
    /* … nine dimensions */
  ],
  "runtime": { "cpu_seconds": 461.6, "cpu_utilization": 0.87, "peak_memory_mb": 1660, "wall_seconds": 530.6, "epochs": 48,
               "decisions": 50002, "decisions_per_second": 471, "restarts": 1, "checkpoints_written": 9,
               "failed_workloads": 1, "recovery_seconds": 69.8 },
  "policy_summary": [ /* PolicyHeadline per policy */ ],
  "config": { "fit": { /* FitConfig */ }, "simulation": { /* SimulationConfig */ } }
}
```

`RuntimeStats.cpu_utilization` is mean cores used by the worker (1.0 = one core);
`decisions_per_second` counts policy decisions (borrowers × policies) per wall second.

---

## 9. Sentinel reviews (human in the loop)

`GET /sentinel/reviews/?simulation=&status=pending|acknowledged` → `Paginated<Review>`
(newest first). A review is created whenever a breaker trips (also in `auto_ack` mode, where
it is acknowledged automatically with `acknowledged_by = "auto_ack"`, `auto = true`).

`POST /sentinel/reviews/{id}/acknowledge/` → 200 `Review`; 409 if already acknowledged,
400 for an unknown action.

```json
{ "action": "resegment", "note": "FICO and DTI shifted after 2016-06; refine by fico_band" }
```

| Action | Effect on the run |
|---|---|
| `resume` | keep the incumbent policy, unfreeze updates; reference window unchanged (the breaker usually stays half-open while drift persists) |
| `resegment` | `Segmenter.refine()` (next dimension of `refinement_order`), refit the risk model on data to date, re-warm-start the bandits, rebase the reference window; `segmentation_after` set |
| `rollback` | restore the champion snapshot taken before the drift and rebase the reference window |

The breaker moves to `half_open`; a run waiting in `awaiting_review` resumes from the next epoch.

```jsonc
{
  "id": 1, "simulation": 1, "simulation_name": "Governance drill · breaker pause · hybrid feedback", "epoch": "2016-06",
  "status": "acknowledged", "created_at": "2026-10-14T15:21:42Z", "psi_aggregate": 0.282, "level": "breaker",
  "breaker_state": "half_open",                 // "open" while pending
  "reason": "aggregate PSI 0.282 exceeded the breaker threshold 0.25 (fico_range_low 0.282)",
  "top_features": [ { "feature": "fico_range_low", "psi": 0.282 }, { "feature": "dti", "psi": 0.2409 } /* top 3 */ ],
  "action": "resegment", "note": "FICO/DTI shift", "acknowledged_at": "2026-10-16T14:18:47Z", "acknowledged_by": "researcher",
  "auto": false,
  "segmentation_before": ["grade", "purpose_group", "home_ownership"],
  "segmentation_after": ["grade", "purpose_group", "home_ownership", "fico_band"]
}
```

---

## 10. Status codes per endpoint (summary)

| Situation | Code |
|---|---|
| created dataset / fit / simulation | 201 |
| portfolio job accepted | 202 |
| validation error (incl. LeakageError on shadow context) | 400 |
| dataset / fit not ready, invalid simulation transition, review already acknowledged, simulation sub-resource before the first epoch | 409 |
| unknown id | 404 |

---

## 11. Additions and decisions beyond SPEC §3 (the backend must honour these)

1. **New endpoints**: `GET /jobs/` (filters `status`, `kind`); `GET /fits/{id}/shadow/portfolio/`
   (run list); `GET /simulations/{id}/posteriors/`; `GET /simulations/{id}/simpson/`;
   `GET /simulations/{id}/sentinel/buckets/`.
2. **New query params**: `GET /fits/?dataset=`, `GET /simulations/?fit=`,
   `GET /simulations/{id}/arms/?group=`, `GET /simulations/{id}/events/?order=desc`,
   `GET /sentinel/reviews/?status=`.
3. **Epoch strings everywhere**: `train_end`, `start`, `end`, `issue_from`, `issue_to`,
   `shift.at` are `"YYYY-MM"`; `train_end` is exclusive (fit on `issue_d` before its first
   day), `start`/`end` inclusive. The core's `"2015-01-01"`-style dates are converted server-side.
4. **`shift`** is `{ at, deltas }` (additive deltas per feature), mapped to
   `SyntheticLendingClub.generate(shift_at=…, shift=…)`.
5. **Loan records are stage-structured** (`context` = PRE_DECISION only, `policy`, `outcome`),
   and include `historical_arm` and `realized_reward`.
6. **Groups level `group` includes unobserved groups** (cartesian product of observed
   dimension values) so shrinkage to the prior is visible; leaf rows carry `dims.arm`.
7. **Reliability bins are prediction deciles** (equal frequency); ECE is computed on them.
8. **MCTM payload** adds `structural_zeros`, `matrix_low` / `matrix_high`, `row_prediction_error`
   and `maturity` (collapse at `t = horizon`); `structure` may differ from the fit's (estimated on demand).
9. **Simulation lifecycle** adds `awaiting_review` and `interrupted`; `chaos` → `interrupted`,
   `resume` restarts from the last checkpoint; `Simulation.last_recovery` reports the G.6
   recovery including `deterministic_match`; `worker` exposes pid / liveness.
10. **Timeseries units**: `reward`/`regret`/`realized` are per-epoch **sums**, `rolling_reward`
    is **per decision** over 6 epochs.
11. **Events**: lifecycle event types (`RunStarted`, `RunPaused`, `RunResumed`, `WorkerKilled`,
    `RunCompleted`, `RunCancelled`, `RunFailed`) join the SPEC types; default listing excludes
    per-borrower events; events carry `level` and a one-line `message`.
12. **Reviews** are also recorded for auto-acknowledged trips (`auto = true`) and report
    `segmentation_before/after`.
13. **Evaluation report** is grouped by the nine Table 11 rows with `axis`, per-metric `unit`,
    `better`, `baseline` and `by_policy`; `status` `"partial"` while the run is live.
14. **Lists use the DRF LimitOffset envelope**; computed sub-resources are bare objects/arrays.
15. **Shadow evaluation** reports `leakage_safe`, `context_columns`, `n_samples`; a context with
    non-PRE_DECISION keys is a 400 (LeakageError), not silently dropped.
16. **`historical_expected_reward`** is computed for the exact issued policy (amount, rate, term),
    with the risk of the nearest arm's leaf.

---

## 12. Mock mode (frontend)

`pnpm dev` sets `VITE_API_MOCK=1`: every request above is answered in the browser by
`frontend/src/mocks`, built from a seeded synthetic Lending Club twin (200k loans, 2012–2018,
PSI shift at 2016-06). Fits, calibration, MCTM, shadow reports, simulations, sentinel and the
report are computed from the same loan index, so the numbers agree across pages. New jobs
progress in wall-clock time (ingest ≈ 5 s, fit ≈ 7 s, simulation ≈ 0.4 s per epoch, portfolio
≈ 5 s); state resets on page reload. The mock also serves the §13.1 extras (guardrails and
reference rates, the structured leaf prior, prepayment ρ, recovery by term, the portfolio
decomposition and λ sweep, report `details`), with reward moments from the core's closed form
(docs/MATH.md §2.4, §2.8). `pnpm dev:live` (or `VITE_API_MOCK=0`) talks to Django through the
`/api` proxy instead (default target `http://127.0.0.1:8010`, override with
`VITE_API_PROXY_TARGET`).

---

## 13. Backend implementation notes (deviations)

The Django backend (`api/`) serves every interface of `types.ts` with the documented keys and
nullability (`api/tests/test_contract.py` checks this against the served payloads) and adds extra
keys where the core provides more. Deviations, detailed in [`api/README.md`](../api/README.md#contract-notes-and-deviations):

* Events: `BorrowerArrived`, `PolicySelected` and `PosteriorUpdated` are never emitted (decisions and
  posterior snapshots are served by `/arms/` and `/posteriors/`); the runner adds `EpochStarted`
  (hidden by default) and `ResegmentationApplied`. Payloads are the runner's plus the §8.8 aliases.
* Portfolio `counts` carry an extra core class, `downsized` (mutually exclusive with the six others).
* `policies` accept any core spec (`thompson_se`, `epsilon_greedy(epsilon=0.05)`, …).
* Posterior snapshots cover the pooled level and the 10 largest groups per policy (every 3 epochs and
  the last), which bounds `/posteriors/`, `/simpson/` and grouped `/arms/`.
* `Simulation.sentinel_level` / `psi_aggregate` follow the latest reading (possibly the epoch in progress).
* `Calibration.ece` uses the decile bins; `FitMetrics.ece` the core's equal-width bins (both reported).
* `rolling_reward` is recomputed over 6 epochs (the runner's 3-epoch value is `rolling_reward_runner`).
* PSI bucket tables are recomputed from the run's reference and window (`psi_runner` = runner's PSI).
* Simulation defaults: `start` = the fit's `train_end`, `end` = the dataset's last epoch.
* Dataset `path` must lie under the server's `DATA_ROOT`.

### 13.1 Model extras (structured leaf prior, guardrails, prepayment, recovery by term)

Added after the first end-to-end runs (SPEC §7, docs/MATH.md §2.6–2.9). Every key is
additive; existing keys keep their meaning, except that `arms[].eligible` now reflects all
the guardrails.

**Fit configuration** (`POST /fits/` body and `Fit.config`; defaults in §6.1):
`arm_space.max_payment_to_income` (default now **0.20**, was 0.4), `arm_space.max_amount_to_income`
(0.50), `arm_space.max_total_dti` (0.43), `arm_space.rate_corridor` (0.05) — each may be
`null` to switch the guardrail off; `hierarchy.policy_effect` (true),
`hierarchy.policy_effect_cfg` (object); `recovery.term_level` (true);
`reward.prepayment_adjustment` (`"empirical"` | `null`).

**Simulation configuration**: `hindsight_sample` (`"elapsed"` default | `"resolved"`).

**`Fit.metrics`** gains:

| key | content |
|---|---|
| `interest_realization` | `{global, by_term {"36", "60"}, by_grade_term {grade: {term: ρ}}, n_fit, sample, min_count, clip, prepayment_adjustment}` — realised share of the scheduled interest of fully-paid training loans |
| `policy_effect` | `{status, model_status, installed, beta {int_rate, term60, pti, log_amount}, std_errors, n_fit, sample, converged, n_iter, max_abs_logit_shift, reason}` |
| `guardrails` | `{max_payment_to_income, rate_corridor, max_amount_to_income, max_total_dti}` |
| `reference_rates` | `{n_fit, by_grade {grade: mean int_rate}, global}` |

**`ShadowReport`** (`POST /fits/{id}/shadow/evaluate/`) gains `eligibility`:
`reference_rate` (training mean `int_rate` of the borrower's sub-grade → grade → global;
null when the fit has no reference rates), `n_eligible` (arms in the final mask, decline and
an admitted historical arm included), `n_eligible_lending` (lending arms passing the
guardrails), `historical_admitted` (the historical arm failed the guardrails but is kept),
`guardrails` and `rate_corridor_applied`. Each `arms[]` row gains `realization` (ρ).

**`RiskEstimate`** (`GET /fits/{id}/risk/`): `reward.realization` (ρ) and
`recovery.core_source` (`leaf` | `group_term` | `group` | `global` | `prior`; `recovery.level`
keeps the `HierarchyLevel` vocabulary and reports `group_term` as `group`). The same
`core_source` appears in `HierarchyRow.recovery`.

**`HierarchyRow.prior_tilt_logit`** (`GET /fits/{id}/groups/`) is served: the leaf's structured-prior
tilt t(g, a) on the logit scale, from the fit's `hierarchy_summary.parquet`; `null` on global and
group rows (the core writes 0 / NaN there, they have no tilt) and on leaves of fits saved before
the tilt existed.

**`PortfolioSummary`** (`GET /fits/{id}/shadow/portfolio/{job_id}/` → `summary`) gains:

| key | content |
|---|---|
| `recommendation_mix` | `recommended` / `historical`: `{n, n_lending, decline_share, by_term {"36","60"}, by_rate {"<10%", "10-15%", "15-20%", "20-25%", ">=25%"}, by_amount {"<$10k", "$10k-20k", "$20k-30k", ">=$30k"}, mean_rate, mean_amount, share_36m, share_at_max_rate, share_at_max_amount}`; flat `mean_rate_recommended/historical`, `mean_amount_recommended/historical`, `share_36m_recommended/historical`, `share_at_max_rate_recommended`, `share_at_max_amount_recommended`; `reference {n, mean_reference_rate, mean_rate_minus_reference, share_above_reference, share_within_corridor, historical_mean_rate_minus_reference}` |
| `uplift_decomposition` | `expected_profit_historical`; `repricing_only` / `repricing_and_resizing` / `unrestricted`: `{expected_profit, uplift, uplift_pct}` — the best admissible arm keeping the historical term and amount / keeping the term / unrestricted |
| `by_risk_aversion` | `{"0.0": …, "0.5": …, "1.0": …}`, each `{risk_aversion, expected_profit_shadow, expected_profit_historical, uplift, uplift_pct, mean_amount_recommended, mean_rate_recommended, share_36m_recommended, share_at_max_amount_recommended, share_at_max_rate_recommended, decline_share}` — the shadow policy under `mean − λ·std` on the same tables and eligibility |
| `eligibility` | `{guardrails, reference_rates (bool), share_with_reference_rate, mean_n_eligible_lending, share_single_fallback_arm}` |

Like the summary's own `uplift_pct`, every `uplift_pct` inside `uplift_decomposition` and
`by_risk_aversion` is a **fraction** of `|expected_profit_historical|` (the core reports
percent; the API converts). The core's per-loan parquet (`per_loan.parquet` in the job
directory) adds `reference_rate` and `n_eligible_lending` columns.

**`EvaluationReport`** (`GET /simulations/{id}/report/`): `dimensions.counterfactual_evaluation`
adds `recommendation_mix`, `uplift_decomposition`, `by_risk_aversion`, `eligibility` and
`interest_realization` (when the shadow summary is present);
`dimensions.bandit_performance.by_policy[*]` adds `selection_mix` (the same structure as
`recommendation_mix.recommended`, over the last 6 epochs) and `top_arm_labels_recent`;
`fit` adds `policy_effect`, `reference_rates`, `guardrails`, `interest_realization`.

**Report `details` (backend change).** The runner writes `report.json` without a shadow summary, so the
API now rebuilds the report with `pipeline.build_report(..., shadow_summary=…)` from the run directory
and the **latest succeeded** shadow-portfolio run of the simulation's fit (the `summary.json` of its most
recently finished `shadow_portfolio` job). `details.counterfactual_evaluation` then carries
`available: true`, `portfolio_job_id`, `portfolio_request`, the summary scalars, `recommendation_mix`,
`uplift_decomposition`, `by_risk_aversion`, `eligibility`, `interest_realization`, `by_group_top` and
`top_loans` (first 10, mapped to `PortfolioLoanRow`); the Table 11 `counterfactual_evaluation` metrics
are filled from the same summary. Without a succeeded run it stays
`{available: false, note: "shadow portfolio not run"}`. The rebuilt report is cached on the identity
of the run files and of that portfolio summary, so a portfolio run that finishes after the simulation
shows up on the next `GET report/`. `details.computational_feasibility.process_cpu_seconds` is dropped
(it measured the API process; the worker's CPU is in `runtime`).

**Units:** every `uplift_pct` under `details` is now a **fraction** of `|expected_profit_historical|`
(the core reports percent; e.g. 146.27 → 1.4627), the same unit as `PortfolioSummary.uplift_pct`.
Converted keys:
`details.policy_performance.by_policy.<policy>.uplift_pct`,
`details.counterfactual_evaluation.uplift_pct`,
`details.counterfactual_evaluation.uplift_decomposition.{repricing_only,repricing_and_resizing,unrestricted}.uplift_pct`,
`details.counterfactual_evaluation.by_risk_aversion.<λ>.uplift_pct`
(the converter rewrites any `uplift_pct` key under `details`; these are the only ones the core emits).

Compatibility: fits saved before these changes still load (no tilt, no rate corridor, ρ = 1,
no recovery term level). Simulation run directories created before them cannot be resumed,
because the configuration hash changed (`CheckpointError`); completed runs and reports are
unaffected.

### 13.2 How the console reads the §13.1 keys (checked against the served payloads)

* The report's extra blocks are served under `details.<dimension>` (the core's per-dimension
  dict); `dimensions` stays the Table 11 metric list. The console reads
  `details.bandit_performance.by_policy[*].selection_mix` / `top_arm_labels_recent` and
  `details.counterfactual_evaluation`.
* Every `uplift_pct` under `details.*` (counterfactual top level, `uplift_decomposition.*`, `by_risk_aversion.*`, `details.policy_performance.by_policy[*]`) is a **fraction**, the same unit as the portfolio endpoint (converted by the API; see §13.1).
* `details.counterfactual_evaluation` is populated from the fit's latest succeeded shadow-portfolio run (`available: true`, `portfolio_job_id`, the five blocks); it is `{available: false, note: "shadow portfolio not run"}` only when no portfolio run exists.
* `EvaluationReport.fit` is the fit id. The fit extras (`policy_effect`, `reference_rates`,
  `guardrails`, `interest_realization`) are read from `GET /fits/{id}/` `metrics`.
* `HierarchyRow.prior_tilt_logit` is served on leaf rows (`null` on global/group rows); the leaves table shows it as the "Prior tilt" column.
* The shadow heatmap explains why an arm is not admissible by re-checking G1–G4 in the browser
  from `context.annual_inc`, `context.dti`, `eligibility.reference_rate` and
  `eligibility.guardrails`. This reproduces the served `eligible` flags. The exceptions are the
  admitted historical arm and the fallback arm, which are eligible by construction.
