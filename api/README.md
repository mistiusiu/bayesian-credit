# Bayesian Credit API

Django 6 + Django REST Framework backend of the shadow-lending-engine research console.
It exposes the `bayesian_credit` modelling library (`../core`) over the JSON contract in
[`frontend/src/api/types.ts`](../frontend/src/api/types.ts) (authoritative) and
[`docs/API.md`](../docs/API.md): datasets, fits (Two-Stage Beta-Beta + MCTM), the shadow
engine, replay simulations with the PSI Sentinel, and the human-in-the-loop review queue.
Long-running work runs as background jobs; simulations talk to the API only through files
in a run directory, so a worker can be SIGKILLed (chaos) and resumed from its checkpoint.

## Quick start

```bash
cd <repo> && . .venv/bin/activate          # Python 3.14; core installed: pip install -e core
pip install -r api/requirements.txt
cd api
python manage.py migrate
python manage.py demo                      # ~25 s: 20k synthetic loans, fit, portfolio, 2 simulations
python manage.py runserver 8010            # http://127.0.0.1:8010/api/docs/  (port 8000 is taken on the dev box)
pytest -q                                  # ~18 s, in-memory SQLite, jobs inline
```

| URL | |
|---|---|
| `/api/v1/` | REST API (index of collections) |
| `/api/docs/` | Swagger UI · `/api/schema/` OpenAPI 3 (drf-spectacular) |
| `/api/v1/health/` | `{status, version, python, core_version}` |
| `/api/v1/metrics/` | Prometheus text: jobs by kind/status, simulations by status, live workers, restarts, reviews |
| `/admin/` | Django admin (create a user with `createsuperuser`) |

The console (`frontend/`) proxies `/api` to the backend; point it here with
`VITE_API_PROXY_TARGET=http://127.0.0.1:8010` (or `VITE_API_MOCK=0`).

## Configuration (environment)

| Variable | Default | |
|---|---|---|
| `DJANGO_SECRET_KEY` | insecure dev key | set in production |
| `DJANGO_DEBUG` | `1` | `0` in containers |
| `DJANGO_ALLOWED_HOSTS` | `*` | comma list |
| `DATABASE_URL` | SQLite `api/db.sqlite3` (`api/db/db.sqlite3` when an `api/db/` volume exists; `SQLITE_PATH` overrides) | `sqlite:////abs/path`, `sqlite:///rel/to/api`, `postgres://user:pass@host:5432/db?sslmode=require` (parsed in `config/settings.py`, psycopg 3) |
| `DATA_ROOT` | `<repo>/data` | |
| `ARTIFACT_ROOT` | `<DATA_ROOT>/artifacts` | datasets, fits, run dirs, job logs |
| `LENDING_CLUB_PATH` | `<DATA_ROOT>/raw/accepted_2007_to_2018Q4.csv.gz` | Kaggle file (`scripts/download_data.sh`) |
| `CORS_ALLOW_ALL_ORIGINS` | on when debug | `CORS_ALLOWED_ORIGINS` = comma list otherwise |
| `JOBS_SYNC` | `0` | `1` runs jobs inline in the request (tests, `manage.py demo`) |
| `JOB_PROGRESS_INTERVAL` | `2` | seconds between progress writes of a running job |
| `SHADOW_SAMPLES` | `2000` | posterior-predictive draws per arm in the shadow engine |
| `FIT_CACHE_SIZE` | `4` | fits kept loaded per process (LRU) |
| `SIMULATION_EXECUTOR` | `subprocess` | only the subprocess executor is implemented by the API (see k8s note below) |

SQLite runs in WAL mode with `IMMEDIATE` transactions so the API and its worker
subprocesses can write concurrently; use Postgres before scaling to several API replicas.

## Layout

```
api/
  config/          settings.py (env), test_settings.py (pytest), urls.py, wsgi.py, asgi.py
  apps/common/     Job model + runner (jobs.py), pagination, exception handler, utils, file caches,
                   contract serializers (schemas.py mirrors types.ts), /health/ /metrics/ /jobs/,
                   management commands run_job · demo · gc_artifacts
  apps/datasets/   Dataset model, ingest job, Table 3 variables, summary, DuckDB loan browser
  apps/fits/       Fit model, fit + shadow-portfolio jobs, hierarchy / risk / calibration / MCTM / arms,
                   shadow engine (evaluate, portfolio)
  apps/simulations/ Simulation model, worker job, run-directory readers, control actions
  apps/sentinel/   Review model mirrored from <run_dir>/reviews/, acknowledge
  tests/           end-to-end workflow, contract keys vs types.ts, control/chaos, validation, schema
```

## Endpoints (`/api/v1/`)

| Method & path | Returns |
|---|---|
| `GET /health/`, `GET /metrics/` | health JSON · Prometheus text |
| `GET /jobs/?status=queued,running&kind=simulation` · `GET /jobs/{id}/` | `Paginated<Job>` · `Job` |
| `POST /datasets/` · `GET /datasets/?source=` · `GET /datasets/{id}/` | 201 `Dataset` (+ ingest job) · list · detail |
| `GET /datasets/{id}/variables/` | `VariableRow[]` (Table 3 order) |
| `GET /datasets/{id}/summary/?features=a,b` | `DatasetSummary` (cached JSON per feature set) |
| `GET /datasets/{id}/loans/?limit&offset&grade&purpose&home_ownership&group&resolved&search&ordering` | `Paginated<LoanRecord>` (DuckDB over the parquet) |
| `POST /fits/` · `GET /fits/?dataset=` · `GET /fits/{id}/` | 201 `Fit` (+ fit job) · list · detail |
| `GET /fits/{id}/groups/?level=global\|group\|leaf&group=` | `HierarchyRow[]` |
| `GET /fits/{id}/risk/?group=&arm=` | `RiskEstimate` |
| `GET /fits/{id}/calibration/` | `Calibration` |
| `GET /fits/{id}/mctm/?structure=proposal\|extended&horizon=60&start=current\|delinquent` | `Mctm` |
| `GET /fits/{id}/arms/` | `FitArms` |
| `POST /fits/{id}/shadow/evaluate/` (`{loan_id}` or `{context, historical_policy?}`) | `ShadowReport` |
| `POST /fits/{id}/shadow/portfolio/` · `GET …/shadow/portfolio/` · `GET …/shadow/portfolio/{job_id}/` | 202 `Job` · `PortfolioRun[]` · `PortfolioResult` |
| `POST /simulations/` · `GET /simulations/?fit=&status=` · `GET /simulations/{id}/` | 201 `Simulation` (+ worker job) · list · detail |
| `GET /simulations/{id}/timeseries/?policy=` | `Timeseries` |
| `GET /simulations/{id}/arms/?policy=&epoch=&group=` | `SimArms` |
| `GET /simulations/{id}/posteriors/?policy=&group=&arms=1,2` | `PosteriorHistory` |
| `GET /simulations/{id}/simpson/?policy=&epoch=` | `SimpsonPanel` |
| `GET /simulations/{id}/events/?type=A,B&after=&limit=&order=asc\|desc` | `EventPage` |
| `GET /simulations/{id}/sentinel/` · `GET …/sentinel/buckets/?feature=&epoch=` | `SentinelTimeline` · `PsiBucketTable` |
| `GET /simulations/{id}/report/` | `EvaluationReport` (`partial` while running, 409 before the first epoch) |
| `POST /simulations/{id}/pause/` · `resume/` · `cancel/` · `chaos/` | `Simulation` (409 for invalid transitions) |
| `GET /sentinel/reviews/?simulation=&status=` · `POST /sentinel/reviews/{id}/acknowledge/` | `Paginated<Review>` · `Review` (409 if already acknowledged) |

Errors: `{"detail": "...", "code": "..."}` or DRF field errors `{"field": ["..."]}` (dotted keys
for fit config fields, e.g. `hierarchy.kappa_group`). Framework exceptions map by class:
`LeakageError`/`ConfigurationError` → 400, `GovernanceError`/`CircuitBreakerOpenError`/
`CheckpointError` → 409, `DataSourceError`/numerical errors → 422, with `code` = the class name.
Unknown `/api/` routes return JSON 404s. Collections use the LimitOffset envelope (page 50,
max 500); timestamps are `YYYY-MM-DDTHH:MM:SSZ`; NaN/inf never appear (→ `null`).

## Jobs

`Job(kind: ingest|fit|simulation|shadow_portfolio, status: queued|running|succeeded|failed|cancelled,
progress, message, log_tail[≤50], pid, started_at, finished_at, error, target_type/target_id, params)`.

* `JobRunner.launch(job)`: with `JOBS_SYNC` the job runs inline (`apps.common.jobs.execute`);
  otherwise `python manage.py run_job <id>` starts in a new session (`start_new_session=True`),
  stdout/stderr → `<ARTIFACT_ROOT>/jobs/<id>.log`, and its pid is stored on the row. A daemon
  thread waits on the child so exited workers never linger as zombies.
* `run_job` executes `execute(job)`: marks the job running, dispatches on `kind` to the app
  handler (`apps/*/jobs.py`), records success, cancellation or failure (traceback in `error`,
  last lines in `log_tail`). Progress is written at most every `JOB_PROGRESS_INTERVAL` seconds;
  log lines (milestones) immediately.
* Handlers call the core pipeline, never re-implement it: ingest → `load_loans(..., with_outcomes=True)`
  + `summarize`; fit → `pipeline.fit_models` (holdout scored inside) + `FittedModels.save`;
  portfolio → `pipeline.run_shadow_portfolio`; simulation → `pipeline.run_simulation` /
  `SimulationRunner.resume` **in the job's own process** (the job's pid is the simulation worker).
* Lost workers are detected lazily (job/dataset/fit/simulation list and detail views, metrics):
  an active job whose pid is gone (or reused by another program, checked via `/proc/<pid>/cmdline`)
  becomes `failed` with `message = "worker lost"`. Nothing runs at import time.
* Datasets and fits derive their status from their job (`queued → running → ready|failed|cancelled`).

## Run directory protocol (simulations)

`POST /simulations/` creates `<ARTIFACT_ROOT>/simulations/<id>/run/` with
`bayesian_credit.simulation.store.RunDirectory.create` (the same class the worker and the CLI use):

| File | Writer | Content |
|---|---|---|
| `config.json` | API (then worker) | `{framework: FrameworkConfig, simulation: SimulationConfig, dataset: {path, …}, fit_dir, created_at}` |
| `progress.json` | worker | `status, epoch_index, n_epochs, current_epoch, pct, message, updated_at, breaker_state, pid, started_at, finished_at, error` |
| `control.json` | API | `{"command": "pause"\|"resume"\|"cancel", "at"}` — read by the worker at every epoch boundary |
| `reviews/<id>.json` | worker | breaker review record (trip epoch, PSI per feature, segmentation, auto flag, …) |
| `reviews/<id>.ack.json` | API (or worker for auto acks) | `{"action", "note", "at", "by"}` |
| `events.jsonl` | worker (+ API `WorkerKilled`, `RunCancelled` for dead runs) | one event per line |
| `decisions/`, `feedback/`, `arms/epoch=YYYY-MM.parquet` | worker | per-decision logs, released feedback, posterior snapshots |
| `timeseries.parquet`, `sentinel.parquet`, `runtime.json` | worker | per-epoch metrics, PSI per feature, CPU/RSS per session |
| `checkpoints/latest.pkl` (+ last 3 epoch-stamped) | worker | atomic resume state |
| `report.json` | worker at completion | core `EvaluationReport` without a shadow summary (the API rebuilds the report, see deviation 9) |

The API never imports a running worker's state; it reads these files (atomic replaces), so the
same run directory works with a subprocess worker or a Kubernetes Job
(`python -m bayesian_credit.simulation.cli resume --run-dir <dir>`, `deploy/k8s/job-template.yaml`).

### Status, pause, review, chaos and resume

`Simulation.status` reconciles `progress.json` with the worker's liveness:
`queued → running → completed`, `paused`, `awaiting_review` (breaker tripped with
`breaker_mode="pause"`), `interrupted` (worker died: chaos SIGKILL, OOM, node loss),
`failed` (exception), `cancelled`.

* **pause** (from `queued`/`running`) writes `control.json {"command":"pause"}`; the worker checkpoints
  at the next epoch boundary and waits. **resume** on a paused run writes `{"command":"resume"}`.
* **Review**: on a trip the worker writes `reviews/<id>.json`; `GET /sentinel/reviews/` mirrors it into
  a `Review` row; `POST …/acknowledge/` writes `<id>.ack.json`; a worker in `awaiting_review` picks it
  up and continues (resegment / rollback / resume). Auto-acknowledged trips are recorded too
  (`auto = true`, `acknowledged_by = "auto_ack"`).
* **chaos** (live worker only) sends SIGKILL to the worker's process group, marks the job failed with
  `message = "chaos: worker killed"`, records `last_recovery.killed_at / killed_at_epoch`, appends a
  `WorkerKilled` event; the run is `interrupted`.
* **resume** on an interrupted / failed / cancelled run deletes a stale `control.json`, starts a *new*
  job with `params.mode = "resume"` (`restarts += 1`); the worker restores `checkpoints/latest.pkl`,
  truncates per-epoch outputs after the checkpoint and recomputes them deterministically
  (per-epoch seeded RNGs). `last_recovery` then reports `resumed_from_epoch` (from `RunResumed`),
  `replayed_epochs`, `deterministic_match` (replayed `EpochCompleted` metrics vs the pre-kill ones)
  and `recovery_seconds` (kill → first epoch completed by the new worker).
* **cancel** writes `{"command":"cancel"}` (the worker checkpoints and stops); a run whose worker is
  already dead is marked `cancelled` directly.

`deploy/chaos/chaos-monkey.sh --api http://127.0.0.1:8010 --simulation <id>` drives the same
endpoints in a loop.

## Artifacts and caches

```
<ARTIFACT_ROOT>/datasets/<id>/loans.parquet, summary_*.json
<ARTIFACT_ROOT>/fits/<id>/  FittedModels.save() files + api_*.json (arm table, calibration deciles,
                            holdout recovery, MCTM counts for the other structure)
<ARTIFACT_ROOT>/fits/<id>/portfolio/<job_id>/summary.json, per_loan.parquet, request.json
<ARTIFACT_ROOT>/simulations/<id>/run/   (run directory above)
<ARTIFACT_ROOT>/jobs/<id>.log
```

A resource's directory is reset when the resource is created (primary keys can be reused after a
database reset, stale caches are never served). Loaded fits and shadow engines are kept in a
per-process LRU keyed by fit id and file mtimes; run-directory tables are memoised by file identity.
`python manage.py gc_artifacts [--apply]` lists/removes directories no row refers to.

## Tests

`cd api && pytest -q` (settings `config.test_settings`: in-memory SQLite, `JOBS_SYNC`, temp
`ARTIFACT_ROOT`, 500 shadow samples). A session fixture runs the whole workflow through the API
once — synthetic dataset (6k loans, 2016-06 shift) → fit (train_end 2015-01) → shadow evaluate
(by loan and by context) → shadow portfolio (1000 loans) → simulation (thompson/historical/random,
2015-01..2016-06, 1500 loans, auto_ack, checkpoint_every 2) → a short low-threshold simulation
(`thompson_se`) whose breaker trips and leaves a pending review — and the tests check every
endpoint, acknowledge the review, exercise pause/cancel/chaos/resume against a stand-in worker
process, worker-loss detection, validation errors, framework-exception mapping, Prometheus
metrics, the OpenAPI schema (`spectacular --validate --fail-on-warn`) and a contract check that
every property of the `types.ts` interfaces appears in the served payloads. 147 tests, ~18 s.

Lint: `ruff check api/` (config in `api/ruff.toml`).

## Production notes

`deploy/docker/Dockerfile.api` runs `gunicorn config.wsgi:application` (entrypoint `serve`), with
`DJANGO_DEBUG=0`, the data volume at `/app/data` and SQLite in the `/app/api/db` volume (or
`DATABASE_URL`). Worker subprocesses run inside the API container/pod and survive API restarts
(own session); a restarted API re-attaches by pid. The k8s Job executor is not driven by the API:
render `deploy/k8s/job-template.yaml` with the run directory to run or resume a simulation as a Job.

## Contract notes and deviations

Everything in `types.ts` is served with the documented names and nullability (checked by
`tests/test_contract.py`); extra keys are added where the core provides more. Where `types.ts`,
`docs/API.md` and the build brief disagreed, `types.ts` / `docs/API.md` won. Known deviations:

1. **Event types**: the runner never emits per-borrower `BorrowerArrived` / `PolicySelected` or sampled
   `PosteriorUpdated` events (decisions live in `decisions/*.parquet`, posteriors in `arms/*.parquet`,
   served by `/arms/` and `/posteriors/`). It emits two types outside `SimulationEventType`:
   `EpochStarted` (hidden unless requested with `type=`) and `ResegmentationApplied` (listed).
   Payloads are the runner's, plus the `docs/API.md §8.8` aliases (`aggregate`, `top_feature`,
   `review`, `psi`, `interim`, `terminal`, `reward`, `from_checkpoint`, …).
2. **Portfolio outcome classes**: the core has a seventh mutually exclusive class `downsized`; it is
   returned as an extra key in `counts` (absent from `PortfolioOutcomeClass`).
3. **Policies**: any core policy spec is accepted (`thompson_se`, `thompson_flat`,
   `epsilon_greedy(epsilon=0.05)`, …) and normalised to its label (lower case, no spaces); `PolicyName`
   only lists the six base names.
4. **Posterior snapshots** are written by the runner every 3 epochs (and the last) for the pooled
   level `*` and the 10 largest groups per policy, so `/posteriors/`, `/simpson/` and grouped `/arms/`
   cover those groups only. The Simpson panel's pooled best arm uses the policy's own pooled
   posterior when present (decision-weighted mean of the group posteriors otherwise); a group's best
   arm needs ≥ 3 effective observations.
5. **Simulation `sentinel_level` / `psi_aggregate`** follow the latest PSI reading, which may belong to
   the epoch in progress (e.g. while `awaiting_review`), like `breaker_state` from `progress.json`.
6. **Calibration**: reliability bins are prediction deciles and `Calibration.ece` is computed on them
   (`docs/API.md §11.7`); `FitMetrics.ece` keeps the core's 10 equal-width bins (`types.ts`), so the two
   ECEs can differ (`Calibration.ece_equal_width` repeats the latter). Predictive metrics use the core's
   primary holdout: resolved loans whose term elapsed by the 2019-03 snapshot.
7. **`rolling_reward`** is recomputed by the API over 6 epochs (contract); the runner's 3-epoch value is
   in `rolling_reward_runner`.
8. **PSI bucket tables** are recomputed (the runner stores per-feature PSI only) from the same
   reference (`issue_d < start`, or the window a rebased acknowledgement switched to) and the same
   window (arrivals of the last `window_months` epochs); `psi_runner` carries the runner's value.
9. **Report**: rebuilt by the API with `pipeline.build_report(..., shadow_summary=…)` from the run
   directory and the fit's **latest succeeded** shadow-portfolio summary (the runner's `report.json` has
   none and is only the fallback), cached on the identity of the run files and of that summary, so a
   later portfolio run shows up on the next request; `status: partial` while running; 409 before the
   first epoch (the brief said 404 until finished). `dimensions` are the core's Table 11 rows (metric
   keys follow the core); `details` are the core's per-dimension blocks with
   `details.counterfactual_evaluation.{available, portfolio_job_id, portfolio_request, …,
   top_loans[≤10 PortfolioLoanRow]}` and **every `uplift_pct` under `details` converted from percent
   to a fraction** (`policy_performance.by_policy.*`, `counterfactual_evaluation`,
   `counterfactual_evaluation.uplift_decomposition.*`, `counterfactual_evaluation.by_risk_aversion.*`).
10. **Shadow**: `historical_expected_reward` is the exact issued policy's expected reward with the
    nearest arm's risk (`docs/API.md §11.16`) and `uplift_expected` = recommended `reward_mean` − that;
    the engine's nearest-arm figure is `historical_arm_expected_reward`. LeakageError bodies carry both
    `detail`/`code` and `context: ["LeakageError: …"]`.
11. **`GET /fits/{id}/groups/`** returns a bare array (contract; the brief suggested pagination) and
    includes unobserved groups (cartesian product of observed dimension values).
12. **Dataset `path`** must resolve under `DATA_ROOT` (or be `LENDING_CLUB_PATH`); both sources default
    to the 2012-01..2018-12 issue window.
13. **Simulation defaults**: `start` = the fit's `train_end`, `end` = the dataset's last epoch (equal to
    `docs/API.md`'s 2015-01 / 2018-12 for default fits on full-range datasets); `pause` is also accepted
    while `queued`.
14. **`RecoveryInfo.killed_at`** is the SIGKILL time for chaos, or the detection time for a worker
    that died on its own; `recovery_seconds` runs from it to the first epoch completed after resume.
15. **Model extras** (`docs/API.md §13.1`): the fit config accepts and returns the lending guardrails
    `arm_space.{max_payment_to_income (0.20), max_amount_to_income (0.50), max_total_dti (0.43),
    rate_corridor (0.05)}` (each `null`-able), `hierarchy.policy_effect` / `policy_effect_cfg`,
    `recovery.term_level` and `reward.prepayment_adjustment` (`"empirical"` | `null`); simulations
    accept `hindsight_sample` (`"elapsed"` | `"resolved"`). `Fit.metrics` adds `interest_realization`,
    `policy_effect`, `guardrails`, `reference_rates`; `ShadowReport` adds `eligibility` and
    `arms[].realization`; `RiskEstimate` and `HierarchyRow` add `recovery.core_source` (`group_term`
    included), `HierarchyRow` adds `prior_tilt_logit` (leaf tilt t(g, a), null on global/group rows),
    `RiskEstimate` adds `reward.realization`; the
    portfolio `summary` adds `recommendation_mix`, `uplift_decomposition`, `by_risk_aversion` and
    `eligibility`. Reward figures served by `/risk/`, `/arms/` and the shadow endpoints all apply the
    fit's prepayment factor ρ(grade, term), like the engine.
