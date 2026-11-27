# Reproducing the ablation study

The DSA4900 ablation study asks whether the framework still lends well as borrower information is
removed. It runs four variable subsets, each with a warm and a fully cold bandit-lab run, at two
scales (pilot and full). One script reproduces it on any machine in any of the three ways the
platform runs: Docker Compose, the local Poetry environment ("CLI"), or Kubernetes.

```bash
scripts/download_data.sh                        # once: data/raw/Loan_status_2007-2020Q3.csv (Kaggle token)
scripts/reproduce_ablation.sh docker pilot      # or: cli | k8s, and: pilot | full | both
python3 scripts/ablation/clean.py               # afterwards: list the caches it can remove (--yes deletes)
```

## What runs

`scripts/ablation/study.py` holds the whole definition (`study.py plan --scale pilot|full` prints
every request it sends) and drives the platform's HTTP API, so the study is identical whatever
hosts the API:

| | |
|---|---|
| Data | Kaggle `ethon0426/lending-club-20072020q1`, `Loan_status_2007-2020Q3.csv`, 1,773,470,505 bytes, SHA-256 `5878af2a088f8ab5214c9337289fb8b5eb6c6338fd3f417b6cdc18513dc6f35f` (checked before the run; `--skip-checksum` checks the size only). Loans observed 2020-10-31 |
| Datasets | issued 2007-06 .. 2020-09, seed 42; pilot: a seeded random 500,000 loans, full: all 2,925,492 |
| Subsets | All variables; No FICO (4 FICO fields removed); No FICO, no LC grade (+ grade, sub_grade); Thin file (+ 25 credit-report fields) |
| Segmentation | purpose × home ownership × income band, plus grade where kept, plus FICO band in "All". Breaker resegmentation adds at most 2 dimensions, from DTI band, employment band, verification, application type (FICO band first in "All") |
| Warm fit | trained on loans issued before 2013-06 (six years) |
| Bandit lab | 2013-06 .. 2020-09 (88 months), all 7 policies (Thompson, Thompson σ/√n, UCB1, ε-greedy, Greedy, Historical, Random), hybrid feedback (interim 12 months, weight 0.5), seed 42; pilot streams 50,000 borrowers, full the number in `SCALES["full"]` |
| Cold run | fit-less cold start with the same segmentation; its replay world is fitted on the pre-2013-06 loans, never the warm fit's |
| PSI Sentinel | the pre-decision variables recorded throughout the window (22 / 20 / 18 / 8 per subset; fields Lending Club only started reporting in 2012 or 2016 would read as drift), 3-month window, watch 0.10, breaker 0.25, aggregate max; trips auto-acknowledged with `resegment` |
| Checkpoints | every 6 months (pilot) or 12 (full); they do not change results |

Jobs run strictly one at a time, lightest segmentation first. `run` records each job in
`state.json` as soon as it is queued, so re-running the same command resumes (a job still running
is waited for, never duplicated). The study stops at the first failed run instead of starting a
heavier one.

## The three modes

| | `docker` | `cli` | `k8s` |
|---|---|---|---|
| Needs | Docker + Compose v2 | `make setup` (Poetry `.venv`), Docker for Postgres/Temporal | an applied `deploy/k8s` kustomization, `kubectl` |
| Starts | builds the images, `up` with `deploy/docker/docker-compose.ablation.yml` | `make infra`, then the API on `:API_PORT` (8010) and one worker as local processes (stopped on exit) | nothing: sizes the worker Deployment, copies the CSV onto the PVC if missing, port-forwards `svc/api` to `:LOCAL_PORT` (18000) |
| Worker | 1 job at a time, `ABLATION_WORKER_MEMORY` (9g) / `ABLATION_WORKER_CPUS` (10), no swap | 1 job at a time, no memory limit (the host's) | 1 job at a time, limits = requests = `ABLATION_WORKER_MEMORY` (as Gi) / `ABLATION_WORKER_CPUS` |
| Memory probe | the worker container's cgroup | RSS of the `run_job` processes | the worker pod's cgroup |
| Console | http://localhost:8080 | `make web` (http://localhost:5173) | the Ingress (`bayesian-credit.local`) |

Results always land on the machine running the script, in `data/runs/ablation-<scale>/`:
`RESULTS.md`, `results.json` / `results.csv` (every policy × run), `state.json` (job ids and the
manifest: commit, API and core versions, data checksum, the full settings, start and end time) and,
when `reportlab` is importable (`pip install reportlab`), `<Scale>_Ablation_Study_Results.pdf`
(Times New Roman 12 pt, 1.5 spacing).

## Resources

Measured on a 16 GB / 12-thread laptop (worker 9 GiB, 10 CPUs):

| | per simulation | worker peak |
|---|---|---|
| Pilot, thin file / no grade | 1–1.5 min | 1.5–2 GB |
| Pilot, no FICO | 2–3 min | 3.5 GB |
| Pilot, all variables | 3.5–4 min | 6 GB (the most groups) |
| Full dataset: ingest, fit | 15–30 s each | 2.4–5.2 GB |

Memory grows with the number of borrower groups and of borrowers streamed: the setup holds the
whole dataset, the streamed loans with their eligibility matrix (borrowers × 129 arms) and the
hindsight model. Streaming all 2.7 million borrowers of the window does not fit in 9 GiB (the
worker is stopped before the first month), which is why the full scale streams a sample.

## Checking a reproduction

```bash
python3 scripts/ablation/study.py compare data/runs/ablation-pilot path/to/their/ablation-pilot
```

compares the fit metrics, every policy's decisions, expected reward, regret, realised reward and
decline rate, and the breaker-trip months (relative tolerance `--rtol`, default 1e-9); it exits 1
on any difference. Same commit, same data file and the same image or `poetry.lock` give the same
numbers: every random draw is seeded and the simulation is deterministic (a kill-and-resume replay
is bit-identical, `scripts/run_pipeline.py`'s resilience check). Different CPU architectures or
BLAS builds may differ in the last floating-point digits; use `--rtol 1e-6` there.

## Cleaning up

`scripts/ablation/clean.py` removes the intermediate files an analysis leaves behind and keeps
every result and everything the console's graphs read. Dry run by default:

| Removed | Kept |
|---|---|
| `checkpoints/` of completed simulations (the bulk: 0.3–3 GB per pilot run; `--unfinished` adds failed / cancelled runs, which then cannot be resumed) | each run's `timeseries.parquet`, `events.jsonl`, `sentinel.parquet`, `arms/`, `decisions/`, `feedback/`, `reviews/`, `report.json`, `progress.json`, `config.json`, a fit-less run's `fit/` |
| the loader caches `data/processed/*.canonical.v*.parquet` (rebuilt from the CSV in seconds; skipped while a simulation runs; `--keep-loader-cache`) | fit directories, dataset `loans.parquet` and summaries, job logs |
| `checkpoints/` and the `resilience/` scratch copy of finished `run_pipeline.py` runs | `data/runs/*` results (`state.json`, `results.*`, `RESULTS.md`, PDFs, logs), `data/raw/` |
| with `--code-caches`: `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `frontend/node_modules/.vite` | `.venv`, `node_modules` |

Runs that are queued, running, paused or awaiting review are never touched.

```bash
python3 scripts/ablation/clean.py --yes                                       # host (docker or cli: ./data is the data root)
docker compose -f deploy/docker/docker-compose.yml exec worker \
  python /app/scripts/ablation/clean.py --data-root /app/data --yes          # inside the worker container
kubectl -n bayesian-credit exec -i deploy/worker -- \
  python - --data-root /app/data --yes < scripts/ablation/clean.py           # on the Kubernetes PVC
```
