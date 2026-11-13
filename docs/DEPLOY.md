# Deployment — containers, K3s/Talos, chaos

The proposal (§3.5, Appendix G.6) targets a resource-constrained bare-metal Kubernetes
cluster (K3s on Talos Linux, 12–16 cores / 32–44 GB RAM aggregate, consumer hardware) and
evaluates whether the system **keeps working or recovers** when nodes, pods, power or the
network fail. Everything here is built around one property: *every simulation is
event-sourced and checkpointed on persistent storage, so a killed worker resumes
bit-for-bit from its last checkpoint* (`docs/SIMULATION.md`).

## Images

| Image | Dockerfile | Contents |
|---|---|---|
| `bayesian-credit-api` | `deploy/docker/Dockerfile.api` | `python:3.14-slim`; a Poetry build stage installs `poetry.lock` (main group: the `bayesian_credit` library, Django API, gunicorn, Temporal SDK) into `/opt/venv`; Temporal job worker, simulation CLI. Entrypoint modes: `serve` (API), `worker` (Temporal worker), `run-job <job_id>`, `simulate <run_dir>`, `resume <run_dir>`, `demo`. |
| `postgres:17-alpine` | upstream | One PostgreSQL instance: the API database `bc` and Temporal's `temporal` (persistence) and `temporal_visibility` databases. |
| `temporalio/server:1.32.2` · `admin-tools:1.32.2` · `ui:2.56.1` | upstream | The workflow engine every background job runs on (on PostgreSQL), its schema / namespace tooling and the web UI. |
| `bayesian-credit-web` | `deploy/docker/Dockerfile.frontend` | Vite production build served by nginx; `/api/` is proxied to `API_UPSTREAM` (default `http://api:8000`). |

Build from the repository root:

```bash
docker build -f deploy/docker/Dockerfile.api      -t bayesian-credit-api:local .
docker build -f deploy/docker/Dockerfile.frontend -t bayesian-credit-web:local .
```

## docker compose (laptop / single node)

Needs Docker Compose v2 (`sudo dnf install docker-compose-plugin` on Fedora; Docker Desktop / Rancher Desktop ship it).

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
# console  http://localhost:8080      API docs  http://localhost:8000/api/docs/      Temporal UI  http://localhost:8233
docker compose -f deploy/docker/docker-compose.yml exec api /app/entrypoint.sh demo   # seed a demo dataset/fit/simulation
docker compose -f deploy/docker/docker-compose.yml up -d --scale worker=2             # more job workers
```

Services:

| Service | Role |
|---|---|
| `postgres` | PostgreSQL 17 (`pgdata` volume). First boot runs `deploy/k8s/config/postgres-init.sh`: role + database `bc` for the API, role `temporal` (CREATEDB). Published on `127.0.0.1:5432`. |
| `temporal-schema` | one-shot (`admin-tools`): creates `temporal` + `temporal_visibility` and applies every schema version (`deploy/k8s/config/temporal-setup.sh schema`; idempotent, re-run on upgrades) |
| `temporal` | Temporal server 1.32 on PostgreSQL (`DB=postgres12`; dynamic config `deploy/k8s/config/temporal-dynamicconfig.yaml`), `127.0.0.1:7233` |
| `temporal-namespace` | one-shot: registers the `default` namespace (30-day retention) |
| `temporal-ui` | web UI on `127.0.0.1:8233` |
| `api` | gunicorn; migrates the `bc` database, starts workflows, runs no jobs |
| `worker` | `manage.py temporal_worker`, 2 jobs at a time; scale with `--scale worker=N` |
| `web` | the console (nginx), `:8080` |

For local development `make infra` starts only `postgres` + Temporal + UI, and the API and
worker run from the Poetry environment against `localhost` (`make api`, `make worker`).

`./data` is bind-mounted into the API container: put the Kaggle file at
`data/raw/accepted_2007_to_2018Q4.csv.gz` (or run `scripts/download_data.sh`) and it is
visible to the API. Fit artifacts and simulation run directories (checkpoints included) are
written under `data/artifacts/`, shared by the API and the workers. Passwords default to
development values; set `POSTGRES_PASSWORD`, `BC_DB_PASSWORD`, `TEMPORAL_DB_PASSWORD` and
`DJANGO_SECRET_KEY` in the environment (or an `.env` file next to the compose file) before the
first `up` — role passwords are applied when the `pgdata` volume is initialised.

The API and worker containers run as the owner of `./data` (`HOST_UID` / `HOST_GID`, default
`1000:1000`; export `HOST_UID=$(id -u) HOST_GID=$(id -g)` otherwise) and mount it with `:z`, so
on SELinux hosts (Fedora) the directory is relabelled once as shared container content. Static
files are collected at image build time, so the containers need no writable image paths.
`API_PORT` / `WEB_PORT` move the published API (8000) and console (8080) ports.

## Kubernetes (K3s on Talos)

Manifests are a kustomization in `deploy/k8s/`:

```
namespace.yaml   configmap.yaml   secret.yaml   pvc.yaml   postgres.yaml   temporal.yaml   api.yaml   worker.yaml   web.yaml   ingress.yaml
config/          postgres-init.sh · temporal-setup.sh · temporal-dynamicconfig.yaml (→ ConfigMap bc-config-files, also used by compose)
```

```bash
# 1. cluster: Talos + Omni (or plain K3s). Confirm the default StorageClass is local-path.
kubectl get storageclass
# 2. secrets: edit deploy/k8s/secret.yaml (api-secrets: DJANGO_SECRET_KEY, DATABASE_URL;
#    postgres-secrets: POSTGRES_PASSWORD, BC_DB_PASSWORD, TEMPORAL_DB_PASSWORD) — never commit real ones
# 3. images: push to a registry the nodes can pull from (edit images: in kustomization.yaml)
docker tag bayesian-credit-api:local ghcr.io/<you>/bayesian-credit-api:latest && docker push ...
# 4. deploy
kubectl apply -k deploy/k8s
kubectl -n bayesian-credit get pods -w
# 5. data: copy the Lending Club file onto the PVC once
kubectl -n bayesian-credit cp data/raw/accepted_2007_to_2018Q4.csv.gz \
  $(kubectl -n bayesian-credit get pod -l app=api -o name | cut -d/ -f2):/app/data/raw/
# 6. open http://bayesian-credit.local and http://temporal.bayesian-credit.local (point both hosts at a
#    node IP; Traefik is K3s' default ingress)
# 7. more job throughput: kubectl -n bayesian-credit scale deploy/worker --replicas=2
```

Sizing: the API pod requests 0.5 CPU / 1 GiB and is limited to 2 CPU / 4 GiB (it serves
requests and in-request shadow evaluations; it runs no jobs). Each worker pod requests 1 CPU /
2 GiB and is limited to 4 CPU / 8 GiB for `TEMPORAL_WORKER_CONCURRENCY` (2) jobs. Temporal
needs ~0.1 CPU / 256 MiB. A 50 k-borrower, 6-policy replay runs comfortably inside one
worker node of the proposal's hardware. The API is a single replica and the workers are
pinned next to it (pod affinity) because the RWO `bc-data` volume (artifacts, run directories)
is shared; with a ReadWriteMany volume the workers can spread over nodes. PostgreSQL is a
single-replica StatefulSet (20 GiB local-path volume, 0.25–2 CPU / 0.5–2 GiB); Temporal takes
0.25–2 CPU / 0.5–2 GiB.

### Background jobs on Temporal

Every ingest, fit, shadow portfolio and simulation is a Temporal workflow (`BackgroundJob`,
`api/apps/common/temporal/workflows.py`) on the task queue `TEMPORAL_TASK_QUEUE`:

```
POST /fits/ ──► API: Job row + start workflow bc-fit-<id>-<suffix> ──► Temporal server (temporal.yaml)
                                                                          │ task queue bayesian-credit-jobs
                         worker pod: manage.py temporal_worker ◄──────────┘
                           └─ activity run_job: spawns `manage.py run_job <id> --supervised`,
                              heartbeats every 0.5 s; the child writes progress / outcome to the Job row
```

* **Placement and scaling.** Any number of worker pods poll the queue; Temporal hands each job
  to exactly one. `TEMPORAL_WORKER_CONCURRENCY` bounds the jobs per pod; scale the Deployment
  for more throughput. Jobs wait (`queued`) while no worker runs; nothing is dropped.
* **Lost workers.** A worker that stops heartbeating for `TEMPORAL_HEARTBEAT_TIMEOUT` (60 s) or
  whose child dies without recording an outcome is *lost*. Ingest / fit / portfolio jobs are
  retried on another worker (3 attempts); a simulation is marked `worker lost` → `interrupted`
  and is resumed from its checkpoint by the operator (`/resume/`), which keeps the G.6
  experiment measurable. An orphaned `run_job` child exits within a second of its supervisor
  dying, so a retried job never runs twice.
* **Chaos.** `POST /simulations/{id}/chaos/` is a Temporal *update* on the job's workflow: the
  activity is cancelled, its supervisor SIGKILLs the `run_job` process on whichever node runs it,
  and the request returns once the process is gone.
* **Graceful stop.** SIGTERM on a worker pod (rollout, drain) sends SIGTERM to its jobs: a
  simulation checkpoints at the epoch boundary and stops (`cancelled`, resumable); other jobs
  are retried on another worker.
* **Maintenance.** Workers upsert a Temporal Schedule that runs the `ArtifactGC` workflow
  (`TEMPORAL_GC_SCHEDULE`, daily 03:00 by default; empty disables it), deleting artifact
  directories and job logs no row refers to (`manage.py gc_artifacts --apply` by hand).
* **Visibility.** Every job links to its workflow in the Temporal UI (`Job.temporal_url`); the
  console's side navigation shows Temporal's status and the number of polling workers.

The Temporal server is a single `temporalio/server` pod (all four services in one process)
whose persistence and visibility stores are PostgreSQL databases (`temporal`,
`temporal_visibility`): workflow histories, schedules and list/search queries (`temporal
workflow list -q 'WorkflowType="BackgroundJob"'`) survive pod and node loss with the database.
The `schema` initContainer applies new schema versions on upgrade; the `temporal-namespace` Job
registers the namespace. To scale Temporal itself use the official helm chart against the same
databases, or Temporal Cloud (`TEMPORAL_API_KEY` turns on TLS + API-key auth), and delete
`temporal.yaml` from the kustomization.

The core CLI still runs a run directory without the API or Temporal
(`entrypoint.sh simulate <run_dir>` / `resume <run_dir>`), for experiments outside the console.

## Resilience experiments (Table 10)

| Experiment | How to run it here | What should happen |
|---|---|---|
| Workload termination | `POST /api/v1/simulations/{id}/chaos/` then `/resume/`, or `deploy/chaos/chaos-monkey.sh --api … --simulation {id}` | progress continues from the last checkpoint; decisions/timeseries are identical to an uninterrupted run (tested in `core/tests/test_runner.py`) |
| Node / pod failure | `deploy/chaos/chaos-monkey.sh --k8s --api … --simulation {id}` (deletes a worker pod; `--selector app=api` / `app=temporal` for the other pods) | replacement pod becomes Ready; Temporal reports the lost worker, the run is `interrupted` and resumes from the PVC; a lost ingest / fit is retried automatically |
| Multiple-node failure | `talosctl -n <ip> shutdown` on two workers | pods reschedule once a node is back; no state is lost because state lives on the PVC + run dir, never in memory only |
| Resource constraint | lower `resources.limits` in `worker.yaml` | run slows but completes; `report.json → computational_feasibility` records CPU seconds and peak RSS |
| Network interruption | `talosctl -n <ip> … ` link down for a minute, or cut the switch | API returns when the network returns; running workers are unaffected (local disk) |
| Controlled power interruption | power-cycle a worker node | as node failure; the worst case loses at most `checkpoint_every` epochs of work |
| State recovery | kill the worker between `CheckpointWritten` events and resume | the runner deletes partial epoch files after the checkpoint and recomputes them deterministically (per-epoch seeded RNG) |

Timing of recovery, number of failed workloads and whether state recovered successfully are
the metrics in Table 11 → *Infrastructure resilience*; the chaos script prints them and the
simulation's `events.jsonl` records every `CheckpointWritten` / `RunResumed` event.

## Observability

The API exposes `GET /api/v1/metrics/` in Prometheus text format (jobs by kind/status,
running simulations, `bc_temporal_up`, `bc_temporal_workers`) and the pod carries
`prometheus.io/*` scrape annotations, so a
kube-prometheus-stack / Grafana install picks it up without extra configuration. Resource
utilisation per pod comes from the kubelet / cAdvisor metrics that Prometheus already scrapes.

## Event bus and workflow engine

The proposal names Kafka (append-only event state) and Temporal (workflow orchestration).
The prototype implements the *semantics* with a file-backed append-only `events.jsonl` and
atomic checkpoints behind an `EventSink` protocol (`core/bayesian_credit/simulation/events.py`:
`JSONLSink` is the default; `KafkaSink` publishes the same events to a topic and needs
`poetry install --with kafka` (or `pip install "bayesian-credit[kafka]"`) plus a `confluent_kafka` producer config such as
`{"bootstrap.servers": "broker:9092"}`). Swapping the sink is a constructor argument of the
runner's `EventStore`; the checkpoint/resume mechanism does not depend on the sink. Workflow
orchestration is Temporal (see *Background jobs on Temporal* above): every background job is
a durable workflow, so job placement, retries, lost-worker detection, the chaos kill and the
maintenance schedule all go through it.
