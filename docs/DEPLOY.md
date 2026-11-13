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
| `bayesian-credit-api` | `deploy/docker/Dockerfile.api` | `python:3.14-slim`, the `bayesian_credit` library, Django API (gunicorn), the Temporal job worker, simulation CLI. Entrypoint modes: `serve` (API), `worker` (Temporal worker), `run-job <job_id>`, `simulate <run_dir>`, `resume <run_dir>`, `demo`. |
| `temporalio/temporal:1.10.0` | upstream | Temporal dev server (single binary, SQLite persistence) + web UI: the workflow engine every background job runs on. |
| `bayesian-credit-web` | `deploy/docker/Dockerfile.frontend` | Vite production build served by nginx; `/api/` is proxied to `API_UPSTREAM` (default `http://api:8000`). |

Build from the repository root:

```bash
docker build -f deploy/docker/Dockerfile.api      -t bayesian-credit-api:local .
docker build -f deploy/docker/Dockerfile.frontend -t bayesian-credit-web:local .
```

## docker compose (laptop / single node)

Needs the compose v2 plugin (`sudo dnf install docker-compose-plugin` on Fedora; Docker Desktop / Rancher Desktop ship it).

```bash
docker compose -f deploy/docker/docker-compose.yml up --build
# console  http://localhost:8080      API docs  http://localhost:8000/api/docs/      Temporal UI  http://localhost:8233
docker compose -f deploy/docker/docker-compose.yml exec api /app/entrypoint.sh demo   # seed a demo dataset/fit/simulation
docker compose -f deploy/docker/docker-compose.yml up -d --scale worker=2             # more job workers
```

Four services: `temporal` (server + UI, state in the `temporal-data` volume), `api`
(gunicorn; starts workflows, runs no jobs), `worker` (`manage.py temporal_worker`, 2 jobs at
a time) and `web`. The API and the workers share `./data` and the SQLite volume.

`./data` is bind-mounted into the API container: put the Kaggle file at
`data/raw/accepted_2007_to_2018Q4.csv.gz` (or run `scripts/download_data.sh`) and it is
visible to the API. Fit artifacts and simulation run directories (checkpoints included) are
written under `data/artifacts/`. Add `--profile postgres` and set `DATABASE_URL` to move off
SQLite.

## Kubernetes (K3s on Talos)

Manifests are a kustomization in `deploy/k8s/`:

```
namespace.yaml   configmap.yaml   secret.yaml   pvc.yaml   temporal.yaml   api.yaml   worker.yaml   web.yaml   ingress.yaml
```

```bash
# 1. cluster: Talos + Omni (or plain K3s). Confirm the default StorageClass is local-path.
kubectl get storageclass
# 2. secrets
kubectl -n bayesian-credit create secret generic api-secrets \
  --from-literal=DJANGO_SECRET_KEY=$(openssl rand -hex 32) --dry-run=client -o yaml > deploy/k8s/secret.yaml
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
pinned next to it (pod affinity) because SQLite and the RWO volumes are shared; switch to
Postgres (`DATABASE_URL`) and a ReadWriteMany volume before spreading workers over nodes.

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

The Temporal server here is the single-binary dev server with SQLite persistence on a PVC,
adequate for one research cluster. For a production deployment point `TEMPORAL_ADDRESS` at a
Temporal cluster (helm chart) or Temporal Cloud (`TEMPORAL_API_KEY` turns on TLS + API-key
auth) and delete `temporal.yaml` from the kustomization.

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
`pip install "bayesian-credit[kafka]"` plus a `confluent_kafka` producer config such as
`{"bootstrap.servers": "broker:9092"}`). Swapping the sink is a constructor argument of the
runner's `EventStore`; the checkpoint/resume mechanism does not depend on the sink. Workflow
orchestration is Temporal (see *Background jobs on Temporal* above): every background job is
a durable workflow, so job placement, retries, lost-worker detection, the chaos kill and the
maintenance schedule all go through it.
