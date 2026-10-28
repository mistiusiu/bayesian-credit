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
| `bayesian-credit-api` | `deploy/docker/Dockerfile.api` | `python:3.14-slim`, the `bayesian_credit` library, Django API (gunicorn), simulation CLI. Entrypoint modes: `serve`, `worker <job_id>`, `simulate <run_dir>`, `resume <run_dir>`, `demo`. |
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
# console  http://localhost:8080      API docs  http://localhost:8000/api/docs/
docker compose -f deploy/docker/docker-compose.yml exec api /app/entrypoint.sh demo   # seed a demo dataset/fit/simulation
```

`./data` is bind-mounted into the API container: put the Kaggle file at
`data/raw/accepted_2007_to_2018Q4.csv.gz` (or run `scripts/download_data.sh`) and it is
visible to the API. Fit artifacts and simulation run directories (checkpoints included) are
written under `data/artifacts/`. Add `--profile postgres` and set `DATABASE_URL` to move off
SQLite.

## Kubernetes (K3s on Talos)

Manifests are a kustomization in `deploy/k8s/`:

```
namespace.yaml   configmap.yaml   secret.yaml   pvc.yaml   api.yaml   web.yaml   ingress.yaml
job-template.yaml   (not applied by kustomize — one Job per simulation, optional executor)
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
# 6. open http://bayesian-credit.local (point the host at a node IP; Traefik is K3s' default ingress)
```

Sizing: the API pod requests 0.5 CPU / 1 GiB and is limited to 3 CPU / 5 GiB; the console
is tiny. A 50 k-borrower, 6-policy replay runs comfortably inside one worker node of the
proposal's hardware. The API is a single replica because SQLite and the RWO volume are
shared; switch to Postgres (`DATABASE_URL`) before scaling.

### Two ways to run simulation workers

1. **Subprocess executor (default).** The API process spawns `python manage.py run_job <id>`
   for each job; the worker runs inside the API pod. Simple, and the chaos endpoint can
   SIGKILL it. Resource limits of the API pod bound the worker.
2. **Kubernetes Jobs.** Render `deploy/k8s/job-template.yaml` with `SIM_ID` and `RUN_DIR`
   (the run directory the API created under `/app/data/artifacts/simulations/<id>/run`) and
   `kubectl apply` it. The container runs `resume <run_dir>`, which behaves like `run` when no
   checkpoint exists and otherwise continues from `checkpoints/latest.pkl`. With
   `restartPolicy: OnFailure` a killed pod is restarted by Kubernetes and resumes on its own.

## Resilience experiments (Table 10)

| Experiment | How to run it here | What should happen |
|---|---|---|
| Workload termination | `POST /api/v1/simulations/{id}/chaos/` then `/resume/`, or `deploy/chaos/chaos-monkey.sh --api … --simulation {id}` | progress continues from the last checkpoint; decisions/timeseries are identical to an uninterrupted run (tested in `core/tests/test_runner.py`) |
| Node / pod failure | `deploy/chaos/chaos-monkey.sh --k8s --selector app=simulation` (or `app=api`) | replacement pod becomes Ready; worker resumes from the PVC |
| Multiple-node failure | `talosctl -n <ip> shutdown` on two workers | pods reschedule once a node is back; no state is lost because state lives on the PVC + run dir, never in memory only |
| Resource constraint | lower `resources.limits` in `api.yaml` / `job-template.yaml` | run slows but completes; `report.json → computational_feasibility` records CPU seconds and peak RSS |
| Network interruption | `talosctl -n <ip> … ` link down for a minute, or cut the switch | API returns when the network returns; running workers are unaffected (local disk) |
| Controlled power interruption | power-cycle a worker node | as node failure; the worst case loses at most `checkpoint_every` epochs of work |
| State recovery | kill the worker between `CheckpointWritten` events and resume | the runner deletes partial epoch files after the checkpoint and recomputes them deterministically (per-epoch seeded RNG) |

Timing of recovery, number of failed workloads and whether state recovered successfully are
the metrics in Table 11 → *Infrastructure resilience*; the chaos script prints them and the
simulation's `events.jsonl` records every `CheckpointWritten` / `RunResumed` event.

## Observability

The API exposes `GET /api/v1/metrics/` in Prometheus text format (jobs by kind/status,
running simulations) and the pod carries `prometheus.io/*` scrape annotations, so a
kube-prometheus-stack / Grafana install picks it up without extra configuration. Resource
utilisation per pod comes from the kubelet / cAdvisor metrics that Prometheus already scrapes.

## Event bus and workflow engine

The proposal names Kafka (append-only event state) and Temporal (workflow orchestration).
The prototype implements the *semantics* with a file-backed append-only `events.jsonl` and
atomic checkpoints behind an `EventSink` protocol (`core/bayesian_credit/simulation/events.py`:
`JSONLSink` is the default; `KafkaSink` publishes the same events to a topic and needs
`pip install "bayesian-credit[kafka]"` plus a `confluent_kafka` producer config such as
`{"bootstrap.servers": "broker:9092"}`). Swapping the sink is a constructor argument of the
runner's `EventStore`; the checkpoint/resume mechanism does not depend on the sink. Durable
workflow orchestration (Temporal) is the natural next step for the job runner and is
intentionally out of scope for the prototype.
