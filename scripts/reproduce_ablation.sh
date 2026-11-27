#!/usr/bin/env bash
# Reproduce the DSA4900 ablation study (docs/REPRODUCE.md) in one of the three deployment modes.
#
#   scripts/reproduce_ablation.sh <docker|cli|k8s> <pilot|full|both> [extra study.py run args]
#
#   docker  Docker Compose on this machine (builds the images, starts Postgres + Temporal + API +
#           worker + console with deploy/docker/docker-compose.ablation.yml: one job at a time,
#           worker limited to ABLATION_WORKER_MEMORY / ABLATION_WORKER_CPUS, default 9g / 10)
#   cli     the Poetry environment: `make infra` (Postgres + Temporal in Docker), then the API on
#           :API_PORT (default 8010) and one worker (concurrency 1) as local processes, stopped on exit
#   k8s     an existing `kubectl apply -k deploy/k8s` install: copies the CSV onto the data PVC if
#           missing, sizes the worker Deployment, port-forwards the API to :LOCAL_PORT (default 18000)
#
# Needs: python3 >= 3.10, the Lending Club CSV at data/raw/Loan_status_2007-2020Q3.csv
# (scripts/download_data.sh), and docker / poetry (.venv via `make setup`) / kubectl for the mode.
# Results land in data/runs/ablation-<scale>/ (RESULTS.md, results.json/.csv, state.json); the
# PDF is built when reportlab is importable. Re-running resumes where it stopped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"; SCALE="${2:-}"
shift $(( $# >= 2 ? 2 : $# ))
case "$MODE" in docker|cli|k8s) ;; *) sed -n '2,20p' "$0"; exit 2 ;; esac
case "$SCALE" in pilot|full) SCALES=("$SCALE") ;; both) SCALES=(pilot full) ;; *) sed -n '2,20p' "$0"; exit 2 ;; esac

CSV="$ROOT/data/raw/Loan_status_2007-2020Q3.csv"
[[ -f "$CSV" ]] || { echo "missing $CSV - run scripts/download_data.sh first" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

ABLATION_WORKER_MEMORY="${ABLATION_WORKER_MEMORY:-9g}"
ABLATION_WORKER_CPUS="${ABLATION_WORKER_CPUS:-10}"
export ABLATION_WORKER_MEMORY ABLATION_WORKER_CPUS
ANON='grep "^anon " /sys/fs/cgroup/memory.stat | cut -d" " -f2'   # job memory without page cache
CLEANUP=()
on_exit() { for c in "${CLEANUP[@]}"; do eval "$c" || true; done; }
trap on_exit EXIT

wait_health() {  # $1 = API base URL
  for _ in $(seq 1 90); do
    python3 - "$1" <<'PY' && return 0
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/health/", timeout=5)
except Exception:
    sys.exit(1)
PY
    sleep 2
  done
  echo "API at $1 did not become healthy" >&2; exit 1
}

case "$MODE" in
  docker)
    command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
    export HOST_UID="${HOST_UID:-$(id -u)}" HOST_GID="${HOST_GID:-$(id -g)}"
    COMPOSE=(docker compose -f deploy/docker/docker-compose.yml -f deploy/docker/docker-compose.ablation.yml)
    "${COMPOSE[@]}" build api web
    "${COMPOSE[@]}" up -d --wait api worker web   # pulls in Postgres, Temporal and its schema/namespace jobs
    API="http://localhost:${API_PORT:-8000}/api/v1"
    WORKER="$("${COMPOSE[@]}" ps -q worker)"
    MEM_CMD="docker exec $WORKER sh -c '$ANON'"
    echo "console: http://localhost:${WEB_PORT:-8080}  worker: ${ABLATION_WORKER_MEMORY} / ${ABLATION_WORKER_CPUS} CPUs"
    ;;
  cli)
    PY="$ROOT/.venv/bin/python"
    [[ -x "$PY" ]] || { echo "missing .venv - run: make setup" >&2; exit 1; }
    make infra
    PORT="${API_PORT:-8010}"
    LOGS="$ROOT/data/runs/ablation-logs"; mkdir -p "$LOGS"
    (cd api && "$PY" manage.py migrate --noinput)
    (cd api && exec "$PY" manage.py runserver --noreload "0.0.0.0:$PORT" >"$LOGS/api.log" 2>&1) &
    CLEANUP+=("kill $! 2>/dev/null")
    (cd api && TEMPORAL_WORKER_CONCURRENCY=1 exec "$PY" manage.py temporal_worker >"$LOGS/worker.log" 2>&1) &
    WORKER_PID=$!
    CLEANUP+=("kill -TERM $WORKER_PID 2>/dev/null; wait $WORKER_PID 2>/dev/null")
    API="http://localhost:$PORT/api/v1"
    # no container limit here: report the resident memory of the worker's job processes
    MEM_CMD="ps -eo rss=,args= | awk '/manage.py run_job/ && !/awk/ {s+=\$1*1024} END {print s+0}'"
    echo "API http://localhost:$PORT (logs in $LOGS); the job runs as a local process, without a memory limit"
    ;;
  k8s)
    command -v kubectl >/dev/null || { echo "kubectl is required" >&2; exit 1; }
    NS="${NAMESPACE:-bayesian-credit}"
    K=(kubectl -n "$NS")
    "${K[@]}" get deploy/api deploy/worker >/dev/null || { echo "deploy first: kubectl apply -k deploy/k8s" >&2; exit 1; }
    if ! "${K[@]}" exec deploy/worker -- test -f /app/data/raw/Loan_status_2007-2020Q3.csv; then
      echo "copying the CSV onto the data PVC (1.8 GB)"
      "${K[@]}" exec deploy/worker -- mkdir -p /app/data/raw
      POD="$("${K[@]}" get pod -l app=worker -o jsonpath='{.items[0].metadata.name}')"
      kubectl cp "$CSV" "$NS/$POD:/app/data/raw/Loan_status_2007-2020Q3.csv"
    fi
    MEM_K8S="${ABLATION_WORKER_MEMORY%g}Gi"
    "${K[@]}" set resources deploy/worker --limits="cpu=${ABLATION_WORKER_CPUS},memory=${MEM_K8S}" \
      --requests="cpu=1,memory=${MEM_K8S}"
    "${K[@]}" set env deploy/worker TEMPORAL_WORKER_CONCURRENCY=1
    "${K[@]}" rollout status deploy/worker --timeout=300s
    LOCAL_PORT="${LOCAL_PORT:-18000}"
    "${K[@]}" port-forward svc/api "$LOCAL_PORT:8000" >/dev/null 2>&1 &
    CLEANUP+=("kill $! 2>/dev/null")
    API="http://localhost:$LOCAL_PORT/api/v1"
    MEM_CMD="kubectl -n $NS exec deploy/worker -- sh -c '$ANON'"
    echo "worker Deployment: limits cpu=${ABLATION_WORKER_CPUS} memory=${MEM_K8S} (the node needs that much allocatable)"
    ;;
esac

wait_health "$API"
for S in "${SCALES[@]}"; do
  OUT="$ROOT/data/runs/ablation-$S"
  python3 scripts/ablation/study.py run --scale "$S" --api "$API" --out "$OUT" --data-file "$CSV" \
    --mem-cmd "$MEM_CMD" "$@"
  if python3 -c "import reportlab" 2>/dev/null; then
    python3 scripts/ablation/report.py "$OUT"
  else
    echo "PDF skipped (pip install reportlab, then: python3 scripts/ablation/report.py $OUT)"
  fi
done
echo "done: data/runs/ablation-{${SCALES[*]// /,}}/RESULTS.md; compare with another run: python3 scripts/ablation/study.py compare A B"
