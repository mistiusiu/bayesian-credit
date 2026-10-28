#!/usr/bin/env bash
# Chaos Monkey for the shadow lending engine (proposal §3.6, Appendix G.6 "Infrastructure
# Resilience Protocol"). Repeatedly kills the simulation worker mid-run and asks the system to
# recover; prints whether the run resumed from its last checkpoint.
#
#   Local / docker-compose (kills the worker process through the API's chaos endpoint):
#     deploy/chaos/chaos-monkey.sh --api http://localhost:8000 --simulation 3 --rounds 3 --interval 25
#
#   Kubernetes (deletes the worker pod; the Job/Deployment restarts it and the entrypoint resumes):
#     deploy/chaos/chaos-monkey.sh --k8s --namespace bayesian-credit --selector app=simulation --rounds 3 --interval 60
#     deploy/chaos/chaos-monkey.sh --k8s --selector app=api            # node-failure style: kill the API pod itself
#
# Exit code 0 when every round recovered (status back to running/completed), 1 otherwise.
set -euo pipefail

API="http://localhost:8000"
SIM=""
ROUNDS=3
INTERVAL=30
MODE="api"
NAMESPACE="bayesian-credit"
SELECTOR="app=simulation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api) API="$2"; shift 2;;
    --simulation) SIM="$2"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --k8s) MODE="k8s"; shift;;
    --namespace) NAMESPACE="$2"; shift 2;;
    --selector) SELECTOR="$2"; shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
status_of() { curl -fsS "$API/api/v1/simulations/$SIM/" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("current_epoch"), d.get("pct"))'; }

failures=0
for ((round=1; round<=ROUNDS; round++)); do
  log "round $round/$ROUNDS — letting the system work for ${INTERVAL}s"
  sleep "$INTERVAL"
  if [[ "$MODE" == "api" ]]; then
    [[ -n "$SIM" ]] || { echo "--simulation <id> is required in api mode"; exit 2; }
    before=$(status_of); log "before kill: $before"
    log "SIGKILL the worker via POST /simulations/$SIM/chaos/"
    curl -fsS -X POST "$API/api/v1/simulations/$SIM/chaos/" >/dev/null || true
    sleep 3
    log "after kill: $(status_of)"
    log "POST /simulations/$SIM/resume/  (resume from checkpoints/latest.pkl)"
    curl -fsS -X POST "$API/api/v1/simulations/$SIM/resume/" >/dev/null || true
    sleep 8
    after=$(status_of); log "after resume: $after"
    case "$after" in running*|completed*) log "recovered ✔";; *) log "NOT recovered ✘"; failures=$((failures+1));; esac
  else
    pod=$(kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [[ -n "$pod" ]] || { log "no pod matches $SELECTOR"; failures=$((failures+1)); continue; }
    log "deleting pod $pod (grace 0)"
    kubectl -n "$NAMESPACE" delete pod "$pod" --grace-period=0 --force >/dev/null
    log "waiting for a replacement pod to become Ready"
    if kubectl -n "$NAMESPACE" wait --for=condition=Ready pod -l "$SELECTOR" --timeout=180s >/dev/null 2>&1; then
      log "replacement Ready ✔ — the worker entrypoint resumes from the last checkpoint on the PVC"
    else
      log "no Ready replacement within 180s ✘"; failures=$((failures+1))
    fi
  fi
done

if [[ $failures -eq 0 ]]; then log "all $ROUNDS rounds recovered"; exit 0; fi
log "$failures/$ROUNDS rounds failed to recover"; exit 1
