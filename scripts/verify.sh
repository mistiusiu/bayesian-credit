#!/usr/bin/env bash
# Full verification: lint, core tests, API tests, frontend typecheck/lint/build, quick synthetic pipeline.
# Usage: scripts/verify.sh [--fast]   (--fast skips the frontend build and the pipeline run)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FAST=0; [[ "${1:-}" == "--fast" ]] && FAST=1
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "missing .venv — run: make setup"; exit 1; }

step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
fail=0
run() { if "$@"; then printf '\033[32m   ok\033[0m\n'; else printf '\033[31m   FAILED: %s\033[0m\n' "$*"; fail=1; fi; }
# run in a directory; the subshell only scopes the cd, so a failure still reaches $fail
run_in() {
  local dir="$1"; shift
  if (cd "$dir" && "$@"); then printf '\033[32m   ok\033[0m\n'; else printf '\033[31m   FAILED (in %s): %s\033[0m\n' "$dir" "$*"; fail=1; fi
}

step "ruff (core, api, scripts)";         run "$ROOT/.venv/bin/ruff" check core api scripts
step "core tests";                         run_in core "$PY" -m pytest -q -x --no-header -p no:cacheprovider
if [[ -f api/manage.py ]]; then
  step "api tests";                        run_in api "$PY" -m pytest -q -x --no-header -p no:cacheprovider
  step "django system checks";             run_in api "$PY" manage.py check
fi
if [[ -f frontend/package.json ]]; then
  step "frontend typecheck";               run_in frontend pnpm -s typecheck
  step "frontend lint";                    run_in frontend pnpm -s lint
  if [[ $FAST -eq 0 ]]; then
    step "frontend build";                 run_in frontend pnpm -s build
  fi
fi
if [[ $FAST -eq 0 && -f scripts/run_pipeline.py ]]; then
  step "synthetic pipeline (small)"
  run "$PY" scripts/run_pipeline.py --source synthetic --n 30000 --max-loans 5000 --out data/runs/verify --seed 1
fi

printf '\n'
if [[ $fail -eq 0 ]]; then printf '\033[1;32mALL CHECKS PASSED\033[0m\n'; else printf '\033[1;31mSOME CHECKS FAILED\033[0m\n'; exit 1; fi
