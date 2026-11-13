SHELL := /bin/bash
VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: help setup lock test lint verify infra infra-down worker api web pipeline synthetic data docker k8s clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install the Poetry environment in .venv (core editable + API + dev tools) and the frontend deps
	poetry install --with dev
	cd frontend && pnpm install

lock: ## Re-resolve poetry.lock after editing pyproject.toml
	poetry lock

test: ## Run core + API tests
	cd core && ../$(PY) -m pytest -q
	cd api && ../$(PY) -m pytest -q

lint: ## Ruff + frontend typecheck/lint
	$(VENV)/bin/ruff check core api scripts
	cd frontend && pnpm typecheck && pnpm lint

verify: ## Everything: lint, core + API tests, frontend typecheck/lint/build, quick synthetic pipeline
	bash scripts/verify.sh

API_PORT ?= 8010

COMPOSE := docker compose -f deploy/docker/docker-compose.yml
INFRA := postgres temporal temporal-ui

infra: ## Start PostgreSQL + Temporal (on PostgreSQL) + Temporal UI for local work: :5432, :7233, http://localhost:8233
	$(COMPOSE) up -d --wait $(INFRA)
	$(COMPOSE) run --rm temporal-namespace

infra-down: ## Stop the local infrastructure (data stays in the pgdata volume)
	$(COMPOSE) stop $(INFRA)

worker: ## Run a Temporal worker that executes the background jobs (needs `make infra`)
	cd api && ../$(PY) manage.py temporal_worker

api: ## Run the Django API on :$(API_PORT) (8000 is often taken on dev machines; needs `make infra`, jobs need `make worker`)
	cd api && ../$(PY) manage.py migrate --noinput && ../$(PY) manage.py runserver 0.0.0.0:$(API_PORT)

web: ## Run the Vite dev server on :5173 against the live API on :$(API_PORT)
	cd frontend && VITE_API_PROXY_TARGET=http://localhost:$(API_PORT) pnpm dev:live

synthetic: ## Generate a 200k-loan synthetic Lending Club twin
	$(PY) scripts/make_synthetic.py --n 200000 --seed 7 --out data/processed/synthetic.parquet

data: ## Download the Lending Club accepted-loans file from Kaggle (needs ~/.kaggle/access_token)
	bash scripts/download_data.sh

pipeline: ## End-to-end: ingest → fit → shadow portfolio → simulation → report (synthetic)
	$(PY) scripts/run_pipeline.py --source synthetic

pipeline-lc: ## Same on 200k real loans
	$(PY) scripts/run_pipeline.py --source lending_club --max-rows 200000

docker: ## Build and start Temporal, api, worker and frontend containers
	docker compose -f deploy/docker/docker-compose.yml up --build

k8s: ## Apply the K3s manifests
	kubectl apply -k deploy/k8s

clean: ## Remove caches and build outputs
	rm -rf core/.pytest_cache api/.pytest_cache .ruff_cache frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
