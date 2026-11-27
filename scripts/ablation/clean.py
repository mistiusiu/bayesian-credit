#!/usr/bin/env python3
"""Remove the intermediate caches an analysis leaves behind, keeping every result.

Removed (each regenerable, or only needed to resume an unfinished run):

* ``checkpoints/`` of finished simulation runs (``<ARTIFACT_ROOT>/simulations/<id>/run/`` whose
  ``progress.json`` status is ``completed``; add ``--unfinished`` for ``failed`` / ``cancelled``
  runs, which can then no longer be resumed). These are the bulk: 0.3-3 GB per
  pilot run. Runs that are ``running`` / ``paused`` / ``queued`` are never touched;
* the loader's parquet caches ``<DATA_ROOT>/processed/*.canonical.v*.parquet`` (+ ``.json``),
  rebuilt from the raw CSV in seconds (``--keep-loader-cache`` keeps them);
* ``checkpoints/`` and the ``resilience/`` kill-and-resume scratch copy of finished
  ``scripts/run_pipeline.py`` runs under ``<DATA_ROOT>/runs/``;
* with ``--code-caches``: ``__pycache__``, ``.pytest_cache``, ``.ruff_cache``, ``.mypy_cache`` and
  ``frontend/node_modules/.vite`` under the repository (never inside ``.venv`` / ``node_modules``).

Kept: everything the console reads and every result - per run ``timeseries.parquet``,
``events.jsonl``, ``sentinel.parquet``, ``arms/``, ``decisions/``, ``feedback/``, ``reviews/``,
``report.json``, ``progress.json``, ``config.json``, a fit-less run's ``fit/``; fit directories
(and their ``api_*.json``); dataset ``loans.parquet`` / summaries; job logs; ``data/runs/*``
results (``state.json``, ``results.*``, ``RESULTS.md``, PDFs, logs); the raw CSV.

Dry run by default; ``--yes`` deletes. Standard library only, so it runs anywhere:

    python3 scripts/ablation/clean.py                                  # host / local CLI install
    docker compose -f deploy/docker/docker-compose.yml exec worker python /app/scripts/ablation/clean.py --data-root /app/data --yes
    kubectl -n bayesian-credit exec -i deploy/worker -- python - --data-root /app/data --yes < scripts/ablation/clean.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

#: progress.json statuses (core simulation.store.RUN_STATUSES). A worker killed mid-run leaves
#: "running" behind (the API shows it as interrupted), so such a run is skipped as in use.
ACTIVE = {"queued", "running", "paused", "awaiting_review"}
FINISHED = {"completed"}
UNFINISHED = {"failed", "cancelled"}
CODE_CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_DIRS = {".venv", "node_modules", ".git"}


def size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).lstat().st_size
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def run_status(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "progress.json").read_text()).get("status")
    except (OSError, ValueError):
        return None


def targets(data_root: Path, artifact_root: Path, repo: Path | None, args: argparse.Namespace) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    allowed = FINISHED | (UNFINISHED if args.unfinished else set())
    busy = False
    sims = artifact_root / "simulations"
    for run_dir in sorted(sims.glob("*/run")) if sims.is_dir() else []:
        ck = run_dir / "checkpoints"
        if not ck.is_dir():
            continue
        status = run_status(run_dir)
        if status in allowed:
            found.append((ck, f"simulation {run_dir.parent.name} ({status}) checkpoints"))
        elif status in ACTIVE:
            busy = True
            print(f"skip  simulation {run_dir.parent.name}: {status} (still in use)")
        else:
            print(f"skip  simulation {run_dir.parent.name}: {status or 'no progress.json'}"
                  + (" (resumable; --unfinished removes its checkpoints)" if status in UNFINISHED else ""))
    if busy and not args.keep_loader_cache:
        print("skip  loader parquet cache: a simulation is still running (jobs may read it)")
    elif not args.keep_loader_cache:
        for p in sorted((data_root / "processed").glob("*.canonical.v*.*")):
            if p.suffix in (".parquet", ".json"):
                found.append((p, "loader parquet cache"))
    runs = data_root / "runs"
    for sim_dir in sorted(runs.glob("*/simulation")) if runs.is_dir() else []:
        status = run_status(sim_dir)
        if status in ACTIVE:
            print(f"skip  pipeline run {sim_dir.parent.name}: {status}")
            continue
        for name in ("checkpoints",):
            if (sim_dir / name).is_dir():
                found.append((sim_dir / name, f"pipeline run {sim_dir.parent.name} checkpoints"))
        if (sim_dir.parent / "resilience").is_dir():
            found.append((sim_dir.parent / "resilience", f"pipeline run {sim_dir.parent.name} kill-resume scratch"))
    if args.code_caches and repo is not None:
        for root, dirs, _files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (Path(root) / d).is_symlink()]
            for d in list(dirs):
                if d in CODE_CACHES:
                    found.append((Path(root) / d, "code cache"))
                    dirs.remove(d)
        vite = repo / "frontend" / "node_modules" / ".vite"
        if vite.is_dir():
            found.append((vite, "code cache"))
    return found


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve() if "__file__" in globals() else None
    repo_guess = here.parents[2] if here and len(here.parents) > 2 else None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT") or
                                                          (repo_guess / "data" if repo_guess else "data")))
    ap.add_argument("--artifact-root", type=Path, default=os.environ.get("ARTIFACT_ROOT"),
                    help="default <data-root>/artifacts")
    ap.add_argument("--repo", type=Path, default=repo_guess, help="repository for --code-caches")
    ap.add_argument("--unfinished", action="store_true", help="also failed / cancelled runs (no longer resumable)")
    ap.add_argument("--keep-loader-cache", action="store_true", help="keep data/processed parquet caches")
    ap.add_argument("--code-caches", action="store_true", help="also __pycache__, .pytest_cache, ...")
    ap.add_argument("--yes", action="store_true", help="delete (default: dry run)")
    args = ap.parse_args(argv)
    data_root = args.data_root.resolve()
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else data_root / "artifacts"
    if not data_root.is_dir():
        print(f"no data root at {data_root}", file=sys.stderr)
        return 2
    found = targets(data_root, artifact_root, args.repo, args)
    total = 0
    for path, why in found:
        n = size(path)
        total += n
        print(f"{'delete' if args.yes else 'would delete'}  {human(n):>9}  {path}  [{why}]")
        if args.yes:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    verb = "freed" if args.yes else "would free"
    print(f"{len(found)} item(s), {verb} {human(total)}" + ("" if args.yes else "  (dry run; --yes deletes)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
