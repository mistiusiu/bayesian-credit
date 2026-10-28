"""Simulation worker job: the job's process *is* the simulation worker.

``mode="run"`` calls :func:`bayesian_credit.pipeline.run_simulation` in-process (so a
SIGKILL on the job's pid kills the run mid-epoch — the chaos experiment); ``mode="resume"``
calls :meth:`SimulationRunner.resume`, which continues from ``checkpoints/latest.pkl``
(fresh start when no checkpoint exists). Pause / review waits happen inside the runner,
which polls ``control.json`` and ``reviews/<id>.ack.json`` written by the API.
"""

from __future__ import annotations

from typing import Any

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.pipeline import load_dataset
from bayesian_credit.pipeline import run_simulation as pipeline_run_simulation
from bayesian_credit.simulation.runner import SimulationRunner
from bayesian_credit.simulation.store import RunDirectory

from apps.common.jobs import JobCancelled, ProgressReporter
from apps.common.models import Job
from apps.common.utils import iso, utcnow
from apps.fits.services import load_fitted

from .models import Simulation

POLL_INTERVAL = 0.5


def run_simulation(job: Job, reporter: ProgressReporter) -> None:
    sim = Simulation.objects.select_related("fit", "fit__dataset").get(pk=job.target_id)
    mode = (job.params or {}).get("mode", "run")
    inline = job.pid is None
    rd = RunDirectory(sim.directory)
    conf = rd.read_config()

    def progress(info: dict[str, Any]) -> None:
        n = max(int(info.get("n_epochs") or 1), 1)
        done = int(info.get("epoch_index", -1)) + 1
        reporter.progress(done / n, f"epoch {info.get('epoch')} ({done}/{n}), breaker {info.get('breaker_state')}")

    common = {"progress_callback": progress, "install_signal_handlers": not inline, "poll_interval": POLL_INTERVAL}
    if mode == "resume":
        ck = rd.checkpoints.load()
        where = f"checkpoint {ck.get('epoch')}" if ck else "the start (no checkpoint yet)"
        reporter.log(f"resuming simulation #{sim.pk} from {where}", 0.0)
        result = SimulationRunner.resume(rd.path, **common)
    else:
        framework = FrameworkConfig.from_dict(conf["framework"])
        sim_cfg = SimulationConfig.from_dict(conf["simulation"])
        reporter.log(f"loading dataset #{sim.fit.dataset_id} and fit #{sim.fit_id}", 0.0)
        df = load_dataset(sim.fit.dataset.parquet)
        fitted = load_fitted(sim.fit)
        reporter.log(
            f"running {len(sim_cfg.policies)} policies over {sim_cfg.start[:7]}..{sim_cfg.end[:7]} "
            f"(max_loans={sim_cfg.max_loans}, feedback={sim_cfg.feedback_mode.value}, breaker={sim_cfg.breaker_mode})",
            0.01,
        )
        result = pipeline_run_simulation(
            df,
            fitted,
            sim_cfg,
            rd.path,
            cfg=framework,
            dataset_path=str(sim.fit.dataset.parquet),
            dataset_info=conf.get("dataset") or {},
            fit_dir=str(sim.fit.directory),
            **common,
        )
    reporter.log(
        f"simulation {result.status}: {result.epochs_completed}/{result.n_epochs} epochs, "
        f"{result.n_decisions:,} decisions in {result.wall_seconds:.1f}s",
        result.epochs_completed / max(result.n_epochs, 1),
    )
    if result.status == "cancelled":
        raise JobCancelled()


def on_failure(job: Job, exc: BaseException) -> None:
    """Make sure ``progress.json`` does not claim the run is still alive."""
    sim = Simulation.objects.filter(pk=job.target_id).first()
    if sim is None or not sim.run_dir:
        return
    rd = RunDirectory(sim.directory)
    status = (rd.read_progress() or {}).get("status")
    if status not in ("completed", "failed", "cancelled"):
        rd.write_progress(
            status="failed", finished_at=iso(utcnow()), error=f"{type(exc).__name__}: {exc}", message="failed"
        )
