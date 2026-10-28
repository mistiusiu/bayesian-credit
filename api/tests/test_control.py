"""Simulation control actions (pause / resume / cancel / chaos) and worker-loss detection.

A stand-in worker (``python -c 'sleep' run_job <id>``) plays the live simulation process so
the API's liveness checks, SIGKILL and control files can be exercised without a real run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.django_db


@pytest.fixture
def live_sim(e2e):
    from bayesian_credit.simulation.store import RunDirectory

    from apps.common import artifacts
    from apps.common.models import Job
    from apps.fits.models import Fit
    from apps.simulations import services
    from apps.simulations.models import Simulation

    fit = Fit.objects.get(pk=e2e["fit"]["id"])
    config = dict(e2e["simulation"]["config"], start="2015-01", end="2015-06", max_loans=400, checkpoint_every=2)
    sim = Simulation.objects.create(name="stand-in", fit=fit, config=config, policies=list(config["policies"]))
    sim.run_dir = str(artifacts.run_dir(sim.pk))
    artifacts.reset_dir(artifacts.simulation_dir(sim.pk))
    framework, simulation = services.core_configs(fit, config)
    rd = RunDirectory.create(
        sim.run_dir,
        framework,
        simulation,
        dataset={"id": fit.dataset_id, "path": str(fit.dataset.parquet)},
        fit_dir=str(fit.directory),
    )
    rd.write_progress(status="running", epoch_index=1, current_epoch="2015-02", n_epochs=6)
    job = Job.objects.create(
        kind="simulation", status="running", target_type="simulation", target_id=sim.pk, params={"mode": "run"}
    )
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "run_job", str(job.pk)])
    job.pid = proc.pid
    job.save(update_fields=["pid"])
    sim.job = job
    sim.save(update_fields=["run_dir", "job"])
    yield sim, proc, rd
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def test_live_worker_is_reported(api, live_sim):
    sim, proc, _ = live_sim
    body = api.get(f"/api/v1/simulations/{sim.pk}/").json()
    assert body["status"] == "running"
    assert body["worker"] == {"pid": proc.pid, "alive": True, "last_heartbeat": body["worker"]["last_heartbeat"]}


def test_pause_and_cancel_write_control_json(api, live_sim):
    from apps.common.utils import read_json

    sim, _, rd = live_sim
    r = api.post(f"/api/v1/simulations/{sim.pk}/pause/")
    assert r.status_code == 200 and r.json()["id"] == sim.pk
    assert read_json(rd.control_path)["command"] == "pause"
    r = api.post(f"/api/v1/simulations/{sim.pk}/cancel/")
    assert r.status_code == 200
    assert read_json(rd.control_path)["command"] == "cancel"


def test_resume_paused_writes_resume(api, live_sim):
    from apps.common.utils import read_json

    sim, _, rd = live_sim
    rd.write_progress(status="paused")
    assert api.get(f"/api/v1/simulations/{sim.pk}/").json()["status"] == "paused"
    r = api.post(f"/api/v1/simulations/{sim.pk}/resume/")
    assert r.status_code == 200
    assert read_json(rd.control_path)["command"] == "resume"


def test_awaiting_review_cannot_be_resumed(api, live_sim):
    sim, _, rd = live_sim
    rd.write_progress(status="awaiting_review")
    r = api.post(f"/api/v1/simulations/{sim.pk}/resume/")
    assert r.status_code == 409 and "review" in r.json()["detail"]


def test_chaos_kills_the_worker_then_resume_recovers(api, live_sim):
    sim, proc, rd = live_sim
    r = api.post(f"/api/v1/simulations/{sim.pk}/chaos/")
    assert r.status_code == 200, r.content
    body = r.json()
    from apps.common.jobs import pid_alive

    proc.wait(timeout=10)  # (the API already reaped the SIGKILLed child, so the code is not observable here)
    assert not pid_alive(proc.pid)
    assert body["status"] == "interrupted" and body["worker"]["alive"] is False
    assert body["job"]["status"] == "failed" and body["job"]["message"] == "chaos: worker killed"
    assert body["last_recovery"]["killed_at"] and body["last_recovery"]["killed_at_epoch"] == "2015-02"
    events = api.get(f"/api/v1/simulations/{sim.pk}/events/", {"type": "WorkerKilled"}).json()["results"]
    assert events and events[0]["level"] == "critical" and events[0]["payload"]["pid"] == proc.pid
    assert api.post(f"/api/v1/simulations/{sim.pk}/chaos/").status_code == 409
    assert api.post(f"/api/v1/simulations/{sim.pk}/pause/").status_code == 409

    # resume launches a new worker in resume mode (inline under JOBS_SYNC): no checkpoint yet,
    # so the runner starts from the beginning and runs to completion
    r = api.post(f"/api/v1/simulations/{sim.pk}/resume/")
    assert r.status_code == 200, r.content
    body = r.json()
    assert body["status"] == "completed" and body["restarts"] == 1
    assert body["job"]["status"] == "succeeded" and body["epochs_done"] == body["epochs_total"]
    assert body["last_recovery"]["resumed_at"] is not None
    assert api.post(f"/api/v1/simulations/{sim.pk}/resume/").status_code == 409


def test_cancel_dead_worker_marks_cancelled(api, live_sim):
    sim, proc, rd = live_sim
    proc.kill()
    proc.wait(timeout=10)
    r = api.post(f"/api/v1/simulations/{sim.pk}/cancel/")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert rd.read_progress()["status"] == "cancelled"
    assert api.post(f"/api/v1/simulations/{sim.pk}/cancel/").status_code == 409


def test_lost_worker_detected_lazily(api, live_sim):
    sim, proc, _ = live_sim
    proc.kill()
    proc.wait(timeout=10)
    job = api.get(f"/api/v1/jobs/{sim.job_id}/").json()
    assert job["status"] == "failed" and job["message"] == "worker lost" and job["pid"] is None
    body = api.get(f"/api/v1/simulations/{sim.pk}/").json()
    assert body["status"] == "interrupted" and body["last_recovery"]["killed_at"]


def test_completed_simulation_rejects_controls(api, e2e):
    sid = e2e["simulation"]["id"]
    for action in ("pause", "resume", "cancel", "chaos"):
        r = api.post(f"/api/v1/simulations/{sid}/{action}/")
        assert r.status_code == 409, action
        assert r.json()["code"] == "conflict"


def test_sub_resources_409_before_first_epoch(api, live_sim):
    sim, _, _ = live_sim
    for part in ("report", "arms", "posteriors", "simpson", "sentinel/buckets"):
        r = api.get(f"/api/v1/simulations/{sim.pk}/{part}/")
        assert r.status_code == 409 and r.json()["code"] == "conflict", part
    assert api.get(f"/api/v1/simulations/{sim.pk}/timeseries/").json()["rows"] == []
    assert Path(sim.run_dir, "config.json").is_file()


def _ev(seq, etype, epoch, ts, payload=None):
    return {"seq": seq, "type": etype, "epoch": epoch, "ts": ts, "payload": payload or {}}


@pytest.mark.parametrize(
    ("post", "expected_match", "expected_seconds"),
    [
        ([("2015-03", 10.0), ("2015-04", 7.0)], True, 30.0),  # replay reproduced the killed epoch
        ([("2015-03", 11.0)], False, 30.0),  # replayed metrics differ
        ([], None, None),  # the new worker has not passed the kill point yet
    ],
)
def test_recovery_info_compares_replayed_epochs(post, expected_match, expected_seconds):
    from types import SimpleNamespace

    from apps.simulations.services import recovery_info

    sim = SimpleNamespace(
        recovery={"killed_at": "2026-10-16T10:00:00Z", "killed_at_epoch": "2015-03"},
        config={"start": "2015-01"},
    )
    events = [
        _ev(1, "EpochCompleted", "2015-01", "2026-10-16T09:59:50Z", {"per_policy": {"thompson": {"reward": 1.0}}}),
        _ev(2, "CheckpointWritten", "2015-02", "2026-10-16T09:59:55Z"),
        _ev(3, "EpochCompleted", "2015-02", "2026-10-16T09:59:55Z", {"per_policy": {"thompson": {"reward": 2.0}}}),
        _ev(4, "EpochCompleted", "2015-03", "2026-10-16T09:59:58Z", {"per_policy": {"thompson": {"reward": 10.0}}}),
        _ev(5, "WorkerKilled", "2015-03", "2026-10-16T10:00:00Z", {"pid": 1, "signal": "SIGKILL"}),
        _ev(6, "RunResumed", "2015-02", "2026-10-16T10:00:20Z", {"from_epoch": "2015-02"}),
    ]
    for k, (epoch, reward) in enumerate(post):
        events.append(
            _ev(
                7 + k,
                "EpochCompleted",
                epoch,
                f"2026-10-16T10:00:{30 + k:02d}Z",
                {"per_policy": {"thompson": {"reward": reward}}},
            )
        )
    info = recovery_info(sim, events)
    assert info["resumed_from_epoch"] == "2015-02" and info["replayed_epochs"] == 1
    assert info["deterministic_match"] is expected_match
    assert info["recovery_seconds"] == expected_seconds
