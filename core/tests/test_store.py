"""Run-directory protocol, event store/codec and atomic checkpoints (SPEC §2.11.5, G.6)."""

import json
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.data import DEFAULT_SHIFT
from bayesian_credit.exceptions import CheckpointError
from bayesian_credit.pipeline import fit_models, prepare_dataset
from bayesian_credit.simulation import events as ev
from bayesian_credit.simulation.runner import SimulationRunner
from bayesian_credit.simulation.store import (
    PROGRESS_KEYS,
    Checkpoint,
    EventStore,
    RunDirectory,
    atomic_write_json,
    atomic_write_parquet,
    config_hash,
)


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=3, shift_at="2016-06", shift=dict(DEFAULT_SHIFT))
    return cfg, df, fit_models(df, cfg, "2016-01-01", evaluate=False)


# ----- atomic writes -------------------------------------------------------------------------
def test_atomic_json_and_parquet_leave_no_temp_files(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": np.float64(1.5), "y": float("nan"), "z": np.int64(3)})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1.5, "y": None, "z": 3}
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    atomic_write_parquet(tmp_path / "t.parquet", df)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "t.parquet"), df, check_dtype=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "t.parquet"]


def test_failed_atomic_write_keeps_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    atomic_write_json(target, {"status": "running"})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"status": "corrupt"})
    monkeypatch.undo()
    assert json.loads(target.read_text()) == {"status": "running"}
    assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]


# ----- checkpoints ---------------------------------------------------------------------------------
def test_checkpoint_keeps_last_three_and_survives_a_corrupt_latest(tmp_path):
    ck = Checkpoint(tmp_path / "checkpoints", keep=3)
    assert ck.load() is None
    for i, e in enumerate(["2015-01", "2015-02", "2015-03", "2015-04"]):
        ck.write({"epoch_index": i, "epoch": e, "payload": np.arange(3) * i}, e)
    assert [e for e, _ in ck.list()] == ["2015-02", "2015-03", "2015-04"]
    state = ck.load()
    assert state["epoch_index"] == 3 and state["payload"].tolist() == [0, 3, 6]
    ck.latest_path.write_bytes(b"\x80\x05garbage")  # e.g. a disk error after the fact
    assert ck.load()["epoch"] == "2015-04"  # newest readable epoch-stamped copy
    with pytest.raises(CheckpointError):
        ck.load(ck.latest_path)
    bad = tmp_path / "x.pkl"
    bad.write_bytes(pickle.dumps({"no_epoch": 1}))
    with pytest.raises(CheckpointError):
        ck.load(bad)


# ----- events ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("etype", ev.EVENT_TYPES)
def test_event_codec_round_trip(etype):
    klass = getattr(ev, etype)
    e = klass(epoch="2015-03", payload={"n": np.int64(3), "psi": np.float64(0.2), "bad": float("inf"), "arr": np.arange(2)})
    line = ev.encode_event(e)
    back = ev.decode_event(line)
    assert type(back) is klass and back.type == etype and back.epoch == "2015-03" and back.ts == e.ts
    assert back.payload == {"n": 3, "psi": 0.2, "bad": None, "arr": [0, 1]}
    assert json.loads(line).keys() == {"ts", "type", "epoch", "payload"}


def test_event_store_filters_and_skips_a_truncated_line(tmp_path):
    store = EventStore(tmp_path / "events.jsonl")
    store.append(ev.RunStarted(payload={"n_epochs": 2}))
    store.append(ev.EpochStarted(epoch="2015-01"))
    store.append(ev.EpochCompleted(epoch="2015-01", payload={"n_decisions": 5}))
    with open(store.path, "a") as fh:
        fh.write('{"ts": "2026-01-01", "type": "EpochSta')  # SIGKILL mid-write
    rows = store.read()
    assert [r["type"] for r in rows] == ["RunStarted", "EpochStarted", "EpochCompleted"]
    assert [r["seq"] for r in rows] == [1, 2, 3]
    assert [r["seq"] for r in store.read(types="EpochCompleted")] == [3]
    assert [r["type"] for r in store.read(after=1, limit=1)] == ["EpochStarted"]
    assert ev.decode_event({"type": "Unknown", "payload": {}}).payload["_type"] == "Unknown"


# ----- run directory protocol -------------------------------------------------------------------------------
def test_run_directory_files_and_keys(tmp_path):
    rd = RunDirectory.create(tmp_path / "run", FrameworkConfig(), SimulationConfig(), {"path": "x.parquet"}, None)
    cfg = json.loads(rd.config_path.read_text())
    assert set(cfg) == {"framework", "simulation", "dataset", "fit_dir", "created_at"}
    assert rd.config_hash() == config_hash(FrameworkConfig().to_dict(), SimulationConfig().to_dict())
    prog = rd.read_progress()
    assert set(prog) == set(PROGRESS_KEYS) and prog["status"] == "queued"
    assert rd.read_control() is None
    rd.write_control("pause")
    assert rd.read_control()["command"] == "pause" and "at" in rd.read_control()
    with pytest.raises(ValueError):
        rd.write_control("explode")
    rd.write_review({"id": "review-0001", "epoch": "2016-06", "psi_aggregate": 0.4})
    assert rd.read_ack("review-0001") is None
    rd.write_ack("review-0001", "resegment", "looks like a new segment")
    ack = json.loads(rd.ack_path("review-0001").read_text())
    assert set(ack) == {"action", "note", "at"} and ack["action"] == "resegment"
    assert rd.list_reviews()[0]["ack"]["action"] == "resegment"
    for e in ("2015-01", "2015-02", "2015-03"):
        rd.write_decisions(e, pd.DataFrame({"epoch": [e], "policy": ["p"], "loan_id": [1], "simulated_reward": [np.nan]}))
    assert rd.truncate_after("2015-01") == 2 and rd.decision_epochs() == ["2015-01"]


def test_protocol_files_after_a_run(tmp_path):
    cfg, df, fitted = _data()
    sim = SimulationConfig(start="2016-01-01", end="2016-04-30", max_loans=600, seed=1, interim_months=2,
                           checkpoint_every=2, policies=("thompson", "historical"))
    res = SimulationRunner(df, cfg, sim, tmp_path / "run", fitted=fitted, install_signal_handlers=False).run()
    rd = RunDirectory(tmp_path / "run")
    assert res.status == "completed"
    for p in (rd.config_path, rd.progress_path, rd.events_path, rd.timeseries_path, rd.sentinel_path, rd.report_path,
              rd.runtime_path, rd.checkpoints.latest_path):
        assert p.is_file(), p
    epochs = ["2016-01", "2016-02", "2016-03", "2016-04"]
    assert rd.decision_epochs() == epochs
    assert sorted(p.name for p in rd.arms_dir.iterdir()) == ["epoch=2016-03.parquet", "epoch=2016-04.parquet"]
    prog = rd.read_progress()
    assert set(prog) == set(PROGRESS_KEYS)
    assert prog["status"] == "completed" and prog["epoch_index"] == 3 and prog["pct"] == 100.0 and prog["n_epochs"] == 4
    assert prog["current_epoch"] == "2016-04" and prog["finished_at"] and prog["error"] is None and prog["pid"] == os.getpid()
    conf = rd.read_config()
    assert conf["dataset"]["path"].endswith("dataset.parquet") and conf["fit_dir"].endswith("fit")
    events = rd.events.read()
    for line in rd.events_path.read_text().splitlines():
        assert set(json.loads(line)) == {"ts", "type", "epoch", "payload"}
    types = [e["type"] for e in events]
    assert types[0] == "RunStarted" and types[-1] == "RunCompleted"
    assert types.count("EpochCompleted") == 4 and types.count("CheckpointWritten") == 2
    ck = rd.checkpoints.load()
    assert set(ck) >= {"epoch_index", "policies", "governed", "sentinel", "environment", "queue", "segmenter",
                       "metrics_state", "config_hash"}
    assert ck["epoch_index"] == 3 and set(ck["policies"]) == {"thompson", "historical"}
    sen = rd.read_sentinel()
    assert list(sen.columns[:6]) == ["epoch", "feature", "psi", "level", "aggregate", "breaker_state"]
    assert set(sen["feature"]) == set(cfg.psi.features)
    runtime = rd.read_runtime()
    assert runtime["n_decisions"] == res.n_decisions and runtime["sessions"][0]["pid"] == os.getpid()

def test_runner_keeps_an_api_prepared_config(tmp_path):
    cfg, df, fitted = _data()
    from bayesian_credit.data import LendingClubLoader

    data_path = LendingClubLoader.to_parquet(df, tmp_path / "shared" / "dataset.parquet")
    fitted.save(tmp_path / "shared" / "fit")
    sim = SimulationConfig(start="2016-01-01", end="2016-02-29", max_loans=300, policies=("thompson",), interim_months=1)
    rd = RunDirectory.create(tmp_path / "run", cfg, sim, {"path": str(data_path), "simulation_id": 17, "name": "demo"},
                             tmp_path / "shared" / "fit")
    before = rd.read_config()
    SimulationRunner(df, cfg, sim, rd.path, fitted=fitted, dataset_info=before["dataset"], install_signal_handlers=False).run()
    after = rd.read_config()
    assert after["created_at"] == before["created_at"] and after["dataset"]["simulation_id"] == 17
    assert after["dataset"]["path"] == str(data_path) and after["fit_dir"] == str(tmp_path / "shared" / "fit")
    assert not (rd.path / "dataset.parquet").exists() and not (rd.path / "fit").exists()
