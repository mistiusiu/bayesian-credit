"""SimulationRunner: run-dir protocol, leakage guard, yardstick, feedback modes, breaker
review protocol, pause / cancel / SIGTERM / SIGKILL and bit-for-bit resume (SPEC §2.11, G.6)."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayesian_credit.config import FrameworkConfig, SimulationConfig
from bayesian_credit.data import DEFAULT_SHIFT, LendingClubLoader
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS, BorrowerContext, LeakageGuard
from bayesian_credit.exceptions import CheckpointError, LeakageError
from bayesian_credit.monitoring import GovernedPolicy
from bayesian_credit.pipeline import fit_models, prepare_dataset
from bayesian_credit.simulation import (
    DECISION_COLUMNS,
    TIMESERIES_COLUMNS,
    RunDirectory,
    SimulationRunner,
)
from bayesian_credit.types import Stage

POLICIES = ("thompson", "epsilon_greedy", "ucb1", "historical", "random")
CORE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _data():
    cfg = FrameworkConfig()
    df, _ = prepare_dataset("synthetic", None, cfg=cfg, n=5000, seed=3, shift_at="2016-06", shift=dict(DEFAULT_SHIFT))
    fitted = fit_models(df, cfg, "2016-01-01", evaluate=False)
    return cfg, df, fitted


def sim_cfg(**kw) -> SimulationConfig:
    base = {
        "start": "2016-01-01",
        "end": "2016-08-31",
        "max_loans": 1500,
        "seed": 5,
        "policies": POLICIES,
        "interim_months": 3,
        "checkpoint_every": 3,
        "breaker_mode": "auto_ack",
    }
    base.update(kw)
    return SimulationConfig(**base)


def make_runner(path, sim: SimulationConfig | None = None, **kw) -> SimulationRunner:
    cfg, df, fitted = _data()
    kw.setdefault("install_signal_handlers", False)
    kw.setdefault("poll_interval", 0.02)
    return SimulationRunner(df, cfg, sim or sim_cfg(), path, fitted=fitted, **kw)


def assert_same_run(a: Path, b: Path) -> None:
    ra, rb = RunDirectory(a), RunDirectory(b)
    pd.testing.assert_frame_equal(ra.read_decisions(), rb.read_decisions(), check_exact=True)
    pd.testing.assert_frame_equal(ra.read_timeseries(), rb.read_timeseries(), check_exact=True)
    pd.testing.assert_frame_equal(ra.read_sentinel(), rb.read_sentinel(), check_exact=True)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """One completed auto_ack run on the shifted stream; records every context shown to a policy."""
    path = tmp_path_factory.mktemp("finished") / "run"
    runner = make_runner(path)
    keys: set[str] = set()
    count = [0]
    orig = GovernedPolicy.select

    def spy(self, context, eligible, rng, *, historical_arm=None):
        keys.update(context.features.keys())
        count[0] += 1
        return orig(self, context, eligible, rng, historical_arm=historical_arm)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GovernedPolicy, "select", spy)
        result = runner.run()
    return {"path": path, "runner": runner, "result": result, "context_keys": keys, "n_selects": count[0]}


# ----- completion & schema ------------------------------------------------------------------------
def test_run_completes_with_the_exact_decision_columns(finished):
    res, runner = finished["result"], finished["runner"]
    rd = RunDirectory(finished["path"])
    assert res.status == "completed" and res.epochs_completed == res.n_epochs == 8
    dec = rd.read_decisions()
    assert tuple(dec.columns) == DECISION_COLUMNS
    assert len(dec) == len(runner.stream) * len(POLICIES) == res.n_decisions
    assert sorted(dec["epoch"].unique()) == runner.epochs
    assert set(dec["policy"]) == set(POLICIES)
    assert not dec.duplicated(["policy", "loan_id"]).any()
    ts = rd.read_timeseries()
    assert tuple(ts.columns) == TIMESERIES_COLUMNS and len(ts) == 8 * len(POLICIES)
    assert rd.read_progress()["status"] == "completed"
    assert finished["n_selects"] == len(dec)


def test_policies_only_ever_see_pre_decision_columns(finished):
    keys = finished["context_keys"]
    assert keys and keys <= set(PRE_DECISION_COLUMNS)
    LeakageGuard.assert_allowed(keys, Stage.PRE_DECISION)
    runner = finished["runner"]
    LeakageGuard.assert_allowed(runner.ctx_frame.columns, Stage.PRE_DECISION)
    for i in range(len(runner.epochs)):
        for ctx in runner.contexts(i)[:5]:
            assert set(ctx.features) <= set(PRE_DECISION_COLUMNS) and ctx.loan_id is not None
    for col in ("loan_status", "total_rec_prncp", "last_pymnt_d", "y_default", "out_prncp"):
        with pytest.raises(LeakageError):
            BorrowerContext(1, {"grade": "B", col: 1})


def test_historical_policy_is_scored_at_the_hindsight_value_of_its_arm(finished):
    runner = finished["runner"]
    dec = RunDirectory(finished["path"]).read_decisions()
    hist = dec.loc[dec["policy"] == "historical"].set_index("loan_id")
    rows = pd.Series(np.arange(len(runner.stream)), index=runner.stream["loan_id"]).loc[hist.index].to_numpy()
    h = runner.stream["y_historical_arm"].to_numpy()[rows]
    assert (hist["arm"].to_numpy() == h).all() and hist["historical_match"].all()
    expected = runner.hind_score[runner.hind_code[rows], h]
    np.testing.assert_array_equal(hist["expected_reward"].to_numpy(), expected)
    # its realized reward is the loan's own terminal reward (where resolved)
    np.testing.assert_array_equal(hist["realized_reward"].to_numpy(), runner.realized[rows])


def test_regret_is_non_negative_against_the_hindsight_oracle(finished):
    runner = finished["runner"]
    dec = RunDirectory(finished["path"]).read_decisions()
    assert (dec["oracle_reward"] - dec["expected_reward"] >= -1e-6).all()
    one = dec.loc[dec["policy"] == "thompson"]
    rows = pd.Series(np.arange(len(runner.stream)), index=runner.stream["loan_id"]).loc[one["loan_id"]].to_numpy()
    scores = np.where(runner.elig_hist[rows], runner.hind_score[runner.hind_code[rows]], -np.inf)
    np.testing.assert_array_equal(one["oracle_reward"].to_numpy(), scores.max(axis=1))
    lent = ~dec["declined"]
    assert (dec.loc[~lent, "expected_reward"] == 0).all() and (dec.loc[~lent, "amount"] == 0).all()
    assert (dec.loc[dec["realized_reward"].notna(), "historical_match"]).all()


def test_decisions_respect_every_lending_guardrail(finished):
    runner = finished["runner"]
    _, _, fitted = _data()
    stream = runner.stream
    ref = fitted.reference_rates.for_frame(stream)
    np.testing.assert_array_equal(runner.reference_rate, ref)
    np.testing.assert_array_equal(runner.elig, fitted.arm_space.eligible_many(stream["annual_inc"], stream["dti"], ref))
    dec = RunDirectory(finished["path"]).read_decisions()
    rows = pd.Series(np.arange(len(stream)), index=stream["loan_id"]).loc[dec["loan_id"]].to_numpy()
    arms = dec["arm"].to_numpy()
    allowed = np.where(dec["policy"].to_numpy() == "historical", runner.elig_hist[rows, arms], runner.elig[rows, arms])
    assert allowed.all()  # every policy - fallbacks included - chose among the same admissible arms
    lend = ~dec["declined"].to_numpy() & (dec["policy"].to_numpy() != "historical")
    assert (np.abs(dec["rate"].to_numpy()[lend] - ref[rows][lend]) <= fitted.arm_space.rate_corridor + 1e-9).all()


def test_yardstick_and_learning_environment_share_the_prepayment_table(finished):
    runner = finished["runner"]
    _, _, fitted = _data()
    tab = fitted.interest_realization
    assert tab is not None and runner.interest_realization is tab
    assert runner.hindsight.interest_realization is tab
    g = runner.hind_groups[0]
    a60 = fitted.arm_space.nearest(10_000, 0.16, 60)
    assert runner.hindsight.realization(g, a60) == fitted.realization(g, a60) < 1.0
    ck = RunDirectory(finished["path"]).checkpoints.load()
    assert ck["environment"]["interest_realization"] == tab.to_dict()  # the learning env carries it too
    dec = RunDirectory(finished["path"]).read_decisions()
    one = dec.loc[(dec["policy"] == "thompson") & ~dec["declined"]].head(50)
    rows = pd.Series(np.arange(len(runner.stream)), index=runner.stream["loan_id"]).loc[one["loan_id"]].to_numpy()
    for row, arm, exp in zip(rows, one["arm"], one["expected_reward"], strict=True):
        assert exp == runner.hindsight.score(runner.stream["y_group"].iloc[row], int(arm))


def test_hindsight_yardstick_is_fitted_on_elapsed_term_loans(tmp_path, finished):
    from bayesian_credit.exceptions import ConfigurationError
    from bayesian_credit.pipeline import elapsed_term_mask

    runner = finished["runner"]
    t = runner.table
    usable = t["y_resolved"].to_numpy(dtype=bool) & t["y_default"].notna().to_numpy() & (t["y_historical_arm"] >= 0).to_numpy()
    elapsed = elapsed_term_mask(t, runner.snapshot)
    d, n = runner.hindsight.two_stage.default_model.global_counts()
    assert d + n == int((usable & elapsed).sum())
    biased = make_runner(tmp_path / "resolved", sim_cfg(hindsight_sample="resolved"))
    d2, n2 = biased.hindsight.two_stage.default_model.global_counts()
    assert d2 + n2 == int(usable.sum()) > d + n
    assert d2 / (d2 + n2) > d / (d + n)  # recent vintages' resolved loans are mostly early defaults
    np.testing.assert_array_equal(biased.hindsight.mctm.matrix, runner.hindsight.mctm.matrix)  # MCTM: full panel
    with pytest.raises(ConfigurationError):
        SimulationConfig(hindsight_sample="everything").validate()


def test_kill_and_resume_is_exact_with_the_policy_effect_and_corridor(tmp_path):
    from bayesian_credit.pipeline import resilience_check

    cfg, df, _ = _data()
    tilted = FrameworkConfig.from_dict(cfg.to_dict())
    tilted.hierarchy.policy_effect_cfg.min_rows = 300  # small data: let the structured prior switch on
    fitted = fit_models(df, tilted, "2016-01-01", evaluate=False)
    assert fitted.two_stage.policy_effect_info["status"] == "fitted" and fitted.reference_rates is not None
    res = resilience_check(df, fitted, sim_cfg(policies=("thompson", "historical", "random")), tmp_path / "res",
                           n_epochs=5, kill_after=2, checkpoint_every=2, max_loans=700)
    assert res["passed"] and res["decisions_identical"] and res["timeseries_identical"]
    ck = RunDirectory(tmp_path / "res" / "killed").checkpoints.load()
    env_hm = ck["environment"]["two_stage"]["default_model"]
    assert env_hm["leaf_prior_tilt"]["type"] == "PolicyEffectTilt"  # the as-of environment carries its own tilt


def test_breaker_trips_on_the_shift_and_the_frozen_epoch_uses_the_fallback(finished):
    rd = RunDirectory(finished["path"])
    events = rd.events.read()
    trips = [e for e in events if e["type"] == "BreakerTripped"]
    assert trips, "the 2016-06 shift must trip the breaker"
    first = trips[0]["epoch"]
    assert first >= "2016-06"
    dec = rd.read_decisions()
    epochs = sorted(dec["epoch"].unique())
    frozen = dec.loc[dec["epoch"] == first]
    assert frozen["fallback"].all()
    nxt = epochs[epochs.index(first) + 1]
    assert not dec.loc[dec["epoch"] == nxt, "fallback"].any()  # auto_ack after one epoch
    acks = [e for e in events if e["type"] == "ReviewAcknowledged"]
    assert acks[0]["epoch"] == nxt and acks[0]["payload"]["action"] == "resume" and acks[0]["payload"]["auto"]
    assert acks[0]["payload"]["rebased"] is True
    ts = rd.read_timeseries()
    assert set(ts.loc[ts["epoch"] == first, "breaker_state"]) == {"open"}
    assert (ts.loc[ts["epoch"] == first, "fallback_rate"] == 1.0).all()
    review = rd.list_reviews()[0]
    assert review["id"] == trips[0]["payload"]["review_id"] and review["ack"]["action"] == "resume"
    assert review["policies"] == list(POLICIES) and review["epoch"] == first
    sen = rd.read_sentinel()
    assert (sen.loc[sen["epoch"] == first, "breaker_state"] == "open").all()
    assert sen.loc[sen["epoch"] == first, "aggregate"].iloc[0] > 0.25
    # the historical reference keeps its own arm even while the breaker is open
    hist = frozen.loc[frozen["policy"] == "historical"]
    assert hist["historical_match"].all()


def test_simulated_feedback_fills_simulated_reward_and_metrics_add_up(finished):
    rd = RunDirectory(finished["path"])
    dec = rd.read_decisions()
    fb = rd.read_feedback()
    assert set(fb["source"]) == {"simulated"}
    filled = dec["simulated_reward"].notna()
    assert filled.any() and filled.sum() == len(fb)
    assert not dec.loc[filled, "declined"].any()
    ts = rd.read_timeseries()
    for name, grp in ts.groupby("policy"):
        sub = dec.loc[dec["policy"] == name]
        assert grp["n_decisions"].sum() == len(sub)
        np.testing.assert_allclose(grp["cum_reward"].to_numpy(), grp["reward"].cumsum().to_numpy())
        assert grp["cum_regret"].iloc[-1] == pytest.approx((sub["oracle_reward"] - sub["expected_reward"]).sum())
        assert grp["cum_realized"].iloc[-1] == pytest.approx(np.nansum(sub["realized_reward"]))
        assert grp["n_feedback_interim"].sum() > 0
    assert ts.loc[ts["epoch"] == ts["epoch"].min(), "n_feedback_interim"].eq(0).all()  # nothing is due yet


# ----- feedback modes --------------------------------------------------------------------------------
def test_replay_mode_only_updates_on_historical_matches(tmp_path):
    res = make_runner(tmp_path / "run", sim_cfg(feedback_mode="replay")).run()
    rd = RunDirectory(tmp_path / "run")
    assert res.status == "completed"
    fb = rd.read_feedback()
    assert set(fb["source"]) <= {"realized", "skipped_replay"}
    dec = rd.read_decisions().set_index(["policy", "loan_id"])
    match = dec.loc[list(zip(fb["policy"], fb["loan_id"], strict=True)), "historical_match"].to_numpy()
    np.testing.assert_array_equal(fb["source"].to_numpy() == "realized", match)
    assert (fb.loc[fb["source"] == "skipped_replay", "delivered"] == False).all()  # noqa: E712
    assert dec["simulated_reward"].isna().all()
    ts = rd.read_timeseries()
    skipped = ts.groupby("policy")["n_skipped_replay"].sum()
    assert skipped["historical"] == 0 and skipped["thompson"] > 0 and skipped["random"] > 0
    delivered = ts.groupby("policy")[["n_feedback_interim", "n_feedback_terminal"]].sum().sum(axis=1)
    assert delivered["historical"] > delivered["random"]


def test_hybrid_mode_mixes_realized_and_simulated_rewards(tmp_path):
    make_runner(tmp_path / "run", sim_cfg(feedback_mode="hybrid", policies=("thompson", "historical"))).run()
    rd = RunDirectory(tmp_path / "run")
    fb = rd.read_feedback()
    dec = rd.read_decisions().set_index(["policy", "loan_id"])
    match = dec.loc[list(zip(fb["policy"], fb["loan_id"], strict=True)), "historical_match"].to_numpy()
    np.testing.assert_array_equal(fb["source"].to_numpy(), np.where(match, "realized", "simulated"))
    assert fb["delivered"].all()


# ----- governance protocol --------------------------------------------------------------------------------
def _watch(rd: RunDirectory, action, timeout: float = 90.0) -> threading.Thread:
    def loop():
        deadline = time.time() + timeout
        while time.time() < deadline:
            if action():
                return
            time.sleep(0.01)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def test_pause_and_resume_via_control_file(tmp_path, finished):
    path = tmp_path / "run"
    rd = RunDirectory(path)
    state = {"phase": "wait_running", "paused_at": None}

    def driver() -> bool:
        p = rd.read_progress()
        if state["phase"] == "wait_running" and p.get("status") == "running" and (p.get("epoch_index") or -1) >= 1:
            rd.write_control("pause")
            state["phase"] = "wait_paused"
        elif state["phase"] == "wait_paused" and p.get("status") == "paused":
            state["paused_at"] = p["epoch_index"]
            time.sleep(0.1)
            rd.write_control("resume")
            return True
        return False

    t = _watch(rd, driver)
    res = make_runner(path).run()
    t.join(5)
    assert res.status == "completed" and state["paused_at"] is not None
    types = [e["type"] for e in rd.events.read()]
    assert "RunPaused" in types
    paused_at = state["paused_at"]
    assert rd.checkpoints.list()  # a checkpoint was written at the pause boundary
    assert any(e["payload"].get("epoch_index") == paused_at for e in rd.events.read("CheckpointWritten"))
    assert_same_run(finished["path"], path)  # pausing never changes the outcome


def test_cancel_via_control_file_checkpoints_then_resume_completes(tmp_path, finished):
    path = tmp_path / "run"
    rd = RunDirectory(path)

    def driver() -> bool:
        p = rd.read_progress()
        if p.get("status") == "running" and (p.get("epoch_index") or -1) >= 2:
            rd.write_control("cancel")
            return True
        return False

    t = _watch(rd, driver)
    res = make_runner(path).run()
    t.join(5)
    assert res.status == "cancelled" and res.epochs_completed < 8
    prog = rd.read_progress()
    assert prog["status"] == "cancelled" and prog["finished_at"]
    ck = rd.checkpoints.load()
    assert ck["epoch_index"] == prog["epoch_index"] == res.epochs_completed - 1
    cancelled = rd.events.read("RunCancelled")
    assert cancelled[-1]["payload"]["reason"] == "control" and cancelled[-1]["payload"]["checkpoint"]
    rd.write_control("resume")
    done = SimulationRunner.resume(path, install_signal_handlers=False)
    assert done.status == "completed"
    assert_same_run(finished["path"], path)


def test_breaker_pause_mode_waits_for_a_human_review_and_resegments(tmp_path):
    path = tmp_path / "run"
    rd = RunDirectory(path)
    acked: list[tuple[str, str]] = []

    def reviewer() -> bool:
        p = rd.read_progress()
        if p.get("status") == "awaiting_review":
            for r in rd.list_reviews():
                if r.get("ack") is None and r["id"] not in {a for a, _ in acked}:
                    action = "resegment" if not acked else "resume"
                    rd.write_ack(r["id"], action, "reviewed in test")
                    acked.append((r["id"], action))
        return p.get("status") in ("completed", "failed", "cancelled")

    t = _watch(rd, reviewer)
    res = make_runner(path, sim_cfg(breaker_mode="pause")).run()
    t.join(5)
    assert res.status == "completed" and acked and acked[0][1] == "resegment"
    events = rd.events.read()
    paused = [e for e in events if e["type"] == "RunPaused"]
    assert paused and paused[0]["payload"]["reason"] == "awaiting_review"
    reseg = [e for e in events if e["type"] == "ResegmentationApplied"]
    assert reseg and reseg[0]["payload"]["dimensions_after"] == ["grade", "purpose_group", "home_ownership", "fico_band"]
    dec = rd.read_decisions()
    assert not dec["fallback"].any()  # the run waited instead of falling back
    after = dec.loc[(dec["epoch"] > reseg[0]["epoch"]) & (dec["policy"] == "thompson"), "group"]
    assert after.str.count(r"\|").eq(3).all()  # four-dimensional groups after re-segmentation
    ck = rd.checkpoints.load()
    assert ck["segmenter"]["dimensions"][-1] == "fico_band"


def test_breaker_mode_off_only_monitors(tmp_path):
    make_runner(tmp_path / "run", sim_cfg(breaker_mode="off", policies=("thompson", "historical"))).run()
    rd = RunDirectory(tmp_path / "run")
    types = [e["type"] for e in rd.events.read()]
    assert "BreakerTripped" not in types and "ReviewAcknowledged" not in types
    assert not rd.read_decisions()["fallback"].any()
    sen = rd.read_sentinel()
    assert sen["aggregate"].max() > 0.25 and set(sen["breaker_state"]) == {"closed"}


@pytest.mark.parametrize("action", ["resegment", "rollback"])
def test_auto_ack_applies_the_configured_action(tmp_path, action):
    cfg, df, fitted = _data()
    cfg2 = FrameworkConfig.from_dict(cfg.to_dict())
    cfg2.psi.auto_action = action
    sim = sim_cfg(policies=("thompson", "thompson_flat", "historical"))
    res = SimulationRunner(df, cfg2, sim, tmp_path / "run", fitted=fitted, install_signal_handlers=False).run()
    assert res.status == "completed"
    rd = RunDirectory(tmp_path / "run")
    acks = rd.events.read("ReviewAcknowledged")
    assert acks and {a["payload"]["action"] for a in acks} == {action}
    reseg = rd.events.read("ResegmentationApplied")
    assert bool(reseg) == (action == "resegment")
    if action == "resegment":
        assert rd.checkpoints.load()["segmenter"]["dimensions"][3] == "fico_band"


# ----- resume ----------------------------------------------------------------------------------------------
class _Kill(RuntimeError):
    pass


@pytest.mark.parametrize("kill_after", [4, 5])
def test_kill_and_resume_reproduces_the_uninterrupted_run(tmp_path, finished, kill_after):
    path = tmp_path / "killed"

    def killer(info):
        if info["epoch_index"] == kill_after:
            raise _Kill("simulated crash")

    with pytest.raises(_Kill):
        make_runner(path, progress_callback=killer).run()
    rd = RunDirectory(path)
    assert rd.read_progress()["status"] == "failed"
    assert rd.checkpoints.load()["epoch_index"] == (2 if kill_after == 4 else 5)
    assert len(rd.decision_epochs()) == kill_after + 1  # outputs beyond the checkpoint exist ...
    res = SimulationRunner.resume(path, install_signal_handlers=False)
    assert res.status == "completed"
    assert_same_run(finished["path"], path)  # ... and were recomputed identically
    resumed = rd.events.read("RunResumed")
    assert resumed[-1]["payload"]["from_epoch_index"] == (2 if kill_after == 4 else 5)
    again = SimulationRunner.resume(path, install_signal_handlers=False)  # resuming a finished run is a no-op
    assert again.status == "completed"
    assert_same_run(finished["path"], path)


def test_resume_rejects_a_checkpoint_from_another_configuration(tmp_path):
    path = tmp_path / "run"
    make_runner(path, sim_cfg(end="2016-03-31", checkpoint_every=1)).run()
    rd = RunDirectory(path)
    conf = rd.read_config()
    conf["simulation"]["seed"] = 999
    rd.config_path.write_text(json.dumps(conf))
    with pytest.raises(CheckpointError):
        SimulationRunner.resume(path, install_signal_handlers=False)


# ----- real signals (subprocess worker, as launched by the API) ------------------------------------------------
def _prepare_worker_dir(tmp_path: Path, name: str, sim: SimulationConfig) -> RunDirectory:
    cfg, df, fitted = _data()
    shared = tmp_path / "shared"
    if not (shared / "dataset.parquet").exists():
        LendingClubLoader.to_parquet(df, shared / "dataset.parquet")
        fitted.save(shared / "fit")
    return RunDirectory.create(tmp_path / name, cfg, sim, {"path": str(shared / "dataset.parquet")}, shared / "fit")


def _spawn(rd: RunDirectory) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CORE) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "bayesian_credit.simulation.cli", "run", "--run-dir", str(rd.path), "--poll-interval", "0.05"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _wait_for(pred, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_sigterm_checkpoints_and_marks_the_run_cancelled(tmp_path):
    rd = _prepare_worker_dir(tmp_path, "term", sim_cfg(end="2016-04-30"))
    rd.write_control("pause")  # the worker pauses at its first epoch boundary
    proc = _spawn(rd)
    try:
        assert _wait_for(lambda: rd.read_progress().get("status") == "paused"), proc.stderr.read().decode() if proc.poll() is not None else "timeout"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rd.read_progress()["status"] == "cancelled"
    assert rd.events.read("RunCancelled")[-1]["payload"]["reason"] == "signal"


def test_sigkill_mid_run_then_resume_matches_an_uninterrupted_run(tmp_path):
    sim = sim_cfg(start="2015-01-01", end="2016-12-31", max_loans=900, interim_months=2, checkpoint_every=3)
    rd = _prepare_worker_dir(tmp_path, "chaos", sim)
    proc = _spawn(rd)
    try:
        assert _wait_for(lambda: (rd.read_progress().get("epoch_index") or -1) >= 4 or proc.poll() is not None)
        proc.send_signal(signal.SIGKILL)  # the API's "chaos" button
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    killed = rd.read_progress()
    assert proc.returncode == -signal.SIGKILL and killed["status"] == "running"  # really killed mid-run
    assert killed["epoch_index"] < killed["n_epochs"] - 1
    res = SimulationRunner.resume(rd.path, install_signal_handlers=False)
    assert res.status == "completed"
    ref = _prepare_worker_dir(tmp_path, "reference", sim)
    assert SimulationRunner.from_run_dir(ref.path, install_signal_handlers=False).run().status == "completed"
    assert_same_run(ref.path, rd.path)
