"""Event-sourced, resumable replay simulation (SPEC §2.11; proposal §3.6, G.3, G.5, G.6).

Historical Lending Club loans are replayed month by month as a stream of borrowers. In
every **epoch** (issue month ``e``):

(a) **Sentinel** - the PSI Sentinel scores the arrivals of the last
    ``psi_window_months`` epochs (pre-decision features only, checked by
    :class:`~bayesian_credit.data.schema.LeakageGuard`) against the reference window
    (``issue_d < start``). A trip opens a review; ``breaker_mode="auto_ack"`` acknowledges
    it with ``cfg.psi.auto_action`` after ``auto_ack_epochs`` epochs (a visible frozen
    period with fallback decisions), ``"pause"`` stops and waits for
    ``reviews/<id>.ack.json``, ``"off"`` only monitors.
(b) **Feedback** - everything due in the :class:`DelayedFeedbackQueue` is released:
    interim signals (monthly, from the borrower's real repayment path) and terminal
    rewards (at the loan's resolution; never for loans still open - real censoring),
    valued per ``feedback_mode`` and passed to ``policy.update``. Independently, the
    :class:`BayesianEnvironment` absorbs the *realized* outcome of every historical loan
    that resolves in ``e`` (it learns from reality).
(c) **Decisions** - every arriving borrower's pre-decision context is shown to every
    policy (each wrapped in its own :class:`GovernedPolicy`, all sharing one
    :class:`PSISentinel`); decisions are logged and their feedback scheduled.

**Determinism.** Randomness is drawn from generators seeded per epoch:
``default_rng([seed, epoch_index, policy_index])`` for policy ``policy_index``'s
selections and ``default_rng([seed, epoch_index, 7])`` for the environment's draws. A
checkpoint therefore never needs a bit-generator state, and a run killed at any point and
resumed from its last checkpoint reproduces the uninterrupted run bit for bit (tested).

**Yardstick.** ``expected_reward`` of a decision is the *hindsight* environment's
risk-adjusted expected reward of the chosen arm for the loan's (original) group - one fixed
model fitted on all resolved loans, so it is comparable across policies and over time;
``oracle_reward`` is the best admissible arm under the same model and
``regret = oracle − expected ≥ 0``.

See ``docs/SIMULATION.md`` for the full protocol (run directory, control files, resume).
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..bandits.base import POOLED_GROUP, Decision
from ..config import FrameworkConfig, SimulationConfig
from ..data.panel import LoanStatePanel
from ..data.schema import PRE_DECISION_COLUMNS, BorrowerContext, LeakageGuard
from ..data.segmentation import Segmenter
from ..exceptions import CheckpointError, ConfigurationError, GovernanceError
from ..models.mctm import MarkovChainTransitionMatrix
from ..models.two_stage import TwoStageBetaBeta
from ..monitoring.psi import PSICalculator
from ..monitoring.sentinel import GovernedPolicy, PSISentinel, SentinelReading
from ..rewards.compound import CompoundRewardFunction
from ..types import BreakerState, FeedbackKind, FeedbackMode, ReviewAction, Stage, State
from .environment import BayesianEnvironment, month_index_of
from .events import (
    BreakerTripped,
    CheckpointWritten,
    EpochCompleted,
    EpochStarted,
    FeedbackReleased,
    ResegmentationApplied,
    ReviewAcknowledged,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    SentinelEvaluated,
    SimulationEvent,
    utc_now,
)
from .feedback import DelayedFeedbackQueue
from .policies import PolicySpec, build_policies, parse_policy_specs
from .store import RunDirectory, atomic_write_json, config_hash, read_json

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline import FittedModels

__all__ = [
    "ARM_SNAPSHOT_COLUMNS",
    "DECISION_COLUMNS",
    "ENV_RNG_STREAM",
    "SENTINEL_COLUMNS",
    "TIMESERIES_COLUMNS",
    "SimulationResult",
    "SimulationRunner",
    "epoch_of_month",
    "month_of_epoch",
]

#: Columns of ``decisions/epoch=YYYY-MM.parquet`` (``evaluation/policy.py`` log + extras).
DECISION_COLUMNS = (
    "epoch",
    "policy",
    "loan_id",
    "group",
    "arm",
    "amount",
    "rate",
    "term",
    "declined",
    "expected_reward",
    "oracle_reward",
    "realized_reward",
    "fallback",
    "simulated_reward",
    "historical_match",
    "meta_group_new",
)
#: Columns of ``timeseries.parquet`` (one row per epoch x policy).
TIMESERIES_COLUMNS = (
    "epoch",
    "policy",
    "n_decisions",
    "reward",
    "cum_reward",
    "realized",
    "cum_realized",
    "regret",
    "cum_regret",
    "rolling_reward",
    "decline_rate",
    "fallback_rate",
    "mean_amount",
    "mean_rate",
    "n_feedback_interim",
    "n_feedback_terminal",
    "pending_feedback",
    "breaker_state",
    "psi_aggregate",
    "n_groups",
    "epoch_index",
    "sentinel_level",
    "feedback_reward",
    "n_skipped_replay",
    "n_dropped_frozen",
)
#: Columns of ``arms/epoch=YYYY-MM.parquet``.
ARM_SNAPSHOT_COLUMNS = ("epoch", "policy", "group", "arm", "mu", "sigma", "n", "selections")
#: Columns of ``sentinel.parquet``.
SENTINEL_COLUMNS = ("epoch", "feature", "psi", "level", "aggregate", "breaker_state", "n_current", "skipped")
#: Third seed component of the environment's per-epoch generator.
ENV_RNG_STREAM = 7
#: Rolling-reward window (epochs).
ROLLING_EPOCHS = 3
#: Re-derive the environment's Stage-1 child priors every this many epochs.
REFRESH_EVERY = 12
#: Groups (by arrivals so far) whose posteriors are snapshotted, besides the pooled level.
SNAPSHOT_TOP_GROUPS = 10
CHECKPOINT_FORMAT = 1

_INTERIM = FeedbackKind.INTERIM
_TERMINAL = FeedbackKind.TERMINAL
_TRANSIENT = (int(State.CURRENT), int(State.DELINQUENT))


def month_of_epoch(epoch: str) -> int:
    """``'YYYY-MM'`` -> calendar month index ``year·12 + month − 1``."""
    y, m = str(epoch)[:7].split("-")
    return int(y) * 12 + int(m) - 1


def epoch_of_month(mi: int) -> str:
    y, m = divmod(int(mi), 12)
    return f"{y:04d}-{m + 1:02d}"


class _PhaseClock:
    """Accumulates wall-clock seconds per phase into a dict (diagnostics only)."""

    __slots__ = ("store", "t")

    def __init__(self, store: dict[str, float]):
        self.store = store
        self.t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.store[name] = self.store.get(name, 0.0) + (now - self.t)
        self.t = now


class _Cancelled(Exception):  # noqa: N818 - control-flow signal, not an error
    def __init__(self, reason: str, checkpoint: bool):
        super().__init__(reason)
        self.reason = reason
        self.checkpoint = checkpoint


@dataclass
class SimulationResult:
    """Outcome of :meth:`SimulationRunner.run` / :meth:`SimulationRunner.resume`."""

    status: str
    run_dir: Path
    epochs_completed: int
    n_epochs: int
    n_decisions: int
    wall_seconds: float
    last_epoch: str | None

    def decisions(self) -> pd.DataFrame:
        return RunDirectory(self.run_dir).read_decisions()

    def timeseries(self) -> pd.DataFrame:
        return RunDirectory(self.run_dir).read_timeseries()


class SimulationRunner:
    """Replay-simulation driver: run / resume / pause / cancel / breaker review protocol.

    Parameters
    ----------
    table:
        LoanTable with ``y_*`` outcome columns (derived when missing).
    cfg, sim_cfg:
        :class:`FrameworkConfig` (bandit / PSI / reward settings) and
        :class:`SimulationConfig` (window, policies, feedback & breaker modes, cadence).
    run_dir:
        Run directory (see :class:`RunDirectory`); created if needed.
    fitted:
        Reference :class:`~bayesian_credit.pipeline.FittedModels` (arm space, segmenter,
        MCTM); fitted on ``issue_d < cfg.train_end`` when omitted.
    progress_callback:
        Called after every completed epoch with a dict (``epoch_index, epoch, n_epochs,
        pct, n_decisions, breaker_state``). An exception raised by the callback aborts the
        run like a crash (used by the kill-and-resume test).
    dataset_path, fit_dir:
        Where the dataset parquet / saved fit live (stored in ``config.json`` for
        :meth:`resume`). Without them the table and the fit are written into the run dir.
    poll_interval:
        Seconds between polls of ``control.json`` / review acks while paused.
    install_signal_handlers:
        SIGTERM → checkpoint + status ``cancelled`` at the next epoch boundary.
    min_window_size:
        PSI windows with fewer arrivals are recorded but do not step the breaker.
    rebase_on_ack:
        Review actions after which the Sentinel's reference is re-based on the window that
        tripped it (default: all three - see docs/SIMULATION.md).
    """

    def __init__(
        self,
        table: pd.DataFrame,
        cfg: FrameworkConfig,
        sim_cfg: SimulationConfig,
        run_dir: str | os.PathLike[str] | RunDirectory,
        fitted: FittedModels | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        dataset_path: str | os.PathLike[str] | None = None,
        dataset_info: Mapping[str, Any] | None = None,
        fit_dir: str | os.PathLike[str] | None = None,
        poll_interval: float = 1.0,
        install_signal_handlers: bool = True,
        write_config: bool = True,
        min_window_size: int = 50,
        rebase_on_ack: tuple[str, ...] = ("resume", "resegment", "rollback"),
    ):
        from ..pipeline import ensure_outcomes, fit_models

        cfg.validate()
        sim_cfg.validate()
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.run_dir = run_dir if isinstance(run_dir, RunDirectory) else RunDirectory(run_dir)
        self.run_dir.ensure()
        self.progress_callback = progress_callback
        self.poll_interval = float(poll_interval)
        self.install_signal_handlers = bool(install_signal_handlers)
        self.min_window_size = int(min_window_size)
        self.rebase_on_ack = tuple(str(a) for a in rebase_on_ack)
        self.specs: list[PolicySpec] = parse_policy_specs(sim_cfg.policies)
        self.policy_names = [s.label for s in self.specs]
        LeakageGuard.assert_allowed(cfg.psi.features, Stage.PRE_DECISION)

        if fitted is None:
            fitted = fit_models(table, cfg, cfg.train_end, evaluate=False)
        self.fitted = fitted
        self.arm_space = fitted.arm_space
        self.segmenter0: Segmenter = fitted.segmenter
        self.mctm = fitted.mctm
        self.reward_fn = CompoundRewardFunction(cfg.reward)
        self.table = ensure_outcomes(table, cfg, arm_space=self.arm_space, segmenter=self.segmenter0)

        self.config_hash = config_hash(cfg.to_dict(), sim_cfg.to_dict())
        if write_config:
            self._ensure_config(dataset_path, dataset_info, fit_dir)

        self._signal_cancel = False
        self._old_sigterm: Any = None
        self._session_t0 = time.perf_counter()
        self._session_cpu0 = time.process_time()
        self._session_started = utc_now()
        self._last_checkpoint_index: int | None = None
        self.phase_seconds: dict[str, float] = {}
        t_prep = time.perf_counter()
        self._prepare_static()
        self.phase_seconds["prepare"] = time.perf_counter() - t_prep

    def _ensure_config(
        self,
        dataset_path: str | os.PathLike[str] | None,
        dataset_info: Mapping[str, Any] | None,
        fit_dir: str | os.PathLike[str] | None,
    ) -> None:
        """Make ``config.json`` describe this run so :meth:`resume` can rebuild it.

        A ``config.json`` already holding the same (framework, simulation) configuration -
        e.g. prepared by the API with :meth:`RunDirectory.create` - is kept as is (extra keys
        preserved); only a missing dataset path / fit directory is filled in. Otherwise a
        fresh file is written. Without a usable dataset parquet / saved fit, the in-memory
        table / models are written into the run directory.
        """
        existing = read_json(self.run_dir.config_path)
        same = isinstance(existing, dict) and config_hash(
            existing.get("framework") or {}, existing.get("simulation") or {}
        ) == self.config_hash
        prev_ds = dict((existing or {}).get("dataset") or {}) if same else {}
        if dataset_path is None and prev_ds.get("path") and Path(prev_ds["path"]).is_file():
            dataset_path = prev_ds["path"]
        if dataset_path is None:
            from ..data.loader import LendingClubLoader

            dataset_path = LendingClubLoader.to_parquet(self.table, self.run_dir.path / "dataset.parquet")
        prev_fit = (existing or {}).get("fit_dir") if same else None
        if fit_dir is None and prev_fit and (Path(prev_fit) / "two_stage.json").is_file():
            fit_dir = prev_fit
        if fit_dir is None:
            fit_dir = self.fitted.save(self.run_dir.path / "fit")
        if same:
            doc = dict(existing)
            ds = {**dict(dataset_info or {}), **prev_ds, "path": str(dataset_path)}
            ds.setdefault("n", int(len(self.table)))
            if ds != prev_ds or doc.get("fit_dir") != str(fit_dir):
                doc["dataset"] = ds
                doc["fit_dir"] = str(fit_dir)
                atomic_write_json(self.run_dir.config_path, doc)
            return
        info = dict(dataset_info or {})
        info["path"] = str(dataset_path)
        info.setdefault("n", int(len(self.table)))
        self.run_dir.write_config(self.cfg, self.sim_cfg, info, fit_dir)

    # ==================================================================================
    # construction from a run directory
    # ==================================================================================
    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | os.PathLike[str] | RunDirectory,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> SimulationRunner:
        """Build a runner from ``config.json`` (dataset parquet + optional ``fit_dir``)."""
        from ..pipeline import FittedModels, load_dataset

        rd = run_dir if isinstance(run_dir, RunDirectory) else RunDirectory(run_dir)
        conf = rd.read_config()
        framework = FrameworkConfig.from_dict(conf.get("framework"))
        sim = SimulationConfig.from_dict(conf.get("simulation"))
        dataset = conf.get("dataset") or {}
        path = dataset.get("path")
        if not path:
            raise CheckpointError(f"{rd.config_path} has no dataset.path")
        df = load_dataset(path)
        fit_dir = conf.get("fit_dir")
        fitted = FittedModels.load(fit_dir) if fit_dir else None
        return cls(
            df,
            framework,
            sim,
            rd,
            fitted=fitted,
            progress_callback=progress_callback,
            dataset_info=dataset,
            fit_dir=fit_dir,
            write_config=False,
            **kwargs,
        )

    @classmethod
    def resume(
        cls,
        run_dir: str | os.PathLike[str] | RunDirectory,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """Continue a run from its latest checkpoint (fresh start if there is none)."""
        return cls.from_run_dir(run_dir, progress_callback, **kwargs).run(resume=True)

    # ==================================================================================
    # static preparation (pure functions of data + config; recomputed on resume)
    # ==================================================================================
    def _prepare_static(self) -> None:
        cfg, sim = self.cfg, self.sim_cfg
        t = self.table
        k = self.arm_space.n_arms
        self.snapshot = pd.Timestamp(cfg.data.snapshot_date)
        start, end = pd.Timestamp(sim.start), pd.Timestamp(sim.end)
        self.start_mi = start.year * 12 + start.month - 1
        self.end_mi = end.year * 12 + end.month - 1
        issue = t["issue_d"]

        # ---- stream: start <= issue_d <= end, seeded subsample, (issue_d, loan_id) order ----
        stream_idx = np.flatnonzero(((issue >= start) & (issue <= end)).to_numpy())
        if sim.max_loans is not None and stream_idx.size > int(sim.max_loans):
            pick = np.random.default_rng(sim.seed).choice(stream_idx.size, size=int(sim.max_loans), replace=False)
            stream_idx = np.sort(stream_idx[pick])
        if stream_idx.size == 0:
            raise ConfigurationError(f"no loans issued between {sim.start} and {sim.end}")
        stream = t.iloc[stream_idx].sort_values(["issue_d", "loan_id"], kind="stable").reset_index(drop=True)
        self.stream = stream
        n = len(stream)
        issue_mi = month_index_of(stream["issue_d"]).astype(np.int64)
        self.issue_mi = issue_mi
        months = np.unique(issue_mi)
        self.epoch_mi = months
        self.epochs = [epoch_of_month(m) for m in months]
        los = np.searchsorted(issue_mi, months, side="left")
        his = np.searchsorted(issue_mi, months, side="right")
        self.bounds = list(zip(los.tolist(), his.tolist(), strict=True))
        self._mi_base = int(min(self.start_mi, months.min()))
        top = int(max(self.end_mi, months.max())) + 2
        self._mi_to_epoch = np.array([epoch_of_month(m) for m in range(self._mi_base, top + 1)], dtype=object)

        # ---- contexts (pre-decision only) and PSI windows ----
        self.ctx_columns = [c for c in PRE_DECISION_COLUMNS if c in stream.columns]
        self.ctx_frame = stream.loc[:, ["loan_id", *self.ctx_columns]]
        LeakageGuard.assert_allowed(self.ctx_frame.columns, Stage.PRE_DECISION)
        self.psi_features = [f for f in cfg.psi.features if f in t.columns]
        if not self.psi_features:
            raise ConfigurationError(f"none of the PSI features {cfg.psi.features} is in the data")
        self.stream_psi = stream.loc[:, self.psi_features]
        ref = t.loc[(issue < start).to_numpy(), self.psi_features]
        if len(ref) == 0:
            raise ConfigurationError(f"empty PSI reference window (no loans issued before {sim.start})")
        self.reference = ref

        # ---- per-row decision inputs: lending guardrails (pre-decision income, DTI and the
        # reference rate of the sub-grade; no rate corridor for fits without reference rates) ----
        self.loan_ids = stream["loan_id"].to_numpy(dtype=np.int64)
        hist = stream["y_historical_arm"].to_numpy(dtype=np.int64)
        self.hist = hist
        self.valid_hist = (hist >= 0) & (hist < k)
        self.reference_rates = getattr(self.fitted, "reference_rates", None)
        self.reference_rate = (
            self.reference_rates.for_frame(stream) if self.reference_rates is not None else np.full(n, np.nan)
        )
        self.elig = self.arm_space.eligible_many(
            stream["annual_inc"] if "annual_inc" in stream.columns else np.full(n, np.nan),
            stream["dti"] if "dti" in stream.columns else None,
            self.reference_rate if self.reference_rates is not None else None,
        )
        self.elig_hist = self.elig.copy()
        vh = np.flatnonzero(self.valid_hist)
        self.elig_hist[vh, hist[vh]] = True
        self.realized = self.reward_fn.terminal_many(stream)

        # ---- hindsight yardstick (original segmentation; the same prepayment table as the
        # learning environment, so both use the same reward formula). Its risk model uses the
        # resolved loans whose term elapsed by the snapshot (hindsight_sample="elapsed"), its
        # MCTM the full panel ----
        self.interest_realization = getattr(self.fitted, "interest_realization", None)
        hind_rows = t
        if sim.hindsight_sample == "elapsed":
            from ..pipeline import elapsed_term_mask

            hind_rows = t.loc[elapsed_term_mask(t, self.snapshot)]
        hind_panel = LoanStatePanel.from_table(t, cfg.mctm, self.snapshot)
        hind_mctm = MarkovChainTransitionMatrix.from_config(cfg.mctm).fit_counts(hind_panel.counts)
        self.hindsight = BayesianEnvironment.fit(
            hind_rows, cfg, arm_space=self.arm_space, segmenter=self.segmenter0, reward_fn=self.reward_fn,
            mctm=hind_mctm, snapshot_date=self.snapshot, interest_realization=self.interest_realization,
        )
        codes, uniq = pd.factorize(stream["y_group"].astype(str))
        self.hind_code = codes.astype(np.int64)
        self.hind_groups = list(uniq)
        _, _, self.hind_score = self.hindsight.expected_table(self.hind_groups)
        oracle = np.empty(n)
        for lo in range(0, n, 25_000):
            hi = min(lo + 25_000, n)
            s = self.hind_score[self.hind_code[lo:hi]]
            oracle[lo:hi] = np.max(np.where(self.elig_hist[lo:hi], s, -np.inf), axis=1)
        self.oracle = oracle

        # ---- feedback timing: observation month = max(resolution month, issue month + 1) ----
        res_mi = month_index_of(stream["y_resolution_date"])
        resolved = stream["y_resolved"].to_numpy(dtype=bool) & np.isfinite(res_mi)
        obs_mi = np.where(resolved, np.maximum(np.nan_to_num(res_mi, nan=0.0), issue_mi + 1), np.inf)
        self.term_rel = np.where(obs_mi <= self.end_mi, obs_mi, -1).astype(np.int64)
        max_w = max(
            (sp.bandit_config(cfg.bandit).interim_weight for sp in self.specs), default=0.0
        )
        self.schedule_interim = sim.interim_months > 0 and max_w > 0
        if self.schedule_interim:
            m_count = int(sim.interim_months)
            panel = LoanStatePanel.from_table(stream, cfg.mctm, self.snapshot)
            states = np.empty((n, m_count), dtype=np.int8)
            paid = np.empty((n, m_count), dtype=np.int16)
            for m in range(1, m_count + 1):
                s, p = panel.interim_states(stream, m)
                states[:, m - 1] = s
                paid[:, m - 1] = p
            transient = np.isin(states, _TRANSIENT)
            alive = np.cumprod(transient, axis=1).astype(bool)  # stop after an absorbing state
            rel = issue_mi[:, None] + np.arange(1, m_count + 1)[None, :]
            ok = alive & (rel <= self.end_mi) & (rel < obs_mi[:, None])
            rows_i, cols_m = np.nonzero(ok)
            self.plan_row = rows_i.astype(np.int64)
            self.plan_month = (cols_m + 1).astype(np.int16)
            self.plan_state = states[rows_i, cols_m]
            self.plan_paid = paid[rows_i, cols_m]
            self.plan_rel = rel[rows_i, cols_m].astype(np.int64)
            self.plan_bounds = np.searchsorted(self.plan_row, np.arange(n + 1))
        else:
            self.plan_row = np.empty(0, dtype=np.int64)
            self.plan_bounds = np.zeros(n + 1, dtype=np.int64)

        # ---- environment learning schedule over the whole table ----
        t_issue = month_index_of(issue)
        t_res = month_index_of(t["y_resolution_date"])
        t_def = t["y_default"].to_numpy(dtype=float, na_value=np.nan)
        t_ok = t["y_resolved"].to_numpy(dtype=bool) & np.isfinite(t_res) & np.isfinite(t_def)
        t_obs = np.where(t_ok, np.maximum(np.nan_to_num(t_res, nan=0.0), t_issue + 1), np.inf)
        self.asof_rows = np.flatnonzero(t_ok & (t_obs < self.start_mi))
        during = np.flatnonzero(t_ok & (t_obs >= self.start_mi) & (t_obs <= self.end_mi))
        order = np.argsort(t_obs[during], kind="stable")
        self.obs_rows = during[order]
        self.obs_mi_sorted = t_obs[self.obs_rows].astype(np.int64)
        self.obs_arm = t["y_historical_arm"].to_numpy(dtype=np.int64)[self.obs_rows]
        self.obs_default = t_def[self.obs_rows]
        self.obs_recovery = t["y_recovery_rate"].to_numpy(dtype=float, na_value=np.nan)[self.obs_rows]
        self._y_group_all = t["y_group"].astype(str).to_numpy(dtype=object)
        self._y_group_stream = stream["y_group"].astype(str).to_numpy(dtype=object)

    def _epoch_strings(self, months: np.ndarray) -> np.ndarray:
        return self._mi_to_epoch[np.asarray(months, dtype=np.int64) - self._mi_base]

    # ==================================================================================
    # mutable state
    # ==================================================================================
    def _set_env_groups(self) -> None:
        """Group keys of the stream / observed loans under the environment's segmentation."""
        seg = self.env.segmenter
        if seg is None or seg == self.segmenter0:
            self._stream_env_groups = self._y_group_stream
            self._obs_groups = self._y_group_all[self.obs_rows]
        else:
            self._stream_env_groups = seg.assign(self.stream).astype(str).to_numpy(dtype=object)
            self._obs_groups = seg.assign(self.table.iloc[self.obs_rows]).astype(str).to_numpy(dtype=object)

    def _new_sentinel(self) -> PSISentinel:
        return PSISentinel.from_reference(
            self.reference,
            self.cfg.psi,
            monitor_only=self.sim_cfg.breaker_mode == "off",
            min_window_size=self.min_window_size,
        )

    def _wrap(self, policies: Mapping[str, Any]) -> dict[str, GovernedPolicy]:
        out = {}
        for spec in self.specs:
            fallback = "historical" if spec.is_historical else None
            out[spec.label] = GovernedPolicy(policies[spec.label], self.sentinel, self.arm_space, fallback=fallback)
        return out

    def _init_state(self) -> None:
        cfg = self.cfg
        asof = self.table.iloc[self.asof_rows]
        # as of the stream start: only outcomes observed before it; the policy effect's
        # "elapsed-term" loans are those whose term had elapsed by the start
        two_stage = TwoStageBetaBeta.from_config(cfg, self.arm_space).fit(asof, snapshot_date=self.sim_cfg.start)
        self.env = BayesianEnvironment(
            two_stage, self.mctm, self.reward_fn, self.arm_space, self.segmenter0,
            interest_realization=self.interest_realization,
        )
        self.segmenter = self.segmenter0
        self._set_env_groups()
        self.sentinel = self._new_sentinel()
        policies = build_policies(self.specs, self.arm_space, self.segmenter, cfg.bandit, self.env.prior_provider)
        self.governed = self._wrap(policies)
        self.queue = DelayedFeedbackQueue()
        self.ms: dict[str, Any] = {
            "cum": {name: {"n": 0, "reward": 0.0, "realized": 0.0, "regret": 0.0, "feedback_reward": 0.0} for name in self.policy_names},
            "group_arrivals": {name: {} for name in self.policy_names},
            "timeseries": [],
            "sentinel_rows": [],
            "pending_acks": [],
            "obs_pointer": 0,
            "n_decisions": 0,
            "rebased_at": [],
            "resegmentations": [],
            "acks": [],
            "checkpoints_written": 0,
            "restarts": 0,
        }

    def _checkpoint_state(self, epoch_index: int) -> dict[str, Any]:
        policies, governed = {}, {}
        for name, gov in self.governed.items():
            st = gov.state()
            policies[name] = st.pop("policy")
            governed[name] = st
        return {
            "format": CHECKPOINT_FORMAT,
            "epoch_index": int(epoch_index),
            "epoch": self.epochs[epoch_index],
            "policies": policies,
            "governed": governed,
            "sentinel": self.sentinel.to_dict(),
            "environment": self.env.snapshot(),
            "queue": self.queue.to_dict(),
            "segmenter": self.segmenter.to_dict(),
            "metrics_state": self._copy_ms(),
            "config_hash": self.config_hash,
            "created_at": utc_now(),
        }

    def _copy_ms(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self.ms)

    def _restore(self, ck: Mapping[str, Any]) -> None:
        if ck.get("config_hash") != self.config_hash:
            raise CheckpointError(
                f"checkpoint config hash {ck.get('config_hash')} does not match the run config {self.config_hash}"
            )
        if int(ck.get("format", 1)) != CHECKPOINT_FORMAT:
            raise CheckpointError(f"unsupported checkpoint format {ck.get('format')}")
        self.sentinel = PSISentinel.from_dict(ck["sentinel"])
        self.segmenter = Segmenter.from_dict(ck["segmenter"])
        self.env = BayesianEnvironment.from_snapshot(ck["environment"], self.reward_fn, self.arm_space)
        self._set_env_groups()
        self.governed = {}
        for spec in self.specs:
            st = dict(ck["governed"][spec.label])
            st["policy"] = ck["policies"][spec.label]
            self.governed[spec.label] = GovernedPolicy.from_state(
                st, self.arm_space, self.segmenter, self.env.prior_provider, sentinel=self.sentinel
            )
        self.queue = DelayedFeedbackQueue.from_dict(ck["queue"])
        self.ms = dict(ck["metrics_state"])

    # ==================================================================================
    # public driver
    # ==================================================================================
    def run(self, resume: bool = False) -> SimulationResult:
        """Run (fresh) or continue (``resume=True``) the simulation.

        Returns when the run completes, is cancelled (control file or SIGTERM) or - for
        ``breaker_mode="pause"`` / a ``pause`` command - after it has been resumed and
        finished. Exceptions mark the run ``failed`` and are re-raised.
        """
        self._session_t0 = time.perf_counter()
        self._session_cpu0 = time.process_time()
        self._session_started = utc_now()
        self._install_signals()
        start_index = 0
        try:
            if resume:
                start_index = self._resume_state()
            else:
                self.run_dir.clear_outputs()
                self._init_state()
                self._emit(
                    RunStarted(
                        payload={
                            "n_epochs": len(self.epochs),
                            "first_epoch": self.epochs[0],
                            "last_epoch": self.epochs[-1],
                            "n_loans": int(len(self.stream)),
                            "policies": self.policy_names,
                            "feedback_mode": FeedbackMode(self.sim_cfg.feedback_mode).value,
                            "breaker_mode": self.sim_cfg.breaker_mode,
                            "pid": os.getpid(),
                        }
                    )
                )
            self._progress("running", start_index - 1, message="starting" if start_index == 0 else "resuming")
            return self._loop(start_index)
        except _Cancelled as c:
            return self._finish_cancel(c)
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            self._restore_signals()

    def pause(self) -> None:
        """Ask the running loop to pause at the next epoch boundary (writes control.json)."""
        self.run_dir.write_control("pause")

    def cancel(self) -> None:
        """Ask the running loop to checkpoint and stop at the next epoch boundary."""
        self.run_dir.write_control("cancel")

    def resume_paused(self) -> None:
        self.run_dir.write_control("resume")

    # ==================================================================================
    # loop
    # ==================================================================================
    def _resume_state(self) -> int:
        ck = self.run_dir.checkpoints.load()
        if ck is None:
            self.run_dir.truncate_after(None)
            for p in (self.run_dir.timeseries_path, self.run_dir.sentinel_path, self.run_dir.report_path):
                if p.exists():
                    p.unlink()
            self._init_state()
            self.ms["restarts"] = 1
            self._emit(RunResumed(payload={"from_epoch_index": -1, "from_epoch": None, "checkpoint": None,
                                           "n_epochs": len(self.epochs), "pid": os.getpid()}))
            return 0
        self._restore(ck)
        idx = int(ck["epoch_index"])
        epoch = str(ck["epoch"])
        if idx >= len(self.epochs) or self.epochs[idx] != epoch:
            raise CheckpointError(f"checkpoint epoch {epoch} (index {idx}) does not match this run's epochs")
        self.run_dir.truncate_after(epoch)
        self.ms["timeseries"] = [r for r in self.ms["timeseries"] if r["epoch"] <= epoch]
        self.ms["sentinel_rows"] = [r for r in self.ms["sentinel_rows"] if r["epoch"] <= epoch]
        self.ms["restarts"] = int(self.ms.get("restarts", 0)) + 1
        self._write_tables()
        if self.run_dir.report_path.exists() and idx < len(self.epochs) - 1:
            self.run_dir.report_path.unlink()
        self._last_checkpoint_index = idx
        self._emit(
            RunResumed(
                epoch=epoch,
                payload={
                    "from_epoch_index": idx,
                    "from_epoch": epoch,
                    "checkpoint": str(self.run_dir.checkpoints.latest_path),
                    "n_epochs": len(self.epochs),
                    "pid": os.getpid(),
                },
            )
        )
        return idx + 1

    def _loop(self, start_index: int) -> SimulationResult:
        n_epochs = len(self.epochs)
        self._boundary(start_index - 1)
        for i in range(start_index, n_epochs):
            self._run_epoch(i)
            self._boundary(i)
        return self._finish_completed()

    def _boundary(self, last_index: int) -> None:
        """Epoch boundary: honour SIGTERM and control.json (pause / cancel)."""
        if self._signal_cancel:
            raise _Cancelled("signal", checkpoint=last_index >= 0)
        ctl = self.run_dir.read_control()
        command = ctl.get("command") if ctl else None
        if command == "cancel":
            raise _Cancelled("control", checkpoint=last_index >= 0)
        if command != "pause":
            return
        if last_index >= 0 and self._last_checkpoint_index != last_index:
            self._write_checkpoint(last_index)
        self._progress("paused", last_index, message="paused by control.json")
        self._emit(RunPaused(epoch=self.epochs[last_index] if last_index >= 0 else None, payload={"reason": "control", "review_id": None}))
        while True:
            time.sleep(self.poll_interval)
            if self._signal_cancel:
                raise _Cancelled("signal", checkpoint=last_index >= 0)
            ctl = self.run_dir.read_control()
            command = ctl.get("command") if ctl else None
            if command == "cancel":
                raise _Cancelled("control", checkpoint=last_index >= 0)
            if command != "pause":
                break
        self._progress("running", last_index, message="resumed")

    # ==================================================================================
    # one epoch
    # ==================================================================================
    def _run_epoch(self, i: int) -> None:
        e = self.epochs[i]
        lo, hi = self.bounds[i]
        self._emit(EpochStarted(epoch=e, payload={"epoch_index": i, "n_arrivals": hi - lo}))

        clock = _PhaseClock(self.phase_seconds)
        # (a) sentinel ----------------------------------------------------------------
        self._process_auto_acks(i)
        reading = self._evaluate_sentinel(i)
        if reading.tripped:
            self._on_trip(i, reading)
        breaker_state = self.sentinel.state.value
        clock.lap("sentinel")

        # (b) feedback + environment learning ------------------------------------------
        fb = self._release_feedback(i)
        clock.lap("feedback")
        n_observed = self._observe_outcomes(i)
        if (i + 1) % REFRESH_EVERY == 0:
            self.env.refresh()
        clock.lap("environment")
        self._emit(
            FeedbackReleased(
                epoch=e,
                payload={
                    "n_interim": int(sum(v["n_interim"] for v in fb.values())),
                    "n_terminal": int(sum(v["n_terminal"] for v in fb.values())),
                    "n_skipped_replay": int(sum(v["n_skipped_replay"] for v in fb.values())),
                    "n_dropped_frozen": int(sum(v["n_dropped_frozen"] for v in fb.values())),
                    "n_observed_outcomes": int(n_observed),
                    "pending": int(len(self.queue)),
                    "by_policy": fb,
                },
            )
        )

        # (c) decisions -------------------------------------------------------------------
        dec = self._decide(i)
        clock.lap("decide")
        self._schedule_feedback(i, dec)
        clock.lap("schedule")
        frame = self._decision_frame(i, dec)
        self.run_dir.write_decisions(e, frame)

        # metrics / outputs ---------------------------------------------------------------
        per_policy = self._update_metrics(i, frame, fb, breaker_state, reading)
        if (i + 1) % max(int(self.sim_cfg.posterior_snapshot_every), 1) == 0 or i == len(self.epochs) - 1:
            self._snapshot_arms(i)
        self._write_tables()
        clock.lap("outputs")
        if (i + 1) % int(self.sim_cfg.checkpoint_every) == 0 or i == len(self.epochs) - 1:
            self._write_checkpoint(i)
        clock.lap("checkpoint")
        n_dec = int(len(frame))
        self._emit(
            EpochCompleted(
                epoch=e,
                payload={
                    "epoch_index": i,
                    "n_arrivals": hi - lo,
                    "n_decisions": n_dec,
                    "breaker_state": breaker_state,
                    "psi_aggregate": reading.psi_aggregate,
                    "pending_feedback": int(len(self.queue)),
                    "per_policy": per_policy,
                },
            )
        )
        self._progress("running", i, message=f"epoch {e} ({i + 1}/{len(self.epochs)})")
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "epoch_index": i,
                    "epoch": e,
                    "n_epochs": len(self.epochs),
                    "pct": 100.0 * (i + 1) / len(self.epochs),
                    "n_decisions": n_dec,
                    "breaker_state": self.sentinel.state.value,
                }
            )

    # ----- (a) sentinel ---------------------------------------------------------------------
    def _window(self, i: int) -> pd.DataFrame:
        w = max(int(self.sim_cfg.psi_window_months), 1)
        lo = self.bounds[max(0, i - w + 1)][0]
        hi = self.bounds[i][1]
        return self.stream_psi.iloc[lo:hi]

    def _evaluate_sentinel(self, i: int) -> SentinelReading:
        e = self.epochs[i]
        window = self._window(i)
        LeakageGuard.assert_allowed(window.columns, Stage.PRE_DECISION)
        reading = self.sentinel.evaluate(window, e)
        self._emit(SentinelEvaluated(epoch=e, payload=reading.to_dict()))
        for f, v in reading.per_feature.items():
            self.ms["sentinel_rows"].append(
                {
                    "epoch": e,
                    "feature": f,
                    "psi": float(v),
                    "level": self.sentinel.classify(v).value,
                    "aggregate": float(reading.psi_aggregate),
                    "breaker_state": reading.breaker_after.value,
                    "n_current": int(reading.n_current),
                    "skipped": bool(reading.skipped),
                }
            )
        return reading

    def _on_trip(self, i: int, reading: SentinelReading) -> None:
        e = self.epochs[i]
        review = self.sentinel.review(str(reading.review_id))
        mode = self.sim_cfg.breaker_mode
        due = None
        if mode == "auto_ack":
            due = i + max(int(self.cfg.psi.auto_ack_epochs), 0)
        record = {
            **review.to_dict(),
            "epoch_index": i,
            "policies": self.policy_names,
            "breaker_mode": mode,
            "breaker_state": self.sentinel.state.value,
            "segmentation": list(self.segmenter.dimensions),
            "auto": mode == "auto_ack",
            "auto_action": self.cfg.psi.auto_action if mode == "auto_ack" else None,
            "auto_ack_epoch": None if due is None or due >= len(self.epochs) else self.epochs[due],
            "created_at": utc_now(),
        }
        self.run_dir.write_review(record)
        self._emit(
            BreakerTripped(
                epoch=e,
                payload={
                    "review_id": review.id,
                    "psi_aggregate": reading.psi_aggregate,
                    "per_feature": reading.per_feature,
                    "breaker_mode": mode,
                    "auto_ack_epoch": record["auto_ack_epoch"],
                },
            )
        )
        if mode == "auto_ack":
            if due is not None and due <= i:
                self._acknowledge(i, review.id, self.cfg.psi.auto_action, "auto_ack", auto=True)
            else:
                self.ms["pending_acks"].append({"review_id": review.id, "due_index": int(due)})
        elif mode == "pause":
            self._await_review(i, review.id)

    def _process_auto_acks(self, i: int) -> None:
        if not self.ms["pending_acks"]:
            return
        keep = []
        for item in self.ms["pending_acks"]:
            if int(item["due_index"]) <= i:
                rid = str(item["review_id"])
                if self.sentinel.review(rid).status == "pending":
                    self._acknowledge(i, rid, self.cfg.psi.auto_action, "auto_ack", auto=True)
            else:
                keep.append(item)
        self.ms["pending_acks"] = keep

    def _await_review(self, i: int, review_id: str) -> None:
        e = self.epochs[i]
        self._progress("awaiting_review", i - 1, message=f"breaker tripped at {e}: waiting for {review_id}", current_epoch=e)
        self._emit(RunPaused(epoch=e, payload={"reason": "awaiting_review", "review_id": review_id}))
        while True:
            ack = self.run_dir.read_ack(review_id)
            if ack is not None:
                try:
                    self._acknowledge(i, review_id, str(ack.get("action")), str(ack.get("note") or ""), auto=False)
                    break
                except (GovernanceError, ValueError) as exc:
                    bad = self.run_dir.ack_path(review_id)
                    if bad.exists():
                        bad.replace(bad.with_name(f"{review_id}.ack.invalid.json"))
                    self._progress("awaiting_review", i - 1, message=f"invalid acknowledgement for {review_id}: {exc}")
            if self._signal_cancel:
                raise _Cancelled("signal", checkpoint=False)
            ctl = self.run_dir.read_control()
            if ctl and ctl.get("command") == "cancel":
                raise _Cancelled("control", checkpoint=False)
            time.sleep(self.poll_interval)
        self._progress("running", i - 1, message=f"review {review_id} acknowledged", current_epoch=e)

    def _acknowledge(self, i: int, review_id: str, action: str, note: str, auto: bool) -> None:
        e = self.epochs[i]
        try:
            act = ReviewAction(action)
        except ValueError as exc:
            raise GovernanceError(f"unknown review action {action!r}") from exc
        self.sentinel.acknowledge(review_id, act, note, epoch=e)  # governed policies unfreeze / roll back
        rebased = False
        if act.value in self.rebase_on_ack:
            rebased = self._rebase_reference(i)
        reseg = None
        if act == ReviewAction.RESEGMENT:
            reseg = self._resegment(i, review_id)
        if auto:
            self.run_dir.write_ack(review_id, act.value, note, auto=True, epoch=e)
        rec = self.run_dir.read_review(review_id) or {}
        rec.update(
            {
                **self.sentinel.review(review_id).to_dict(),
                "ack": {"action": act.value, "note": note, "auto": auto, "epoch": e, "at": utc_now()},
                "rebased": rebased,
                "segmentation_after": None if reseg is None else reseg["dimensions_after"],
            }
        )
        self.run_dir.write_review(rec)
        self.ms["acks"].append({"review_id": review_id, "action": act.value, "epoch": e, "auto": auto, "rebased": rebased})
        self._emit(
            ReviewAcknowledged(
                epoch=e,
                payload={
                    "review_id": review_id,
                    "action": act.value,
                    "note": note,
                    "auto": auto,
                    "rebased": rebased,
                    "breaker_state": self.sentinel.state.value,
                },
            )
        )

    def _rebase_reference(self, i: int) -> bool:
        """Re-fit the Sentinel's reference on the most recently evaluated window."""
        last = self.sentinel.history[-1].epoch if self.sentinel.history else None
        idx = i if last == self.epochs[i] else i - 1
        if idx < 0:
            return False
        window = self._window(idx)
        if len(window) < self.min_window_size:
            return False
        self.sentinel.calculator = PSICalculator.from_config(self.cfg.psi).fit(window)
        self.ms["rebased_at"].append(self.epochs[i])
        return True

    def _resegment(self, i: int, review_id: str) -> dict[str, Any]:
        e = self.epochs[i]
        before = list(self.segmenter.dimensions)
        new_seg = self.segmenter.refine()
        info: dict[str, Any] = {"review_id": review_id, "dimensions_before": before, "dimensions_after": list(new_seg.dimensions)}
        if new_seg == self.segmenter:
            info.update(n_refit_loans=0, n_groups=None, note="no refinement dimension left")
            self._emit(ResegmentationApplied(epoch=e, payload=info))
            return info
        ptr = int(self.ms["obs_pointer"])
        rows = np.concatenate([self.asof_rows, self.obs_rows[:ptr]])
        sub = self.table.iloc[rows]
        groups = new_seg.assign(sub).astype(str).to_numpy(dtype=object)
        two_stage = TwoStageBetaBeta.from_config(self.cfg, self.arm_space).fit(
            sub.assign(y_group=groups), snapshot_date=f"{e}-01"
        )
        self.env.resegment(new_seg, two_stage)
        self._set_env_groups()
        for gov in self.governed.values():
            if getattr(gov.policy, "contextual", False):  # flat policies have no groups to refine
                gov.policy.resegment(new_seg, self.env.prior_provider)
        self.segmenter = new_seg
        info.update(n_refit_loans=int(len(rows)), n_groups=int(len(set(groups.tolist()))))
        self.ms["resegmentations"].append({"epoch": e, **info})
        self._emit(ResegmentationApplied(epoch=e, payload=info))
        return info

    # ----- (b) feedback ------------------------------------------------------------------------
    def _release_feedback(self, i: int) -> dict[str, dict[str, Any]]:
        e = self.epochs[i]
        stats = {
            name: {"n_interim": 0, "n_terminal": 0, "n_skipped_replay": 0, "n_dropped_frozen": 0, "feedback_reward": 0.0}
            for name in self.policy_names
        }
        batch = self.queue.pop_due_batch(e)
        n = len(batch)
        if n == 0:
            return stats
        c = batch.columns
        kind, rows, arms, match = c["kind"], c["row"], c["arm"], c["match"]
        env_g = self._stream_env_groups[rows]
        rewards = np.full(n, np.nan)
        mode = FeedbackMode(self.sim_cfg.feedback_mode)
        interim = kind == 0
        terminal = ~interim
        d_int = interim & match if mode == FeedbackMode.REPLAY else interim.copy()
        if d_int.any():
            rewards[d_int] = self.env.interim_rewards(
                env_g[d_int], arms[d_int], c["state"][d_int], c["month"][d_int], c["months_paid"][d_int]
            )
        if mode == FeedbackMode.SIMULATED:
            sim_m, real_m = terminal, np.zeros(n, dtype=bool)
        elif mode == FeedbackMode.REPLAY:
            sim_m, real_m = np.zeros(n, dtype=bool), terminal & match
        else:  # hybrid
            sim_m, real_m = terminal & ~match, terminal & match
        rng = np.random.default_rng([int(self.sim_cfg.seed), int(i), ENV_RNG_STREAM])
        if sim_m.any():
            rewards[sim_m] = self.env.draw_terminal_rewards(env_g[sim_m], arms[sim_m], rng)
        if real_m.any():
            rewards[real_m] = self.realized[rows[real_m]]
        deliver = (d_int | sim_m | real_m) & np.isfinite(rewards)
        skipped = (interim & ~d_int) | (terminal & ~(sim_m | real_m))
        pol = c["policy"]
        groups, segv, fbk = c["group"], c["seg_version"], c["fallback"]
        for name in self.policy_names:
            mine = pol == name
            s = stats[name]
            s["n_skipped_replay"] = int(np.count_nonzero(mine & skipped))
            sel = np.flatnonzero(mine & deliver)
            if sel.size == 0:
                continue
            gov = self.governed[name]
            w_int = float(gov.policy.cfg.interim_weight)
            dropped0 = gov.dropped_updates
            update = gov.update
            for a, g, sv, fl, r, k in zip(
                arms[sel].tolist(), groups[sel].tolist(), segv[sel].tolist(), fbk[sel].tolist(),
                rewards[sel].tolist(), kind[sel].tolist(), strict=True,
            ):
                d = Decision(arm=a, group=g, policy_name=name, fallback=fl, meta={"seg_version": sv} if sv >= 0 else {})
                update(d, r, _TERMINAL if k else _INTERIM, None)
            n_term = int(np.count_nonzero(kind[sel]))
            s["n_terminal"] = n_term
            s["n_interim"] = int(sel.size - n_term)
            s["n_dropped_frozen"] = int(gov.dropped_updates - dropped0)
            w = np.where(kind[sel] == 1, 1.0, w_int)
            s["feedback_reward"] = float(np.sum(w * rewards[sel]))
        if terminal.any():
            src = np.where(sim_m, "simulated", np.where(real_m, "realized", "skipped_replay"))
            tm = np.flatnonzero(terminal)
            log = pd.DataFrame(
                {
                    "epoch": pd.array([e] * tm.size, dtype="string"),
                    "policy": pd.array(pol[tm].astype(str), dtype="string"),
                    "loan_id": c["loan_id"][tm].astype(np.int64),
                    "decision_epoch": pd.array(c["decision_epoch"][tm].astype(str), dtype="string"),
                    "arm": arms[tm].astype(np.int64),
                    "reward": rewards[tm],
                    "source": pd.array(src[tm].astype(str), dtype="string"),
                    "delivered": deliver[tm],
                }
            )
            self.run_dir.write_feedback(e, log)
        return stats

    def _observe_outcomes(self, i: int) -> int:
        ptr = int(self.ms["obs_pointer"])
        end = int(np.searchsorted(self.obs_mi_sorted, self.epoch_mi[i], side="right"))
        n = 0
        if end > ptr:
            n = self.env.observe_outcomes(
                self._obs_groups[ptr:end], self.obs_arm[ptr:end], self.obs_default[ptr:end], self.obs_recovery[ptr:end]
            )
        self.ms["obs_pointer"] = end
        return n

    # ----- (c) decisions --------------------------------------------------------------------------
    def contexts(self, i: int) -> list[BorrowerContext]:
        """The :class:`BorrowerContext` objects shown to the policies in epoch ``i``."""
        lo, hi = self.bounds[i]
        ctx = self.ctx_frame.iloc[lo:hi]
        LeakageGuard.assert_allowed(ctx.columns, Stage.PRE_DECISION)
        cols = self.ctx_columns
        values = [ctx[c].tolist() for c in cols]
        issue = self.stream["issue_d"].iloc[lo:hi].tolist()
        return [
            BorrowerContext(loan_id=int(lid), features=dict(zip(cols, vals, strict=True)), issue_date=pd.Timestamp(ts))
            for lid, vals, ts in zip(self.loan_ids[lo:hi].tolist(), zip(*values, strict=True), issue, strict=True)
        ]

    def _decide(self, i: int) -> dict[str, Any]:
        lo, hi = self.bounds[i]
        n = hi - lo
        contexts = self.contexts(i)
        ctx_frame = self.ctx_frame.iloc[lo:hi]
        hist = [int(h) if ok else None for h, ok in zip(self.hist[lo:hi].tolist(), self.valid_hist[lo:hi].tolist(), strict=True)]
        n_pol = len(self.specs)
        arms = np.zeros((n_pol, n), dtype=np.int64)
        groups = np.empty((n_pol, n), dtype=object)
        fallback = np.zeros((n_pol, n), dtype=bool)
        new = np.zeros((n_pol, n), dtype=bool)
        segv = np.full((n_pol, n), -1, dtype=np.int32)
        for p, spec in enumerate(self.specs):
            gov = self.governed[spec.label]
            gov.prime_groups(ctx_frame)
            rng = np.random.default_rng([int(self.sim_cfg.seed), int(i), int(p)])
            masks = (self.elig_hist if spec.is_historical else self.elig)[lo:hi]
            select = gov.select
            arm_row, grp_row, fb_row, new_row, sv_row = arms[p], groups[p], fallback[p], new[p], segv[p]
            for j in range(n):
                d = select(contexts[j], masks[j], rng, historical_arm=hist[j])
                arm_row[j] = d.arm
                grp_row[j] = d.group
                fb_row[j] = d.fallback
                meta = d.meta
                new_row[j] = bool(meta.get("group_new", False))
                sv = meta.get("seg_version")
                if sv is not None:
                    sv_row[j] = int(sv)
        return {"arms": arms, "groups": groups, "fallback": fallback, "group_new": new, "seg_version": segv}

    def _decision_frame(self, i: int, dec: dict[str, Any]) -> pd.DataFrame:
        e = self.epochs[i]
        lo, hi = self.bounds[i]
        n = hi - lo
        rows = np.arange(lo, hi)
        null = self.arm_space.null_arm
        frames = []
        for p, name in enumerate(self.policy_names):
            a = dec["arms"][p]
            match = a == self.hist[lo:hi]
            expected = self.hind_score[self.hind_code[rows], a]
            frames.append(
                pd.DataFrame(
                    {
                        "epoch": pd.array([e] * n, dtype="string"),
                        "policy": pd.array([name] * n, dtype="string"),
                        "loan_id": self.loan_ids[lo:hi],
                        "group": pd.array(dec["groups"][p].astype(str), dtype="string"),
                        "arm": a.astype(np.int64),
                        "amount": self.arm_space.amounts[a],
                        "rate": self.arm_space.rates[a],
                        "term": self.arm_space.terms[a].astype(np.int64),
                        "declined": a == (null if null is not None else -1),
                        "expected_reward": expected,
                        "oracle_reward": self.oracle[lo:hi],
                        "realized_reward": np.where(match, self.realized[lo:hi], np.nan),
                        "fallback": dec["fallback"][p],
                        "simulated_reward": np.full(n, np.nan),
                        "historical_match": match,
                        "meta_group_new": dec["group_new"][p],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True).loc[:, list(DECISION_COLUMNS)]

    def _schedule_feedback(self, i: int, dec: dict[str, Any]) -> None:
        e = self.epochs[i]
        lo, hi = self.bounds[i]
        null = self.arm_space.null_arm
        p_lo, p_hi = int(self.plan_bounds[lo]), int(self.plan_bounds[hi])
        local = self.plan_row[p_lo:p_hi] - lo
        term_local = np.flatnonzero(self.term_rel[lo:hi] >= 0)
        hist = self.hist[lo:hi]
        for p, name in enumerate(self.policy_names):
            a = dec["arms"][p]
            lend = a != (null if null is not None else -1)
            if self.schedule_interim and local.size:
                sel = lend[local]
                if sel.any():
                    jj = local[sel]
                    self.queue.schedule_batch(
                        self._epoch_strings(self.plan_rel[p_lo:p_hi][sel]),
                        kind=np.zeros(jj.size, dtype=np.int8),
                        policy=name,
                        loan_id=self.loan_ids[lo + jj],
                        row=lo + jj,
                        arm=a[jj],
                        group=dec["groups"][p][jj],
                        seg_version=dec["seg_version"][p][jj],
                        fallback=dec["fallback"][p][jj],
                        month=self.plan_month[p_lo:p_hi][sel],
                        months_paid=self.plan_paid[p_lo:p_hi][sel],
                        state=self.plan_state[p_lo:p_hi][sel],
                        match=a[jj] == hist[jj],
                        decision_epoch=e,
                    )
            tj = term_local[lend[term_local]]
            if tj.size:
                self.queue.schedule_batch(
                    self._epoch_strings(self.term_rel[lo + tj]),
                    kind=np.ones(tj.size, dtype=np.int8),
                    policy=name,
                    loan_id=self.loan_ids[lo + tj],
                    row=lo + tj,
                    arm=a[tj],
                    group=dec["groups"][p][tj],
                    seg_version=dec["seg_version"][p][tj],
                    fallback=dec["fallback"][p][tj],
                    match=a[tj] == hist[tj],
                    decision_epoch=e,
                )

    # ----- metrics & outputs -----------------------------------------------------------------------
    def _update_metrics(
        self,
        i: int,
        frame: pd.DataFrame,
        fb: Mapping[str, Mapping[str, Any]],
        breaker_state: str,
        reading: SentinelReading,
    ) -> dict[str, Any]:
        e = self.epochs[i]
        pending = self.queue.pending_by_policy()
        per_policy: dict[str, Any] = {}
        for name in self.policy_names:
            sub = frame.loc[frame["policy"] == name]
            n = int(len(sub))
            exp = sub["expected_reward"].to_numpy(dtype=float)
            orc = sub["oracle_reward"].to_numpy(dtype=float)
            realized = float(np.nansum(sub["realized_reward"].to_numpy(dtype=float)))
            reward = float(exp.sum())
            regret = float((orc - exp).sum())
            declined = sub["declined"].to_numpy(dtype=bool)
            lent = ~declined
            cum = self.ms["cum"][name]
            cum["n"] += n
            cum["reward"] += reward
            cum["realized"] += realized
            cum["regret"] += regret
            cum["feedback_reward"] += float(fb[name]["feedback_reward"])
            prev = [r for r in self.ms["timeseries"] if r["policy"] == name][-(ROLLING_EPOCHS - 1):] if ROLLING_EPOCHS > 1 else []
            roll_n = n + sum(r["n_decisions"] for r in prev)
            roll_r = reward + sum(r["reward"] for r in prev)
            gov = self.governed[name]
            counts = self.ms["group_arrivals"][name]
            if n:
                g_names, g_counts = np.unique(sub["group"].to_numpy(dtype=str), return_counts=True)
                for g, cnt in zip(g_names.tolist(), g_counts.tolist(), strict=True):
                    counts[g] = counts.get(g, 0) + int(cnt)
            row = {
                "epoch": e,
                "policy": name,
                "n_decisions": n,
                "reward": reward,
                "cum_reward": cum["reward"],
                "realized": realized,
                "cum_realized": cum["realized"],
                "regret": regret,
                "cum_regret": cum["regret"],
                "rolling_reward": roll_r / roll_n if roll_n else float("nan"),
                "decline_rate": float(declined.mean()) if n else float("nan"),
                "fallback_rate": float(sub["fallback"].to_numpy(dtype=bool).mean()) if n else float("nan"),
                "mean_amount": float(sub["amount"].to_numpy()[lent].mean()) if lent.any() else float("nan"),
                "mean_rate": float(sub["rate"].to_numpy()[lent].mean()) if lent.any() else float("nan"),
                "n_feedback_interim": int(fb[name]["n_interim"]),
                "n_feedback_terminal": int(fb[name]["n_terminal"]),
                "pending_feedback": int(pending.get(name, 0)),
                "breaker_state": breaker_state,
                "psi_aggregate": float(reading.psi_aggregate),
                "n_groups": int(len(gov.group_keys())),
                "epoch_index": int(i),
                "sentinel_level": reading.level.value,
                "feedback_reward": float(fb[name]["feedback_reward"]),
                "n_skipped_replay": int(fb[name]["n_skipped_replay"]),
                "n_dropped_frozen": int(fb[name]["n_dropped_frozen"]),
            }
            self.ms["timeseries"].append(row)
            per_policy[name] = {
                "n": n,
                "reward": reward,
                "regret": regret,
                "decline_rate": row["decline_rate"],
                "fallback_rate": row["fallback_rate"],
            }
        self.ms["n_decisions"] = int(self.ms["n_decisions"]) + int(len(frame))
        return per_policy

    def _snapshot_arms(self, i: int) -> None:
        e = self.epochs[i]
        frames = []
        for name in self.policy_names:
            gov = self.governed[name]
            pooled = gov.arm_table(POOLED_GROUP)
            if len(pooled):
                frames.append(pooled.assign(group=POOLED_GROUP))
            keys = set(gov.group_keys())
            counts = self.ms["group_arrivals"][name]
            ranked = sorted((g for g in counts if g in keys), key=lambda g: (-counts[g], g))[:SNAPSHOT_TOP_GROUPS]
            for g in ranked:
                tab = gov.arm_table(g)
                if len(tab):
                    frames.append(tab)
            frames = [f.assign(policy=name) if "policy" not in f.columns else f for f in frames]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        df = df.assign(epoch=e)
        df["group"] = df["group"].astype("string")
        df["policy"] = df["policy"].astype("string")
        df["epoch"] = df["epoch"].astype("string")
        self.run_dir.write_arms(e, df.loc[:, list(ARM_SNAPSHOT_COLUMNS)])

    def _write_tables(self) -> None:
        ts = pd.DataFrame(self.ms["timeseries"], columns=list(TIMESERIES_COLUMNS))
        for col in ("epoch", "policy", "breaker_state", "sentinel_level"):
            ts[col] = ts[col].astype("string")
        self.run_dir.write_timeseries(ts)
        sen = pd.DataFrame(self.ms["sentinel_rows"], columns=list(SENTINEL_COLUMNS))
        for col in ("epoch", "feature", "level", "breaker_state"):
            sen[col] = sen[col].astype("string")
        self.run_dir.write_sentinel(sen)

    def _write_checkpoint(self, i: int) -> None:
        t0 = time.perf_counter()
        self.ms["checkpoints_written"] = int(self.ms["checkpoints_written"]) + 1
        state = self._checkpoint_state(i)
        path, nbytes = self.run_dir.checkpoints.write(state, self.epochs[i])
        self._last_checkpoint_index = i
        self._emit(
            CheckpointWritten(
                epoch=self.epochs[i],
                payload={"epoch_index": i, "path": str(path), "bytes": int(nbytes), "seconds": round(time.perf_counter() - t0, 4)},
            )
        )
        self._write_runtime()

    # ==================================================================================
    # status, events, signals
    # ==================================================================================
    def _emit(self, event: SimulationEvent) -> None:
        self.run_dir.events.append(event)

    def _progress(self, status: str, epoch_index: int, message: str = "", **extra: Any) -> None:
        n_epochs = len(self.epochs)
        done = max(epoch_index + 1, 0)
        fields = {
            "status": status,
            "epoch_index": int(epoch_index),
            "n_epochs": n_epochs,
            "current_epoch": self.epochs[epoch_index] if 0 <= epoch_index < n_epochs else None,
            "pct": round(100.0 * done / n_epochs, 2) if n_epochs else 0.0,
            "message": message,
            "breaker_state": self.sentinel.state.value if hasattr(self, "sentinel") else BreakerState.CLOSED.value,
            "pid": os.getpid(),
            "started_at": self._session_started,
            "finished_at": None,
            "error": None,
        }
        fields.update(extra)
        self.run_dir.write_progress(**fields)

    def _runtime_session(self) -> dict[str, Any]:
        from ..pipeline import peak_rss_mb

        return {
            "pid": os.getpid(),
            "started_at": self._session_started,
            "wall_seconds": round(time.perf_counter() - self._session_t0, 3),
            "cpu_seconds": round(time.process_time() - self._session_cpu0, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }

    def _write_runtime(self, status: str | None = None) -> None:
        prev = self.run_dir.read_runtime() or {}
        cur = self._runtime_session()
        sessions = [s for s in prev.get("sessions", []) if not (s.get("pid") == cur["pid"] and s.get("started_at") == cur["started_at"])]
        sessions.append(cur)
        n_dec = int(self.ms.get("n_decisions", 0)) if hasattr(self, "ms") else 0
        wall = float(sum(s["wall_seconds"] for s in sessions))
        doc = {
            "sessions": sessions,
            "wall_seconds": round(wall, 3),
            "cpu_seconds": round(float(sum(s["cpu_seconds"] for s in sessions)), 3),
            "peak_rss_mb": max(float(s["peak_rss_mb"]) for s in sessions),
            "n_decisions": n_dec,
            "decisions_per_second": round(n_dec / wall, 1) if wall > 0 else None,
            "epochs": len(self.epochs),
            "restarts": int(self.ms.get("restarts", 0)) if hasattr(self, "ms") else 0,
            "checkpoints_written": int(self.ms.get("checkpoints_written", 0)) if hasattr(self, "ms") else 0,
            "status": status or prev.get("status"),
            "phase_seconds": {k: round(v, 3) for k, v in self.phase_seconds.items()},
        }
        self.run_dir.write_runtime(doc)

    def _install_signals(self) -> None:
        self._signal_cancel = False
        if not self.install_signal_handlers:
            return
        try:
            self._old_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)
        except ValueError:  # not in the main thread
            self._old_sigterm = None

    def _restore_signals(self) -> None:
        if self._old_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._old_sigterm)
            except ValueError:  # pragma: no cover
                pass
            self._old_sigterm = None

    def _on_sigterm(self, signum: int, frame: Any) -> None:  # pragma: no cover - exercised via subprocess
        self._signal_cancel = True

    # ----- terminal states ---------------------------------------------------------------------
    def _completed_epochs(self) -> int:
        return len({r["epoch"] for r in self.ms.get("timeseries", [])}) if hasattr(self, "ms") else 0

    def _result(self, status: str) -> SimulationResult:
        done = self._completed_epochs()
        return SimulationResult(
            status=status,
            run_dir=self.run_dir.path,
            epochs_completed=done,
            n_epochs=len(self.epochs),
            n_decisions=int(self.ms.get("n_decisions", 0)) if hasattr(self, "ms") else 0,
            wall_seconds=round(time.perf_counter() - self._session_t0, 3),
            last_epoch=self.epochs[done - 1] if done else None,
        )

    def _finish_completed(self) -> SimulationResult:
        last = len(self.epochs) - 1
        report_path = None
        report_error = None
        try:
            from ..pipeline import build_report

            build_report(self.run_dir.path, fitted=self.fitted)
            report_path = str(self.run_dir.report_path)
        except Exception as exc:  # noqa: BLE001 - the run itself succeeded
            report_error = f"{type(exc).__name__}: {exc}"
        self._write_runtime("completed")
        self._emit(
            RunCompleted(
                epoch=self.epochs[last],
                payload={
                    "n_epochs": len(self.epochs),
                    "n_decisions": int(self.ms["n_decisions"]),
                    "wall_seconds": round(time.perf_counter() - self._session_t0, 3),
                    "report": report_path,
                    "report_error": report_error,
                },
            )
        )
        self._progress("completed", last, message="completed", finished_at=utc_now())
        return self._result("completed")

    def _finish_cancel(self, c: _Cancelled) -> SimulationResult:
        done = self._completed_epochs()
        last = done - 1
        ck = None
        if c.checkpoint and last >= 0 and self._last_checkpoint_index != last:
            self._write_checkpoint(last)
        if self.run_dir.checkpoints.exists():
            ck = str(self.run_dir.checkpoints.latest_path)
        self._write_runtime("cancelled")
        self._emit(RunCancelled(epoch=self.epochs[last] if last >= 0 else None, payload={"reason": c.reason, "checkpoint": ck}))
        self._progress("cancelled", last, message=f"cancelled ({c.reason})", finished_at=utc_now())
        return self._result("cancelled")

    def _fail(self, exc: BaseException) -> None:
        try:
            done = self._completed_epochs()
            self._write_runtime("failed")
            self._emit(
                RunFailed(
                    epoch=self.epochs[done - 1] if done else None,
                    payload={"error": str(exc), "error_type": type(exc).__name__, "epoch_index": done - 1},
                )
            )
            self._progress(
                "failed", done - 1, message=f"failed: {type(exc).__name__}", finished_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001,S110 - never mask the original error
            pass

