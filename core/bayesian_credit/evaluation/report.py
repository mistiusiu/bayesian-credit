"""Evaluation report organised by the dimensions of Table 11 (SPEC §2.12; proposal §3.7,
G.7).

:class:`EvaluationReport` gathers the outputs of a run - the predictive metrics of the fit,
the simulation's decision log / time series / Sentinel readings / event log, the shadow
portfolio summary and runtime measurements - and :meth:`EvaluationReport.build` returns
one JSON-ready dict:

``dimensions``
    Nested, detailed results per Table 11 dimension: ``default_prediction``,
    ``recovery_prediction``, ``credit_state_modeling``, ``policy_performance``,
    ``bandit_performance``, ``counterfactual_evaluation``, ``distributional_monitoring``,
    ``computational_feasibility``, ``infrastructure_resilience``.
``table11``
    The same results flattened into ``{key, label, axis, metrics: [{key, label, value,
    unit, better, baseline, baseline_label, by_policy, note}]}`` rows (what the research
    console renders).
``policy_summary``, ``runtime``, ``config``, ``config_hashes``, ``dataset``,
``generated_at``, ``status`` (``final`` once the run completed, else ``partial``).

Predictive metrics follow the calibration protocol of :mod:`bayesian_credit.pipeline`:
**primary** = resolved holdout loans whose term has fully elapsed by the snapshot,
**secondary** = all resolved holdout loans.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import policy as pol

__all__ = ["DIMENSIONS", "EvaluationReport"]

#: (key, label, G.7 axis) of the nine Table 11 rows.
DIMENSIONS: tuple[tuple[str, str, str], ...] = (
    ("default_prediction", "Default prediction", "predictive"),
    ("recovery_prediction", "Recovery prediction", "predictive"),
    ("credit_state_modeling", "Credit-state modelling (MCTM)", "predictive"),
    ("policy_performance", "Policy performance", "decision"),
    ("bandit_performance", "Bandit performance", "decision"),
    ("counterfactual_evaluation", "Counterfactual evaluation (shadow engine)", "decision"),
    ("distributional_monitoring", "Distributional monitoring (PSI Sentinel)", "adaptability"),
    ("computational_feasibility", "Computational feasibility", "feasibility"),
    ("infrastructure_resilience", "Infrastructure resilience", "feasibility"),
)

_LAST_EPOCHS_CONCENTRATION = 6
_TOP_ARMS = 3


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean(value: Any) -> Any:
    """JSON-safe copy (numpy → python, NaN/inf → None)."""
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    if value is None or isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _hash(obj: Any) -> str:
    blob = json.dumps(_clean(obj), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _metric(
    key: str,
    label: str,
    value: Any,
    unit: str,
    better: str | None = None,
    *,
    baseline: Any = None,
    baseline_label: str | None = None,
    by_policy: Mapping[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": _f(value) if not isinstance(value, str) else value,
        "unit": unit,
        "better": better,
        "baseline": _f(baseline),
        "baseline_label": baseline_label,
        "by_policy": None if by_policy is None else {str(k): _f(v) for k, v in by_policy.items()},
        "note": note,
    }


class EvaluationReport:
    """Table 11 evaluation report of one pipeline run (see module docstring).

    Every input is optional; missing inputs produce ``None`` metrics (and a ``note``)
    rather than errors, so a partial report can be rendered while a run is in progress.
    """

    def __init__(
        self,
        *,
        decisions: pd.DataFrame | None = None,
        timeseries: pd.DataFrame | None = None,
        sentinel: pd.DataFrame | None = None,
        events: list[dict[str, Any]] | None = None,
        reviews: list[dict[str, Any]] | None = None,
        config: Mapping[str, Any] | None = None,
        progress: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        fit_metrics: Mapping[str, Any] | None = None,
        predictive: Mapping[str, Any] | None = None,
        shadow_summary: Any = None,
        timings: Mapping[str, float] | None = None,
        resilience: Mapping[str, Any] | None = None,
        dataset_summary: Mapping[str, Any] | None = None,
        arm_space: Any = None,
        run_dir: str | os.PathLike[str] | None = None,
    ):
        self.decisions = decisions if decisions is not None else pd.DataFrame()
        self.timeseries = timeseries if timeseries is not None else pd.DataFrame()
        self.sentinel = sentinel if sentinel is not None else pd.DataFrame()
        self.events = list(events or [])
        self.reviews = list(reviews or [])
        self.config = dict(config or {})
        self.progress = dict(progress or {})
        self.runtime = dict(runtime or {})
        self.fit_metrics = dict(fit_metrics or {})
        self.predictive = dict(predictive or self.fit_metrics.get("predictive") or {})
        if shadow_summary is not None and hasattr(shadow_summary, "to_dict"):
            shadow_summary = shadow_summary.to_dict()
        self.shadow = dict(shadow_summary or {})
        self.timings = dict(timings or {})
        self.resilience = dict(resilience or {})
        self.dataset_summary = dict(dataset_summary or self.config.get("dataset", {}).get("summary") or {})
        self.arm_space = arm_space
        self.run_dir = None if run_dir is None else Path(run_dir)

    # ----- construction ----------------------------------------------------------------------
    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | os.PathLike[str],
        *,
        fitted: Any = None,
        predictive: Mapping[str, Any] | None = None,
        shadow_summary: Any = None,
        timings: Mapping[str, float] | None = None,
        resilience: Mapping[str, Any] | None = None,
        dataset_summary: Mapping[str, Any] | None = None,
    ) -> EvaluationReport:
        """Load everything a run directory holds (plus optional in-memory extras).

        Fit metrics come from ``fitted.metrics`` or ``<fit_dir>/metrics.json``; a shadow
        summary from the argument or ``<run_dir>/shadow_summary.json``; resilience results
        from the argument or ``<run_dir>/resilience.json``.
        """
        from ..simulation.store import RunDirectory, read_json

        rd = RunDirectory(run_dir)
        config = read_json(rd.config_path, {}) or {}
        fit_metrics: dict[str, Any] = {}
        arm_space = None
        if fitted is not None:
            fit_metrics = dict(getattr(fitted, "metrics", {}) or {})
            arm_space = getattr(fitted, "arm_space", None)
        elif config.get("fit_dir"):
            fit_metrics = read_json(Path(config["fit_dir"]) / "metrics.json", {}) or {}
            arm_d = read_json(Path(config["fit_dir"]) / "arm_space.json")
            if arm_d:
                from ..data.arms import ArmSpace

                arm_space = ArmSpace.from_dict(arm_d)
        if shadow_summary is None:
            shadow_summary = read_json(rd.path / "shadow_summary.json")
        if resilience is None:
            resilience = read_json(rd.path / "resilience.json")
        return cls(
            decisions=rd.read_decisions(),
            timeseries=rd.read_timeseries(),
            sentinel=rd.read_sentinel(),
            events=rd.events.read(),
            reviews=rd.list_reviews(),
            config=config,
            progress=rd.read_progress(),
            runtime=rd.read_runtime() or {},
            fit_metrics=fit_metrics,
            predictive=predictive,
            shadow_summary=shadow_summary,
            timings=timings,
            resilience=resilience,
            dataset_summary=dataset_summary,
            arm_space=arm_space,
            run_dir=run_dir,
        )

    # ----- sections -------------------------------------------------------------------------------
    def default_prediction(self) -> dict[str, Any]:
        d = self.predictive.get("default") or {}
        if not d:
            return {"available": False, "note": "no predictive metrics (fit without a holdout)"}
        return {
            "available": True,
            "primary": d.get("primary"),
            "secondary": d.get("secondary"),
            "by_vintage": d.get("by_vintage"),
            "snapshot_date": self.predictive.get("snapshot_date"),
            "n_holdout": self.predictive.get("n_holdout"),
            "n_resolved": self.predictive.get("n_resolved"),
            "n_elapsed_resolved": self.predictive.get("n_elapsed_resolved"),
        }

    def recovery_prediction(self) -> dict[str, Any]:
        r = self.predictive.get("recovery") or {}
        if not r:
            return {"available": False}
        return {"available": True, **r}

    def credit_state_modeling(self) -> dict[str, Any]:
        cs = self.predictive.get("credit_state") or {}
        mc = self.fit_metrics.get("mctm") or {}
        if not cs and not mc:
            return {"available": False}
        return {"available": True, "holdout": cs, "train": mc}

    def _policies(self) -> list[str]:
        sim = self.config.get("simulation") or {}
        names = [str(p).replace(" ", "").lower() for p in sim.get("policies", [])]
        if len(self.decisions):
            present = list(dict.fromkeys(self.decisions["policy"].astype(str).tolist()))
            names = [n for n in names if n in present] + [p for p in present if p not in names]
        return names

    def policy_performance(self) -> dict[str, Any]:
        dec = self.decisions
        if len(dec) == 0:
            return {"available": False, "note": "no decisions logged"}
        prof = pol.profitability_vs_historical(dec)
        dr = pol.decline_rate(dec, by_epoch=False)
        rows: dict[str, Any] = {}
        for name in self._policies():
            sub = dec.loc[dec["policy"].astype(str) == name]
            p = prof.loc[prof["policy"] == name]
            lent = ~sub["declined"].to_numpy(dtype=bool)
            realized = sub["realized_reward"].to_numpy(dtype=float)
            rows[name] = {
                "n_decisions": int(len(sub)),
                "total_expected_reward": _f(p["total_expected"].iloc[0]) if len(p) else None,
                "mean_expected_reward": _f(sub["expected_reward"].mean()),
                "total_realized_reward": _f(np.nansum(realized)),
                "n_realized": int(np.isfinite(realized).sum()),
                "realized_coverage": _f(np.isfinite(realized).mean()),
                "uplift_vs_historical": _f(p["uplift_vs_historical"].iloc[0]) if len(p) else None,
                "uplift_pct": _f(p["uplift_pct"].iloc[0]) if len(p) else None,
                "decline_rate": _f(dr.loc[dr["policy"] == name, "decline_rate"].iloc[0]) if (dr["policy"] == name).any() else None,
                "fallback_rate": _f(sub["fallback"].to_numpy(dtype=bool).mean()),
                "historical_match_rate": _f(sub["historical_match"].to_numpy(dtype=bool).mean()) if "historical_match" in sub else None,
                "mean_amount": _f(sub["amount"].to_numpy(dtype=float)[lent].mean()) if lent.any() else None,
                "mean_rate": _f(sub["rate"].to_numpy(dtype=float)[lent].mean()) if lent.any() else None,
                "mean_simulated_reward": _f(np.nanmean(sub["simulated_reward"].to_numpy(dtype=float)))
                if "simulated_reward" in sub and np.isfinite(sub["simulated_reward"].to_numpy(dtype=float)).any()
                else None,
            }
        best = None
        adaptive = {k: v for k, v in rows.items() if k not in ("historical", "random") and v["total_expected_reward"] is not None}
        if adaptive:
            best = max(adaptive, key=lambda k: adaptive[k]["total_expected_reward"])
        return {
            "available": True,
            "by_policy": rows,
            "best_adaptive_policy": best,
            "historical_present": "historical" in rows,
            "note": "expected_reward = hindsight-model risk-adjusted expected reward of the chosen arm; realized only where the chosen arm equals the historical arm and the loan resolved",
        }

    def bandit_performance(self) -> dict[str, Any]:
        dec = self.decisions
        if len(dec) == 0:
            return {"available": False}
        reg = pol.cumulative_regret(dec)
        conv = pol.convergence_epoch(reg)
        epochs = sorted(dec["epoch"].astype(str).unique())
        last = set(epochs[-_LAST_EPOCHS_CONCENTRATION:])
        recent = dec.loc[dec["epoch"].astype(str).isin(last)]
        max_rate, max_amount = self._grid_maxima()
        rows: dict[str, Any] = {}
        for name in self._policies():
            r = reg.loc[reg["policy"] == name]
            c = conv.loc[conv["policy"] == name]
            sub = recent.loc[recent["policy"].astype(str) == name]
            counts = sub["arm"].value_counts()
            conc = float(counts.iloc[:_TOP_ARMS].sum() / counts.sum()) if counts.sum() else None
            mix = pol.arm_mix(
                sub["amount"].to_numpy(dtype=float),
                sub["rate"].to_numpy(dtype=float),
                sub["term"].to_numpy(dtype=float),
                ~sub["declined"].to_numpy(dtype=bool),
                max_rate=max_rate,
                max_amount=max_amount,
            )
            rows[name] = {
                "final_cumulative_regret": _f(r["cum_regret"].iloc[-1]) if len(r) else None,
                "average_regret": _f(r["cum_avg_regret"].iloc[-1]) if len(r) else None,
                "final_epoch_avg_regret": _f(r["avg_regret"].iloc[-1]) if len(r) else None,
                "convergence_epoch": None if not len(c) else c["convergence_epoch"].iloc[0],
                "converged": bool(c["converged"].iloc[0]) if len(c) else False,
                "selection_concentration_top3": conc,
                "top_arms_recent": [int(a) for a in counts.index[:_TOP_ARMS].tolist()],
                "top_arm_labels_recent": [self._arm_label(int(a)) for a in counts.index[:_TOP_ARMS].tolist()],
                "n_distinct_arms": int(dec.loc[dec["policy"].astype(str) == name, "arm"].nunique()),
                "selection_mix": mix,
            }
        return {
            "available": True,
            "by_policy": rows,
            "concentration_window_epochs": sorted(last),
            "note": "regret = oracle − expected under the hindsight model (pseudo-regret); concentration = share of the top-3 arms "
            "in the last 6 epochs; selection_mix = the same window's lending decisions by term / rate bucket / amount bucket",
        }

    def _grid_maxima(self) -> tuple[float | None, float | None]:
        """Top rate / amount of the arm grid (from the arm space, else the decision log)."""
        space = self.arm_space
        if space is not None and getattr(space, "rate_grid", None):
            return float(max(space.rate_grid)), float(max(space.amount_grid))
        dec = self.decisions
        if len(dec):
            lend = ~dec["declined"].to_numpy(dtype=bool)
            if lend.any():
                return float(dec["rate"].to_numpy(dtype=float)[lend].max()), float(dec["amount"].to_numpy(dtype=float)[lend].max())
        return None, None

    def _arm_label(self, arm: int) -> str | None:
        space = self.arm_space
        if space is None or not 0 <= arm < getattr(space, "n_arms", 0):
            return None
        return space[arm].label()

    def counterfactual_evaluation(self) -> dict[str, Any]:
        if not self.shadow:
            return {"available": False, "note": "shadow portfolio not run"}
        out = {k: v for k, v in self.shadow.items() if k not in ("top_loans", "by_group")}
        out["top_loans"] = (self.shadow.get("top_loans") or [])[:10]
        out["by_group_top"] = (self.shadow.get("by_group") or [])[:15]
        out["recommendation_mix"] = self.shadow.get("recommendation_mix") or None
        out["eligibility"] = self.shadow.get("eligibility") or None
        out["uplift_decomposition"] = self.shadow.get("uplift_decomposition") or None
        out["by_risk_aversion"] = self.shadow.get("by_risk_aversion") or None
        out["interest_realization"] = self.fit_metrics.get("interest_realization")
        return {"available": True, **out}

    def distributional_monitoring(self) -> dict[str, Any]:
        sen = self.sentinel
        trips = [e for e in self.events if e.get("type") == "BreakerTripped"]
        acks = [e for e in self.events if e.get("type") == "ReviewAcknowledged"]
        reseg = [e for e in self.events if e.get("type") == "ResegmentationApplied"]
        out: dict[str, Any] = {"available": bool(len(sen)), "n_trips": len(trips)}
        if len(sen):
            psi = sen.assign(psi=pd.to_numeric(sen["psi"], errors="coerce"))
            per_feature = (
                psi.groupby("feature", sort=True)["psi"].agg(["max", "mean"]).rename(columns={"max": "max_psi", "mean": "mean_psi"})
            )
            out["per_feature"] = {f: {"max_psi": _f(r.max_psi), "mean_psi": _f(r.mean_psi)} for f, r in per_feature.iterrows()}
            by_epoch = psi.groupby("epoch", sort=True).agg(
                aggregate=("aggregate", "first"), breaker_state=("breaker_state", "first"), skipped=("skipped", "first")
            )
            sim = self.config.get("framework", {}).get("psi", {})
            watch = float(sim.get("watch_threshold", 0.10))
            breaker = float(sim.get("breaker_threshold", 0.25))
            agg = pd.to_numeric(by_epoch["aggregate"], errors="coerce")
            valid = ~by_epoch["skipped"].astype(bool) & agg.notna()
            out["epochs"] = int(len(by_epoch))
            out["epochs_normal"] = int((valid & (agg < watch)).sum())
            out["epochs_watch"] = int((valid & (agg >= watch) & (agg <= breaker)).sum())
            out["epochs_breaker_level"] = int((valid & (agg > breaker)).sum())
            out["epochs_breaker_open"] = int((by_epoch["breaker_state"].astype(str) == "open").sum())
            out["max_aggregate_psi"] = _f(agg.max())
            out["mean_aggregate_psi"] = _f(agg.mean())
            out["thresholds"] = {"watch": watch, "breaker": breaker}
            first = by_epoch.loc[valid & (agg > breaker)]
            out["first_breaker_epoch"] = None if not len(first) else str(first.index[0])
        out["trips"] = [{"epoch": e.get("epoch"), **(e.get("payload") or {})} for e in trips]
        out["reviews"] = [
            {
                "id": r.get("id"),
                "epoch": r.get("epoch"),
                "psi_aggregate": r.get("psi_aggregate"),
                "status": r.get("status"),
                "action": (r.get("ack") or {}).get("action") or r.get("action"),
                "auto": r.get("auto"),
                "rebased": r.get("rebased"),
            }
            for r in self.reviews
        ]
        actions: dict[str, int] = {}
        for e in acks:
            a = (e.get("payload") or {}).get("action")
            actions[str(a)] = actions.get(str(a), 0) + 1
        out["actions"] = actions
        out["resegmentations"] = [e.get("payload") for e in reseg]
        return out

    def computational_feasibility(self) -> dict[str, Any]:
        rt = self.runtime
        n_dec = int(len(self.decisions)) if len(self.decisions) else int(rt.get("n_decisions") or 0)
        n_loans = int(self.decisions["loan_id"].nunique()) if len(self.decisions) else None
        sim_wall = _f(rt.get("wall_seconds"))
        out = {
            "stage_seconds": {k: _f(v) for k, v in self.timings.items()},
            "total_seconds": _f(sum(v for v in self.timings.values() if _f(v) is not None)) if self.timings else None,
            "simulation_wall_seconds": sim_wall,
            "simulation_cpu_seconds": _f(rt.get("cpu_seconds")),
            "peak_rss_mb": _f(rt.get("peak_rss_mb")),
            "n_loans_streamed": n_loans,
            "n_decisions": n_dec,
            "decisions_per_second": _f(n_dec / sim_wall) if sim_wall else None,
            "n_epochs": int(self.timeseries["epoch"].nunique()) if len(self.timeseries) else None,
            "simulation_phase_seconds": rt.get("phase_seconds"),
            "sessions": rt.get("sessions"),
        }
        out["report_process_time_seconds"] = _f(time.process_time())
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            out["process_cpu_seconds"] = _f(usage.ru_utime + usage.ru_stime)
            out["process_peak_rss_mb"] = _f(usage.ru_maxrss / 1024.0)
        except (ImportError, OSError):  # pragma: no cover
            pass
        return out

    def infrastructure_resilience(self) -> dict[str, Any]:
        types = [e.get("type") for e in self.events]
        out = {
            "checkpoints_written": types.count("CheckpointWritten"),
            "restarts": types.count("RunResumed"),
            "failures": types.count("RunFailed"),
            "cancellations": types.count("RunCancelled"),
            "pauses": types.count("RunPaused"),
            "checkpoint_every": (self.config.get("simulation") or {}).get("checkpoint_every"),
            "resume_test": self.resilience or None,
        }
        return out

    # ----- assembly ------------------------------------------------------------------------------------
    def policy_summary(self) -> list[dict[str, Any]]:
        ts = self.timeseries
        dec = self.decisions
        out = []
        for name in self._policies():
            rows = ts.loc[ts["policy"].astype(str) == name] if len(ts) else ts
            sub = dec.loc[dec["policy"].astype(str) == name] if len(dec) else dec
            lent = ~sub["declined"].to_numpy(dtype=bool) if len(sub) else np.zeros(0, dtype=bool)
            n = int(len(sub))
            cum_reward = float(sub["expected_reward"].sum()) if n else 0.0
            out.append(
                {
                    "policy": name,
                    "n_decisions": n,
                    "cum_reward": _f(cum_reward),
                    "cum_realized": _f(np.nansum(sub["realized_reward"].to_numpy(dtype=float))) if n else None,
                    "cum_regret": _f(rows["cum_regret"].iloc[-1]) if len(rows) else None,
                    "mean_reward": _f(cum_reward / n) if n else None,
                    "decline_rate": _f(sub["declined"].mean()) if n else None,
                    "mean_amount": _f(sub["amount"].to_numpy(dtype=float)[lent].mean()) if lent.any() else None,
                    "mean_rate": _f(sub["rate"].to_numpy(dtype=float)[lent].mean()) if lent.any() else None,
                }
            )
        return out

    def _table11(self, dims: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = []
        dp = dims["default_prediction"]
        prim = (dp.get("primary") or {}) if dp.get("available") else {}
        sec = (dp.get("secondary") or {}) if dp.get("available") else {}
        rows.append(
            [
                _metric("brier", "Brier score (elapsed-term holdout)", prim.get("brier"), "score", "lower",
                        baseline=prim.get("brier_climatology"), baseline_label="base-rate forecast"),
                _metric("auc", "ROC AUC (elapsed-term holdout)", prim.get("auc"), "ratio", "higher", baseline=0.5, baseline_label="random"),
                _metric("ece", "Expected calibration error", prim.get("ece"), "fraction", "lower"),
                _metric("log_loss", "Log loss", prim.get("log_loss"), "nats", "lower"),
                _metric("brier_all_resolved", "Brier score (all resolved, secondary)", sec.get("brier"), "score", "lower",
                        note="all resolved holdout loans; recent vintages skew to early defaults"),
                _metric("auc_all_resolved", "ROC AUC (all resolved, secondary)", sec.get("auc"), "ratio", "higher"),
                _metric("n_test", "Holdout loans scored (primary)", prim.get("n"), "count"),
            ]
        )
        rp = dims["recovery_prediction"]
        rprim = (rp.get("primary") or {}) if rp.get("available") else {}
        rows.append(
            [
                _metric("recovery_mae", "Recovery MAE", rprim.get("mae"), "fraction", "lower"),
                _metric("recovery_rmse", "Recovery RMSE", rprim.get("rmse"), "fraction", "lower"),
                _metric("recovery_n", "Defaults scored", rprim.get("n"), "count"),
            ]
        )
        cs = dims["credit_state_modeling"]
        hold = ((cs.get("holdout") or {}).get("all") or {}) if cs.get("available") else {}
        rows.append(
            [
                _metric("transition_log_likelihood", "Holdout transition log-likelihood per transition",
                        hold.get("log_likelihood_per_transition"), "nats", "higher"),
                _metric("transition_row_error", "Mean transient-row L1 error", hold.get("row_error_mean"), "fraction", "lower"),
                _metric("n_transitions", "Holdout transitions", hold.get("n_transitions"), "count"),
            ]
        )
        pp = dims["policy_performance"].get("by_policy", {}) if dims["policy_performance"].get("available") else {}
        hist_total = (pp.get("historical") or {}).get("total_expected_reward")
        rows.append(
            [
                _metric("total_expected_reward", "Total expected reward", None, "usd", "higher", baseline=hist_total,
                        baseline_label="historical policy", by_policy={k: v["total_expected_reward"] for k, v in pp.items()}),
                _metric("uplift_vs_historical", "Uplift vs historical", None, "usd", "higher",
                        by_policy={k: v["uplift_vs_historical"] for k, v in pp.items()}),
                _metric("total_realized_reward", "Realized reward (where observed)", None, "usd", "higher",
                        by_policy={k: v["total_realized_reward"] for k, v in pp.items()}),
                _metric("decline_rate", "Decline rate", None, "fraction", None, by_policy={k: v["decline_rate"] for k, v in pp.items()}),
            ]
        )
        bp = dims["bandit_performance"].get("by_policy", {}) if dims["bandit_performance"].get("available") else {}
        rows.append(
            [
                _metric("cumulative_regret", "Final cumulative pseudo-regret", None, "usd", "lower",
                        by_policy={k: v["final_cumulative_regret"] for k, v in bp.items()}),
                _metric("average_regret", "Average regret per decision", None, "usd_per_decision", "lower",
                        by_policy={k: v["average_regret"] for k, v in bp.items()}),
                _metric("selection_concentration", "Top-3 arm share (last 6 epochs)", None, "fraction", None,
                        by_policy={k: v["selection_concentration_top3"] for k, v in bp.items()}),
            ]
        )
        ce = dims["counterfactual_evaluation"]
        rows.append(
            [
                _metric("shadow_expected_profit", "Shadow expected profit", ce.get("expected_profit_shadow"), "usd", "higher",
                        baseline=ce.get("expected_profit_historical"), baseline_label="historical policy"),
                _metric("shadow_uplift", "Shadow uplift", ce.get("uplift"), "usd", "higher"),
                _metric("share_lendable", "Share lendable", ce.get("share_lendable"), "fraction", "higher"),
                _metric("rescued", "Rescued loans (historical loss → positive shadow offer)", (ce.get("counts") or {}).get("n_rescued"), "count"),
            ]
        )
        dm = dims["distributional_monitoring"]
        rows.append(
            [
                _metric("max_psi", "Max aggregate PSI", dm.get("max_aggregate_psi"), "score", None, baseline=0.25, baseline_label="breaker threshold"),
                _metric("epochs_watch", "Epochs at watch level", dm.get("epochs_watch"), "count"),
                _metric("epochs_breaker", "Epochs at breaker level", dm.get("epochs_breaker_level"), "count"),
                _metric("trips", "Breaker trips", dm.get("n_trips"), "count"),
            ]
        )
        cf = dims["computational_feasibility"]
        rows.append(
            [
                _metric("wall_seconds", "Pipeline wall time", cf.get("total_seconds") or cf.get("simulation_wall_seconds"), "seconds", "lower"),
                _metric("simulation_seconds", "Simulation wall time", cf.get("simulation_wall_seconds"), "seconds", "lower"),
                _metric("peak_memory", "Peak RSS", cf.get("peak_rss_mb"), "megabytes", "lower"),
                _metric("decisions_per_second", "Decisions per second", cf.get("decisions_per_second"), "ratio", "higher"),
            ]
        )
        ir = dims["infrastructure_resilience"]
        rt = ir.get("resume_test") or {}
        rows.append(
            [
                _metric("checkpoints", "Checkpoints written", ir.get("checkpoints_written"), "count"),
                _metric("restarts", "Restarts from checkpoint", ir.get("restarts"), "count"),
                _metric("resume_deterministic", "Kill-and-resume reproduces the run", None if not rt else float(bool(rt.get("passed"))),
                        "ratio", "higher", note=None if rt else "resume test not run"),
                _metric("recovery_seconds", "Recovery time", rt.get("recovery_seconds"), "seconds", "lower"),
            ]
        )
        return [
            {"key": key, "label": label, "axis": axis, "metrics": metrics}
            for (key, label, axis), metrics in zip(DIMENSIONS, rows, strict=True)
        ]

    def build(self) -> dict[str, Any]:
        dims = {
            "default_prediction": self.default_prediction(),
            "recovery_prediction": self.recovery_prediction(),
            "credit_state_modeling": self.credit_state_modeling(),
            "policy_performance": self.policy_performance(),
            "bandit_performance": self.bandit_performance(),
            "counterfactual_evaluation": self.counterfactual_evaluation(),
            "distributional_monitoring": self.distributional_monitoring(),
            "computational_feasibility": self.computational_feasibility(),
            "infrastructure_resilience": self.infrastructure_resilience(),
        }
        framework = self.config.get("framework") or {}
        simulation = self.config.get("simulation") or {}
        status = "final" if self.progress.get("status") == "completed" else "partial"
        report = {
            "generated_at": _now(),
            "status": status,
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "config_hashes": {
                "framework": _hash(framework) if framework else None,
                "simulation": _hash(simulation) if simulation else None,
                "run": _hash({"framework": framework, "simulation": simulation}) if framework or simulation else None,
            },
            "config": {"framework": framework, "simulation": simulation, "fit_dir": self.config.get("fit_dir")},
            "dataset": self.dataset_summary or {k: v for k, v in (self.config.get("dataset") or {}).items() if k != "summary"},
            "fit": {k: v for k, v in self.fit_metrics.items() if k != "predictive"},
            "dimensions": dims,
            "table11": self._table11(dims),
            "policy_summary": self.policy_summary(),
            "runtime": self.runtime,
        }
        return _clean(report)

    def to_json(self, path: str | os.PathLike[str] | None = None, indent: int = 2) -> str:
        text = json.dumps(self.build(), indent=indent, allow_nan=False, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text
