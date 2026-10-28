"""End-to-end pipeline functions shared by the CLI, ``scripts/run_pipeline.py`` and the
Django API (SPEC §2.12, §3, §6).

    df, summary = prepare_dataset("synthetic", "data/runs/x/dataset.parquet", n=200_000, seed=7)
    fitted      = fit_models(df, cfg, train_end="2015-01-01")       # Two-Stage Beta-Beta + MCTM
    fitted.save("data/runs/x/fit"); fitted = FittedModels.load("data/runs/x/fit")
    metrics     = evaluate_predictive(fitted, holdout_df, snapshot_date="2019-03-31")
    per_loan, portfolio = run_shadow_portfolio(fitted, df, max_loans=50_000, seed=7)
    result      = run_simulation(df, fitted, SimulationConfig(...), "data/runs/x/sim")
    report      = build_report("data/runs/x/sim", fitted=fitted, shadow_summary=portfolio)

**Calibration protocol [decision].** Among loans issued after ``train_end`` only those
resolved by the snapshot have a known outcome, and for recent vintages those skew heavily
towards early defaults (a loan issued in 2017 can have defaulted by 2019-03 but rarely
have repaid a 36-month term). Predictive metrics are therefore reported on two holdout
subsets: **primary** = resolved loans whose term has fully elapsed by the snapshot
(``issue_d + term months <= snapshot_date``; nearly complete outcome information) and
**secondary** = all resolved holdout loans (biased towards defaults; kept for
transparency), plus a per-vintage breakdown.
"""

from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FrameworkConfig, SimulationConfig
from .data import LendingClubLoader, load_loans, summarize
from .data.arms import ArmSpace
from .data.loader import STRING_DTYPE
from .data.outcomes import add_months, derive_outcomes, historical_arms
from .data.panel import LoanStatePanel
from .data.reference_rates import ReferenceRates
from .data.schema import OUTCOME_COLUMNS
from .data.segmentation import Segmenter
from .evaluation import predictive as pm
from .exceptions import DataSourceError
from .models.mctm import START_CURRENT, START_DELINQUENT, MarkovChainTransitionMatrix
from .models.two_stage import TwoStageBetaBeta
from .rewards.compound import CompoundRewardFunction
from .rewards.realization import InterestRealization, RealizationLookup
from .shadow.engine import RISK_AVERSION_SWEEP, PortfolioSummary, ShadowLendingEngine
from .simulation.events import jsonable

__all__ = [
    "FittedModels",
    "build_report",
    "elapsed_term_mask",
    "ensure_outcomes",
    "evaluate_predictive",
    "fit_models",
    "load_dataset",
    "peak_rss_mb",
    "prepare_dataset",
    "resilience_check",
    "run_shadow_portfolio",
    "run_simulation",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (``ru_maxrss`` is KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / (1024.0 * 1024.0) if os.uname().sysname == "Darwin" else float(rss) / 1024.0


def _write_json(path: Path, obj: Any) -> None:
    from .simulation.store import atomic_write_json

    atomic_write_json(path, obj, fsync=False)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ======================================================================================
# Data
# ======================================================================================
def ensure_outcomes(
    df: pd.DataFrame,
    cfg: FrameworkConfig | None = None,
    *,
    arm_space: ArmSpace | None = None,
    segmenter: Segmenter | None = None,
) -> pd.DataFrame:
    """Return ``df`` with ``y_*`` outcome columns consistent with ``cfg``.

    Missing outcome columns are derived (:func:`derive_outcomes`); when they exist,
    ``y_group`` and ``y_historical_arm`` are recomputed with the given segmenter / arm space
    (cheap), so a dataset prepared under one configuration can be fitted under another.
    """
    cfg = cfg or FrameworkConfig()
    arm_space = arm_space or ArmSpace.from_config(cfg.arms)
    segmenter = segmenter or Segmenter.from_config(cfg.segmentation)
    if any(c not in df.columns for c in OUTCOME_COLUMNS):
        return derive_outcomes(df, arm_space, segmenter, cfg.data)
    out = df.reset_index(drop=True)
    groups = segmenter.assign(out).reset_index(drop=True)
    arms = historical_arms(out, arm_space)
    return out.assign(y_group=groups.astype(STRING_DTYPE).to_numpy(), y_historical_arm=arms)


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a LoanTable parquet written by :func:`prepare_dataset` (dtypes restored)."""
    return LendingClubLoader.from_parquet(path)


def prepare_dataset(
    source: str,
    out_parquet: str | os.PathLike[str] | None = None,
    *,
    cfg: FrameworkConfig | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest a LoanTable with outcomes and (optionally) write it to parquet.

    ``source`` is ``"synthetic"`` or ``"lending_club"``; ``kwargs`` go to
    :func:`bayesian_credit.data.load_loans` (``n``, ``seed``, ``shift_at``, ``shift`` /
    ``max_rows``, ``sample_fraction``, ``issue_from``, ``order`` ...). For the real file a
    ``max_rows`` subset defaults to ``order="random"`` (the loader's ``"issue_d"`` default
    would keep only the earliest loans and leave the 2015+ stream empty).
    """
    cfg = cfg or FrameworkConfig()
    t0 = time.perf_counter()
    kw = {k: v for k, v in kwargs.items() if v is not None}
    src = source.lower().replace("-", "_")
    if src in ("lending_club", "lc", "kaggle") and "max_rows" in kw:
        kw.setdefault("order", "random")
    df = load_loans(
        source,
        with_outcomes=True,
        arm_space=ArmSpace.from_config(cfg.arms),
        segmenter=Segmenter.from_config(cfg.segmentation),
        data_cfg=cfg.data,
        **kw,
    )
    summary = summarize(df, cfg.data)
    summary.update(
        {
            "source": source,
            "params": jsonable(kw),
            "n_columns": int(df.shape[1]),
            "load_seconds": round(time.perf_counter() - t0, 3),
            "path": None,
        }
    )
    if "synthetic" in df.attrs:
        summary["synthetic"] = jsonable(df.attrs["synthetic"])
    if out_parquet is not None:
        path = LendingClubLoader.to_parquet(df, out_parquet)
        summary["path"] = str(path)
    return df, summary


def elapsed_term_mask(df: pd.DataFrame, snapshot_date: Any) -> np.ndarray:
    """``issue_d + term months <= snapshot_date`` (the loan's full term is observable)."""
    end = add_months(df["issue_d"].reset_index(drop=True), df["term"].to_numpy(dtype=float))
    return (end <= pd.Timestamp(snapshot_date)).to_numpy(dtype=bool)


# ======================================================================================
# Fitted models
# ======================================================================================
@dataclass
class FittedModels:
    """Everything fitted on the training window: risk model, MCTM and decision context.

    ``reference_rates`` (training mean ``int_rate`` by sub-grade) anchors the rate-corridor
    guardrail of the arm space; it is ``None`` for fits saved before it existed, in which
    case the corridor is not applied (the other guardrails are). ``interest_realization``
    (prepayment: realised share of the scheduled interest by grade and term) feeds the
    reward's ``realization`` factor; ``None`` (older fits) means the contractual schedule.
    """

    two_stage: TwoStageBetaBeta
    mctm: MarkovChainTransitionMatrix
    segmenter: Segmenter
    arm_space: ArmSpace
    reward_fn: CompoundRewardFunction
    cfg: FrameworkConfig
    train_end: str
    metrics: dict[str, Any] = field(default_factory=dict)
    fitted_at: str = field(default_factory=_now)
    calibration: dict[str, Any] = field(default_factory=dict)
    mctm_summary: dict[str, Any] = field(default_factory=dict)
    reference_rates: ReferenceRates | None = None
    interest_realization: InterestRealization | None = None

    # ----- persistence ------------------------------------------------------------------------
    FILES = (
        "two_stage.json",
        "mctm.json",
        "segmenter.json",
        "arm_space.json",
        "config.json",
        "metrics.json",
        "hierarchy_summary.parquet",
        "recovery_summary.parquet",
        "calibration.json",
        "mctm_summary.json",
        "reference_rates.json",
        "interest_realization.json",
    )

    @property
    def policy_effect(self) -> Any:
        """The Stage-1 :class:`~bayesian_credit.models.policy_effect.PolicyEffectModel` (or ``None``)."""
        return self.two_stage.policy_effect

    def realization_lookup(self) -> RealizationLookup:
        """``(group, arm) -> ρ`` under this fit's reward config, table, arm space and groups
        (memoised; rebuilt when any of them is replaced)."""
        key = (self.cfg.reward.prepayment_adjustment, id(self.interest_realization), id(self.arm_space), id(self.segmenter))
        cached = self.__dict__.get("_realization_cache")
        if cached is None or cached[0] != key:
            cached = (key, RealizationLookup.from_config(self.cfg.reward, self.interest_realization, self.arm_space,
                                                         self.segmenter))
            self.__dict__["_realization_cache"] = cached
        return cached[1]

    def realization(self, group: str, arm: int | None = None, *, term: int | None = None) -> float:
        """Prepayment factor ρ of ``(group, arm)`` - or of ``(group, term)`` for a policy off
        the grid - to pass as ``reward_fn.expected(..., realization=ρ)`` (1.0 when disabled)."""
        lookup = self.realization_lookup()
        if term is not None:
            return lookup.for_term(group, term)
        return 1.0 if arm is None else lookup(group, int(arm))

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Write the fit to ``directory`` (JSON + parquet; see :attr:`FILES` -
        ``reference_rates.json`` only when the fit has reference rates)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        _write_json(d / "two_stage.json", self.two_stage.to_dict())
        _write_json(d / "mctm.json", self.mctm.to_dict())
        _write_json(d / "segmenter.json", self.segmenter.to_dict())
        _write_json(d / "arm_space.json", self.arm_space.to_dict())
        _write_json(
            d / "config.json", {"framework": self.cfg.to_dict(), "train_end": self.train_end, "fitted_at": self.fitted_at}
        )
        _write_json(d / "metrics.json", self.metrics)
        self.two_stage.default_model.summary().to_parquet(d / "hierarchy_summary.parquet", index=False)
        self.two_stage.recovery_model.summary().to_parquet(d / "recovery_summary.parquet", index=False)
        _write_json(d / "calibration.json", self.calibration)
        _write_json(d / "mctm_summary.json", self.mctm_summary or mctm_summary(self.mctm))
        if self.reference_rates is not None:
            _write_json(d / "reference_rates.json", self.reference_rates.to_dict())
        if self.interest_realization is not None:
            _write_json(d / "interest_realization.json", self.interest_realization.to_dict())
        return d

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> FittedModels:
        d = Path(directory)
        if not (d / "two_stage.json").is_file():
            raise DataSourceError(f"{d} does not contain a saved fit (two_stage.json missing)")
        conf = _read_json(d / "config.json")
        cfg = FrameworkConfig.from_dict(conf["framework"])
        opt = {}
        for name in ("metrics", "calibration", "mctm_summary"):
            p = d / f"{name}.json"
            opt[name] = _read_json(p) if p.is_file() else {}
        ref_path = d / "reference_rates.json"
        ref = ReferenceRates.from_dict(_read_json(ref_path)) if ref_path.is_file() else None
        rho_path = d / "interest_realization.json"
        rho = InterestRealization.from_dict(_read_json(rho_path)) if rho_path.is_file() else None
        return cls(
            two_stage=TwoStageBetaBeta.from_dict(_read_json(d / "two_stage.json")),
            mctm=MarkovChainTransitionMatrix.from_dict(_read_json(d / "mctm.json")),
            segmenter=Segmenter.from_dict(_read_json(d / "segmenter.json")),
            arm_space=ArmSpace.from_dict(_read_json(d / "arm_space.json")),
            reward_fn=CompoundRewardFunction(cfg.reward),
            cfg=cfg,
            train_end=str(conf.get("train_end", cfg.train_end)),
            metrics=opt["metrics"],
            fitted_at=str(conf.get("fitted_at", _now())),
            calibration=opt["calibration"],
            mctm_summary=opt["mctm_summary"],
            reference_rates=None if ref is None or ref.is_empty else ref,
            interest_realization=None if rho is None or rho.is_empty else rho,
        )

    def shadow_engine(self, n_samples: int = 2000, seed: int = 0) -> ShadowLendingEngine:
        return ShadowLendingEngine.from_fitted(self, n_samples=n_samples, seed=seed)


def mctm_summary(mctm: MarkovChainTransitionMatrix, horizon: int = 60) -> dict[str, Any]:
    """JSON view of the chain for the API (``/fits/{id}/mctm/``)."""
    out: dict[str, Any] = {
        "states": ["Current", "Delinquent", "Recovered", "Absorbed Loss"],
        "structure": mctm.structure.value,
        "prior_strength": mctm.prior_strength,
        "matrix": mctm.matrix.tolist(),
        "counts": mctm.counts.tolist(),
        "n_transitions": mctm.n_transitions,
        "horizon": horizon,
    }
    try:
        out["absorption_probabilities"] = mctm.absorption_probabilities().tolist()
        out["expected_time_to_absorption"] = mctm.expected_time_to_absorption().tolist()
    except Exception as exc:  # noqa: BLE001 - singular (I - Q) is reported, not fatal
        out["absorption_error"] = str(exc)
    out["trajectory_current"] = mctm.trajectory(START_CURRENT, horizon).tolist()
    out["trajectory_delinquent"] = mctm.trajectory(START_DELINQUENT, horizon).tolist()
    out["p_default_curve"] = mctm.p_default_curve(horizon).tolist()
    out["default_timing_cdf"] = {str(t): mctm.default_timing_cdf(t).tolist() for t in (36, 60)}
    out["p_loss_given_state"] = {
        "current_36": mctm.p_loss_given_state(0, 36),
        "current_60": mctm.p_loss_given_state(0, 60),
        "delinquent_36": mctm.p_loss_given_state(1, 36),
    }
    return jsonable(out)


def fit_models(
    df: pd.DataFrame,
    cfg: FrameworkConfig | None = None,
    train_end: Any = None,
    *,
    evaluate: bool = True,
    snapshot_date: Any = None,
) -> FittedModels:
    """Fit the Two-Stage Beta-Beta model and the MCTM on ``issue_d < train_end``.

    The Stage-1/Stage-2 posteriors use the training loans resolved by the snapshot (with the
    structured leaf prior of :class:`~bayesian_credit.models.policy_effect.PolicyEffectModel`
    when ``cfg.hierarchy.policy_effect``); the MCTM is fitted on the reconstructed monthly
    state panel of the training loans; the reference rates (mean ``int_rate`` by sub-grade)
    on all training loans; the interest-realization table (prepayment) on the fully-paid
    training loans whose term elapsed by the snapshot. With ``evaluate`` the holdout
    (``issue_d >= train_end``) is scored by :func:`evaluate_predictive` and the results are
    stored in ``metrics`` / ``calibration`` / ``mctm_summary``.
    """
    cfg = cfg or FrameworkConfig()
    train_end = str(train_end if train_end is not None else cfg.train_end)
    snapshot = snapshot_date if snapshot_date is not None else cfg.data.snapshot_date
    t0 = time.perf_counter()
    arm_space = ArmSpace.from_config(cfg.arms)
    segmenter = Segmenter.from_config(cfg.segmentation)
    data = ensure_outcomes(df, cfg, arm_space=arm_space, segmenter=segmenter)
    cut = pd.Timestamp(train_end)
    train_mask = (data["issue_d"] < cut).to_numpy()
    train = data.loc[train_mask]
    if len(train) == 0:
        raise DataSourceError(f"no training loans issued before {train_end}")
    two_stage = TwoStageBetaBeta.from_config(cfg, arm_space).fit(train, snapshot_date=snapshot)
    panel = LoanStatePanel.from_table(train, cfg.mctm, snapshot)
    mctm = MarkovChainTransitionMatrix.from_config(cfg.mctm).fit_counts(panel.counts)
    reference_rates = ReferenceRates.fit(train) if "int_rate" in train.columns else None
    realization = InterestRealization.fit(train, snapshot_date=snapshot) if "total_rec_int" in train.columns else None
    if realization is not None and realization.is_empty:
        realization = None
    fit_seconds = time.perf_counter() - t0
    fitted = FittedModels(
        two_stage=two_stage,
        mctm=mctm,
        segmenter=segmenter,
        arm_space=arm_space,
        reward_fn=CompoundRewardFunction(cfg.reward),
        cfg=cfg,
        train_end=train_end,
        reference_rates=reference_rates,
        interest_realization=realization,
    )
    metrics: dict[str, Any] = {
        "train_end": train_end,
        "snapshot_date": str(snapshot),
        "n_train": int(len(train)),
        "n_train_resolved": int(train["y_resolved"].sum()),
        "n_holdout": int((~train_mask).sum()),
        "fit_seconds": round(fit_seconds, 3),
        "hierarchy": jsonable(two_stage.default_model.diagnostics),
        "recovery": jsonable(two_stage.recovery_model.diagnostics),
        "policy_effect": jsonable(two_stage.policy_effect_info),
        "reference_rates": None if reference_rates is None else jsonable(
            {"n_fit": reference_rates.n_fit, "by_grade": reference_rates.by_grade, "global": reference_rates.global_rate}
        ),
        "guardrails": jsonable(arm_space.guardrails),
        "interest_realization": None if realization is None else jsonable(
            {**realization.summary(), "prepayment_adjustment": cfg.reward.prepayment_adjustment}
        ),
        "mctm": {
            "n_transitions": mctm.n_transitions,
            "train_log_likelihood": mctm.log_likelihood(panel.counts),
            "train_log_likelihood_per_transition": pm.transition_log_likelihood(mctm.matrix, panel.counts, normalize=True),
        },
    }
    fitted.metrics = metrics
    summ = mctm_summary(mctm)
    if evaluate and (~train_mask).any():
        t1 = time.perf_counter()
        pred = evaluate_predictive(fitted, data.loc[~train_mask], snapshot)
        fitted.calibration = pred.pop("calibration", {})
        metrics["predictive"] = pred
        metrics["evaluate_seconds"] = round(time.perf_counter() - t1, 3)
        metrics.update(_headline_metrics(pred, metrics))
        cs = pred.get("credit_state", {}).get("all", {})
        summ["holdout_log_likelihood"] = cs.get("log_likelihood")
        summ["holdout_row_prediction_error"] = cs.get("row_error")
    fitted.mctm_summary = summ
    return fitted


def _headline_metrics(pred: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Flat headline numbers for the API's fit list (``/fits/{id}/``)."""
    primary = pred.get("default", {}).get("primary", {})
    rec = pred.get("recovery", {}).get("primary", {})
    cs = pred.get("credit_state", {}).get("all", {})
    return {
        "brier": primary.get("brier"),
        "auc": primary.get("auc"),
        "ece": primary.get("ece"),
        "log_loss": primary.get("log_loss"),
        "recovery_mae": rec.get("mae"),
        "recovery_rmse": rec.get("rmse"),
        "transition_log_likelihood": cs.get("log_likelihood_per_transition"),
        "n_test": primary.get("n"),
    }


# ======================================================================================
# Predictive evaluation
# ======================================================================================
def _default_block(y: np.ndarray, p: np.ndarray, with_tables: bool = False) -> dict[str, Any]:
    keep = np.isfinite(y) & np.isfinite(p)
    yy, pp = y[keep], p[keep]
    n = int(yy.size)
    out: dict[str, Any] = {"n": n, "n_defaults": int(yy.sum()) if n else 0}
    if n == 0:
        out.update(base_rate=None, mean_pred=None, brier=None, brier_climatology=None, brier_skill=None, auc=None, ece=None, log_loss=None)
        return out
    base = float(yy.mean())
    brier = pm.brier(yy, pp)
    clim = float(np.mean((base - yy) ** 2))
    ece, table = pm.expected_calibration_error(yy, pp, 10)
    out.update(
        base_rate=base,
        mean_pred=float(pp.mean()),
        brier=brier,
        brier_climatology=clim,
        brier_skill=(1.0 - brier / clim) if clim > 0 else None,
        auc=pm.roc_auc(yy, pp),
        ece=ece,
        log_loss=pm.log_loss(yy, pp),
    )
    if with_tables:
        out["reliability"] = jsonable(table.to_dict("records"))
        fpr, tpr, thr = pm.roc_curve(yy, pp)
        if fpr.size > 201:
            pick = np.unique(np.linspace(0, fpr.size - 1, 201).round().astype(int))
            fpr, tpr, thr = fpr[pick], tpr[pick], thr[pick]
        out["roc"] = [{"fpr": float(a), "tpr": float(b), "threshold": None if not np.isfinite(c) else float(c)} for a, b, c in zip(fpr, tpr, thr, strict=True)]
    return jsonable(out)


def _recovery_block(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    keep = np.isfinite(y) & np.isfinite(p)
    if not keep.any():
        return {"n": 0, "mae": None, "rmse": None, "mean_actual": None, "mean_pred": None}
    return jsonable(
        {
            "n": int(keep.sum()),
            "mae": pm.mae(y[keep], p[keep]),
            "rmse": pm.rmse(y[keep], p[keep]),
            "mean_actual": float(y[keep].mean()),
            "mean_pred": float(p[keep].mean()),
        }
    )


def _transition_block(P: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    row_err = pm.transition_row_error(P, counts)
    return jsonable(
        {
            "n_transitions": float(counts.sum()),
            "log_likelihood": pm.transition_log_likelihood(P, counts),
            "log_likelihood_per_transition": pm.transition_log_likelihood(P, counts, normalize=True),
            "row_error": row_err[:2].tolist(),
            "row_error_mean": float(np.nanmean(row_err[:2])) if np.isfinite(row_err[:2]).any() else None,
            "counts": counts.tolist(),
        }
    )


def evaluate_predictive(fitted: FittedModels, holdout_df: pd.DataFrame, snapshot_date: Any = None) -> dict[str, Any]:
    """Table 11 predictive metrics of a fit on a holdout LoanTable (with ``y_*`` columns).

    * ``default``: Brier (+ climatology and skill), AUC, ECE (+ reliability bins), log loss,
      base rate and mean prediction for ``primary`` (resolved, elapsed term),
      ``secondary`` (all resolved) and ``by_vintage`` (issue year, all resolved; with the
      elapsed share);
    * ``recovery``: MAE / RMSE of ``E[R]`` vs realized recovery of defaulted loans
      (``primary`` elapsed term, ``secondary`` all);
    * ``credit_state``: holdout transition log-likelihood (total and per transition), per-row
      L1 error and counts of the reconstructed panel (``all`` and ``elapsed``);
    * ``calibration``: reliability table + ROC points of the primary subset (popped into
      ``calibration.json`` by :func:`fit_models`).
    """
    snapshot = snapshot_date if snapshot_date is not None else fitted.cfg.data.snapshot_date
    df = ensure_outcomes(holdout_df, fitted.cfg, arm_space=fitted.arm_space, segmenter=fitted.segmenter)
    pred = fitted.two_stage.predict_table(df)
    y = df["y_default"].to_numpy(dtype=float, na_value=np.nan)
    p = pred["p_default_mean"].to_numpy(dtype=float)
    elapsed = elapsed_term_mask(df, snapshot)
    resolved = np.isfinite(y)
    out: dict[str, Any] = {
        "snapshot_date": str(snapshot),
        "n_holdout": int(len(df)),
        "n_resolved": int(resolved.sum()),
        "n_elapsed": int(elapsed.sum()),
        "n_elapsed_resolved": int((elapsed & resolved).sum()),
        "resolved_share": float(resolved.mean()) if len(df) else None,
    }
    primary = _default_block(np.where(elapsed, y, np.nan), p, with_tables=True)
    secondary = _default_block(y, p, with_tables=True)
    by_vintage: dict[str, Any] = {}
    years = df["issue_d"].dt.year.to_numpy()
    for yr in np.unique(years):
        m = years == yr
        blk = _default_block(np.where(m, y, np.nan), p)
        blk["n_loans"] = int(m.sum())
        blk["elapsed_share"] = float(elapsed[m].mean())
        blk["resolved_share"] = float(resolved[m].mean())
        by_vintage[str(int(yr))] = blk
    out["default"] = {
        "primary": {"subset": "resolved & term elapsed by snapshot", **{k: v for k, v in primary.items() if k not in ("reliability", "roc")}},
        "secondary": {"subset": "all resolved", **{k: v for k, v in secondary.items() if k not in ("reliability", "roc")}},
        "by_vintage": by_vintage,
    }
    rr = df["y_recovery_rate"].to_numpy(dtype=float, na_value=np.nan)
    er = pred["expected_recovery"].to_numpy(dtype=float)
    dflt = y == 1.0
    out["recovery"] = {
        "primary": {"subset": "defaults, term elapsed", **_recovery_block(np.where(dflt & elapsed, rr, np.nan), er)},
        "secondary": {"subset": "all resolved defaults", **_recovery_block(np.where(dflt, rr, np.nan), er)},
    }
    panel_cfg = fitted.cfg.mctm
    counts_all = LoanStatePanel.from_table(df, panel_cfg, snapshot).counts
    counts_el = LoanStatePanel.from_table(df.loc[elapsed], panel_cfg, snapshot).counts if elapsed.any() else np.zeros((4, 4))
    out["credit_state"] = {
        "all": _transition_block(fitted.mctm.matrix, counts_all),
        "elapsed": _transition_block(fitted.mctm.matrix, counts_el),
        "structure": fitted.mctm.structure.value,
    }
    out["calibration"] = {
        "primary": {"reliability": primary.get("reliability", []), "roc": primary.get("roc", [])},
        "secondary": {"reliability": secondary.get("reliability", []), "roc": secondary.get("roc", [])},
    }
    return jsonable(out)


# ======================================================================================
# Shadow portfolio
# ======================================================================================
def run_shadow_portfolio(
    fitted: FittedModels,
    df: pd.DataFrame,
    max_loans: int | None = None,
    seed: int = 0,
    *,
    n_samples: int = 2000,
    holdout_only: bool = True,
    risk_aversions: tuple[float, ...] | list[float] | None = RISK_AVERSION_SWEEP,
) -> tuple[pd.DataFrame, PortfolioSummary]:
    """Shadow engine over the post-``train_end`` loans (or all loans with
    ``holdout_only=False``); see :meth:`ShadowLendingEngine.evaluate_portfolio`
    (``risk_aversions`` = the λ values of the ``by_risk_aversion`` sweep)."""
    data = ensure_outcomes(df, fitted.cfg, arm_space=fitted.arm_space, segmenter=fitted.segmenter)
    if holdout_only:
        data = data.loc[data["issue_d"] >= pd.Timestamp(fitted.train_end)]
    engine = fitted.shadow_engine(n_samples=n_samples, seed=seed)
    return engine.evaluate_portfolio(data, max_loans=max_loans, seed=seed, risk_aversions=risk_aversions)


# ======================================================================================
# Simulation & report
# ======================================================================================
def run_simulation(
    df: pd.DataFrame,
    fitted: FittedModels | None,
    sim_cfg: SimulationConfig,
    run_dir: str | os.PathLike[str],
    *,
    cfg: FrameworkConfig | None = None,
    dataset_path: str | os.PathLike[str] | None = None,
    dataset_info: dict[str, Any] | None = None,
    fit_dir: str | os.PathLike[str] | None = None,
    progress_callback: Any = None,
    **runner_kwargs: Any,
) -> Any:
    """Prepare ``run_dir`` and run the replay simulation to completion (or pause/cancel).

    Returns the :class:`~bayesian_credit.simulation.runner.SimulationResult`.
    """
    from .simulation.runner import SimulationRunner

    framework = cfg or (fitted.cfg if fitted is not None else FrameworkConfig())
    runner = SimulationRunner(
        df,
        framework,
        sim_cfg,
        run_dir,
        fitted=fitted,
        progress_callback=progress_callback,
        dataset_path=dataset_path,
        dataset_info=dataset_info,
        fit_dir=fit_dir,
        **runner_kwargs,
    )
    return runner.run()


def build_report(
    run_dir: str | os.PathLike[str],
    *,
    fitted: FittedModels | None = None,
    predictive: dict[str, Any] | None = None,
    shadow_summary: PortfolioSummary | dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    resilience: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Assemble the Table 11 :class:`~bayesian_credit.evaluation.report.EvaluationReport`
    for a run directory (and write ``report.json``)."""
    from .evaluation.report import EvaluationReport

    report = EvaluationReport.from_run_dir(
        run_dir,
        fitted=fitted,
        predictive=predictive,
        shadow_summary=shadow_summary,
        timings=timings,
        resilience=resilience,
        dataset_summary=dataset_summary,
    )
    out = report.build()
    if write:
        from .simulation.store import RunDirectory

        RunDirectory(run_dir).write_report(out)
    return out


def resilience_check(
    df: pd.DataFrame,
    fitted: FittedModels,
    sim_cfg: SimulationConfig,
    workdir: str | os.PathLike[str],
    *,
    n_epochs: int = 8,
    kill_after: int = 4,
    checkpoint_every: int = 3,
    max_loans: int = 2_000,
    dataset_path: str | os.PathLike[str] | None = None,
    fit_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """G.6 state-recovery experiment in miniature: run a short simulation twice - once
    uninterrupted, once "killed" after ``kill_after + 1`` epochs (an exception raised in the
    progress callback) and resumed from its last checkpoint - and compare the decision logs
    and time series bit for bit."""
    import dataclasses
    import shutil

    from .simulation.runner import SimulationRunner

    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    start = pd.Timestamp(sim_cfg.start)
    issue = df["issue_d"]
    months = sorted(issue.loc[issue >= start].dt.to_period("M").unique())
    if len(months) < n_epochs:
        n_epochs = len(months)
    end = months[n_epochs - 1].to_timestamp(how="end").normalize()
    small = dataclasses.replace(
        sim_cfg, end=str(end.date()), max_loans=max_loans, checkpoint_every=checkpoint_every, breaker_mode="auto_ack"
    )

    class _Kill(RuntimeError):
        pass

    def killer(info: dict[str, Any]) -> None:
        if info["epoch_index"] == kill_after:
            raise _Kill("simulated SIGKILL")

    t0 = time.perf_counter()
    common = {"fitted": fitted, "install_signal_handlers": False, "dataset_path": dataset_path, "fit_dir": fit_dir}
    SimulationRunner(df, fitted.cfg, small, work / "uninterrupted", **common).run()
    killed = work / "killed"
    killed_at = None
    try:
        SimulationRunner(df, fitted.cfg, small, killed, progress_callback=killer, **common).run()
    except _Kill:
        killed_at = kill_after
    t1 = time.perf_counter()
    SimulationRunner.resume(killed, install_signal_handlers=False)
    recovery_seconds = time.perf_counter() - t1
    from .simulation.store import RunDirectory

    a, b = RunDirectory(work / "uninterrupted"), RunDirectory(killed)
    same_dec = _frames_equal(a.read_decisions(), b.read_decisions())
    same_ts = _frames_equal(a.read_timeseries(), b.read_timeseries())
    return {
        "passed": bool(same_dec and same_ts and killed_at is not None),
        "decisions_identical": bool(same_dec),
        "timeseries_identical": bool(same_ts),
        "n_epochs": int(n_epochs),
        "killed_after_epoch_index": killed_at,
        "checkpoint_every": int(checkpoint_every),
        "max_loans": int(max_loans),
        "recovery_seconds": round(recovery_seconds, 3),
        "total_seconds": round(time.perf_counter() - t0, 3),
    }


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError:
        return False
    return True

