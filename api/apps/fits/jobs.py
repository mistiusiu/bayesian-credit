"""Fit and shadow-portfolio jobs (call the core pipeline; never re-implement fitting)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from bayesian_credit.config import FrameworkConfig
from bayesian_credit.data.outcomes import recovery_rate
from bayesian_credit.pipeline import fit_models, load_dataset, run_shadow_portfolio
from django.conf import settings

from apps.common import artifacts
from apps.common.jobs import ProgressReporter
from apps.common.models import Job
from apps.common.utils import clean, epoch_end, epoch_start, write_json_atomic

from . import services
from .models import Fit


def run_fit(job: Job, reporter: ProgressReporter) -> None:
    fit = Fit.objects.select_related("dataset").get(pk=job.target_id)
    Fit.objects.filter(pk=fit.pk).update(status=Fit.Status.RUNNING)
    cfg = FrameworkConfig.from_dict(fit.framework)
    t_job = time.perf_counter()
    reporter.log(f"loading dataset #{fit.dataset_id} ({fit.dataset.parquet.name})", 0.05)
    df = load_dataset(fit.dataset.parquet)
    if cfg.recovery.basis != "principal":
        rr = recovery_rate(df, cfg.recovery.basis)
        df = df.assign(
            y_recovery_rate=np.where(df["y_default"].to_numpy(dtype=float, na_value=np.nan) == 1.0, rr, np.nan)
        )
    reporter.log(
        f"fitting Two-Stage Beta-Beta + MCTM ({cfg.mctm.structure.value}) on {len(df):,} loans, "
        f"train_end={cfg.train_end}, groups={'×'.join(cfg.segmentation.dimensions)}",
        0.15,
    )
    fitted = fit_models(df, cfg, train_end=cfg.train_end)
    reporter.log(
        f"fitted in {fitted.metrics.get('fit_seconds', 0):.1f}s; holdout scored in {fitted.metrics.get('evaluate_seconds', 0) or 0:.1f}s",
        0.7,
    )
    directory = artifacts.fit_dir(fit.pk)
    fitted.save(directory)
    Fit.objects.filter(pk=fit.pk).update(fit_dir=str(directory))
    fit.fit_dir = str(directory)
    metrics = services.contract_metrics(fitted)
    reporter.log("precomputing arm table, calibration and MCTM views", 0.8)
    try:  # warm the API caches so the first UI request is instant (failures are not fatal)
        services._FITTED.put(services._fit_key(fit), fitted)
        services._historical_counts(fit, fitted)
        services._holdout_recovery(fit, fitted)
    except Exception as exc:  # noqa: BLE001
        reporter.log(f"warning: cache warm-up failed: {exc}")
    metrics["job_seconds"] = round(time.perf_counter() - t_job, 3)
    Fit.objects.filter(pk=fit.pk).update(
        status=Fit.Status.READY,
        metrics=metrics,
        n_train=metrics.get("n_train"),
        n_test=metrics.get("n_test"),
        n_groups=len(fitted.two_stage.default_model.groups),
        n_arms=int(fitted.arm_space.n_arms),
        fit_seconds=metrics.get("fit_seconds"),
    )
    reporter.log(
        f"fit ready: AUC {metrics.get('auc') or float('nan'):.3f}, Brier {metrics.get('brier') or float('nan'):.4f}, "
        f"{len(fitted.two_stage.default_model.groups)} groups, {fitted.arm_space.n_arms} arms",
        1.0,
    )


def on_failure(job: Job, exc: BaseException) -> None:
    Fit.objects.filter(pk=job.target_id).update(status=Fit.Status.FAILED)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col in ("grade", "purpose", "home_ownership"):
        values = filters.get(col)
        if values:
            mask &= df[col].astype(str).isin([str(v) for v in values]).to_numpy()
    if filters.get("issue_from"):
        mask &= (df["issue_d"] >= pd.Timestamp(epoch_start(filters["issue_from"]))).to_numpy()
    if filters.get("issue_to"):
        mask &= (df["issue_d"] <= pd.Timestamp(epoch_end(filters["issue_to"]))).to_numpy()
    if filters.get("resolved_only"):
        mask &= df["y_resolved"].fillna(False).astype(bool).to_numpy()
    return df.loc[mask].reset_index(drop=True)


def run_portfolio(job: Job, reporter: ProgressReporter) -> None:
    fit = Fit.objects.select_related("dataset").get(pk=job.target_id)
    request = (job.params or {}).get("request") or {}
    fitted = services.load_fitted(fit)
    reporter.log(f"loading dataset #{fit.dataset_id}", 0.05)
    df = load_dataset(fit.dataset.parquet)
    df = apply_filters(df, request.get("filters") or {})
    reporter.log(f"evaluating every arm for up to {request.get('max_loans')} of {len(df):,} filtered loans", 0.2)
    per_loan, summary = run_shadow_portfolio(
        fitted,
        df,
        max_loans=request.get("max_loans"),
        seed=int(request.get("seed", 42)),
        n_samples=int(settings.SHADOW_SAMPLES),
    )
    out = artifacts.reset_dir(artifacts.portfolio_dir(fit.pk, job.pk))
    per_loan.to_parquet(out / "per_loan.parquet", index=False)
    write_json_atomic(out / "summary.json", summary.to_dict())
    write_json_atomic(out / "request.json", clean(request))
    s = summary.to_dict()
    reporter.log(
        f"portfolio done: {s['n_loans']:,} loans, uplift ${s['uplift'] or 0:,.0f}, lendable {s['share_lendable'] or 0:.1%}",
        1.0,
    )
