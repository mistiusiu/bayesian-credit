"""Fit read side: loading saved fits (per-process LRU), hierarchy rows, risk estimates,
calibration, MCTM views, arm tables and the shadow engine.

Everything is computed from the artefacts written by ``FittedModels.save`` plus the dataset
parquet; derived views that need a pass over the data (calibration extras, historical arm
counts, MCTM panels for the non-fitted structure) are cached as ``api_*.json`` in the fit
directory so they are computed once per fit.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from bayesian_credit.config import FrameworkConfig, MCTMConfig
from bayesian_credit.data.arms import ArmSpace, LendingPolicy
from bayesian_credit.data.outcomes import recovery_rate
from bayesian_credit.data.panel import LoanStatePanel
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS, VARIABLES_BY_NAME, BorrowerContext, LeakageGuard
from bayesian_credit.models.beta import BetaPosterior
from bayesian_credit.models.mctm import MarkovChainTransitionMatrix, allowed_transitions
from bayesian_credit.types import Stage
from django.conf import settings
from rest_framework.exceptions import NotFound, ValidationError
from scipy import stats

from apps.common import cache
from apps.common.exceptions import Conflict
from apps.common.utils import clean, epoch_start, num, read_json, write_json_atomic
from apps.datasets.services import context_from_row, outcome_from_row, read_loans_by_id

from .models import Fit

logger = logging.getLogger("apps.fits")

STATES = ["current", "delinquent", "recovered", "loss"]
STATE_LABELS = ["Current", "Delinquent", "Recovered", "Absorbed Loss"]
GRID = 101


# ============================================================================ loading
class _LRU:
    def __init__(self, size: int) -> None:
        self.size = max(int(size), 1)
        self.data: OrderedDict[Any, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: Any) -> Any:
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.data.clear()


_FITTED = _LRU(getattr(settings, "FIT_CACHE_SIZE", 4))
_ENGINES = _LRU(getattr(settings, "FIT_CACHE_SIZE", 4))


def _fit_key(fit: Fit) -> tuple:
    d = fit.directory
    return (fit.pk, cache.file_key(d / "two_stage.json", d / "mctm.json", d / "config.json"))


def require_ready(fit: Fit) -> Fit:
    from apps.common.jobs import reap_lost_workers

    if fit.job is not None and fit.job.is_active:
        reap_lost_workers([fit.job])
        fit.job.refresh_from_db()
    if not fit.is_ready:
        raise Conflict(f"fit {fit.pk} is not ready (status {fit.current_status})")
    if not (fit.directory / "two_stage.json").is_file():
        raise Conflict(f"fit {fit.pk} has no saved artefacts in {fit.directory}")
    return fit


def load_fitted(fit: Fit):
    """``FittedModels`` of a fit, loaded once per process (LRU keyed by fit id + file mtimes)."""
    from bayesian_credit.pipeline import FittedModels

    key = _fit_key(fit)
    hit = _FITTED.get(key)
    if hit is None:
        hit = FittedModels.load(fit.directory)
        _FITTED.put(key, hit)
    return hit


def shadow_engine(fit: Fit):
    key = (_fit_key(fit), settings.SHADOW_SAMPLES)
    hit = _ENGINES.get(key)
    if hit is None:
        hit = load_fitted(fit).shadow_engine(n_samples=int(settings.SHADOW_SAMPLES), seed=0)
        _ENGINES.put(key, hit)
    return hit


def fit_arm_space(fit: Fit) -> ArmSpace:
    def build() -> ArmSpace:
        saved = read_json(fit.directory / "arm_space.json")
        if saved:
            return ArmSpace.from_dict(saved)
        return ArmSpace.from_config(FrameworkConfig.from_dict(fit.framework or {}).arms)

    return cache.memo("arm_space", (fit.pk, cache.file_key(fit.directory / "arm_space.json")), build)


def dataset_frame(fit: Fit, columns: list[str] | None = None, *, train: bool | None = None) -> pd.DataFrame:
    """Dataset rows (optionally only the training ``issue_d < train_end`` or holdout part)."""
    ds = fit.dataset
    filters = None
    if train is not None:
        cut = pd.Timestamp(epoch_start(fit.train_end))
        filters = [("issue_d", "<" if train else ">=", cut)]
    cols = None if columns is None else list(dict.fromkeys(columns))
    return pd.read_parquet(ds.parquet, columns=cols, filters=filters).reset_index(drop=True)


# ============================================================================ config
def contract_config(framework: FrameworkConfig, train_end_epoch: str) -> dict[str, Any]:
    """types.ts ``FitConfig`` from a FrameworkConfig."""
    h, r, m, w, a = framework.hierarchy, framework.recovery, framework.mctm, framework.reward, framework.arms
    return clean(
        {
            "segmentation": {"dimensions": list(framework.segmentation.dimensions)},
            "arm_space": {
                "amounts": [float(x) for x in a.amounts],
                "rates": [float(x) for x in a.rates],
                "terms": [int(x) for x in a.terms],
                "include_null": bool(a.include_null),
                "max_payment_to_income": None if a.max_payment_to_income is None else float(a.max_payment_to_income),
                "rate_corridor": None if a.rate_corridor is None else float(a.rate_corridor),
                "max_amount_to_income": None if a.max_amount_to_income is None else float(a.max_amount_to_income),
                "max_total_dti": None if a.max_total_dti is None else float(a.max_total_dti),
            },
            "hierarchy": {
                "kappa_group": h.kappa_group,
                "kappa_leaf": h.kappa_leaf,
                "alpha0": h.alpha0,
                "beta0": h.beta0,
                "policy_effect": bool(h.policy_effect),
                "policy_effect_cfg": h.policy_effect_cfg.to_dict(),
            },
            "recovery": {"min_obs": int(r.min_obs), "basis": r.basis, "term_level": bool(r.term_level)},
            "mctm": {
                "structure": m.structure.value,
                "prior_strength": m.prior_strength,
                "chargeoff_lag_months": int(m.chargeoff_lag_months),
            },
            "reward": {
                "funding_rate": w.funding_rate,
                "risk_aversion": w.risk_aversion,
                "lendable_threshold": w.lendable_threshold,
                "exposure_basis": w.exposure_basis,
                "prepayment_adjustment": w.prepayment_adjustment,
            },
            "train_end": train_end_epoch,
        }
    )


# ============================================================================ metrics
def contract_metrics(fitted) -> dict[str, Any]:
    """types.ts ``FitMetrics`` from ``FittedModels.metrics`` (+ model counts)."""
    m = fitted.metrics or {}
    pred = m.get("predictive") or {}
    primary = (pred.get("default") or {}).get("primary") or {}
    cs = (pred.get("credit_state") or {}).get("all") or {}
    d, n = fitted.two_stage.default_model.global_counts()
    return clean(
        {
            "brier": num(m.get("brier", primary.get("brier"))),
            "brier_baseline": num(primary.get("brier_climatology")),
            "auc": num(m.get("auc", primary.get("auc"))),
            "ece": num(m.get("ece", primary.get("ece"))),
            "log_loss": num(m.get("log_loss", primary.get("log_loss"))),
            "recovery_mae": num(m.get("recovery_mae")),
            "recovery_rmse": num(m.get("recovery_rmse")),
            "transition_log_lik": num(cs.get("log_likelihood")),
            "transition_log_lik_per_obs": num(cs.get("log_likelihood_per_transition")),
            "transition_prediction_error": num(cs.get("row_error_mean")),
            "n_train": int(m.get("n_train_resolved") or m.get("n_train") or 0),
            "n_test": int(m.get("n_test") or primary.get("n") or 0),
            "n_defaults_train": int(d),
            "base_rate_train": num(d / (d + n)) if (d + n) else None,
            "base_rate_test": num(primary.get("base_rate")),
            "fit_seconds": num(m.get("fit_seconds")),
            # extras (not in the contract): secondary holdout subset and holdout sizes
            "brier_secondary": num(((pred.get("default") or {}).get("secondary") or {}).get("brier")),
            "auc_secondary": num(((pred.get("default") or {}).get("secondary") or {}).get("auc")),
            "n_holdout": int(m.get("n_holdout") or 0),
            "n_train_all": int(m.get("n_train") or 0),
            "evaluate_seconds": num(m.get("evaluate_seconds")),
            # model diagnostics behind the shadow engine (docs/MATH.md §2.6-2.9)
            "interest_realization": m.get("interest_realization"),
            "policy_effect": m.get("policy_effect"),
            "guardrails": m.get("guardrails"),
            "reference_rates": m.get("reference_rates"),
        }
    )


# ============================================================================ hierarchy
def _dims(fit: Fit, group: str | None) -> dict[str, str]:
    if not group:
        return {}
    dims = list(((fit.config or {}).get("segmentation") or {}).get("dimensions") or [])
    parts = str(group).split("|")
    return {d: parts[i] if i < len(parts) else "" for i, d in enumerate(dims)}


def _recovery_level(source: str) -> str:
    """Contract ``HierarchyLevel`` of a core recovery source (the core's ``group_term`` level -
    the group restricted to the arm's term - is reported as ``group``; the exact core level is
    served as ``source``)."""
    if source == "group_term":
        return "group"
    return source if source in ("global", "group", "leaf") else "global"


def _recovery_n_obs(rec, group: str | None, arm: int | None, source: str) -> int:
    if source == "leaf" and group is not None and arm is not None:
        return rec.stats(group, arm).n
    if source == "group_term" and group is not None and arm is not None:
        return rec.stats(group, term=rec.term_of(arm)).n
    if source == "group" and group is not None:
        return rec.stats(group).n
    return rec.stats().n


def _recovery_summary(model, group: str | None, arm: int | None) -> dict[str, Any]:
    rec = model.recovery_model
    if group is None:
        post = rec.global_posterior()
        src = rec._resolve_chain(rec._chain(None, None))[1]
    elif arm is None:
        post, src = rec._resolve_chain(rec._chain(group, None))
    else:
        post, src = rec.resolve(group, arm)
    return {
        "mean": round(post.mean, 6),
        "alpha": round(post.alpha, 6),
        "beta": round(post.beta, 6),
        "n_obs": int(_recovery_n_obs(rec, group, arm, src)),
        "source": _recovery_level(src),
        "core_source": src,
    }


def hierarchy_frame(fit: Fit) -> pd.DataFrame:
    path = fit.directory / "hierarchy_summary.parquet"
    return cache.memo("hierarchy", cache.file_key(path), lambda: pd.read_parquet(path))


def hierarchy_rows(fit: Fit, level: str | None = None, group: str | None = None) -> list[dict[str, Any]]:
    """types.ts ``HierarchyRow[]``."""
    if level not in (None, "", "global", "group", "leaf"):
        raise ValidationError({"level": ["must be global, group or leaf"]})
    if level == "leaf" and not group:
        raise ValidationError({"group": ["required when level=leaf"]})
    fitted = load_fitted(fit)
    model = fitted.two_stage
    arm_space = fitted.arm_space
    df = hierarchy_frame(fit)
    hcfg = fitted.cfg.hierarchy
    n_groups = int((df["level"] == "group").sum())
    leaves_per_group = df.loc[df["level"] == "leaf", "group"].astype(str).value_counts().to_dict()

    def row(rec: dict[str, Any]) -> dict[str, Any]:
        lvl = str(rec["level"])
        g = None if lvl == "global" or pd.isna(rec.get("group")) else str(rec["group"])
        a = None if lvl != "leaf" or pd.isna(rec.get("arm")) else int(rec["arm"])
        n_obs = int(rec["n_obs"])
        if lvl == "global":
            kappa = None
            prior_mean = None
            weight = n_obs / (n_obs + hcfg.alpha0 + hcfg.beta0)
            children = n_groups
            parent = None
        else:
            kappa = num(rec.get("kappa"), 4)
            prior_mean = num(rec.get("prior_mean"), 6)
            weight = num(rec.get("shrinkage_weight")) or 0.0
            children = int(leaves_per_group.get(g, 0)) if lvl == "group" else 0
            parent = "global" if lvl == "group" else g
        dims = _dims(fit, g)
        if a is not None:
            dims = {**dims, "arm": arm_space[a].label()}
        return {
            "level": lvl,
            "key": str(rec["key"]),
            "parent": parent,
            "group": g,
            "arm": a,
            "dims": dims,
            "alpha": num(rec["alpha"], 6),
            "beta": num(rec["beta"], 6),
            "n_obs": n_obs,
            "defaults": int(rec["defaults"]),
            "mean": num(rec["mean"], 6),
            "hdi_low": num(rec["hdi_low"], 6),
            "hdi_high": num(rec["hdi_high"], 6),
            "prior_mean": prior_mean,
            "kappa": kappa,
            "shrinkage_weight": round(float(weight), 6),
            "empirical_rate": num(rec.get("empirical_rate"), 6),
            "recovery": _recovery_summary(model, g, a),
            "n_children": children,
            # structured leaf prior: logit shift t(g, a) of the leaf prior mean (the core writes
            # 0 / NaN for global and group rows, which have no tilt → null)
            "prior_tilt_logit": num(rec.get("prior_tilt_logit"), 6) if lvl == "leaf" else None,
        }

    observed = [str(k) for k in df.loc[df["level"] == "group", "key"]]
    all_groups = cartesian_groups(fit, observed)

    def group_rows(keys: list[str]) -> list[dict[str, Any]]:
        recs = {str(r["key"]): r for r in df.loc[df["level"] == "group"].to_dict("records")}
        return [row(recs[k]) if k in recs else unobserved_row(fit, fitted, k) for k in keys]

    if level == "global":
        rows = [row(r) for r in df.loc[df["level"] == "global"].to_dict("records")]
    elif level == "group":
        rows = group_rows([group] if group else all_groups)
    elif level == "leaf":
        sel = df.loc[(df["level"] == "leaf") & (df["group"].astype(str) == group)].sort_values("arm")
        rows = [row(r) for r in sel.to_dict("records")]
    elif group:
        sel = df.loc[(df["level"] == "leaf") & (df["group"].astype(str) == group)].sort_values("arm")
        rows = group_rows([group]) + [row(r) for r in sel.to_dict("records")]
    else:
        rows = [row(r) for r in df.loc[df["level"] == "global"].to_dict("records")] + group_rows(all_groups)
    for r in rows:
        if r["level"] == "global":
            r["n_children"] = len(all_groups)
    return clean(rows)


def cartesian_groups(fit: Fit, observed: list[str]) -> list[str]:
    """Every combination of the dimension values observed in training (docs/API.md §6.3):
    unobserved groups are listed too so shrinkage to the prior is visible."""
    import itertools

    dims = list(((fit.config or {}).get("segmentation") or {}).get("dimensions") or [])
    width = len(dims) or (len(observed[0].split("|")) if observed else 0)
    split = [k.split("|") for k in observed if len(k.split("|")) == width]
    if not split:
        return sorted(observed)
    values = [sorted({parts[i] for parts in split}) for i in range(width)]
    combos = {"|".join(c) for c in itertools.product(*values)}
    return sorted(combos | set(observed))


def unobserved_row(fit: Fit, fitted, group: str) -> dict[str, Any]:
    """HierarchyRow of a group without training data: posterior = the group prior."""
    hm = fitted.two_stage.default_model
    prior = hm.group_prior(group)
    lo, hi = prior.hdi(0.95)
    return {
        "level": "group",
        "key": group,
        "parent": "global",
        "group": group,
        "arm": None,
        "dims": _dims(fit, group),
        "alpha": round(prior.alpha, 6),
        "beta": round(prior.beta, 6),
        "n_obs": 0,
        "defaults": 0,
        "mean": round(prior.mean, 6),
        "hdi_low": round(lo, 6),
        "hdi_high": round(hi, 6),
        "prior_mean": round(prior.mean, 6),
        "kappa": round(float(hm.kappa_group), 4),
        "shrinkage_weight": 0.0,
        "empirical_rate": None,
        "recovery": _recovery_summary(fitted.two_stage, group, None),
        "n_children": 0,
        "prior_tilt_logit": None,
    }


# ============================================================================ risk
def _beta_summary(post: BetaPosterior) -> dict[str, Any]:
    lo, hi = post.hdi(0.95)
    return {
        "alpha": round(post.alpha, 6),
        "beta": round(post.beta, 6),
        "mean": round(post.mean, 6),
        "std": round(post.std, 6),
        "hdi_low": round(lo, 6),
        "hdi_high": round(hi, 6),
    }


def _pdf(post: BetaPosterior) -> list[dict[str, float]]:
    x, y = post.pdf_grid(GRID)
    y = np.where(np.isfinite(y), y, 0.0)
    return [{"x": round(float(a), 4), "density": round(float(b), 6)} for a, b in zip(x, y, strict=True)]


def risk_payload(fit: Fit, group: str, arm: int | None) -> dict[str, Any]:
    """types.ts ``RiskEstimate``."""
    fitted = load_fitted(fit)
    hm = fitted.two_stage.default_model
    arm_space = fitted.arm_space
    known_group = group in set(hm.groups)
    if arm is not None and not 0 <= arm < arm_space.n_arms:
        raise ValidationError({"arm": [f"must be in [0, {arm_space.n_arms - 1}]"]})
    if arm is None:
        post = hm.group_posterior(group)
        prior = hm.group_prior(group) if known_group else hm.global_posterior()
        d, n = hm.group_counts(group) if known_group else (0, 0)
        level = "group" if known_group else "global"
        weight = hm.shrinkage_weight(group) if known_group else 0.0
        rec = fitted.two_stage.recovery_model
        rpost, rsrc = rec._resolve_chain(rec._chain(group, None))
        policy = None
        reward = None
        label = "group posterior"
    else:
        post = hm.posterior(group, arm)
        d, n = hm.counts(group, arm)
        observed = (d + n) > 0
        prior = hm.leaf_prior(group, arm) if known_group else hm.group_prior(group)
        level = "leaf" if observed else ("group" if known_group else "global")
        weight = hm.shrinkage_weight(group, arm) if known_group else 0.0
        rpost, rsrc = fitted.two_stage.recovery_model.resolve(group, arm)
        pol = arm_space[arm]
        policy = None if pol.is_null else pol.to_dict()
        label = pol.label()
        est = fitted.two_stage.risk(group, arm)
        if pol.is_null:
            reward = {
                "mean": 0.0,
                "std": 0.0,
                "revenue": 0.0,
                "expected_loss": 0.0,
                "funding_cost": 0.0,
                "risk_adjusted": 0.0,
            }
        else:
            rho = fitted.realization(group, arm)
            dist = fitted.reward_fn.expected(pol, est, fitted.mctm, realization=rho)
            c = dist.components
            reward = {
                "mean": round(dist.mean, 2),
                "std": round(dist.std, 2),
                "revenue": round(float(c.get("expected_interest", 0.0)), 2),
                "expected_loss": round(float(c.get("expected_loss", 0.0)), 2),
                "funding_cost": round(float(c.get("funding_cost", 0.0)), 2),
                "risk_adjusted": round(fitted.reward_fn.score(dist), 2),
                "gross_interest": round(float(c.get("gross_interest", 0.0)), 2),
                "realization": round(float(rho), 6),
            }
    rec_model = fitted.two_stage.recovery_model
    r_n = _recovery_n_obs(rec_model, group, arm, rsrc)
    return clean(
        {
            "fit": fit.pk,
            "group": group,
            "arm": arm,
            "policy": policy,
            "label": label,
            "p_default": {
                **_beta_summary(post),
                "level": level,
                "n_obs": int(d + n),
                "defaults": int(d),
                "shrinkage_weight": round(float(weight), 6),
            },
            "recovery": {
                **_beta_summary(rpost),
                "level": _recovery_level(rsrc),
                "core_source": rsrc,
                "n_obs": int(r_n),
            },
            "expected_loss_fraction": round(post.mean * (1.0 - rpost.mean), 6),
            "reward": reward,
            "pdf": {"p_default": _pdf(post), "p_default_prior": _pdf(prior), "recovery": _pdf(rpost)},
        }
    )


# ============================================================================ calibration
def _holdout_recovery(fit: Fit, fitted) -> dict[str, Any]:
    """Realized holdout recovery rates of defaulted loans (term elapsed): histogram + by grade."""
    path = fit.directory / "api_recovery_holdout.json"
    cached = read_json(path)
    if cached is not None:
        return cached
    from bayesian_credit.pipeline import elapsed_term_mask

    cols = [
        "issue_d",
        "term",
        "grade",
        "y_default",
        "y_recovery_rate",
        "funded_amnt",
        "loan_amnt",
        "total_rec_prncp",
        "recoveries",
        "collection_recovery_fee",
    ]
    df = dataset_frame(fit, cols, train=False)
    out: dict[str, Any] = {
        "histogram": [{"lo": k / 20, "hi": (k + 1) / 20, "count": 0} for k in range(20)],
        "by_grade": [],
        "n": 0,
    }
    if len(df):
        basis = fitted.cfg.recovery.basis
        rr = (
            recovery_rate(df, basis)
            if basis != "principal"
            else df["y_recovery_rate"].to_numpy(dtype=float, na_value=np.nan)
        )
        dflt = df["y_default"].to_numpy(dtype=float, na_value=np.nan) == 1.0
        elapsed = elapsed_term_mask(df.reset_index(drop=True), fitted.cfg.data.snapshot_date)
        keep = dflt & elapsed & np.isfinite(rr)
        vals = rr[keep]
        counts, _ = np.histogram(np.clip(vals, 0.0, 1.0), bins=np.linspace(0.0, 1.0, 21))
        out["histogram"] = [
            {"lo": round(k / 20, 4), "hi": round((k + 1) / 20, 4), "count": int(c)} for k, c in enumerate(counts)
        ]
        grades = df["grade"].astype(str).to_numpy()[keep]
        by = pd.DataFrame({"grade": grades, "r": vals}).groupby("grade")["r"].agg(["mean", "size"])
        out["by_grade"] = [
            {"grade": str(g), "mean": round(float(r["mean"]), 6), "n": int(r["size"])} for g, r in by.iterrows()
        ]
        out["n"] = int(keep.sum())
    write_json_atomic(path, out)
    return out


def _decile_bins(fit: Fit, fitted) -> dict[str, Any]:
    """Reliability on 10 equal-frequency bins (prediction deciles) of the primary holdout
    (resolved loans whose term elapsed by the snapshot); ``lo``/``hi`` = min/max prediction."""
    path = fit.directory / "api_reliability_deciles.json"
    cached = read_json(path)
    if cached is not None:
        return cached
    from bayesian_credit.pipeline import elapsed_term_mask, ensure_outcomes

    df = dataset_frame(fit, None, train=False)
    out: dict[str, Any] = {"bins": [], "ece": None, "n": 0}
    if len(df):
        df = ensure_outcomes(df, fitted.cfg, arm_space=fitted.arm_space, segmenter=fitted.segmenter)
        y = df["y_default"].to_numpy(dtype=float, na_value=np.nan)
        p = fitted.two_stage.predict_table(df)["p_default_mean"].to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(p) & elapsed_term_mask(df, fitted.cfg.data.snapshot_date)
        y, p = y[keep], p[keep]
        order = np.argsort(p, kind="mergesort")
        bins = []
        ece = 0.0
        for i, idx in enumerate(np.array_split(order, min(10, max(len(order), 1)))):
            if idx.size == 0:
                continue
            pp, yy = p[idx], y[idx]
            bins.append(
                {
                    "bin": i,
                    "lo": round(float(pp.min()), 6),
                    "hi": round(float(pp.max()), 6),
                    "n": int(idx.size),
                    "mean_predicted": round(float(pp.mean()), 6),
                    "observed_rate": round(float(yy.mean()), 6),
                }
            )
            ece += idx.size / len(order) * abs(float(yy.mean()) - float(pp.mean()))
        out = {"bins": bins, "ece": round(ece, 6) if bins else None, "n": int(len(order))}
    write_json_atomic(path, out)
    return out


def calibration_payload(fit: Fit) -> dict[str, Any]:
    """types.ts ``Calibration`` (primary holdout subset: resolved loans with elapsed term)."""
    fitted = load_fitted(fit)
    m = fitted.metrics or {}
    primary = (((m.get("predictive") or {}).get("default") or {}).get("primary")) or {}
    cal = (fitted.calibration or {}).get("primary") or {}
    deciles = _decile_bins(fit, fitted)
    bins = deciles["bins"]
    roc = list(cal.get("roc") or [])
    if len(roc) > 200:
        pick = np.unique(np.linspace(0, len(roc) - 1, 200).round().astype(int))
        roc = [roc[i] for i in pick]
    roc_rows = [
        {
            "fpr": num(p.get("fpr"), 6),
            "tpr": num(p.get("tpr"), 6),
            "threshold": 1.0 if p.get("threshold") is None else min(num(p.get("threshold"), 6) or 0.0, 1.0),
        }
        for p in roc
    ]
    rec_model = fitted.two_stage.recovery_model
    gpost = rec_model.global_posterior()
    hold = _holdout_recovery(fit, fitted)
    return clean(
        {
            "fit": fit.pk,
            "n_test": int(primary.get("n") or 0),
            "base_rate": num(primary.get("base_rate"), 6),
            "brier": num(primary.get("brier"), 6),
            "auc": num(primary.get("auc"), 6),
            "ece": deciles["ece"] if deciles["ece"] is not None else num(primary.get("ece"), 6),
            "ece_equal_width": num(primary.get("ece"), 6),
            "log_loss": num(primary.get("log_loss"), 6),
            "bins": bins,
            "roc": roc_rows,
            "recovery": {
                "alpha": round(gpost.alpha, 6),
                "beta": round(gpost.beta, 6),
                "mean": round(gpost.mean, 6),
                "n_defaults": int(rec_model.stats().n),
                "histogram": hold["histogram"],
                "pdf": _pdf(gpost),
                "mae": num(m.get("recovery_mae"), 6),
                "rmse": num(m.get("recovery_rmse"), 6),
                "by_grade": hold["by_grade"],
                "n_holdout_defaults": hold.get("n", 0),
            },
            "secondary": (((m.get("predictive") or {}).get("default") or {}).get("secondary")) or {},
        }
    )


# ============================================================================ MCTM
def _panel_counts(fit: Fit, fitted, structure: str) -> dict[str, Any]:
    """Training and holdout transition counts of the panel reconstructed under ``structure``."""
    path = fit.directory / f"api_mctm_counts_{structure}.json"
    cached = read_json(path)
    if cached is not None:
        return cached
    snapshot = fitted.cfg.data.snapshot_date
    mcfg = MCTMConfig.from_dict({**fitted.cfg.mctm.to_dict(), "structure": structure})
    from bayesian_credit.data.panel import PANEL_COLUMNS

    cols = ["loan_id", *PANEL_COLUMNS]
    train = dataset_frame(fit, cols, train=True)
    hold = dataset_frame(fit, cols, train=False)
    out = {
        "structure": structure,
        "train": LoanStatePanel.from_table(train, mcfg, snapshot).counts.tolist(),
        "holdout": LoanStatePanel.from_table(hold, mcfg, snapshot).counts.tolist()
        if len(hold)
        else np.zeros((4, 4)).tolist(),
    }
    write_json_atomic(path, out)
    return out


def _credible_bounds(chain: MarkovChainTransitionMatrix, q: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    P = np.array(chain.matrix, dtype=float)
    lo, hi = P.copy(), P.copy()
    alpha = chain.dirichlet_alpha
    if alpha is None:
        return lo, hi
    for i in (0, 1):
        a = alpha[i]
        a0 = a.sum()
        if a0 <= 0:
            continue
        for j in range(4):
            if a[j] <= 0:
                lo[i, j] = hi[i, j] = 0.0
            elif a0 - a[j] <= 0:
                lo[i, j] = hi[i, j] = 1.0
            else:
                lo[i, j] = float(stats.beta.ppf(q, a[j], a0 - a[j]))
                hi[i, j] = float(stats.beta.ppf(1 - q, a[j], a0 - a[j]))
    return lo, hi


def mctm_payload(fit: Fit, structure: str | None, horizon: int, start: str) -> dict[str, Any]:
    """types.ts ``Mctm``; a structure other than the fitted one is refitted on the fly from the
    training panel reconstructed under that structure."""
    fitted = load_fitted(fit)
    fitted_structure = fitted.mctm.structure.value
    structure = structure or fitted_structure
    if structure not in ("proposal", "extended"):
        raise ValidationError({"structure": ["must be proposal or extended"]})
    if start not in ("current", "delinquent"):
        raise ValidationError({"start": ["must be current or delinquent"]})
    prior_strength = float(fitted.cfg.mctm.prior_strength)
    if structure == fitted_structure:
        chain = fitted.mctm
        train_counts = np.asarray(chain.counts, dtype=float)
        cs = (((fitted.metrics or {}).get("predictive") or {}).get("credit_state") or {}).get("all") or {}
        hold_counts = np.asarray(cs.get("counts") or _panel_counts(fit, fitted, structure)["holdout"], dtype=float)
    else:
        counts = _panel_counts(fit, fitted, structure)
        train_counts = np.asarray(counts["train"], dtype=float)
        hold_counts = np.asarray(counts["holdout"], dtype=float)
        chain = cache.memo(
            "mctm_refit",
            (fit.pk, structure, prior_strength, tuple(train_counts.ravel())),
            lambda: MarkovChainTransitionMatrix(structure, prior_strength).fit_counts(train_counts),
        )
    P = np.asarray(chain.matrix, dtype=float)
    lo, hi = _credible_bounds(chain)
    pi0 = np.array([1.0, 0, 0, 0]) if start == "current" else np.array([0, 1.0, 0, 0])
    traj = chain.trajectory(pi0, horizon)
    after = chain.maturity_collapse(traj[-1])
    absorption = chain.absorption_probabilities()
    steps = chain.expected_time_to_absorption()
    ll = chain.log_likelihood(hold_counts)
    row_err = chain.row_prediction_error(hold_counts)
    zeros = ~allowed_transitions(structure)
    return clean(
        {
            "fit": fit.pk,
            "structure": structure,
            "fitted_structure": fitted_structure,
            "states": STATES,
            "state_labels": STATE_LABELS,
            "matrix": np.round(P, 8).tolist(),
            "counts": train_counts.tolist(),
            "structural_zeros": zeros.tolist(),
            "matrix_low": np.round(lo, 8).tolist(),
            "matrix_high": np.round(hi, 8).tolist(),
            "absorption": np.round(absorption, 8).tolist(),
            "expected_steps": np.round(steps, 6).tolist(),
            "start": start,
            "horizon": int(horizon),
            "trajectory": [
                {"t": t, **{s: round(float(traj[t, i]), 8) for i, s in enumerate(STATES)}} for t in range(traj.shape[0])
            ],
            "p_default_curve": [
                {
                    "t": t,
                    "p_default": round(float(traj[t, 1] + traj[t, 3]), 8),
                    "p_loss": round(float(traj[t, 3]), 8),
                    "p_delinquent": round(float(traj[t, 1]), 8),
                }
                for t in range(traj.shape[0])
            ],
            "maturity": {
                "t": int(horizon),
                "before": [round(float(v), 8) for v in traj[-1]],
                "after": [round(float(v), 8) for v in after],
                "p_default": round(float(after[1] + after[3]), 8),
            },
            "log_likelihood": num(ll) if math.isfinite(ll) else None,
            "n_transitions": float(hold_counts.sum()),
            "row_prediction_error": [num(row_err[0], 8) or 0.0, num(row_err[1], 8) or 0.0, 0.0, 0.0],
            "prior_strength": prior_strength,
        }
    )


# ============================================================================ arms
def _historical_counts(fit: Fit, fitted) -> list[int]:
    path = fit.directory / "api_historical_arms.json"
    cached = read_json(path)
    if cached is not None and len(cached) == fitted.arm_space.n_arms:
        return cached
    from bayesian_credit.data.outcomes import historical_arms

    df = dataset_frame(fit, ["issue_d", "loan_amnt", "int_rate", "term"], train=True)
    counts = np.zeros(fitted.arm_space.n_arms, dtype=int)
    if len(df):
        arms = historical_arms(df, fitted.arm_space)
        arms = arms[(arms >= 0) & (arms < fitted.arm_space.n_arms)]
        counts = np.bincount(arms, minlength=fitted.arm_space.n_arms)
    out = [int(c) for c in counts]
    write_json_atomic(path, out)
    return out


def arms_payload(fit: Fit) -> dict[str, Any]:
    """types.ts ``FitArms``: group-share-weighted expected reward of offering each arm to
    every borrower (weights = resolved training loans per group)."""

    def compute() -> dict[str, Any]:
        fitted = load_fitted(fit)
        arm_space = fitted.arm_space
        hm = fitted.two_stage.default_model
        groups = list(hm.groups)
        n_by_group = np.array([sum(hm.group_counts(g)) for g in groups], dtype=float)
        weights = (
            n_by_group / n_by_group.sum() if n_by_group.sum() > 0 else np.full(len(groups), 1.0 / max(len(groups), 1))
        )
        hist = _historical_counts(fit, fitted)
        total_hist = max(sum(hist), 1)
        realize = fitted.realization_lookup()
        rows = []
        for a, pol in enumerate(arm_space.policies):
            mean = second = pd_ = rec = 0.0
            for w, g in zip(weights, groups, strict=True):
                est = fitted.two_stage.risk(g, a)
                pd_ += w * est.p_default.mean
                rec += w * est.recovery.mean
                if not pol.is_null:
                    dist = fitted.reward_fn.expected(pol, est, fitted.mctm, realization=realize(g, a))
                    mean += w * dist.mean
                    second += w * (dist.std**2 + dist.mean**2)
            std = math.sqrt(max(second - mean * mean, 0.0)) if not pol.is_null else 0.0
            rows.append(
                {
                    "arm": a,
                    "amount": float(pol.amount),
                    "rate": float(pol.rate),
                    "term": int(pol.term),
                    "is_null": bool(pol.is_null),
                    "installment": round(float(pol.installment), 2),
                    "label": pol.label(),
                    "expected_reward": round(mean, 2) if not pol.is_null else 0.0,
                    "reward_std": round(std, 2),
                    "p_default": round(pd_, 6) if not pol.is_null else 0.0,
                    "expected_recovery": round(rec, 6),
                    "historical_count": int(hist[a]),
                    "historical_share": round(hist[a] / total_hist, 6),
                }
            )
        return clean(
            {
                "fit": fit.pk,
                "amounts": list(arm_space.amount_grid),
                "rates": list(arm_space.rate_grid),
                "terms": list(arm_space.term_grid),
                "arms": rows,
            }
        )

    path = fit.directory / "api_arms.json"
    cached = read_json(path)
    if cached is not None:
        return cached
    out = compute()
    write_json_atomic(path, out)
    return out


# ============================================================================ shadow
def contract_shadow_report(fit: Fit, report, outcome: dict[str, Any] | None) -> dict[str, Any]:
    """types.ts ``ShadowReport`` from the core ``ShadowReport.to_dict()``."""
    d = report.to_dict()
    ctx = {c: d["context"].get(c) for c in PRE_DECISION_COLUMNS}
    for k, v in ctx.items():
        if isinstance(v, float) and not math.isfinite(v):
            ctx[k] = None
    context = context_from_row(ctx)
    d.update(
        {
            "fit": fit.pk,
            "context": context,
            "context_columns": sorted(c for c, v in context.items() if v is not None),
            "outcome": outcome,
        }
    )
    hp = d.get("historical_policy")
    if hp is not None:
        d["historical_policy"] = {"amount": float(hp["amount"]), "rate": float(hp["rate"]), "term": int(hp["term"])}
    if report.historical_arm is not None and report.historical_policy is not None:
        # docs/API.md §11.16: the exact issued policy, with the risk of the nearest arm's leaf
        fitted = load_fitted(fit)
        est = fitted.two_stage.risk(report.group, int(report.historical_arm))
        rho = fitted.realization(report.group, term=int(report.historical_policy.term))
        exact = fitted.reward_fn.expected(report.historical_policy, est, fitted.mctm, realization=rho)
        rec_mean = float(d["arms"][int(d["recommended_arm"])]["reward_mean"] or 0.0)
        d["historical_arm_expected_reward"] = d.get("historical_expected_reward")
        d["historical_expected_reward"] = round(float(exact.mean), 2)
        d["uplift_expected"] = round(rec_mean - float(exact.mean), 2)
    return clean(d)


def shadow_by_loan(fit: Fit, loan_id: int) -> dict[str, Any]:
    from bayesian_credit.pipeline import ensure_outcomes

    fitted = load_fitted(fit)
    rows = read_loans_by_id(fit.dataset, [int(loan_id)])
    if rows.empty:
        raise NotFound(f"loan {loan_id} not found in dataset {fit.dataset_id}")
    rows = ensure_outcomes(rows, fitted.cfg, arm_space=fitted.arm_space, segmenter=fitted.segmenter)
    row = rows.iloc[0]
    report = shadow_engine(fit).evaluate_loan(row)
    outcome = outcome_from_row(row, report.historical_realized_reward)
    return contract_shadow_report(fit, report, outcome)


def _coerce_context(context: dict[str, Any]) -> dict[str, Any]:
    LeakageGuard.assert_allowed(context.keys(), Stage.PRE_DECISION)  # LeakageError → 400
    out: dict[str, Any] = {}
    errors: dict[str, list[str]] = {}
    for key, value in context.items():
        if key not in VARIABLES_BY_NAME:  # derived pre-decision helpers (purpose_group, …)
            out[key] = value
            continue
        kind = VARIABLES_BY_NAME[key].kind
        if value is None:
            out[key] = np.nan if kind == "numeric" else None
        elif kind == "numeric":
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                errors.setdefault("context", []).append(f"{key} must be a number")
        else:
            out[key] = str(value)
    if errors:
        raise ValidationError(errors)
    return out


def shadow_by_context(fit: Fit, context: dict[str, Any], historical_policy: dict[str, Any] | None) -> dict[str, Any]:
    features = _coerce_context(context)
    policy = None
    if historical_policy:
        try:
            policy = LendingPolicy(
                float(historical_policy["amount"]), float(historical_policy["rate"]), int(historical_policy["term"])
            )
        except (KeyError, TypeError, ValueError):
            raise ValidationError({"historical_policy": ["expected {amount, rate, term}"]}) from None
        if policy.amount <= 0 or not 0 < policy.rate < 1 or policy.term <= 0:
            raise ValidationError({"historical_policy": ["amount > 0, rate in (0, 1) as a fraction, term > 0 months"]})
    report = shadow_engine(fit).evaluate(BorrowerContext(loan_id=None, features=features), historical_policy=policy)
    return contract_shadow_report(fit, report, None)


# ============================================================================ portfolio
CONTRACT_CLASSES = ("rescued", "upsized", "repriced_down", "repriced_up", "declined", "unchanged")


def _class_counts(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    out = {c: int(raw.get(c, 0) or 0) for c in CONTRACT_CLASSES}
    if "downsized" in raw:  # core class not in the contract vocabulary (kept as an extra key)
        out["downsized"] = int(raw.get("downsized") or 0)
    return out


def _uplift_fraction(block: Any) -> Any:
    """Core blocks report ``uplift_pct`` in percent; the contract's ``uplift_pct`` is a fraction."""
    if not isinstance(block, dict):
        return block
    out = dict(block)
    if isinstance(out.get("uplift_pct"), (int, float)):
        out["uplift_pct"] = round(float(out["uplift_pct"]) / 100.0, 6)
    return out


def portfolio_summary(summary: dict[str, Any]) -> dict[str, Any]:
    hist = float(summary.get("expected_profit_historical") or 0.0)
    decomposition = summary.get("uplift_decomposition")
    if isinstance(decomposition, dict):
        decomposition = {k: _uplift_fraction(v) for k, v in decomposition.items()}
    by_lambda = summary.get("by_risk_aversion")
    if isinstance(by_lambda, dict):
        by_lambda = {k: _uplift_fraction(v) for k, v in by_lambda.items()}
    uplift = float(summary.get("uplift") or 0.0)
    return clean(
        {
            "n_loans": int(summary.get("n_loans") or 0),
            "n_lendable": int(summary.get("n_lendable") or 0),
            "share_lendable": num(summary.get("share_lendable"), 6) or 0.0,
            "expected_profit_historical": round(hist, 2),
            "expected_profit_shadow": round(float(summary.get("expected_profit_shadow") or 0.0), 2),
            "uplift": round(uplift, 2),
            "uplift_pct": round(uplift / max(abs(hist), 1.0), 6),
            "realized_profit_historical": num(summary.get("realized_profit_historical"), 2),
            "counts": _class_counts(summary.get("counts")),
            "uplift_risk_adjusted": num(summary.get("uplift_risk_adjusted"), 2),
            "n_realized": int(summary.get("n_realized") or 0),
            "seconds": num(summary.get("seconds"), 3),
            # core extras (docs/API.md §13): where the recommendations land, the guardrails in
            # force, where the uplift comes from and how risk appetite changes the policy
            "recommendation_mix": summary.get("recommendation_mix"),
            "eligibility": summary.get("eligibility"),
            "uplift_decomposition": decomposition,  # uplift_pct as fractions, like the summary's
            "by_risk_aversion": by_lambda,
        }
    )


def portfolio_loan_row(r: dict[str, Any]) -> dict[str, Any]:
    """types.ts ``PortfolioLoanRow`` from a core ``top_loans`` row."""
    return {
        "loan_id": int(r.get("loan_id")),
        "grade": str(r.get("grade")),
        "group": str(r.get("group")),
        "historical_arm": int(r.get("historical_arm")),
        "recommended_arm": int(r.get("recommended_arm")),
        "historical_label": str(r.get("historical_label")),
        "recommended_label": str(r.get("recommended_label")),
        "historical_expected_reward": num(r.get("historical_expected"), 2),
        "shadow_expected_reward": num(r.get("recommended_expected"), 2),
        "realized_reward": num(r.get("historical_realized"), 2),
        "class": str(r.get("outcome_class") or "unchanged"),
    }


def latest_portfolio(fit: Fit) -> tuple[Any, dict[str, Any] | None]:
    """``(job, core summary dict)`` of the fit's most recently finished *succeeded*
    shadow-portfolio job whose ``summary.json`` exists, else ``(None, None)``."""
    from django.db.models import F

    from apps.common import artifacts
    from apps.common.models import Job

    jobs = Job.objects.filter(
        kind=Job.Kind.SHADOW_PORTFOLIO, target_type="fit", target_id=fit.pk, status=Job.Status.SUCCEEDED
    ).order_by(F("finished_at").desc(nulls_last=True), "-id")
    for job in jobs:
        raw = read_json(artifacts.portfolio_dir(fit.pk, job.pk) / "summary.json")
        if isinstance(raw, dict) and raw:
            return job, raw
    return None, None


def portfolio_result(fit: Fit, job) -> dict[str, Any]:
    """types.ts ``PortfolioResult`` (summary/tables empty until the job succeeded)."""
    from apps.common import artifacts
    from apps.common.serializers import JobSerializer

    request = (job.params or {}).get("request") or {}
    out = {
        "job": JobSerializer(job).data,
        "fit": fit.pk,
        "request": request,
        "summary": None,
        "by_grade": [],
        "inclusion": [],
        "top_loans": [],
    }
    if job.status != "succeeded":
        return out
    raw = read_json(artifacts.portfolio_dir(fit.pk, job.pk) / "summary.json")
    if not raw:
        return out
    out["summary"] = portfolio_summary(raw)
    out["by_grade"] = [
        {
            "grade": str(r.get("grade")),
            "n": int(r.get("n") or 0),
            "expected_profit_historical": num(r.get("expected_profit_historical"), 2) or 0.0,
            "expected_profit_shadow": num(r.get("expected_profit_shadow"), 2) or 0.0,
            "realized_profit_historical": num(r.get("realized_profit_historical"), 2),
            "counts": _class_counts(r.get("counts")),
        }
        for r in raw.get("by_grade") or []
    ]
    out["inclusion"] = [
        {
            "grade": str(r.get("grade")),
            "n": int(r.get("n") or 0),
            "share_lendable": num(r.get("share_lendable"), 6) or 0.0,
            "decline_rate_shadow": num(r.get("decline_rate_shadow"), 6) or 0.0,
            "mean_rate_historical": num(r.get("mean_rate_historical"), 6),
            "mean_rate_shadow": num(r.get("mean_rate_shadow"), 6),
            "mean_amount_historical": num(r.get("mean_amount_historical"), 2),
            "mean_amount_shadow": num(r.get("mean_amount_shadow"), 2),
        }
        for r in raw.get("inclusion") or []
    ]
    out["top_loans"] = [portfolio_loan_row(r) for r in raw.get("top_loans") or []]
    return clean(out)
