"""Two-Stage Beta-Beta facade (proposal §3.3; SPEC §2.4).

Stage 1, :class:`~.hierarchical.HierarchicalBetaBernoulli`: ``p_{g,a} | D ~ Beta(α_{g,a} + d,
β_{g,a} + n)``. Stage 2, :class:`~.recovery.BetaMoMRecovery`: ``R_{g,a} | D = 1 ~ Beta(α^R,
β^R)`` by method of moments. Together they give the posterior predictive inputs of the
expected loss ``E[L_t] = L_t · P(D | X, A_t) · (1 − E[R | X, A_t])``. p and R are modelled as
independent given (g, a).

**[decision] Structured leaf prior.** With ``HierarchyConfig.policy_effect`` (default on),
an arm space and the policy columns in the table, :meth:`TwoStageBetaBeta.fit` also fits a
:class:`~.policy_effect.PolicyEffectModel` and installs its tilt
``(g, a) ↦ effect(g, arm_space[a])`` in Stage 1, so a leaf's prior mean reflects the arm's
rate, term and payment-to-income (docs/MATH.md "Structured leaf prior").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd

from ..config import FrameworkConfig, HierarchyConfig, RecoveryConfig
from ..data.arms import ArmSpace
from ..exceptions import DataSourceError
from .beta import BetaPosterior
from .hierarchical import HierarchicalBetaBernoulli, _is_missing, beta_hdi
from .policy_effect import PolicyEffectModel, PolicyEffectTilt, policy_effect_columns
from .recovery import BetaMoMRecovery

__all__ = ["REQUIRED_COLUMNS", "RiskEstimate", "TwoStageBetaBeta"]

#: Derived LoanTable columns read by :meth:`TwoStageBetaBeta.fit` (SPEC §1.2).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "y_resolved",
    "y_default",
    "y_recovery_rate",
    "y_group",
    "y_historical_arm",
)


@dataclass(frozen=True)
class RiskEstimate:
    """Posterior risk of borrower group ``g`` under arm ``a``: ``p ~ Beta`` and ``R ~ Beta``.

    ``n_default_obs`` / ``n_recovery_obs`` count the observations at the *leaf* (the
    hierarchical posterior also borrows strength from the group/global levels);
    ``recovery_source`` is the level the recovery Beta was taken from after the fallback
    chain (``leaf``, ``group_term``, ``group``, ``global`` or ``prior``).
    """

    group: str
    arm: int
    p_default: BetaPosterior
    recovery: BetaPosterior
    n_default_obs: int = 0
    n_recovery_obs: int = 0
    recovery_source: str = "leaf"

    @property
    def expected_default(self) -> float:
        """``E[p_{g,a}]``."""
        return self.p_default.mean

    @property
    def expected_recovery(self) -> float:
        """``E[R_{g,a} | D = 1]``."""
        return self.recovery.mean

    @property
    def expected_loss_fraction(self) -> float:
        """``E[p]·(1 − E[R])`` — expected loss per unit of exposure (p ⫫ R)."""
        return self.p_default.mean * (1.0 - self.recovery.mean)

    def to_dict(self, hdi_mass: float = 0.95) -> dict[str, Any]:
        return {
            "group": self.group,
            "arm": int(self.arm),
            "p_default": self.p_default.to_dict(hdi_mass),
            "recovery": self.recovery.to_dict(hdi_mass),
            "n_default_obs": int(self.n_default_obs),
            "n_recovery_obs": int(self.n_recovery_obs),
            "recovery_source": self.recovery_source,
            "expected_loss_fraction": self.expected_loss_fraction,
        }


def _bool_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        arr = series.to_numpy(dtype=float, na_value=np.nan)
        return np.nan_to_num(arr, nan=0.0) != 0.0
    return series.map(lambda v: (not _is_missing(v)) and bool(v)).to_numpy(dtype=bool)


class TwoStageBetaBeta:
    """Facade over Stage 1 (default) and Stage 2 (recovery) — the risk model of the bandit.

    ``TwoStageBetaBeta(hierarchy_cfg, recovery_cfg)``; a :class:`FrameworkConfig` may also be
    passed as the first argument (its ``hierarchy`` / ``recovery`` sections and the data
    snapshot date are used), or use :meth:`from_config`. ``arm_space`` enables the
    structured leaf prior (policy effect); ``snapshot_date`` defines the elapsed-term loans
    the policy effect prefers.
    """

    def __init__(
        self,
        hierarchy: HierarchyConfig | FrameworkConfig | None = None,
        recovery: RecoveryConfig | None = None,
        *,
        arm_space: ArmSpace | None = None,
        snapshot_date: Any = None,
    ) -> None:
        snapshot = None
        if isinstance(hierarchy, FrameworkConfig):
            recovery = recovery if recovery is not None else hierarchy.recovery
            snapshot = hierarchy.data.snapshot_date
            hierarchy = hierarchy.hierarchy
        self.default_model = HierarchicalBetaBernoulli(hierarchy)
        self.recovery_model = BetaMoMRecovery(recovery)
        self.arm_space = arm_space
        self._install_term_level()
        self.snapshot_date = None if (snapshot_date is None and snapshot is None) else str(
            snapshot_date if snapshot_date is not None else snapshot
        )
        self._policy_effect_status = "not_fitted"

    @classmethod
    def from_config(cls, cfg: FrameworkConfig, arm_space: ArmSpace | None = None) -> TwoStageBetaBeta:
        return cls(cfg.hierarchy, cfg.recovery, arm_space=arm_space, snapshot_date=cfg.data.snapshot_date)

    def _install_term_level(self) -> None:
        """Stage 2's (group, term) level needs the arms' terms (``RecoveryConfig.term_level``)."""
        on = self.arm_space is not None and bool(getattr(self.recovery_model.cfg, "term_level", True))
        self.recovery_model.set_term_of_arm(self.arm_space if on else None)

    # ----- estimation -------------------------------------------------------------------
    def fit(self, table: pd.DataFrame, *, snapshot_date: Any = None) -> Self:
        """Fit both stages on a LoanTable.

        Stage 1 uses resolved rows (``y_resolved``) with a known ``y_default``; Stage 2 the
        resolved defaults with a finite ``y_recovery_rate``. Rows whose ``y_historical_arm``
        is ``-1`` (policy not mapped onto the arm grid) are skipped by both stages.

        The policy-effect model (structured leaf prior) is fitted and installed when
        ``cfg.hierarchy.policy_effect`` is on, an ``arm_space`` was given and the table has
        the policy columns (``int_rate, term, loan_amnt, installment, annual_inc``);
        otherwise any previous tilt is removed. :attr:`policy_effect_info` says which.
        ``snapshot_date`` (default: the constructor's) defines the elapsed-term loans.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise DataSourceError(f"LoanTable is missing derived columns: {missing}")
        resolved = _bool_mask(table["y_resolved"])
        res = table.loc[resolved]
        self.default_model.fit(res["y_group"], res["y_historical_arm"], res["y_default"])
        dflt = res["y_default"].to_numpy(dtype=float, na_value=np.nan) == 1.0
        dres = res.loc[dflt]
        self._install_term_level()
        self.recovery_model.fit(dres["y_group"], dres["y_historical_arm"], dres["y_recovery_rate"])
        self._fit_policy_effect(res, snapshot_date if snapshot_date is not None else self.snapshot_date)
        return self

    def _fit_policy_effect(self, resolved: pd.DataFrame, snapshot_date: Any) -> None:
        hcfg = self.default_model.cfg
        if not hcfg.policy_effect:
            self._policy_effect_status = "disabled"
        elif self.arm_space is None:
            self._policy_effect_status = "no_arm_space"
        else:
            cols = [c for c in policy_effect_columns(hcfg.policy_effect_cfg.features) if c not in resolved.columns]
            if cols:
                self._policy_effect_status = f"missing_columns: {', '.join(cols)}"
            else:
                model = PolicyEffectModel(hcfg.policy_effect_cfg).fit(resolved, snapshot_date=snapshot_date)
                self.default_model.set_leaf_prior_tilt(PolicyEffectTilt(model, self.arm_space))
                self._policy_effect_status = "fitted" if model.has_effect else "fallback"
                return
        self.default_model.set_leaf_prior_tilt(None)

    @property
    def policy_effect(self) -> PolicyEffectModel | None:
        """The fitted :class:`PolicyEffectModel` behind the Stage-1 leaf-prior tilt (or ``None``)."""
        tilt = self.default_model.leaf_prior_tilt
        return tilt.model if isinstance(tilt, PolicyEffectTilt) else None

    @property
    def policy_effect_info(self) -> dict[str, Any]:
        """``status`` (``fitted`` / ``fallback`` / ``disabled`` / ``no_arm_space`` /
        ``missing_columns: …`` / ``not_fitted``) and the tilt's diagnostics (β, n_fit, …)."""
        diag = self.default_model.diagnostics["policy_effect"]
        return {"status": self._policy_effect_status, **{k: v for k, v in diag.items() if k != "status"},
                "model_status": diag.get("status")}

    def update(self, group: str, arm: int, defaulted: bool, recovery_rate: float | None = None) -> None:
        """Online update with one resolved loan; the recovery rate is used only if it defaulted."""
        self.default_model.update(group, arm, defaulted)
        if (
            recovery_rate is not None
            and not _is_missing(defaulted)
            and bool(defaulted)
            and not _is_missing(recovery_rate)
        ):
            self.recovery_model.update(group, arm, recovery_rate)

    def refresh_priors(self) -> None:
        """Re-derive Stage-1 child priors from the current counts (see the hierarchy docs)."""
        self.default_model.refresh_priors()

    # ----- prediction ---------------------------------------------------------------------
    def risk(self, group: str, arm: int) -> RiskEstimate:
        p = self.default_model.posterior(group, arm)
        rec, src = self.recovery_model.resolve(group, arm)
        return RiskEstimate(
            group=str(group),
            arm=int(arm),
            p_default=p,
            recovery=rec,
            n_default_obs=self.default_model.n_obs(group, arm),
            n_recovery_obs=self.recovery_model.n_obs(group, arm),
            recovery_source=src,
        )

    def sample(
        self, group: str, arm: int, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """``n`` independent posterior draws ``(p, R)``."""
        est = self.risk(group, arm)
        return est.p_default.sample(rng, n), est.recovery.sample(rng, n)

    def predict_table(
        self,
        table: pd.DataFrame,
        group_col: str = "y_group",
        arm_col: str = "y_historical_arm",
        hdi_mass: float = 0.95,
    ) -> pd.DataFrame:
        """Per-row ``p_default_mean, p_default_hdi_low, p_default_hdi_high,
        expected_recovery, expected_loss_fraction`` (same index as ``table``).

        Posteriors are computed once per distinct ``(group, arm)`` and the HDIs in one
        vectorised call; rows with a missing group or ``arm < 0`` get NaN.
        """
        cols = ["p_default_mean", "p_default_hdi_low", "p_default_hdi_high", "expected_recovery", "expected_loss_fraction"]
        n = len(table)
        out = {c: np.full(n, np.nan) for c in cols}
        if n:
            g = table[group_col].astype("string").reset_index(drop=True)
            a = pd.to_numeric(table[arm_col].reset_index(drop=True), errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            with np.errstate(invalid="ignore"):
                valid = ~g.isna().to_numpy() & np.isfinite(a) & (a >= 0)
            if valid.any():
                pairs = pd.DataFrame({"g": g[valid].to_numpy(dtype=object), "a": a[valid].astype(np.int64)})
                codes = pairs.groupby(["g", "a"], sort=False).ngroup().to_numpy()
                uniq = pairs.drop_duplicates()
                alphas, betas, recs = [], [], []
                for gk, ak in zip(uniq["g"], uniq["a"], strict=True):
                    post = self.default_model.posterior(gk, int(ak))
                    alphas.append(post.alpha)
                    betas.append(post.beta)
                    recs.append(self.recovery_model.expected_recovery(gk, int(ak)))
                al, be, rc = np.asarray(alphas), np.asarray(betas), np.asarray(recs)
                lo, hi = beta_hdi(al, be, hdi_mass)
                mean = al / (al + be)
                out["p_default_mean"][valid] = mean[codes]
                out["p_default_hdi_low"][valid] = lo[codes]
                out["p_default_hdi_high"][valid] = hi[codes]
                out["expected_recovery"][valid] = rc[codes]
                out["expected_loss_fraction"][valid] = (mean * (1.0 - rc))[codes]
        return pd.DataFrame(out, index=table.index)

    # ----- serialisation ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.default_model.is_fitted

    def to_dict(self) -> dict[str, Any]:
        """JSON state of both stages; the Stage-1 dict carries the leaf-prior tilt
        (policy-effect model + arm space), so :meth:`from_dict` restores the tilt exactly."""
        return {
            "type": "TwoStageBetaBeta",
            "version": 2,
            "default_model": self.default_model.to_dict(),
            "recovery_model": self.recovery_model.to_dict(),
            "arm_space": None if self.arm_space is None else self.arm_space.to_dict(),
            "snapshot_date": self.snapshot_date,
            "policy_effect_status": self._policy_effect_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TwoStageBetaBeta:
        obj = cls()
        obj.default_model = HierarchicalBetaBernoulli.from_dict(data["default_model"])
        obj.recovery_model = BetaMoMRecovery.from_dict(data["recovery_model"])
        if data.get("arm_space"):
            obj.arm_space = ArmSpace.from_dict(data["arm_space"])
        elif isinstance(obj.default_model.leaf_prior_tilt, PolicyEffectTilt):
            obj.arm_space = obj.default_model.leaf_prior_tilt.arm_space
        snap = data.get("snapshot_date")
        obj.snapshot_date = None if snap is None else str(snap)
        default_status = "fitted" if obj.default_model.leaf_prior_tilt is not None else "not_fitted"
        obj._policy_effect_status = str(data.get("policy_effect_status", default_status))
        return obj

    def __repr__(self) -> str:
        return f"TwoStageBetaBeta({self.default_model!r}, {self.recovery_model!r})"
