"""Structured leaf prior: the effect of the lending policy on default (**[decision]**).

The proposal (§3.3) specifies the leaf level ``p_{g,a} ~ Beta(α_{g,a}, β_{g,a})`` of the
hierarchical default model but leaves ``α_{g,a}, β_{g,a}`` open, and its literature review
(§2.2.1, Scott 2010) motivates a *structured* model across the policy factors of an arm
instead of independent cells. Shrinking every leaf to its group mean (SPEC §2.2) makes an
arm nobody in the group ever received - a grade-A borrower at 28 % - look exactly as safe
as the group average, so the reward is maximised by the largest loan at the highest rate.

:class:`PolicyEffectModel` is an **offset logistic regression** fitted on the resolved
training loans::

    logit P(D_i = 1) = logit(m_{g(i)}) + β · (x_i − x̄_{g(i)})
    x_i = [int_rate, 1[term = 60], pti_i, log(loan_amnt)],  pti_i = clip(12·installment_i / annual_inc_i, 0, 1.5)
    m_g = (d_g + 1) / (n_g + 2)                               (shrunk group default rate)

The features are centred **within** each group, so β is a within-group effect, free of the
grade → rate confounding between groups. β is found by Newton-Raphson on the mean
log-likelihood with a tiny ridge on the standardised coefficients (fully vectorised).
:meth:`PolicyEffectModel.effect` evaluates ``β · (x_{g,a} − x̄_g)`` for an arm, with the
arm's own rate / term / amount and its payment-to-income at the group's median income,
clipped to ``±max_abs_logit_shift``. :class:`PolicyEffectTilt` binds a model to an
:class:`~bayesian_credit.data.arms.ArmSpace`; installed in
:class:`~bayesian_credit.models.hierarchical.HierarchicalBetaBernoulli` it moves the leaf
prior mean to ``σ(logit(m_g) + effect(g, a))`` (same κ_leaf, same conjugate update).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import POLICY_EFFECT_FEATURES, PolicyEffectConfig
from ..data.arms import ArmSpace, LendingPolicy
from ..data.outcomes import add_months
from ..exceptions import NumericalInstabilityError

__all__ = [
    "FEATURE_COLUMNS",
    "PolicyEffectModel",
    "PolicyEffectTilt",
    "TiltTable",
    "policy_effect_columns",
    "tilt_from_dict",
]

#: LoanTable columns each policy feature is computed from.
FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "int_rate": ("int_rate",),
    "term60": ("term",),
    "pti": ("installment", "annual_inc"),
    "log_amount": ("loan_amnt",),
}
#: Standardised columns with a smaller within-group sd than this are dropped (no variation).
_SD_FLOOR = 1e-12
#: Newton-Raphson stops when the largest standardised step is below this.
_TOL = 1e-9
#: Hessians with a larger condition number are treated as singular.
_MAX_COND = 1e12


def policy_effect_columns(features: tuple[str, ...] | list[str] = POLICY_EFFECT_FEATURES) -> tuple[str, ...]:
    """LoanTable columns needed to fit a policy-effect model on ``features``."""
    out: list[str] = []
    for f in features:
        for c in FEATURE_COLUMNS[f]:
            if c not in out:
                out.append(c)
    return tuple(out)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _clean_float(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _nan(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _log1pexp(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


class PolicyEffectModel:
    """Offset logistic regression of default on the policy features, centred within
    borrower groups (module docstring; docs/MATH.md "Structured leaf prior").

    ``fit(table)`` on a LoanTable (``y_resolved``, ``y_default``, ``y_group`` and the
    policy columns); ``effect(group, policy)`` → logit shift of the leaf prior; ``beta`` /
    :attr:`coefficients`; :attr:`diagnostics` (status, convergence, sample, n_fit, standard
    errors); ``to_dict`` / ``from_dict``. A fit that is impossible or unstable (too few rows,
    no within-group variation, singular Hessian, no convergence) falls back to ``β = 0``
    (no tilt) and says why in ``diagnostics["reason"]``.
    """

    def __init__(self, cfg: PolicyEffectConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else PolicyEffectConfig()
        self.cfg.validate()
        self.feature_names: tuple[str, ...] = tuple(self.cfg.features)
        k = len(self.feature_names)
        self.beta = np.zeros(k)
        self.group_means: dict[str, np.ndarray] = {}
        self.group_median_income: dict[str, float] = {}
        self.group_n: dict[str, int] = {}
        self.global_means = np.full(k, np.nan)
        self.global_median_income = float("nan")
        self.n_fit = 0
        self.fitted = False
        self.diagnostics: dict[str, Any] = {"status": "not_fitted"}

    # ----- features -----------------------------------------------------------------------
    def _row_features(
        self,
        int_rate: np.ndarray,
        term: np.ndarray,
        loan_amnt: np.ndarray,
        installment: np.ndarray,
        annual_inc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(X (n, k), valid (n,))`` for loans (the features of the configured names)."""
        n = int_rate.size
        cols: list[np.ndarray] = []
        valid = np.ones(n, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in self.feature_names:
                if f == "int_rate":
                    x = int_rate.astype(float)
                    valid &= np.isfinite(x) & (x >= 0)
                elif f == "term60":
                    valid &= np.isfinite(term) & (term > 0)
                    x = (term == 60).astype(float)
                elif f == "pti":
                    ok = np.isfinite(installment) & (installment >= 0) & np.isfinite(annual_inc) & (annual_inc > 0)
                    valid &= ok
                    x = np.clip(np.where(ok, installment * 12.0 / np.where(ok, annual_inc, 1.0), 0.0), 0.0, self.cfg.pti_cap)
                else:  # log_amount
                    ok = np.isfinite(loan_amnt) & (loan_amnt > 0)
                    valid &= ok
                    x = np.log(np.where(ok, loan_amnt, 1.0))
                cols.append(x)
        X = np.column_stack(cols) if cols else np.zeros((n, 0))
        return X, valid

    def arm_features(self, group: str, policy: LendingPolicy) -> list[float]:
        """``x_{g,a}`` of an arm: its rate, term flag, payment-to-income at the group's
        median income (the group's mean PTI when the income is unknown) and log amount."""
        g = str(group)
        means = self.group_means.get(g, self.global_means)
        inc = self.group_median_income.get(g, self.global_median_income)
        out: list[float] = []
        for j, f in enumerate(self.feature_names):
            if f == "int_rate":
                out.append(float(policy.rate))
            elif f == "term60":
                out.append(1.0 if int(policy.term) == 60 else 0.0)
            elif f == "pti":
                if math.isfinite(inc) and inc > 0:
                    out.append(min(max(float(policy.installment) * 12.0 / inc, 0.0), float(self.cfg.pti_cap)))
                else:
                    out.append(float(means[j]))
            else:
                out.append(math.log(float(policy.amount)))
        return out

    # ----- estimation ------------------------------------------------------------------------
    def fit(
        self,
        table: pd.DataFrame,
        *,
        snapshot_date: Any = None,
        group_col: str = "y_group",
    ) -> Self:
        """Fit on the resolved rows of a LoanTable (see :meth:`fit_arrays`).

        Rows used: ``y_resolved`` with a 0/1 ``y_default``, a group and valid policy
        features (and ``y_historical_arm ≥ 0`` when present - the rows Stage 1 is fitted
        on). When ``issue_d`` and ``snapshot_date`` are known, the regression uses only the
        loans whose term has elapsed by the snapshot (``issue_d + term ≤ snapshot``; no
        resolution bias towards early defaults) if at least ``cfg.min_elapsed_rows`` exist.
        """
        n = len(table)
        resolved = np.ones(n, dtype=bool)
        if "y_resolved" in table.columns:
            r = table["y_resolved"]
            resolved = (r.to_numpy(dtype=float, na_value=0.0) != 0.0) if r.dtype != bool else r.to_numpy()
        y = _num(table, "y_default") if "y_default" in table.columns else np.full(n, np.nan)
        groups = table[group_col].astype("string").to_numpy(dtype=object, na_value=None)
        keep = resolved & np.isfinite(y) & np.array([g is not None for g in groups], dtype=bool)
        if "y_historical_arm" in table.columns:
            arm = _num(table, "y_historical_arm")
            keep &= np.isfinite(arm) & (arm >= 0)

        def col(name: str) -> np.ndarray:
            return _num(table, name) if name in table.columns else np.full(n, np.nan)

        elapsed = None
        if snapshot_date is not None and "issue_d" in table.columns and "term" in table.columns:
            end = add_months(table["issue_d"].reset_index(drop=True), col("term"))
            elapsed = (end <= pd.Timestamp(snapshot_date)).to_numpy(dtype=bool)
        idx = np.flatnonzero(keep)
        return self.fit_arrays(
            groups[idx],
            y[idx],
            col("int_rate")[idx],
            col("term")[idx],
            col("loan_amnt")[idx],
            col("installment")[idx],
            col("annual_inc")[idx],
            elapsed=None if elapsed is None else elapsed[idx],
        )

    def fit_arrays(
        self,
        groups: Any,
        defaults: Any,
        int_rate: Any,
        term: Any,
        loan_amnt: Any,
        installment: Any,
        annual_inc: Any,
        *,
        elapsed: Any = None,
    ) -> Self:
        """Fit from aligned arrays of resolved loans (``defaults`` 0/1).

        1. reference rows = rows with valid features: per-group feature means ``x̄_g`` and
           median income (used by :meth:`effect`), global ones for unseen groups;
        2. regression sample = the elapsed rows when ``elapsed`` marks at least
           ``cfg.min_elapsed_rows`` of them, else all reference rows;
        3. offset ``logit((d_g + 1)/(n_g + 2))`` and within-group centring on that sample;
        4. Newton-Raphson; any :class:`NumericalInstabilityError` → ``β = 0``.
        """
        cfg = self.cfg
        k = len(self.feature_names)
        g = np.asarray(pd.Series(groups, dtype=object).astype(str).to_numpy(dtype=object))
        y = np.asarray(defaults, dtype=float).reshape(-1)
        arrays = [np.asarray(a, dtype=float).reshape(-1) for a in (int_rate, term, loan_amnt, installment, annual_inc)]
        if not all(a.size == y.size for a in arrays) or g.size != y.size:
            raise ValueError("fit_arrays: all inputs must have the same length")
        if y.size and not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("defaults must be 0/1")
        X, valid = self._row_features(*arrays)
        inc = arrays[4]
        g, y, X, inc = g[valid], y[valid], X[valid], inc[valid]
        el = None if elapsed is None else np.asarray(elapsed, dtype=bool).reshape(-1)[valid]

        self.beta = np.zeros(k)
        self.fitted = True
        n_ref = int(y.size)
        diag: dict[str, Any] = {
            "status": "fallback",
            "reason": None,
            "converged": False,
            "n_iter": 0,
            "n_reference": n_ref,
            "n_dropped_invalid": int((~valid).sum()),
            "features": list(self.feature_names),
            "ridge": float(cfg.ridge),
            "max_abs_logit_shift": float(cfg.max_abs_logit_shift),
        }
        # ---- reference statistics (effect centring) ----
        codes, uniq = pd.factorize(pd.Series(g, dtype=object), sort=True)
        n_g = np.bincount(codes, minlength=len(uniq)) if n_ref else np.zeros(0, dtype=int)
        self.group_means = {}
        self.group_median_income = {}
        self.group_n = {}
        if n_ref:
            sums = np.stack([np.bincount(codes, X[:, j], minlength=len(uniq)) for j in range(k)], axis=1) if k else np.zeros((len(uniq), 0))
            means = sums / n_g[:, None]
            inc_ok = np.isfinite(inc) & (inc > 0)
            med = (
                pd.Series(np.where(inc_ok, inc, np.nan)).groupby(codes).median().reindex(range(len(uniq))).to_numpy(dtype=float)
            )
            for i, key in enumerate(uniq):
                self.group_means[str(key)] = means[i].copy()
                self.group_median_income[str(key)] = float(med[i])
                self.group_n[str(key)] = int(n_g[i])
            self.global_means = X.mean(axis=0) if k else np.zeros(0)
            self.global_median_income = float(np.median(inc[inc_ok])) if inc_ok.any() else float("nan")
        else:
            self.global_means = np.full(k, np.nan)
            self.global_median_income = float("nan")
        diag["n_groups"] = len(uniq)

        # ---- regression sample ----
        sample = "resolved"
        use = np.ones(n_ref, dtype=bool)
        n_el = int(el.sum()) if el is not None else 0
        if el is not None and n_el >= max(int(cfg.min_elapsed_rows), 1):
            use, sample = el, "elapsed"
        diag["n_elapsed"] = n_el
        diag["sample"] = sample
        n_fit = int(use.sum())
        self.n_fit = n_fit
        diag["n_fit"] = n_fit
        if n_fit < int(cfg.min_rows) or k == 0:
            diag["reason"] = f"insufficient data ({n_fit} usable rows < min_rows {cfg.min_rows})" if k else "no features"
            self.diagnostics = self._finish(diag)
            return self
        try:
            beta, info = self._newton(codes[use], y[use], X[use])
        except NumericalInstabilityError as exc:
            diag["reason"] = str(exc)
            self.diagnostics = self._finish(diag)
            return self
        self.beta = beta
        diag.update(info)
        diag["status"] = "fitted"
        self.diagnostics = self._finish(diag)
        return self

    def _newton(self, codes: np.ndarray, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Newton-Raphson for the centred offset logistic regression (raises
        :class:`NumericalInstabilityError` on a singular design or no convergence)."""
        cfg = self.cfg
        n, k = X.shape
        ng = int(codes.max()) + 1
        cnt = np.bincount(codes, minlength=ng).astype(float)
        d = np.bincount(codes, y, minlength=ng)
        offset = np.log((d + 1.0) / (cnt + 2.0)) - np.log1p(-(d + 1.0) / (cnt + 2.0))
        off = offset[codes]
        centre = np.stack([np.bincount(codes, X[:, j], minlength=ng) / np.maximum(cnt, 1.0) for j in range(k)], axis=1)
        Z = X - centre[codes]
        sd = Z.std(axis=0)
        scale = np.maximum(1.0, np.abs(X).max(axis=0))
        active = sd > _SD_FLOOR * scale
        dropped = [self.feature_names[j] for j in range(k) if not active[j]]
        if not active.any():
            raise NumericalInstabilityError("singular design: no policy feature varies within any group")
        Zs = Z[:, active] / sd[active]
        ka = Zs.shape[1]
        ridge = float(cfg.ridge)
        eye = np.eye(ka)

        def objective(b: np.ndarray) -> float:
            eta = off + Zs @ b
            return float(np.mean(_log1pexp(eta) - y * eta) + 0.5 * ridge * (b @ b))

        b = np.zeros(ka)
        f = objective(b)
        f0 = f
        converged = False
        n_iter = 0
        H = eye
        for it in range(1, int(cfg.max_iter) + 1):
            n_iter = it
            p = expit(off + Zs @ b)
            w = p * (1.0 - p)
            grad = Zs.T @ (y - p) / n - ridge * b
            H = (Zs * w[:, None]).T @ Zs / n + ridge * eye
            if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(H))):
                raise NumericalInstabilityError("non-finite gradient / Hessian in the policy-effect fit")
            ev = np.linalg.eigvalsh(H)
            if ev[0] <= 0 or ev[-1] / ev[0] > _MAX_COND:
                raise NumericalInstabilityError(
                    f"singular design: Hessian condition number {ev[-1] / max(ev[0], 1e-300):.3g} (ridge {ridge:g})"
                )
            step = np.linalg.solve(H, grad)
            t = 1.0
            while True:  # step halving keeps the (convex) objective monotone
                cand = b + t * step
                fc = objective(cand)
                if np.isfinite(fc) and fc <= f + 1e-15:
                    break
                t *= 0.5
                if t < 1e-6:
                    break
            b, f = cand, fc
            if float(np.max(np.abs(t * step))) < _TOL:
                converged = True
                break
        if not converged or not np.all(np.isfinite(b)):
            raise NumericalInstabilityError(f"Newton-Raphson did not converge in {cfg.max_iter} iterations")
        beta = np.zeros(k)
        beta[active] = b / sd[active]
        se = np.full(k, np.nan)
        try:
            cov = np.linalg.inv(H * n)
            se[active] = np.sqrt(np.clip(np.diag(cov), 0.0, None)) / sd[active]
        except np.linalg.LinAlgError:  # pragma: no cover - H was checked above
            pass
        info = {
            "converged": True,
            "n_iter": n_iter,
            "dropped_features": dropped,
            "std_errors": {name: _clean_float(v) for name, v in zip(self.feature_names, se, strict=True)},
            "within_group_sd": {name: float(v) for name, v in zip(self.feature_names, sd, strict=True)},
            "mean_log_loss_offset_only": f0,
            "mean_log_loss": float(np.mean(_log1pexp(off + Zs @ b) - y * (off + Zs @ b))),
        }
        return beta, info

    def _finish(self, diag: dict[str, Any]) -> dict[str, Any]:
        diag["beta"] = self.coefficients
        diag.setdefault("dropped_features", [])
        return diag

    # ----- prediction ---------------------------------------------------------------------------
    @property
    def coefficients(self) -> dict[str, float]:
        """``{feature: β_j}`` (logit change per unit of the feature)."""
        return {name: float(v) for name, v in zip(self.feature_names, self.beta, strict=True)}

    @property
    def has_effect(self) -> bool:
        return bool(np.any(self.beta != 0.0))

    def raw_effect(self, group: str, policy: LendingPolicy | None) -> float:
        """``β · (x_{g,a} − x̄_g)`` before the clip (0 for the null arm / no effect)."""
        if policy is None or policy.is_null or not self.has_effect:
            return 0.0
        g = str(group)
        means = self.group_means.get(g, self.global_means)
        x = self.arm_features(g, policy)
        s = 0.0
        for bj, xj, mj in zip(self.beta.tolist(), x, means.tolist(), strict=True):
            if bj != 0.0:
                s += bj * (xj - mj)
        return s if math.isfinite(s) else 0.0

    def effect(self, group: str, policy: LendingPolicy | None) -> float:
        """Logit shift of the leaf prior of ``(group, policy)``: ``β · (x_{g,a} − x̄_g)``
        clipped to ``±cfg.max_abs_logit_shift``; 0 for the null arm. Unseen groups use the
        global feature means and median income."""
        s = self.raw_effect(group, policy)
        c = float(self.cfg.max_abs_logit_shift)
        return min(max(s, -c), c)

    # ----- serialisation ---------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PolicyEffectModel",
            "version": 1,
            "cfg": self.cfg.to_dict(),
            "feature_names": list(self.feature_names),
            "beta": [float(v) for v in self.beta],
            "group_means": {g: [_clean_float(v) for v in m] for g, m in self.group_means.items()},
            "group_median_income": {g: _clean_float(v) for g, v in self.group_median_income.items()},
            "group_n": {g: int(v) for g, v in self.group_n.items()},
            "global_means": [_clean_float(v) for v in self.global_means],
            "global_median_income": _clean_float(self.global_median_income),
            "n_fit": int(self.n_fit),
            "fitted": bool(self.fitted),
            "diagnostics": _jsonable(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyEffectModel:
        obj = cls(PolicyEffectConfig.from_dict(data.get("cfg")))
        names = tuple(data.get("feature_names") or obj.feature_names)
        obj.feature_names = names
        obj.beta = np.array([float(v) for v in data.get("beta", [0.0] * len(names))], dtype=float)
        obj.group_means = {str(g): np.array([_nan(v) for v in m], dtype=float) for g, m in (data.get("group_means") or {}).items()}
        obj.group_median_income = {str(g): _nan(v) for g, v in (data.get("group_median_income") or {}).items()}
        obj.group_n = {str(g): int(v) for g, v in (data.get("group_n") or {}).items()}
        gm = data.get("global_means")
        obj.global_means = np.array([_nan(v) for v in gm], dtype=float) if gm is not None else np.full(len(names), np.nan)
        obj.global_median_income = _nan(data.get("global_median_income"))
        obj.n_fit = int(data.get("n_fit", 0))
        obj.fitted = bool(data.get("fitted", True))
        obj.diagnostics = dict(data.get("diagnostics") or {})
        return obj

    def __repr__(self) -> str:
        coef = ", ".join(f"{k}={v:.3g}" for k, v in self.coefficients.items())
        return f"PolicyEffectModel({coef}; n_fit={self.n_fit}, status={self.diagnostics.get('status')})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _clean_float(value)
    return value


class PolicyEffectTilt:
    """Leaf-prior tilt ``(group, arm) → effect(group, arm_space[arm])`` of a fitted
    :class:`PolicyEffectModel` over an :class:`ArmSpace` (memoised; arms outside the
    space and the null arm → 0). Serialisable, so a hierarchy carrying it round-trips."""

    def __init__(self, model: PolicyEffectModel, arm_space: ArmSpace) -> None:
        self.model = model
        self.arm_space = arm_space
        self._cache: dict[tuple[str, int], float] = {}

    def __call__(self, group: str, arm: int) -> float:
        key = (str(group), int(arm))
        hit = self._cache.get(key)
        if hit is None:
            a = key[1]
            hit = self.model.effect(key[0], self.arm_space[a]) if 0 <= a < self.arm_space.n_arms else 0.0
            self._cache[key] = hit
        return hit

    @property
    def n_arms(self) -> int:
        return self.arm_space.n_arms

    def diagnostics(self) -> dict[str, Any]:
        d = dict(self.model.diagnostics)
        return {
            "status": d.get("status"),
            "reason": d.get("reason"),
            "beta": self.model.coefficients,
            "std_errors": d.get("std_errors"),
            "n_fit": int(self.model.n_fit),
            "sample": d.get("sample"),
            "converged": bool(d.get("converged", False)),
            "n_iter": d.get("n_iter"),
            "max_abs_logit_shift": float(self.model.cfg.max_abs_logit_shift),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"type": "PolicyEffectTilt", "version": 1, "model": self.model.to_dict(), "arm_space": self.arm_space.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyEffectTilt:
        return cls(PolicyEffectModel.from_dict(data["model"]), ArmSpace.from_dict(data["arm_space"]))

    def __repr__(self) -> str:
        return f"PolicyEffectTilt({self.model!r}, n_arms={self.n_arms})"


class TiltTable:
    """A frozen ``{(group, arm): tilt}`` table used as a tilt (unknown pairs → 0). This is
    what a hierarchy restores when its tilt function was not serialisable."""

    def __init__(self, rows: Any = ()) -> None:
        self.table: dict[tuple[str, int], float] = {(str(g), int(a)): float(t) for g, a, t in rows}

    def __call__(self, group: str, arm: int) -> float:
        return self.table.get((str(group), int(arm)), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "TiltTable", "version": 1, "rows": [[g, a, t] for (g, a), t in sorted(self.table.items())]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TiltTable:
        return cls(data.get("rows") or ())


def tilt_from_dict(data: Mapping[str, Any] | None) -> PolicyEffectTilt | TiltTable | None:
    """Rebuild a serialised tilt (``PolicyEffectTilt`` / ``TiltTable``); ``None`` → ``None``."""
    if not data:
        return None
    kind = data.get("type")
    if kind == "PolicyEffectTilt":
        return PolicyEffectTilt.from_dict(data)
    if kind == "TiltTable":
        return TiltTable.from_dict(data)
    raise ValueError(f"unknown tilt type {kind!r}")
