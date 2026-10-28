"""Stage 2 of the Two-Stage Beta-Beta model: recovery given default, Beta method of moments.

Proposal §3.3 (pp. 26-27)::

    R_{g,a} | D = 1 ~ Beta(α^R_{g,a}, β^R_{g,a})
    α^R_{g,a} = μ (μ(1 − μ)/σ² − 1)
    β^R_{g,a} = (1 − μ) (μ(1 − μ)/σ² − 1)

with μ, σ² and N the observed mean, variance and count of the recovery rates of defaulted
loans in leaf (g, a). The closed form itself lives in :meth:`BetaPosterior.from_moments`.

**[decision] Recovery by term** (docs/MATH.md §2.9). The fallback chain of a sparse leaf is
leaf (g, a) → **(g, term of a)** → g → global → prior. On the principal basis a 60-month
default recovers less (Lending Club: 0.38 vs 0.49 of principal for 36-month loans; less
principal has been amortised when it defaults), so an unobserved 60-month leaf should not
borrow a group mean dominated by 36-month loans. The arm → term map comes from the arm
space (``term_of_arm``; without one the level is skipped and the chain is the SPEC's).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd

from ..config import RecoveryConfig
from ..exceptions import NumericalInstabilityError
from .beta import BetaPosterior
from .hierarchical import _clean_observations, _is_missing

__all__ = ["FALLBACK_PRIOR", "LEVELS", "SOURCES", "BetaMoMRecovery", "RunningMoments"]

#: Weakly informative prior used when not even the global level can be fitted: Beta(2, 5)
#: (mean ≈ 0.29, close to Lending Club's empirical ≈ 0.37 on the principal basis).
FALLBACK_PRIOR: tuple[float, float] = (2.0, 5.0)
LEVELS: tuple[str, ...] = ("leaf", "group_term", "group", "global")
SOURCES: tuple[str, ...] = (*LEVELS, "prior")


@dataclass
class RunningMoments:
    """Welford running count / mean / M2 (sum of squared deviations from the mean).

    ``push(x)``: ``n ← n + 1; δ = x − mean; mean ← mean + δ/n; M2 ← M2 + δ(x − mean)``.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        """Sample variance ``M2/(n − 1)`` (the observed σ² of the MoM); NaN for ``n < 2``."""
        return self.m2 / (self.n - 1) if self.n >= 2 else math.nan

    def to_list(self) -> list[float]:
        return [int(self.n), float(self.mean), float(self.m2)]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> RunningMoments:
        return cls(int(values[0]), float(values[1]), float(values[2]))


class BetaMoMRecovery:
    """Recovery-rate model ``R | D = 1 ~ Beta(α^R, β^R)`` by method of moments (Stage 2).

    Running statistics (count, mean, M2 via Welford) are kept per leaf ``(g, a)``, per group
    and globally, so :meth:`update` works online and posteriors are re-derived from
    ``(mean, var, n)`` on demand. σ² is the sample variance ``M2/(n−1)``.

    Fallback chain (SPEC §2.3, plus the term level): leaf → (group, term) → group → global,
    moving up whenever the level has ``n < cfg.min_obs`` observations or its method of moments
    is invalid
    (:meth:`BetaPosterior.from_moments` raises :class:`NumericalInstabilityError` when
    ``σ² ≥ μ(1−μ)`` or σ² is undefined; the error is caught and the next level is tried).
    μ is clipped to ``[eps, 1 − eps]`` and a near-zero variance caps ``α + β`` at
    ``cfg.max_concentration``. If even the global level is unusable the weakly informative
    :data:`FALLBACK_PRIOR` Beta(2, 5) is returned with source ``"prior"``.

    :attr:`diagnostics` reports, over all known nodes, where each leaf's Beta comes from and
    how many nodes fell back because of too few observations or an invalid MoM.

    ``term_of_arm`` (an :class:`~bayesian_credit.data.arms.ArmSpace`, a sequence of terms by
    arm index, a ``{arm: term}`` mapping or a callable) enables the ``group_term`` level;
    :meth:`set_term_of_arm` installs it later. It survives :meth:`fit` and round-trips
    through :meth:`to_dict` as the ``{arm: term}`` map.
    """

    def __init__(self, cfg: RecoveryConfig | None = None, *, term_of_arm: Any = None) -> None:
        self.cfg = cfg if cfg is not None else RecoveryConfig()
        self.cfg.validate()
        self._arm_terms: dict[int, int] | None = None
        self._term_fn: Callable[[int], Any] | None = None
        self._reset()
        self.set_term_of_arm(term_of_arm)

    def _reset(self) -> None:
        self._global = RunningMoments()
        self._groups: dict[str, RunningMoments] = {}
        self._group_terms: dict[tuple[str, int], RunningMoments] = {}
        self._leaves: dict[tuple[str, int], RunningMoments] = {}
        self._clipped = 0
        self._skipped = 0
        self._fitted = False

    # ----- arm → term ---------------------------------------------------------------------
    def set_term_of_arm(self, term_of_arm: Any) -> None:
        """Install the arm → term map (ArmSpace, sequence, mapping, callable or ``None``).

        Changing it re-buckets nothing: re-``fit`` to rebuild the ``group_term`` statistics.
        """
        self._term_fn = None
        if term_of_arm is None:
            self._arm_terms = None
        elif hasattr(term_of_arm, "terms") and hasattr(term_of_arm, "n_arms"):  # ArmSpace
            self._arm_terms = {i: int(t) for i, t in enumerate(np.asarray(term_of_arm.terms).tolist())}
        elif isinstance(term_of_arm, Mapping):
            self._arm_terms = {int(k): int(v) for k, v in term_of_arm.items()}
        elif callable(term_of_arm):
            self._arm_terms = {}
            self._term_fn = term_of_arm
        else:
            self._arm_terms = {i: int(t) for i, t in enumerate(term_of_arm)}

    @property
    def has_terms(self) -> bool:
        return self._arm_terms is not None

    def term_of(self, arm: int | None) -> int | None:
        """Term of ``arm`` (``None`` without a map, for unknown arms and the null arm)."""
        if self._arm_terms is None or arm is None:
            return None
        a = int(arm)
        t = self._arm_terms.get(a)
        if t is None and self._term_fn is not None:
            raw = self._term_fn(a)
            t = None if raw is None else int(raw)
            if t is not None:
                self._arm_terms[a] = t
        return t if t is not None and t > 0 else None

    # ----- estimation -------------------------------------------------------------------
    def fit(self, groups: Sequence[str], arms: Sequence[int], recovery_rates: Sequence[float]) -> Self:
        """Batch fit on the recovery rates of *defaulted* loans (replaces previous state).

        Rows with a missing group, ``arm < 0`` or a missing rate are skipped; rates outside
        ``[0, 1]`` are clipped (both counted in :attr:`diagnostics`).
        """
        g, a, x, skipped = _clean_observations(groups, arms, recovery_rates)
        self._reset()
        self._skipped = skipped
        clipped = np.clip(x, 0.0, 1.0)
        self._clipped = int(np.count_nonzero(clipped != x))
        x = clipped
        if x.size:
            frame = pd.DataFrame({"g": g, "a": a, "x": x})
            for (gk, ak), stats in self._batch_moments(frame, ["g", "a"]).items():
                self._leaves[(str(gk), int(ak))] = stats
            for gk, stats in self._batch_moments(frame, ["g"]).items():
                self._groups[str(gk[0] if isinstance(gk, tuple) else gk)] = stats
            if self.has_terms:
                terms = np.array([self.term_of(int(v)) or -1 for v in a.tolist()], dtype=np.int64)
                known = terms > 0
                if known.any():
                    sub = frame.loc[known].assign(t=terms[known])
                    for (gk, tk), stats in self._batch_moments(sub, ["g", "t"]).items():
                        self._group_terms[(str(gk), int(tk))] = stats
            mean = float(x.mean())
            self._global = RunningMoments(int(x.size), mean, float(((x - mean) ** 2).sum()))
        self._fitted = True
        return self

    @staticmethod
    def _batch_moments(frame: pd.DataFrame, by: list[str]) -> dict[Any, RunningMoments]:
        grouped = frame.groupby(by, sort=True)["x"]
        dev2 = (frame["x"] - grouped.transform("mean")) ** 2
        keys = [frame[c] for c in by]
        table = pd.DataFrame(
            {"n": grouped.size(), "mean": grouped.mean(), "m2": dev2.groupby(keys, sort=True).sum()}
        )
        return {
            idx: RunningMoments(int(n), float(m), float(m2))
            for idx, n, m, m2 in zip(
                table.index, table["n"].to_numpy(), table["mean"].to_numpy(),
                table["m2"].to_numpy(), strict=True,
            )
        }

    def update(self, group: str, arm: int, recovery_rate: float) -> None:
        """Online Welford update of leaf, group and global statistics with one defaulted loan."""
        if _is_missing(recovery_rate) or _is_missing(arm):
            return
        arm_i = int(arm)
        if arm_i < 0:
            return
        x = float(recovery_rate)
        if not math.isfinite(x):
            return
        if x < 0.0 or x > 1.0:
            x = min(max(x, 0.0), 1.0)
            self._clipped += 1
        g = str(group)
        self._global.push(x)
        self._groups.setdefault(g, RunningMoments()).push(x)
        t = self.term_of(arm_i)
        if t is not None:
            self._group_terms.setdefault((g, t), RunningMoments()).push(x)
        self._leaves.setdefault((g, arm_i), RunningMoments()).push(x)

    # ----- resolution -------------------------------------------------------------------
    def _try_level(self, stats: RunningMoments | None) -> tuple[BetaPosterior | None, str]:
        """Beta for one level, or ``(None, reason)`` with reason ∈ {missing, insufficient, invalid}."""
        if stats is None or stats.n == 0:
            return None, "missing"
        if stats.n < self.cfg.min_obs:
            return None, "insufficient"
        try:
            post = BetaPosterior.from_moments(
                stats.mean, stats.var, eps=self.cfg.eps, max_concentration=self.cfg.max_concentration
            )
        except NumericalInstabilityError:
            return None, "invalid"
        return post, "ok"

    def _chain(self, group: str | None, arm: int | None) -> list[tuple[str, RunningMoments | None]]:
        chain: list[tuple[str, RunningMoments | None]] = []
        if group is not None and arm is not None:
            chain.append(("leaf", self._leaves.get((str(group), int(arm)))))
            t = self.term_of(arm)
            if t is not None:
                chain.append(("group_term", self._group_terms.get((str(group), t))))
        if group is not None:
            chain.append(("group", self._groups.get(str(group))))
        chain.append(("global", self._global))
        return chain

    def _resolve_chain(self, chain: list[tuple[str, RunningMoments | None]]) -> tuple[BetaPosterior, str]:
        for level, stats in chain:
            post, _reason = self._try_level(stats)
            if post is not None:
                return post, level
        return BetaPosterior(*FALLBACK_PRIOR), "prior"

    def resolve(self, group: str, arm: int) -> tuple[BetaPosterior, str]:
        """``(posterior, source)`` with source ∈ {"leaf", "group_term", "group", "global",
        "prior"}."""
        return self._resolve_chain(self._chain(group, arm))

    def group_term_posterior(self, group: str, term: int) -> tuple[BetaPosterior, str]:
        """``(posterior, source)`` of the ``(group, term)`` node and its fallbacks."""
        chain = [("group_term", self._group_terms.get((str(group), int(term))))]
        chain += self._chain(group, None)
        return self._resolve_chain(chain)

    def posterior(self, group: str, arm: int) -> BetaPosterior:
        return self.resolve(group, arm)[0]

    def group_posterior(self, group: str) -> BetaPosterior:
        return self._resolve_chain(self._chain(group, None))[0]

    def global_posterior(self) -> BetaPosterior:
        return self._resolve_chain(self._chain(None, None))[0]

    def source(self, group: str, arm: int) -> str:
        return self.resolve(group, arm)[1]

    def expected_recovery(self, group: str, arm: int) -> float:
        """``E[R_{g,a} | D = 1]``."""
        return self.posterior(group, arm).mean

    def sample(self, group: str, arm: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.posterior(group, arm).sample(rng, n)

    # ----- statistics -------------------------------------------------------------------
    def stats(self, group: str | None = None, arm: int | None = None, *, term: int | None = None) -> RunningMoments:
        """Copy of the running moments of a node (global when ``group`` is None; the
        ``(group, term)`` node when ``term`` is given without ``arm``)."""
        if group is None:
            node = self._global
        elif arm is None and term is not None:
            node = self._group_terms.get((str(group), int(term)), RunningMoments())
        elif arm is None:
            node = self._groups.get(str(group), RunningMoments())
        else:
            node = self._leaves.get((str(group), int(arm)), RunningMoments())
        return RunningMoments(node.n, node.mean, node.m2)

    def n_obs(self, group: str, arm: int) -> int:
        node = self._leaves.get((str(group), int(arm)))
        return 0 if node is None else node.n

    @property
    def groups(self) -> list[str]:
        return sorted(self._groups)

    @property
    def leaves(self) -> list[tuple[str, int]]:
        return sorted(self._leaves)

    @property
    def group_terms(self) -> list[tuple[str, int]]:
        return sorted(self._group_terms)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Fallback accounting over all known nodes (recomputed from the current state).

        ``source_<s>``: number of known leaves whose Beta comes from level ``s`` (incl.
        ``source_prior``); ``insufficient_<level>`` / ``invalid_<level>``: nodes of that
        level with ``0 < n < min_obs`` / whose MoM raised ``NumericalInstabilityError``;
        ``global_source``: ``"global"`` or ``"prior"``; ``clipped`` / ``skipped``: data
        quality counters from fit/update.
        """
        out: dict[str, Any] = {
            "n_obs": self._global.n,
            "n_groups": len(self._groups),
            "n_group_terms": len(self._group_terms),
            "n_leaves": len(self._leaves),
            "clipped": self._clipped,
            "skipped": self._skipped,
            "term_level": self.has_terms,
        }
        for s in SOURCES:
            out[f"source_{s}"] = 0
        for level in LEVELS:
            out[f"insufficient_{level}"] = 0
            out[f"invalid_{level}"] = 0
        nodes: list[tuple[str, RunningMoments]] = [("global", self._global)]
        nodes += [("group", s) for s in self._groups.values()]
        nodes += [("group_term", s) for s in self._group_terms.values()]
        nodes += [("leaf", s) for s in self._leaves.values()]
        for level, stats in nodes:
            _post, reason = self._try_level(stats)
            if reason in ("insufficient", "invalid"):
                out[f"{reason}_{level}"] += 1
        for g, a in self._leaves:
            out[f"source_{self.resolve(g, a)[1]}"] += 1
        out["global_source"] = self._resolve_chain(self._chain(None, None))[1]
        return out

    def summary(self) -> pd.DataFrame:
        """One row per node: ``level, group, arm, n_obs, mean, var, alpha, beta, source,
        expected_recovery, term``.

        ``mean``/``var`` are the node's observed μ and σ² (the MoM inputs; NaN when
        undefined); ``alpha``/``beta``/``expected_recovery`` describe the Beta actually used
        for a query at that node after the fallback chain, whose origin is ``source``.
        ``level`` ∈ {global, group, group_term, leaf}; ``term`` is set for ``group_term``
        rows and for leaves whose arm has a known term.
        """
        rows: list[tuple[Any, ...]] = []

        def add(level: str, group: Any, arm: Any, term: Any, stats: RunningMoments, chain: list) -> None:
            post, src = self._resolve_chain(chain)
            mean = stats.mean if stats.n > 0 else math.nan
            rows.append((level, group, arm, stats.n, mean, stats.var, post.alpha, post.beta, src, post.mean, term))

        add("global", pd.NA, pd.NA, pd.NA, self._global, self._chain(None, None))
        for g in sorted(self._groups):
            add("group", g, pd.NA, pd.NA, self._groups[g], self._chain(g, None))
        for g, t in sorted(self._group_terms):
            chain = [("group_term", self._group_terms[(g, t)]), *self._chain(g, None)]
            add("group_term", g, pd.NA, t, self._group_terms[(g, t)], chain)
        for g, a in sorted(self._leaves):
            t = self.term_of(a)
            add("leaf", g, a, pd.NA if t is None else t, self._leaves[(g, a)], self._chain(g, a))
        cols = list(zip(*rows, strict=True))
        return pd.DataFrame(
            {
                "level": pd.array(cols[0], dtype="string"),
                "group": pd.array(cols[1], dtype="string"),
                "arm": pd.array(cols[2], dtype="Int64"),
                "n_obs": np.asarray(cols[3], dtype=np.int64),
                "mean": np.asarray(cols[4], dtype=float),
                "var": np.asarray(cols[5], dtype=float),
                "alpha": np.asarray(cols[6], dtype=float),
                "beta": np.asarray(cols[7], dtype=float),
                "source": pd.array(cols[8], dtype="string"),
                "expected_recovery": np.asarray(cols[9], dtype=float),
                "term": pd.array(cols[10], dtype="Int64"),
            }
        )

    # ----- serialisation ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "BetaMoMRecovery",
            "version": 2,
            "cfg": self.cfg.to_dict(),
            "fitted": self._fitted,
            "clipped": self._clipped,
            "skipped": self._skipped,
            "global": self._global.to_list(),
            "groups": {g: s.to_list() for g, s in self._groups.items()},
            "group_terms": [[g, int(t), *s.to_list()] for (g, t), s in self._group_terms.items()],
            "leaves": [[g, int(a), *s.to_list()] for (g, a), s in self._leaves.items()],
            "arm_terms": None if self._arm_terms is None else {str(a): int(t) for a, t in sorted(self._arm_terms.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BetaMoMRecovery:
        terms = data.get("arm_terms")
        obj = cls(RecoveryConfig.from_dict(data.get("cfg")), term_of_arm=None if terms is None else
                  {int(a): int(t) for a, t in terms.items()})
        obj._fitted = bool(data.get("fitted", True))
        obj._clipped = int(data.get("clipped", 0))
        obj._skipped = int(data.get("skipped", 0))
        obj._global = RunningMoments.from_list(data["global"])
        obj._groups = {str(g): RunningMoments.from_list(v) for g, v in data["groups"].items()}
        obj._group_terms = {
            (str(row[0]), int(row[1])): RunningMoments.from_list(row[2:]) for row in data.get("group_terms") or []
        }
        obj._leaves = {
            (str(row[0]), int(row[1])): RunningMoments.from_list(row[2:]) for row in data["leaves"]
        }
        return obj

    def __repr__(self) -> str:
        return (
            f"BetaMoMRecovery(n_obs={self._global.n}, groups={len(self._groups)}, "
            f"group_terms={len(self._group_terms)}, leaves={len(self._leaves)}, min_obs={self.cfg.min_obs})"
        )
