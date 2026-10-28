"""Policy specifications of a simulation run (SPEC §2.8, §2.11; G.3 baselines).

``SimulationConfig.policies`` is a list of *specs*. A spec is a policy name, optionally
with ``BanditConfig`` overrides in parentheses::

    "thompson"                         ContextualThompsonSampling, cfg.bandit
    "thompson_se"                      same, posterior_scale="standard_error"  (N(μ, σ²/n))
    "thompson_flat"                    flat GaussianThompsonSampling (no groups)
    "epsilon_greedy" | "ucb1" | "greedy" | "random" | "historical"
    "epsilon_greedy(epsilon=0.05)"     overrides: any BanditConfig field (epsilon, ucb_c,
    "ucb1(ucb_c=1.0)"                  posterior_scale, warm_start, interim_weight,
    "thompson(warm_start=flat)"        prior_mu, prior_sigma, min_sigma)

The spec string itself (whitespace removed) is the policy's label in every output. All
policies except ``thompson_flat`` are contextual (one arm set per borrower group) and are
warm-started from the risk model through ``prior_provider`` (``warm_start="risk_model"``).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..bandits import (
    UCB1,
    ArmPriorProvider,
    ArmSetPolicy,
    ContextualThompsonSampling,
    EpsilonGreedy,
    GaussianThompsonSampling,
    Greedy,
    HistoricalPolicy,
    RandomPolicy,
)
from ..config import BanditConfig
from ..data.arms import ArmSpace
from ..data.segmentation import Segmenter
from ..exceptions import ConfigurationError
from ..models.mctm import MarkovChainTransitionMatrix
from ..models.two_stage import TwoStageBetaBeta
from ..rewards.compound import CompoundRewardFunction

__all__ = [
    "POLICY_KINDS",
    "PolicySpec",
    "build_policies",
    "build_policy_from_spec",
    "parse_policy_spec",
    "parse_policy_specs",
    "risk_model_prior_provider",
]

#: Accepted policy kinds.
POLICY_KINDS = (
    "thompson",
    "thompson_se",
    "thompson_flat",
    "epsilon_greedy",
    "ucb1",
    "greedy",
    "random",
    "historical",
)
_SPEC_RE = re.compile(r"^(?P<kind>[a-z0-9_]+)(?:\((?P<args>.*)\))?$")
_BANDIT_FIELDS = {f.name: f.type for f in dataclasses.fields(BanditConfig)}


@dataclass(frozen=True)
class PolicySpec:
    """Parsed policy spec: ``label`` (output name), ``kind`` and BanditConfig overrides."""

    label: str
    kind: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def bandit_config(self, base: BanditConfig) -> BanditConfig:
        changes = dict(self.overrides)
        if self.kind == "thompson_se":
            changes.setdefault("posterior_scale", "standard_error")
        cfg = dataclasses.replace(base, **changes) if changes else base
        cfg.validate()
        return cfg

    @property
    def is_historical(self) -> bool:
        return self.kind == "historical"


def _coerce(name: str, raw: str) -> Any:
    current = getattr(BanditConfig(), name)
    text = raw.strip().strip("'\"")
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        try:
            return float(text) if isinstance(current, float) else int(text)
        except ValueError as exc:
            raise ConfigurationError(f"policy override {name}={raw!r} is not a number") from exc
    return text


def parse_policy_spec(spec: str) -> PolicySpec:
    """Parse ``"kind"`` or ``"kind(field=value, ...)"`` into a :class:`PolicySpec`."""
    label = re.sub(r"\s+", "", str(spec)).lower()
    m = _SPEC_RE.match(label)
    if m is None:
        raise ConfigurationError(f"cannot parse policy spec {spec!r}")
    kind = m.group("kind")
    if kind not in POLICY_KINDS:
        raise ConfigurationError(f"unknown policy {kind!r}; expected one of {POLICY_KINDS}")
    overrides: dict[str, Any] = {}
    args = m.group("args")
    if args:
        for part in args.split(","):
            if not part:
                continue
            if "=" not in part:
                raise ConfigurationError(f"policy override {part!r} must be field=value")
            key, value = part.split("=", 1)
            if key not in _BANDIT_FIELDS:
                raise ConfigurationError(f"unknown BanditConfig field {key!r} in policy spec {spec!r}")
            overrides[key] = _coerce(key, value)
    return PolicySpec(label=label, kind=kind, overrides=overrides)


def parse_policy_specs(specs: Iterable[str]) -> list[PolicySpec]:
    """Parse a list of specs; labels must be unique."""
    out = [parse_policy_spec(s) for s in specs]
    labels = [p.label for p in out]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        raise ConfigurationError(f"duplicate policies in the run: {dupes}")
    if not out:
        raise ConfigurationError("a simulation needs at least one policy")
    return out


def build_policy_from_spec(
    spec: PolicySpec | str,
    arm_space: ArmSpace,
    segmenter: Segmenter | None,
    bandit_cfg: BanditConfig | None = None,
    prior_provider: ArmPriorProvider | None = None,
) -> ArmSetPolicy:
    """Instantiate one policy (named ``spec.label``)."""
    sp = parse_policy_spec(spec) if isinstance(spec, str) else spec
    cfg = sp.bandit_config(bandit_cfg if bandit_cfg is not None else BanditConfig())
    common: dict[str, Any] = {"cfg": cfg, "prior_provider": prior_provider, "name": sp.label}
    kind = sp.kind
    if kind in ("thompson", "thompson_se"):
        if segmenter is None:
            return GaussianThompsonSampling(arm_space, **common)
        return ContextualThompsonSampling(arm_space, segmenter, **common)
    if kind == "thompson_flat":
        return GaussianThompsonSampling(arm_space, **common)
    if kind == "epsilon_greedy":
        return EpsilonGreedy(arm_space, segmenter, epsilon=cfg.epsilon, **common)
    if kind == "ucb1":
        return UCB1(arm_space, segmenter, c=cfg.ucb_c, **common)
    if kind == "greedy":
        return Greedy(arm_space, segmenter, **common)
    if kind == "random":
        return RandomPolicy(arm_space, segmenter, **common)
    if kind == "historical":
        return HistoricalPolicy(arm_space, segmenter, **common)
    raise ConfigurationError(f"unknown policy kind {kind!r}")  # pragma: no cover


def build_policies(
    specs: Iterable[str | PolicySpec],
    arm_space: ArmSpace,
    segmenter: Segmenter | None,
    bandit_cfg: BanditConfig | None = None,
    prior_provider: ArmPriorProvider | None = None,
) -> dict[str, ArmSetPolicy]:
    """``{label: policy}`` in spec order."""
    parsed = [parse_policy_spec(s) if isinstance(s, str) else s for s in specs]
    parse_policy_specs([p.label for p in parsed])  # uniqueness check
    return {
        sp.label: build_policy_from_spec(sp, arm_space, segmenter, bandit_cfg, prior_provider) for sp in parsed
    }


def risk_model_prior_provider(
    two_stage: TwoStageBetaBeta,
    mctm: MarkovChainTransitionMatrix | None,
    reward_fn: CompoundRewardFunction,
    arm_space: ArmSpace,
    *,
    min_sigma: float = 1e-6,
    cache: bool = True,
    realization: Callable[[str, int], float] | None = None,
) -> Callable[[str], tuple[np.ndarray, np.ndarray]]:
    """``group -> (μ_k⁰, σ_k⁰)`` from the Two-Stage Beta-Beta reward distribution (SPEC §2.8
    ``warm_start="risk_model"``): ``reward_fn.expected(arm_k, two_stage.risk(g, k))``
    mean / std for every arm (decline: 0 / ``min_sigma``). Results are memoised per group
    unless ``cache=False`` (use that when the model keeps learning). ``realization`` is the
    prepayment factor ``(group, arm) -> ρ`` (e.g. a ``RealizationLookup``; default 1)."""
    memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    policies = list(arm_space.policies)

    def provider(group: str) -> tuple[np.ndarray, np.ndarray]:
        g = str(group)
        if cache and g in memo:
            mu, sd = memo[g]
            return mu.copy(), sd.copy()
        mu = np.zeros(len(policies))
        sd = np.zeros(len(policies))
        for k, pol in enumerate(policies):
            if pol.is_null:
                continue
            rho = 1.0 if realization is None else realization(g, k)
            d = reward_fn.expected(pol, two_stage.risk(g, k), mctm, realization=rho)
            mu[k], sd[k] = d.mean, d.std
        sd = np.maximum(sd, min_sigma)
        if cache:
            memo[g] = (mu, sd)
        return mu.copy(), sd.copy()

    return provider

