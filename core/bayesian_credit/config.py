"""Configuration dataclasses — the single place where every **[decision]** default lives.

Each dataclass is JSON-round-trippable through :meth:`to_dict` / :meth:`from_dict` so
the Django API can persist and replay a run's exact configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Self

from .exceptions import ConfigurationError
from .types import FeedbackMode, TransitionStructure


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


class ConfigMixin:
    """Serialisation helpers shared by all config dataclasses."""

    _nested: ClassVar[dict[str, type]] = {}

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        data = dict(data or {})
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue  # tolerate unknown keys (forward compatibility)
            nested = cls._nested.get(key)
            if nested is not None and isinstance(value, dict):
                value = nested.from_dict(value)
            kwargs[key] = value
        obj = cls(**kwargs)  # type: ignore[call-arg]
        obj.validate()
        return obj

    def validate(self) -> None:  # pragma: no cover - overridden where needed
        return None


#: Policy features the structured leaf prior can use (``PolicyEffectConfig.features``).
POLICY_EFFECT_FEATURES: tuple[str, ...] = ("int_rate", "term60", "pti", "log_amount")


@dataclass
class PolicyEffectConfig(ConfigMixin):
    """Structured leaf prior (**[decision]**, docs/MATH.md "Structured leaf prior"): an
    offset logistic regression of default on the lending-policy features, centred within
    each borrower group, tilts the leaf prior mean ``m_{g,a} = σ(logit(m_g) + β·(x_{g,a} −
    x̄_g))`` so that arms never observed for a group are not assumed to be as safe as the
    group average (Scott 2010: structure across policy factors instead of independent
    cells)."""

    features: tuple[str, ...] = POLICY_EFFECT_FEATURES
    ridge: float = 1e-3  # L2 penalty on the standardised coefficients (mean log-likelihood scale)
    max_abs_logit_shift: float = 3.0  # clip of the leaf-prior logit tilt
    min_rows: int = 5_000  # fewer usable rows -> beta = 0 (no tilt), recorded in diagnostics
    min_elapsed_rows: int = 20_000  # use only elapsed-term rows when at least this many exist
    max_iter: int = 25  # Newton-Raphson iterations
    pti_cap: float = 1.5  # payment-to-income is clipped to [0, pti_cap]

    def __post_init__(self) -> None:
        self.features = tuple(str(f) for f in self.features)

    def validate(self) -> None:
        unknown = [f for f in self.features if f not in POLICY_EFFECT_FEATURES]
        if unknown:
            raise ConfigurationError(f"unknown policy-effect features {unknown}; expected a subset of {POLICY_EFFECT_FEATURES}")
        if len(set(self.features)) != len(self.features):
            raise ConfigurationError("policy-effect features must be unique")
        if self.ridge < 0:
            raise ConfigurationError("ridge must be >= 0")
        if self.max_abs_logit_shift <= 0:
            raise ConfigurationError("max_abs_logit_shift must be > 0")
        if self.min_rows < 1 or self.min_elapsed_rows < 0 or self.max_iter < 1:
            raise ConfigurationError("min_rows and max_iter must be >= 1, min_elapsed_rows >= 0")
        if self.pti_cap <= 0:
            raise ConfigurationError("pti_cap must be > 0")


@dataclass
class HierarchyConfig(ConfigMixin):
    """Stage 1 priors (§3.3): p0 ~ Beta(alpha0, beta0); child priors are the parent's
    posterior mean scaled by a concentration kappa (pseudo-count). With ``policy_effect``
    the leaf prior mean is tilted by the policy-effect model (``policy_effect_cfg``)."""

    alpha0: float = 1.0
    beta0: float = 1.0
    kappa_group: float | str = 20.0  # or "auto" (Beta-binomial method of moments)
    kappa_leaf: float | str = 10.0
    kappa_min: float = 1.0
    kappa_max: float = 1000.0
    policy_effect: bool = True  # structured leaf prior (needs an arm space + policy columns)
    policy_effect_cfg: PolicyEffectConfig = field(default_factory=PolicyEffectConfig)

    _nested: ClassVar[dict[str, type]] = {"policy_effect_cfg": PolicyEffectConfig}

    def __post_init__(self) -> None:
        if isinstance(self.policy_effect_cfg, dict):
            self.policy_effect_cfg = PolicyEffectConfig.from_dict(self.policy_effect_cfg)
        elif self.policy_effect_cfg is None:
            self.policy_effect_cfg = PolicyEffectConfig()
        self.policy_effect = bool(self.policy_effect)

    def validate(self) -> None:
        if self.alpha0 <= 0 or self.beta0 <= 0:
            raise ConfigurationError("alpha0 and beta0 must be > 0")
        for name in ("kappa_group", "kappa_leaf"):
            v = getattr(self, name)
            if isinstance(v, str):
                if v != "auto":
                    raise ConfigurationError(f"{name} must be a positive float or 'auto'")
            elif v <= 0:
                raise ConfigurationError(f"{name} must be > 0")
        self.policy_effect_cfg.validate()


@dataclass
class RecoveryConfig(ConfigMixin):
    """Stage 2 (Beta method-of-moments) guards. ``term_level`` (**[decision]**, docs/MATH.md
    §2.9) inserts a (group, term) level between the leaf and the group in the fallback chain
    when the model knows the arms' terms (an arm space)."""

    min_obs: int = 5  # below this fall back to the parent level
    max_concentration: float = 1000.0  # cap on alpha+beta when variance ~ 0
    eps: float = 1e-4  # clip mean into [eps, 1-eps]
    basis: str = "principal"  # "principal" | "post_chargeoff"  (SPEC §1.2)
    term_level: bool = True  # leaf -> (group, term) -> group -> global

    def validate(self) -> None:
        if self.basis not in ("principal", "post_chargeoff"):
            raise ConfigurationError("recovery basis must be 'principal' or 'post_chargeoff'")
        if self.min_obs < 1:
            raise ConfigurationError("min_obs must be >= 1")


@dataclass
class MCTMConfig(ConfigMixin):
    structure: TransitionStructure = TransitionStructure.PROPOSAL
    prior_strength: float = 1.0  # Dirichlet pseudo-count per allowed transition; 0 => MLE
    chargeoff_lag_months: int = 4  # LC charges off at 120+ DPD

    def __post_init__(self) -> None:
        if isinstance(self.structure, str):
            self.structure = TransitionStructure(self.structure)

    def validate(self) -> None:
        if self.prior_strength < 0:
            raise ConfigurationError("prior_strength must be >= 0")
        if self.chargeoff_lag_months < 1:
            raise ConfigurationError("chargeoff_lag_months must be >= 1")


@dataclass
class RewardConfig(ConfigMixin):
    """R_t = Revenue_t - Expected Loss_t - Funding Cost_t (§3.3).

    ``prepayment_adjustment="empirical"`` (**[decision]**, docs/MATH.md §2.8) multiplies the
    expected interest and the funding cost of an arm by the realised share ρ(grade, term) of
    the scheduled interest measured on fully-paid training loans (``InterestRealization``);
    ``None`` values the full contractual schedule (the proposal's literal reading)."""

    funding_rate: float = 0.04  # annual cost of funds applied to weighted-average life
    risk_aversion: float = 0.0  # lambda in score = mean - lambda * std ("risk appetite")
    exposure_basis: str = "full"  # "full" (PDF: L_t) | "outstanding"
    lendable_threshold: float = 0.6  # P(reward > 0) needed to call a borrower lendable
    prepayment_adjustment: str | None = "empirical"  # "empirical" | None

    def validate(self) -> None:
        if self.exposure_basis not in ("full", "outstanding"):
            raise ConfigurationError("exposure_basis must be 'full' or 'outstanding'")
        if not 0 <= self.lendable_threshold <= 1:
            raise ConfigurationError("lendable_threshold must be in [0, 1]")
        if self.prepayment_adjustment not in ("empirical", None):
            raise ConfigurationError("prepayment_adjustment must be 'empirical' or null")


@dataclass
class ArmSpaceConfig(ConfigMixin):
    """Candidate lending policies A_k = (L_k, r_k) x term (§3.3, Table 7) and the lending
    guardrails that decide which arms are *eligible* for a borrower (**[decision]**, see
    ``ArmSpace.eligible``; docs/MATH.md §2.7). A guardrail set to ``None`` is switched off.

    Defaults follow US consumer-underwriting conventions:

    * ``max_total_dti = 0.43`` - the 43 % debt-to-income ceiling of a General Qualified
      Mortgage under the CFPB Ability-to-Repay/QM rule, 12 CFR 1026.43(e)(2)(vi) (2013; the
      2020 General QM amendment replaced it with a price test, but 43 % remains the standard
      DTI benchmark). Lending Club's ``dti`` is in percent and excludes mortgage payments, so
      ``dti/100 + 12·installment/annual_inc`` is a lower bound of the full back-end ratio.
    * ``max_payment_to_income = 0.20`` - payment-to-income cap of unsecured instalment
      credit (lender policies in personal and auto lending cap PTI at 15-20 %; a
      convention, not a regulation).
    * ``max_amount_to_income = 0.50`` - unsecured exposure of at most half a year's income
      (a conservative convention, not a regulation).
    * ``rate_corridor = 0.05`` - offers priced within ±5 points of the borrower's sub-grade
      reference rate (the lender's own risk-based pricing schedule)."""

    amounts: tuple[float, ...] = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000)
    rates: tuple[float, ...] = (0.07, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28)
    terms: tuple[int, ...] = (36, 60)
    include_null: bool = True  # arm 0 = decline
    max_payment_to_income: float | None = 0.20  # installment <= 0.20 * annual_inc / 12
    rate_corridor: float | None = 0.05  # |rate - reference rate of the borrower's sub-grade| <= 0.05
    max_amount_to_income: float | None = 0.50  # amount <= 0.50 * annual_inc
    max_total_dti: float | None = 0.43  # dti/100 + installment * 12 / annual_inc <= 0.43 (QM 43 %)

    def __post_init__(self) -> None:
        self.amounts = tuple(float(a) for a in self.amounts)
        self.rates = tuple(float(r) for r in self.rates)
        self.terms = tuple(int(t) for t in self.terms)
        for name in ("max_payment_to_income", "rate_corridor", "max_amount_to_income", "max_total_dti"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, float(v))

    def validate(self) -> None:
        if not self.amounts or not self.rates or not self.terms:
            raise ConfigurationError("amounts, rates and terms must be non-empty")
        if min(self.amounts) <= 0 or min(self.rates) <= 0 or min(self.terms) <= 0:
            raise ConfigurationError("amounts, rates and terms must be positive")
        if max(self.rates) >= 1:
            raise ConfigurationError("rates are fractions (0.13), not percents")
        for name in ("max_payment_to_income", "rate_corridor", "max_amount_to_income", "max_total_dti"):
            v = getattr(self, name)
            if v is not None and not v > 0:
                raise ConfigurationError(f"{name} must be > 0 (or null to switch the guardrail off)")


@dataclass
class SegmentationConfig(ConfigMixin):
    """Borrower group g = combination of pre-decision dimensions (SPEC §1.4)."""

    dimensions: tuple[str, ...] = ("grade", "purpose_group", "home_ownership")
    refinement_order: tuple[str, ...] = (
        "fico_band",
        "dti_band",
        "income_band",
        "emp_band",
        "verification_status",
        "application_type",
        "sub_grade",
    )

    def __post_init__(self) -> None:
        self.dimensions = tuple(self.dimensions)
        self.refinement_order = tuple(self.refinement_order)


@dataclass
class BanditConfig(ConfigMixin):
    posterior_scale: str = "proposal"  # N(mu, sigma^2) as in §3.3 | "standard_error" N(mu, sigma^2/n)
    prior_mu: float = 0.0
    prior_sigma: float = 1_000.0  # used by warm_start="flat"
    warm_start: str = "risk_model"  # "risk_model" | "global" | "flat"
    interim_weight: float = 0.5  # weight of an interim observation (0 => ignore interim)
    epsilon: float = 0.1  # EpsilonGreedy
    ucb_c: float = 2.0  # UCB1 exploration constant
    min_sigma: float = 1e-6

    def validate(self) -> None:
        if self.posterior_scale not in ("proposal", "standard_error"):
            raise ConfigurationError("posterior_scale must be 'proposal' or 'standard_error'")
        if self.warm_start not in ("risk_model", "global", "flat"):
            raise ConfigurationError("warm_start must be 'risk_model', 'global' or 'flat'")
        if not 0 <= self.interim_weight <= 1:
            raise ConfigurationError("interim_weight must be in [0, 1]")


@dataclass
class PSIConfig(ConfigMixin):
    """PSI Sentinel + Bayesian circuit breaker (G.5, Table 9)."""

    features: tuple[str, ...] = (
        "fico_range_low",
        "dti",
        "annual_inc",
        "revol_util",
        "inq_last_6mths",
        "grade",
    )
    n_bins: int = 10
    watch_threshold: float = 0.10
    breaker_threshold: float = 0.25
    window_months: int = 3
    aggregate: str = "max"  # "max" | "mean"
    recovery_readings: int = 3  # consecutive NORMAL readings to close a HALF_OPEN breaker
    fallback: str = "champion_snapshot"  # "champion_snapshot" | "decline" | "historical"
    auto_action: str = "resume"  # action applied in breaker_mode="auto_ack"
    auto_ack_epochs: int = 1
    eps: float = 1e-4

    def validate(self) -> None:
        if not 0 < self.watch_threshold < self.breaker_threshold:
            raise ConfigurationError("need 0 < watch_threshold < breaker_threshold")
        if self.aggregate not in ("max", "mean"):
            raise ConfigurationError("aggregate must be 'max' or 'mean'")
        if self.fallback not in ("champion_snapshot", "decline", "historical"):
            raise ConfigurationError("unknown fallback")


@dataclass
class DataConfig(ConfigMixin):
    snapshot_date: str = "2019-03-31"  # issue_d ends 2018-12, last_pymnt_d runs to 2019-03
    chargeoff_lag_months: int = 4
    recovery_basis: str = "principal"


@dataclass
class SimulationConfig(ConfigMixin):
    """Replay environment (SPEC §2.11).

    ``hindsight_sample`` (**[decision]**, docs/SIMULATION.md §3) chooses the loans the
    hindsight yardstick's risk model is fitted on: ``"elapsed"`` (default) = resolved loans
    whose term had elapsed by the snapshot, an unbiased view of lifetime default;
    ``"resolved"`` = every resolved loan (the original protocol; recent vintages' resolved
    loans are mostly early defaults, which inflates the yardstick's default rates)."""

    start: str = "2015-01-01"
    end: str = "2018-12-31"
    max_loans: int | None = 50_000
    seed: int = 42
    feedback_mode: FeedbackMode = FeedbackMode.SIMULATED
    policies: tuple[str, ...] = ("thompson", "epsilon_greedy", "ucb1", "historical", "random")
    interim_months: int = 12
    breaker_mode: str = "auto_ack"  # "pause" | "auto_ack" | "off"
    checkpoint_every: int = 6  # epochs
    psi_window_months: int = 3
    oracle: str = "hindsight"
    posterior_snapshot_every: int = 3
    hindsight_sample: str = "elapsed"  # "elapsed" | "resolved"

    def __post_init__(self) -> None:
        if isinstance(self.feedback_mode, str):
            self.feedback_mode = FeedbackMode(self.feedback_mode)
        self.policies = tuple(self.policies)

    def validate(self) -> None:
        if self.breaker_mode not in ("pause", "auto_ack", "off"):
            raise ConfigurationError("breaker_mode must be 'pause', 'auto_ack' or 'off'")
        if self.checkpoint_every < 1:
            raise ConfigurationError("checkpoint_every must be >= 1")
        if self.hindsight_sample not in ("elapsed", "resolved"):
            raise ConfigurationError("hindsight_sample must be 'elapsed' or 'resolved'")


@dataclass
class FrameworkConfig(ConfigMixin):
    """Everything needed to fit the risk models and build the decision layer."""

    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    mctm: MCTMConfig = field(default_factory=MCTMConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    arms: ArmSpaceConfig = field(default_factory=ArmSpaceConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    psi: PSIConfig = field(default_factory=PSIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train_end: str = "2015-01-01"  # fit on issue_d < train_end

    _nested: ClassVar[dict[str, type]] = {
        "hierarchy": HierarchyConfig,
        "recovery": RecoveryConfig,
        "mctm": MCTMConfig,
        "reward": RewardConfig,
        "arms": ArmSpaceConfig,
        "segmentation": SegmentationConfig,
        "bandit": BanditConfig,
        "psi": PSIConfig,
        "data": DataConfig,
    }

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, ConfigMixin):
                v.validate()
