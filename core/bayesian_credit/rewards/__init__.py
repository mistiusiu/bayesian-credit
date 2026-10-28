"""Compound risk-adjusted reward (proposal §3.3): ``R_t = Revenue_t − E[Loss]_t − Funding_t``."""

from .amortization import (
    AmortizationSchedule,
    cumulative_opening_balance,
    installment,
    outstanding_balances,
    schedule,
    schedule_arrays,
    total_interest,
    weighted_average_life_years,
)
from .compound import (
    COMPONENT_KEYS,
    CompoundRewardFunction,
    RewardDistribution,
    RewardTerms,
    reward_moments,
)
from .realization import PREPAYMENT_ADJUSTMENTS, InterestRealization, RealizationLookup, group_grade

__all__ = [
    "COMPONENT_KEYS",
    "PREPAYMENT_ADJUSTMENTS",
    "AmortizationSchedule",
    "CompoundRewardFunction",
    "InterestRealization",
    "RealizationLookup",
    "RewardDistribution",
    "RewardTerms",
    "cumulative_opening_balance",
    "group_grade",
    "installment",
    "outstanding_balances",
    "reward_moments",
    "schedule",
    "schedule_arrays",
    "total_interest",
    "weighted_average_life_years",
]
