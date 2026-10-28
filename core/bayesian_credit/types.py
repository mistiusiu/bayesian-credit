"""Shared enumerations (vocabulary used across every layer)."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Stage(StrEnum):
    """Availability stage of a variable in the credit decision process (Appendix F, Table 3).

    The ordering is the lending life cycle. A component operating at stage *s* may only
    see variables whose stage is <= *s*; the bandit decides at ``PRE_DECISION``.
    """

    PRE_DECISION = "pre_decision"
    POLICY = "policy"  # loan_amnt, int_rate, term  — the lending-policy arm
    DISBURSEMENT = "disbursement"  # funded_amnt, issue_d
    INTERIM = "interim"
    TERMINAL = "terminal"

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER = {
    Stage.PRE_DECISION: 0,
    Stage.POLICY: 1,
    Stage.DISBURSEMENT: 2,
    Stage.INTERIM: 3,
    Stage.TERMINAL: 4,
}


class State(IntEnum):
    """MCTM credit states, S = {C, D, R, L} (§3.3, Table 6).

    ``RECOVERED`` and ``LOSS`` are absorbing. Integer values are the row/column index
    into the transition matrix.
    """

    CURRENT = 0
    DELINQUENT = 1
    RECOVERED = 2
    LOSS = 3

    @property
    def is_absorbing(self) -> bool:
        return self in (State.RECOVERED, State.LOSS)

    @property
    def label(self) -> str:
        return {
            State.CURRENT: "Current",
            State.DELINQUENT: "Delinquent",
            State.RECOVERED: "Recovered",
            State.LOSS: "Absorbed Loss",
        }[self]


TRANSIENT_STATES = (State.CURRENT, State.DELINQUENT)
ABSORBING_STATES = (State.RECOVERED, State.LOSS)


class TransitionStructure(StrEnum):
    """Which entries of P are structurally zero.

    ``PROPOSAL`` is the matrix exactly as written in §3.3: a loan can leave *Current*
    only through *Delinquent* (P[C,R] = P[C,L] = 0); *Recovered* means repayment
    following delinquency. Payoff of a never-delinquent loan is right-censoring during
    estimation and the "maturity collapse" during prediction.

    ``EXTENDED`` additionally allows C -> R (prepayment / scheduled payoff) so that the
    chain can also be read as a full loan-outcome model.
    """

    PROPOSAL = "proposal"
    EXTENDED = "extended"


class FeedbackKind(StrEnum):
    """Kind of reward observation delivered to a policy."""

    INTERIM = "interim"
    TERMINAL = "terminal"


class FeedbackMode(StrEnum):
    """How the simulation produces the reward for the arm a policy chose (SPEC §2.11)."""

    SIMULATED = "simulated"  # posterior-predictive draw from the environment model
    REPLAY = "replay"  # rejection-sampling replay: realized reward only when arm == historical
    HYBRID = "hybrid"  # realized when matching, simulated otherwise


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class SentinelLevel(StrEnum):
    """Table 9 interpretation bands."""

    NORMAL = "normal"  # PSI < 0.10
    WATCH = "watch"  # 0.10 <= PSI <= 0.25
    BREAKER = "breaker"  # PSI > 0.25


class ReviewAction(StrEnum):
    RESUME = "resume"
    RESEGMENT = "resegment"
    ROLLBACK = "rollback"
