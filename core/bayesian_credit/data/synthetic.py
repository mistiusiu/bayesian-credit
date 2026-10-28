"""Synthetic Lending Club generator (SPEC §1.3) - a realistic stand-in for the Kaggle file.

Used for tests, CI and whenever ``accepted_2007_to_2018Q4.csv.gz`` is absent. It emits a
canonical ``LoanTable`` (all Table 3 columns + ``loan_id``) whose joint structure mirrors
the real data closely enough for the whole pipeline (risk model, MCTM panel, bandits,
PSI Sentinel, shadow engine) to run end-to-end. Everything is vectorised numpy driven by
one ``numpy.random.Generator``; 200k loans take well under a second.

Generative design (all numbers are defaults; see ``docs/DATA.md``)
-----------------------------------------------------------------
1. **Latent creditworthiness** ``z ~ N(0, 1)``. Each risk-bearing observable is a noisy
   view of it, ``x_j = rho_j z + sqrt(1 - rho_j^2) e_j`` with rho = 0.70 (FICO),
   -0.35 (DTI), 0.30 (log income), -0.45 (revolving utilisation), -0.40 (inquiries).
2. **Observables**: FICO = 655 + exp(3.555 + b x) (b = 0.83 below the median, 0.68 above),
   banded to multiples of 5 in [660, 845] (``high = low + 4``) - matches the LC quantiles
   (1%: 660, 50%: 690, 90%: 745, mean ~ 699, sd ~ 33); income lognormal (median 65k,
   sigma 0.55); DTI lognormal (median 17, sigma 0.45, clipped to [0, 45]); utilisation
   logistic (mean 0.50, sd 0.24, 0.2% over-limit outliers); inquiries Poisson; the other
   bureau attributes are Poisson / negative-binomial / lognormal with LC-like marginals and
   z-dependence; categorical mixes (purpose, emp_length, home_ownership, verification)
   follow the 2007-2018 LC shares; 5% joint applications carry the joint fields.
3. **Distribution shift**: for loans issued on/after ``shift_at`` each entry of ``shift``
   is added to the generated feature *before* grade, rate and default are derived. Because
   default depends on creditworthiness *as revealed by the features*, the latent score
   moves by the regression of z on the shifted observables,
   ``z_eff = z + sum_j a_j dx_j`` (a = 0.52 FICO, -0.15 DTI, 0.12 log-income, -0.21
   utilisation, -0.18 inquiries per latent sd), so e.g. a FICO drop also raises grades'
   risk, rates and defaults. ``shift["grade_mix"] = "worse" | "better" | float`` tilts the
   grade assignment by -0.5 / +0.5 / the given number of sd; ``shift["latent"]`` shifts
   z itself (every feature and outcome moves). Any other numeric pre-decision column can
   be shifted too (it moves, but carries no risk loading). Each feature draws from its own
   named random stream and all calibration constants are fixed, so the pre-shift cohort is
   bit-identical with and without a shift (a clean counterfactual).
4. **Grade / price**: grading score ``g = 0.55 z_eff + 0.835 e`` cut at fixed normal
   quantiles so A..G shares are 19/29/29/14/6/2/1 % (equal split into 5 sub-grades);
   ``int_rate`` = linear sub-grade mean A1 5.3% -> G5 30.9% + N(0, 0.4%), as a fraction.
5. **Loan**: amount lognormal (median 12k, linked to income), rounded to 1000 (45%), 5000
   (15%) or 25 (rest), in [1000, 40000]; term 60 with probability rising in the amount
   (25% overall); installment = annuity; funded_amnt = loan_amnt; issue months on
   [start, end] with linearly growing arrivals (last month 5x the first).
6. **Default**: ``logit p = c - 0.9 z_eff + 6 (int_rate - 0.13) + 0.35 [term = 60]
   + 0.8 loan_amnt / annual_inc + purpose / home / verification effects`` (centred), with
   ``c = -2.45`` (lifetime PD ~15%): resolved default rate by issue year 2012-15
   14-16%, 2016/17/18 22/22/15% (LC: 16-20%, 23/23/16%), ~17.5% overall; by grade A 6% ..
   G 58%, sub-grade AUC 0.70 (LC 6% .. 50%, 0.69). The first missed payment month
   follows a Weibull(k=1.7) with scale 17 (36m) / 24 (60m) truncated to the term, so the
   default density peaks around months 8-18.
7. **Prepayment**: 45% of non-defaulters repay early, at the front-loaded month
   ``6 + floor(Beta(1, 2.5) (term - 6))`` in [6, term-1] (median ~ month 13 of a 36-month
   loan); the rest at maturity. Front-loading puts enough early payoffs into recent,
   mostly-unresolved cohorts that their resolved default rates follow LC's censoring
   pattern instead of being dominated by early charge-offs.
8. **Status at the snapshot**: defaulters whose last payment is more than
   ``chargeoff_lag`` (4) months old are 'Charged Off' (0.5% 'Default'); last payment 1
   month ago -> 'In Grace Period' (65%) / 'Late (16-30 days)'; 2..lag months -> 'Late
   (31-120 days)' (LC's late loans show 0-4 months since the last payment); not yet
   defaulted -> 'Current'. Non-defaulters are 'Fully Paid' once their payoff month has
   passed, else 'Current' - except 0.7% of performing loans that are temporarily 1-3
   months behind (late statuses that later cure).
9. **Cash flows** come from the exact amortisation schedule up to the last paid month
   (payoff adds the remaining principal): total_rec_prncp / total_rec_int / out_prncp /
   last_pymnt_*; next_pymnt_d for unresolved loans; late fees (15-45) on ~2.5% of fully
   paid, ~2% of current, 20% of charged-off and 25% of late loans; recoveries for
   charge-offs = Beta(1.1, 10) x outstanding principal, collection fee = 16% of
   recoveries; total_pymnt = principal + interest + late fees + recoveries; last FICO =
   FICO + N(0, 20), 60-120 points lower for defaulted / late loans.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri

from ..exceptions import DataSourceError
from .loader import CANONICAL_COLUMNS, STRING_DTYPE, canonicalize
from .schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PRE_DECISION_COLUMNS

GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
SUB_GRADES: tuple[str, ...] = tuple(f"{g}{i}" for g in GRADES for i in range(1, 6))
GRADE_SHARES: dict[str, float] = {"A": 0.19, "B": 0.29, "C": 0.29, "D": 0.14, "E": 0.06, "F": 0.02, "G": 0.01}
#: Mean int_rate per sub-grade, A1 5.3% -> G5 30.9% (linear in the 35 sub-grades).
SUB_GRADE_RATES: np.ndarray = np.linspace(0.053, 0.309, len(SUB_GRADES))

#: LC 2007-2018 purpose mix (accepted loans).
PURPOSE_MIX: dict[str, float] = {
    "debt_consolidation": 1_277_877,
    "credit_card": 516_971,
    "home_improvement": 150_457,
    "other": 139_440,
    "major_purchase": 50_445,
    "medical": 27_488,
    "small_business": 24_689,
    "car": 24_013,
    "vacation": 15_525,
    "moving": 15_403,
    "house": 14_136,
    "wedding": 2_355,
    "renewable_energy": 1_445,
    "educational": 424,
}
#: emp_length (years) mix incl. missing (NaN); "< 1 year" -> 0, "10+ years" -> 10.
EMP_LENGTH_MIX: dict[float, float] = {
    0.0: 189_988, 1.0: 148_403, 2.0: 203_677, 3.0: 180_753, 4.0: 136_605, 5.0: 139_698,
    6.0: 102_628, 7.0: 92_695, 8.0: 91_914, 9.0: 79_395, 10.0: 748_005, np.nan: 146_907,
}
VERIFICATION_LEVELS: tuple[str, ...] = ("Not Verified", "Source Verified", "Verified")
VERIFICATION_SHARES: tuple[float, float, float] = (0.33, 0.39, 0.28)

#: Default-logit adjustments by category (centred on the population mix at run time).
PURPOSE_EFFECT: dict[str, float] = {
    "small_business": 0.45, "educational": 0.20, "medical": 0.15, "moving": 0.15,
    "renewable_energy": 0.10, "house": 0.10, "other": 0.08, "debt_consolidation": 0.03,
    "vacation": 0.0, "major_purchase": -0.05, "home_improvement": -0.05,
    "credit_card": -0.12, "wedding": -0.10, "car": -0.15,
}
HOME_EFFECT: dict[str, float] = {"RENT": 0.15, "OTHER": 0.20, "OWN": 0.02, "MORTGAGE": -0.10}
VERIFICATION_EFFECT: dict[str, float] = {"Not Verified": -0.10, "Source Verified": 0.03, "Verified": 0.10}
#: Nominal home_ownership mix (used to centre HOME_EFFECT).
HOME_MIX: dict[str, float] = {"MORTGAGE": 0.49, "RENT": 0.40, "OWN": 0.11, "OTHER": 0.0005}

#: Correlation of each risk-bearing observable's latent view with z.
LATENT_RHO: dict[str, float] = {
    "fico_range_low": 0.70,
    "dti": -0.35,
    "annual_inc": 0.30,
    "revol_util": -0.45,
    "inq_last_6mths": -0.40,
}
#: Raw-unit scale of one latent sd (``annual_inc`` is handled on the log scale).
LATENT_SCALE: dict[str, float] = {
    "fico_range_low": 33.0,
    "dti": 8.9,
    "annual_inc": 0.55,
    "revol_util": 0.245,
    "inq_last_6mths": 0.85,
}


def _risk_loadings() -> dict[str, float]:
    """E[z | x] = sum_j a_j x_j for independent-noise views x_j = rho_j z + ... ."""
    ratio = {k: r * r / (1 - r * r) for k, r in LATENT_RHO.items()}
    denom = 1.0 + sum(ratio.values())
    return {k: (r / (1 - r * r)) / denom for k, r in LATENT_RHO.items()}


#: Regression coefficients of z on the standardised observables (latent sd per sd).
RISK_LOADINGS: dict[str, float] = _risk_loadings()

#: The shift used when ``shift_at`` is given without ``shift`` (and by ``--shift``).
DEFAULT_SHIFT: dict[str, float] = {"fico_range_low": -30.0, "dti": 6.0, "annual_inc": -8000.0}
DEFAULT_SHIFT_AT = "2016-06"

#: Derived columns that cannot be shifted directly (shift their source instead).
_NOT_SHIFTABLE = {"installment": "loan_amnt", "fico_range_high": "fico_range_low"}
_SHIFTABLE = (set(c for c in PRE_DECISION_COLUMNS if c in NUMERIC_COLUMNS) | {"loan_amnt"}) - set(_NOT_SHIFTABLE)
_SHIFT_SPECIAL = {"grade_mix", "latent"}
#: Shifts with bespoke handling inside ``generate`` (risk-loaded or upstream of pricing).
_SHIFT_HANDLED = {"fico_range_low", "annual_inc", "dti", "revol_util", "inq_last_6mths", "loan_amnt"} | _SHIFT_SPECIAL
_INTEGER_VALUED = {
    "delinq_2yrs", "inq_last_12m", "open_acc", "total_acc", "mort_acc", "pub_rec", "pub_rec_bankruptcies",
    "collections_12_mths_ex_med", "num_accts_ever_120_pd", "num_sats", "emp_length",
}
_UNIT_INTERVAL = {"pct_tl_nvr_dlq"}
#: Reference distribution of the standardised log amount (unshifted population) - fixed so
#: calibrations never depend on the sample (or on a shift).
AMOUNT_LZ_MEAN, AMOUNT_LZ_SD = -0.388, 1.02
_GRADE_MIX = {"worse": -0.5, "better": 0.5}


def _mix(rng: np.random.Generator, mix: Mapping[Any, float], n: int) -> np.ndarray:
    keys = list(mix.keys())
    p = np.array([mix[k] for k in keys], dtype=float)
    idx = rng.choice(len(keys), size=n, p=p / p.sum())
    return np.array(keys, dtype=object)[idx]


def _annuity(amount: np.ndarray, rate: np.ndarray, term: np.ndarray) -> np.ndarray:
    i = rate / 12.0
    f = np.power(1.0 + i, term)
    return amount * i * f / (f - 1.0)


def _balance(amount: np.ndarray, rate: np.ndarray, installment: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Outstanding principal after ``k`` scheduled payments."""
    i = rate / 12.0
    g = np.power(1.0 + i, k)
    return np.maximum(amount * g - installment * (g - 1.0) / i, 0.0)


def _month_starts(start: Any, end: Any) -> pd.DatetimeIndex:
    lo = pd.Timestamp(start).to_period("M").to_timestamp()
    hi = pd.Timestamp(end).to_period("M").to_timestamp()
    if hi < lo:
        raise DataSourceError(f"end ({end}) is before start ({start})")
    return pd.date_range(lo, hi, freq="MS")


def _band5(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(x / 5.0) * 5.0, lo, hi)


@dataclass
class SyntheticLendingClub:
    """Realistic Lending Club-schema generator (SPEC §1.3).

    Reproduces the dependencies the framework relies on: sub_grade -> int_rate (A1 5.3%
    -> G5 30.9%), creditworthiness / FICO / DTI / grade -> default probability, default ->
    partial principal repayment + Beta recoveries, 36/60 term mix (~75/25), amounts 1k-40k
    in steps of 25, growing monthly arrivals, loan status consistent with issue date, term
    and snapshot, late fees, and an optional distribution shift for the PSI Sentinel.
    See the module docstring for the full generative design.
    """

    seed: int = 0
    default_intercept: float = -2.33
    chargeoff_lag_months: int = 4
    prepay_share: float = 0.45
    grade_rho: float = 0.55
    default_timing_shape: float = 1.7
    default_timing_scale_36: float = 17.0
    default_timing_scale_60: float = 24.0
    temp_late_share: float = 0.007
    term60_share: float = 0.25
    joint_share: float = 0.05
    prepay_beta_a: float = 1.0  # early payoff month = 6 + floor(Beta(a, b) * (term - 6))
    prepay_beta_b: float = 2.5
    last_meta: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------------------
    def generate(
        self,
        n: int,
        start: Any = "2012-01",
        end: Any = "2018-12",
        snapshot: Any = "2019-03-31",
        shift_at: Any = None,
        shift: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate ``n`` loans issued monthly in ``[start, end]``, observed at ``snapshot``.

        ``shift`` (feature -> additive change, plus optional ``grade_mix`` / ``latent``)
        applies to loans issued on/after ``shift_at``. Giving only one of the two uses
        :data:`DEFAULT_SHIFT` / :data:`DEFAULT_SHIFT_AT` for the other.
        """
        n = int(n)
        if n < 1:
            raise DataSourceError("n must be >= 1")
        if self.chargeoff_lag_months < 1:
            raise DataSourceError("chargeoff_lag_months must be >= 1")
        streams: dict[str, np.random.Generator] = {}

        def R(name: str) -> np.random.Generator:  # noqa: N802 - one named stream per feature
            if name not in streams:
                streams[name] = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
            return streams[name]

        months = _month_starts(start, end)
        snap = pd.Timestamp(snapshot)
        if months[-1] > snap:
            raise DataSourceError("loans cannot be issued after the snapshot date")
        shift_spec, shift_ts = self._resolve_shift(shift, shift_at)

        # ---- arrivals: linearly growing monthly volume, ids increase with time ----
        m_count = len(months)
        weights = 1.0 + 4.0 * np.arange(m_count) / max(m_count - 1, 1)
        month_idx = np.sort(R("month_idx").choice(m_count, size=n, p=weights / weights.sum()))
        issue_d = months.to_numpy()[month_idx]
        loan_id = 1_000_000 + np.cumsum(R("loan_id").integers(1, 40, size=n)).astype(np.int64)
        shifted = (
            np.zeros(n, dtype=bool) if shift_ts is None else (issue_d >= np.datetime64(shift_ts))
        )
        has_shift = bool(shifted.any())

        def shift_of(name: str) -> np.ndarray:
            v = float(shift_spec.get(name, 0.0)) if has_shift else 0.0
            return np.where(shifted, v, 0.0)

        # ---- latent creditworthiness and its noisy views ----
        z = R("z").standard_normal(n) + shift_of("latent")
        latent = {
            k: rho * z + np.sqrt(1.0 - rho * rho) * R(f"latent:{k}").standard_normal(n) for k, rho in LATENT_RHO.items()
        }
        dz = np.zeros(n)  # E[z | shifted observables] - E[z | original observables]

        # FICO
        xf = latent["fico_range_low"]
        fico_raw = 655.0 + np.exp(3.555 + np.where(xf < 0, 0.83, 0.68) * xf)
        fico_low = _band5(fico_raw, 660.0, 845.0)
        if has_shift and "fico_range_low" in shift_spec:
            delta = shift_of("fico_range_low")
            fico_low = _band5(fico_low + delta, 300.0, 845.0)
            dz += RISK_LOADINGS["fico_range_low"] * delta / LATENT_SCALE["fico_range_low"]
        fico_high = fico_low + 4.0

        # income (log scale latent), DTI, utilisation, inquiries
        xi = latent["annual_inc"]
        annual_inc = np.round(np.exp(np.log(65_000.0) + 0.55 * xi), -2)
        annual_inc = np.clip(annual_inc, 4_000.0, 2_000_000.0)
        if has_shift and "annual_inc" in shift_spec:
            new_inc = np.clip(annual_inc + shift_of("annual_inc"), 1_000.0, None)
            dz += RISK_LOADINGS["annual_inc"] * np.log(new_inc / annual_inc) / LATENT_SCALE["annual_inc"]
            annual_inc = new_inc

        dti = np.round(np.clip(np.exp(np.log(17.0) + 0.45 * latent["dti"]), 0.0, 45.0), 2)
        if has_shift and "dti" in shift_spec:
            delta = shift_of("dti")
            new = np.round(np.clip(dti + delta, 0.0, 60.0), 2)
            dz += RISK_LOADINGS["dti"] * (new - dti) / LATENT_SCALE["dti"]
            dti = new

        revol_util = expit(1.35 * latent["revol_util"])
        over = R("over").random(n) < 0.002
        revol_util = np.where(over, R("revol_util").uniform(1.0, 1.4, n), revol_util)
        revol_util = np.round(revol_util, 3)
        if has_shift and "revol_util" in shift_spec:
            delta = shift_of("revol_util")
            new = np.round(np.clip(revol_util + delta, 0.0, 2.0), 3)
            dz += RISK_LOADINGS["revol_util"] * (new - revol_util) / LATENT_SCALE["revol_util"]
            revol_util = new

        inq6 = np.minimum(R("inq6").poisson(0.5 * np.exp(0.45 * latent["inq_last_6mths"])), 8).astype(float)
        if has_shift and "inq_last_6mths" in shift_spec:
            delta = shift_of("inq_last_6mths")
            new = np.clip(np.round(inq6 + delta), 0.0, None)
            dz += RISK_LOADINGS["inq_last_6mths"] * (new - inq6) / LATENT_SCALE["inq_last_6mths"]
            inq6 = new

        z_eff = z + dz

        # ---- categorical context ----
        purpose = _mix(R("purpose"), PURPOSE_MIX, n)
        emp_length = _mix(R("emp_length"), EMP_LENGTH_MIX, n).astype(float)
        p_mort = expit(np.log(0.49 / 0.51) + 0.9 * xi)
        u = R("u").random(n)
        home = np.where(u < p_mort, "MORTGAGE", np.where(u < p_mort + 0.11, "OWN", "RENT")).astype(object)
        home[R("home").random(n) < 0.0005] = "OTHER"

        # ---- loan amount & term ----
        ln_amt = np.log(12_000.0) + 0.62 * (0.35 * xi + np.sqrt(1 - 0.35**2) * R("ln_amt").standard_normal(n))
        amt = np.exp(ln_amt)
        r = R("r").random(n)
        amt = np.where(
            r < 0.45,
            np.round(amt / 1000.0) * 1000.0,
            np.where((r < 0.60) & (amt >= 5000), np.round(amt / 5000.0) * 5000.0, np.round(amt / 25.0) * 25.0),
        )
        loan_amnt = np.clip(amt, 1000.0, 40_000.0)
        # calibration constants are fixed (not sample statistics) and every feature has its
        # own random stream, so a shift never changes the pre-shift cohort
        a60 = self._calibrate_intercept_normal(1.2, AMOUNT_LZ_MEAN, AMOUNT_LZ_SD, self.term60_share)
        lz_mean, lz_sd = AMOUNT_LZ_MEAN, AMOUNT_LZ_SD
        if has_shift and "loan_amnt" in shift_spec:
            loan_amnt = np.clip(np.round((loan_amnt + shift_of("loan_amnt")) / 25.0) * 25.0, 1000.0, 40_000.0)
        lz = (np.log(loan_amnt) - np.log(15_000.0)) / 0.6
        term = np.where(R("term").random(n) < expit(a60 + 1.2 * lz), 60, 36).astype(np.int64)

        # verification: larger loans are verified more often
        v = 0.6 * (lz - lz_mean) / lz_sd + 0.8 * R("v").standard_normal(n)
        q1, q2 = ndtri(VERIFICATION_SHARES[0]), ndtri(VERIFICATION_SHARES[0] + VERIFICATION_SHARES[1])
        verification = np.where(v < q1, VERIFICATION_LEVELS[0], np.where(v < q2, VERIFICATION_LEVELS[1], VERIFICATION_LEVELS[2])).astype(object)

        # ---- grade / sub-grade / price ----
        offset = self._grade_offset(shift_spec.get("grade_mix")) if has_shift else 0.0
        g = self.grade_rho * z_eff + np.sqrt(1 - self.grade_rho**2) * R("g").standard_normal(n)
        g = g + np.where(shifted, offset, 0.0)
        shares = np.repeat([GRADE_SHARES[x] / 5.0 for x in GRADES], 5)
        shares = shares / shares.sum()
        thresholds = ndtri(1.0 - np.cumsum(shares)[:-1])  # decreasing, best sub-grade first
        sub_idx = len(thresholds) - np.searchsorted(thresholds[::-1], g, side="right")
        sub_idx = np.clip(sub_idx, 0, len(SUB_GRADES) - 1)
        sub_grade = np.array(SUB_GRADES, dtype=object)[sub_idx]
        grade = np.array([s[0] for s in SUB_GRADES], dtype=object)[sub_idx]
        int_rate = np.round(np.clip(SUB_GRADE_RATES[sub_idx] + R("int_rate").normal(0.0, 0.004, n), 0.05, 0.31), 4)
        installment = np.round(_annuity(loan_amnt, int_rate, term.astype(float)), 2)
        funded_amnt = loan_amnt.copy()

        # ---- joint applications ----
        joint = R("joint").random(n) < self.joint_share
        application_type = np.where(joint, "Joint App", "Individual").astype(object)
        co_inc = np.round(np.exp(np.log(45_000.0) + 0.6 * R("co_inc").standard_normal(n)), -2)
        annual_inc_joint = np.where(joint, annual_inc + co_inc, np.nan)
        dti_joint = np.where(joint, np.round(dti * R("dti_joint").uniform(0.55, 1.0, n), 2), np.nan)
        verified_joint = np.where(joint, _mix(R("verified_joint"), dict(zip(VERIFICATION_LEVELS, VERIFICATION_SHARES, strict=True)), n), None).astype(object)

        # ---- other bureau attributes ----
        delinq_2yrs = np.minimum(R("delinq_2yrs").poisson(0.25 * np.exp(-0.55 * z)), 15).astype(float)
        delinq_amnt = np.where(
            R("delinq_amnt").random(n) < 0.002 + 0.01 * (delinq_2yrs > 0),
            np.round(np.exp(np.log(800.0) + R("delinq_amnt:1").standard_normal(n))),
            0.0,
        )
        mu_open = 11.6 * np.exp(0.15 * xi - 0.011)
        open_acc = np.clip(R("open_acc").poisson(R("open_acc:1").gamma(6.0, mu_open / 6.0)), 1, 60).astype(float)
        total_acc = open_acc + R("total_acc").poisson(R("total_acc:1").gamma(4.0, 12.5 / 4.0, n)).astype(float)
        num_sats = open_acc - R("num_sats").binomial(open_acc.astype(np.int64), 0.03)
        revol_limit = annual_inc * 0.45 * np.exp(0.6 * R("revol_limit").standard_normal(n) - 0.18)
        revol_bal = np.round(revol_limit * revol_util)
        total_bc_limit = np.round(revol_limit * R("total_bc_limit").uniform(0.55, 0.95, n), -2)
        bc_util = np.round(np.clip(revol_util * np.exp(R("bc_util").normal(0.12, 0.15, n)), 0.0, 2.0), 3)
        is_mort = home == "MORTGAGE"
        tot_cur_bal = np.round(
            np.where(
                is_mort,
                np.exp(np.log(160_000.0) + 0.8 * R("tot_cur_bal").standard_normal(n)),
                np.exp(np.log(35_000.0) + 1.0 * R("tot_cur_bal:1").standard_normal(n)),
            )
            + revol_bal
        )
        tot_hi_cred_lim = np.round(tot_cur_bal * R("tot_hi_cred_lim").uniform(1.05, 1.35, n) + total_bc_limit)
        inst_debt = np.where(R("inst_debt").random(n) < 0.8, np.exp(np.log(25_000.0) + 0.9 * R("inst_debt:1").standard_normal(n)), 0.0)
        total_bal_ex_mort = np.round(revol_bal + inst_debt)
        mort_acc = np.where(
            is_mort,
            1 + R("mort_acc").poisson(1.6, n),
            np.where(home == "OWN", R("mort_acc:1").poisson(0.9, n), R("mort_acc:2").poisson(0.35, n)),
        ).astype(float)
        pub_rec = np.minimum(R("pub_rec").poisson(0.18 * np.exp(-0.35 * z)), 10).astype(float)
        pub_rec_bankruptcies = R("pub_rec_bankruptcies").binomial(pub_rec.astype(np.int64), 0.65).astype(float)
        collections = np.minimum(R("collections").poisson(0.016 * np.exp(-0.4 * z)), 5).astype(float)
        ever120 = np.where(
            R("ever120").random(n) < np.clip(0.21 * np.exp(-0.35 * z), 0, 0.95), 1 + R("ever120:1").poisson(1.2, n), 0
        ).astype(float)
        n_dlq_tl = R("n_dlq_tl").poisson(0.9 * np.exp(-0.4 * z)) + ever120
        pct_tl_nvr_dlq = np.round(np.clip(1.0 - n_dlq_tl / np.maximum(total_acc, 1.0), 0.0, 1.0), 3)
        tot_coll_amt = np.where(
            R("tot_coll_amt").random(n) < np.clip(0.12 * np.exp(-0.4 * z), 0, 0.9),
            np.round(np.exp(np.log(600.0) + 1.1 * R("tot_coll_amt:1").standard_normal(n))),
            0.0,
        )
        new_bureau = issue_d >= np.datetime64("2015-12-01")  # LC added these fields late 2015
        inq_last_12m = np.where(
            new_bureau, inq6 + R("inq_last_12m").poisson(1.3 * np.exp(0.4 * latent["inq_last_6mths"])), np.nan
        )
        all_util = np.where(
            new_bureau, np.round(np.clip(0.35 + 0.45 * revol_util + R("all_util").normal(0, 0.12, n), 0.0, 1.5), 3), np.nan
        )
        has_hist = (delinq_2yrs > 0) | (R("has_hist").random(n) < np.clip(0.36 * np.exp(-0.3 * z), 0, 0.95))
        mths_since_last_delinq = np.where(
            delinq_2yrs > 0,
            R("mths_since_last_delinq").integers(0, 24, n).astype(float),
            np.minimum(24.0 + np.round(R("mths_since_last_delinq:1").gamma(2.0, 20.0, n)), 180.0),
        )
        mths_since_last_delinq = np.where(has_hist, mths_since_last_delinq, np.nan)
        num_tl_90g = np.minimum(R("num_tl_90g").poisson(0.08 * np.exp(-0.6 * z)), 10).astype(float)
        # a little realistic missingness
        bc_util[R("bc_util:1").random(n) < 0.01] = np.nan
        pct_tl_nvr_dlq[R("pct_tl_nvr_dlq").random(n) < 0.01] = np.nan
        revol_util_out = revol_util.copy()
        revol_util_out[R("revol_util_out").random(n) < 0.001] = np.nan

        # ---- default ----
        effects = (
            self._centred_effect(purpose, PURPOSE_EFFECT, PURPOSE_MIX)
            + self._centred_effect(home, HOME_EFFECT, HOME_MIX)
            + self._centred_effect(
                verification, VERIFICATION_EFFECT, dict(zip(VERIFICATION_LEVELS, VERIFICATION_SHARES, strict=True))
            )
        )
        logit = (
            self.default_intercept
            - 0.9 * z_eff
            + 6.0 * (int_rate - 0.13)
            + 0.35 * (term == 60)
            + 0.8 * np.minimum(loan_amnt / annual_inc, 2.0)
            + effects
        )
        p_default = expit(logit)
        defaults = R("defaults").random(n) < p_default

        # first missed payment month tau ~ discrete Weibull(k = default_timing_shape), truncated to the term
        k_w = self.default_timing_shape
        lam = np.where(term == 60, self.default_timing_scale_60, self.default_timing_scale_36)
        f_term = 1.0 - np.exp(-np.power(term / lam, k_w))
        u_w = R("u_w").random(n) * f_term
        t_cont = lam * np.power(-np.log1p(-u_w), 1.0 / k_w)
        tau = np.clip(np.ceil(t_cont), 1, term).astype(np.int64)
        paid_before_default = tau - 1  # payments made by a defaulter

        # early payoff month: front-loaded on [6, term - 1], month = 6 + floor(Beta(a, b) (term - 6))
        prepay = R("prepay").random(n) < self.prepay_share
        u_pp = R("payoff_month").beta(self.prepay_beta_a, self.prepay_beta_b, n)
        early_month = np.minimum(6 + np.floor(u_pp * (term - 6)).astype(np.int64), term - 1)
        payoff_month = np.where(prepay, early_month, term)

        # ---- status at the snapshot ----
        issue_m = issue_d.astype("datetime64[M]").astype(np.int64)
        snap_m = int(np.datetime64(snap.to_datetime64(), "M").astype(np.int64))
        m_snap = snap_m - issue_m  # payment dates elapsed by the snapshot
        lag = int(self.chargeoff_lag_months)
        since_last = m_snap - paid_before_default
        # last payment 1..lag months ago -> delinquent; charged off after that (LC: 120+ DPD)
        charged = defaults & (since_last > lag)
        late = defaults & (since_last >= 1) & (since_last <= lag)
        fully_paid = (~defaults) & (payoff_month <= m_snap)
        # a few performing borrowers are temporarily behind at the snapshot (they cure later)
        temp_late = (~defaults) & (~fully_paid) & (m_snap >= 4) & (R("temp_late").random(n) < self.temp_late_share)
        temp_since = R("temp_since").choice([1, 2, 3], size=n, p=[0.6, 0.25, 0.15])
        since_last = np.where(temp_late, temp_since, since_last)
        late = late | temp_late
        current = ~(charged | late | fully_paid)

        u_default, u_grace = R("u_default").random(n), R("u_default:1").random(n)
        status = np.full(n, "Current", dtype=object)
        status[fully_paid] = "Fully Paid"
        status[charged] = np.where(u_default[charged] < 0.005, "Default", "Charged Off")
        grace = late & (since_last == 1)
        status[grace] = np.where(u_grace[grace] < 0.65, "In Grace Period", "Late (16-30 days)")
        status[late & (since_last >= 2)] = "Late (31-120 days)"

        # payments made (number of scheduled installments honoured)
        k_paid = np.where(
            fully_paid,
            payoff_month,
            np.where(
                charged | (late & defaults),
                paid_before_default,
                np.where(temp_late, np.minimum(m_snap, term) - since_last, np.minimum(m_snap, term)),
            ),
        ).astype(np.int64)
        k_paid = np.maximum(k_paid, 0)
        bal = _balance(loan_amnt, int_rate, installment, k_paid.astype(float))
        sched_prncp = loan_amnt - bal
        interest = np.round(np.maximum(k_paid * installment - sched_prncp, 0.0), 2)
        total_rec_prncp = np.round(np.where(fully_paid, loan_amnt, sched_prncp), 2)
        out_prncp = np.round(np.where(current | late, bal, 0.0), 2)
        last_amt = np.where(k_paid > 0, installment, 0.0)
        last_amt = np.where(fully_paid, installment + bal, last_amt)
        last_pymnt_amnt = np.round(last_amt, 2)
        last_month = issue_m + k_paid
        last_pymnt_d = np.where(k_paid > 0, last_month, -1).astype("datetime64[M]").astype("datetime64[ns]")
        last_pymnt_d[k_paid <= 0] = np.datetime64("NaT", "ns")
        unresolved = current | late
        next_pymnt_d = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        next_pymnt_d[unresolved] = np.datetime64(snap_m + 1, "M").astype("datetime64[ns]")

        # late fees, recoveries
        tilt = np.exp(-0.4 * z - 0.08)
        p_fee = np.select(
            [fully_paid, charged, late, current], [0.025 * tilt, 0.20 * tilt, 0.25 * tilt, 0.02 * tilt], 0.0
        )
        has_fee = R("has_fee").random(n) < np.clip(p_fee, 0.0, 0.95)
        total_rec_late_fee = np.where(has_fee, np.round(R("total_rec_late_fee").uniform(15.0, 45.0, n), 2), 0.0)
        recoveries = np.where(charged, np.round(R("recoveries").beta(1.1, 10.0, n) * bal, 2), 0.0)
        collection_recovery_fee = np.round(0.16 * recoveries, 2)
        total_pymnt = np.round(total_rec_prncp + interest + total_rec_late_fee + recoveries, 2)

        bad = charged | (late & defaults)
        last_fico_low = fico_low + R("last_fico_low").normal(0.0, 20.0, n) - np.where(bad, R("last_fico_low:1").uniform(60.0, 120.0, n), 0.0)
        last_fico_low = _band5(last_fico_low, 300.0, 845.0)
        num_tl_30dpd = np.where(late, 1.0, 0.0)

        data: dict[str, Any] = {
            "loan_id": loan_id,
            "annual_inc": annual_inc,
            "annual_inc_joint": annual_inc_joint,
            "application_type": application_type,
            "dti": dti,
            "dti_joint": dti_joint,
            "emp_length": emp_length,
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "home_ownership": home,
            "purpose": purpose,
            "verification_status": verification,
            "verified_status_joint": verified_joint,
            "delinq_2yrs": delinq_2yrs,
            "delinq_amnt": delinq_amnt,
            "inq_last_6mths": inq6,
            "inq_last_12m": inq_last_12m,
            "open_acc": open_acc,
            "total_acc": total_acc,
            "revol_bal": revol_bal,
            "revol_util": revol_util_out,
            "installment": installment,
            "mort_acc": mort_acc,
            "pub_rec": pub_rec,
            "pub_rec_bankruptcies": pub_rec_bankruptcies,
            "collections_12_mths_ex_med": collections,
            "num_accts_ever_120_pd": ever120,
            "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
            "num_sats": num_sats.astype(float),
            "tot_coll_amt": tot_coll_amt,
            "tot_cur_bal": tot_cur_bal,
            "tot_hi_cred_lim": tot_hi_cred_lim,
            "total_bal_ex_mort": total_bal_ex_mort,
            "total_bc_limit": total_bc_limit,
            "bc_util": bc_util,
            "all_util": all_util,
            "grade": grade,
            "sub_grade": sub_grade,
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "term": term,
            "funded_amnt": funded_amnt,
            "issue_d": issue_d.astype("datetime64[ns]"),
            "last_fico_range_low": last_fico_low,
            "last_fico_range_high": last_fico_low + 4.0,
            "last_pymnt_amnt": last_pymnt_amnt,
            "last_pymnt_d": last_pymnt_d,
            "out_prncp": out_prncp,
            "out_prncp_inv": out_prncp.copy(),
            "num_tl_30dpd": num_tl_30dpd,
            "num_tl_90g_dpd_24m": num_tl_90g,
            "mths_since_last_delinq": mths_since_last_delinq,
            "next_pymnt_d": next_pymnt_d,
            "loan_status": status,
            "recoveries": recoveries,
            "collection_recovery_fee": collection_recovery_fee,
            "total_pymnt": total_pymnt,
            "total_pymnt_inv": total_pymnt.copy(),
            "total_rec_prncp": total_rec_prncp,
            "total_rec_int": interest,
            "total_rec_late_fee": total_rec_late_fee,
        }
        assert tuple(data) == CANONICAL_COLUMNS
        if has_shift:
            for key in set(shift_spec) - _SHIFT_HANDLED:
                moved = np.asarray(data[key], dtype=float) + shift_of(key)
                if key in _INTEGER_VALUED:
                    moved = np.round(moved)
                moved = np.clip(moved, 0.0, 1.0 if key in _UNIT_INTERVAL else (10.0 if key == "emp_length" else None))
                data[key] = moved
        frame = pd.DataFrame(
            {
                k: (pd.array(v, dtype=STRING_DTYPE) if k in CATEGORICAL_COLUMNS else v)
                for k, v in data.items()
            }
        )
        out = canonicalize(frame)
        self.last_meta = {
            "source": "synthetic",
            "seed": self.seed,
            "n": n,
            "start": str(months[0].date()),
            "end": str(months[-1].date()),
            "snapshot": str(snap.date()),
            "shift_at": None if shift_ts is None else str(shift_ts.date()),
            "shift": dict(shift_spec) if shift_ts is not None else None,
            "n_shifted": int(shifted.sum()),
            "default_intercept": self.default_intercept,
            "mean_p_default": float(p_default.mean()),
            "lifetime_default_share": float(defaults.mean()),
            "lifetime_default_share_pre_shift": float(defaults[~shifted].mean()) if (~shifted).any() else None,
            "lifetime_default_share_post_shift": float(defaults[shifted].mean()) if shifted.any() else None,
        }
        out.attrs["synthetic"] = dict(self.last_meta)
        return out

    #: alias used by the API / scripts
    generate_table = generate

    # ------------------------------------------------------------------------------
    @staticmethod
    def _resolve_shift(
        shift: Mapping[str, Any] | None, shift_at: Any
    ) -> tuple[dict[str, Any], pd.Timestamp | None]:
        if shift is None and shift_at is None:
            return {}, None
        spec = dict(DEFAULT_SHIFT if shift is None else shift)
        ts = pd.Timestamp(DEFAULT_SHIFT_AT if shift_at is None else shift_at).to_period("M").to_timestamp()
        derived = [k for k in spec if k in _NOT_SHIFTABLE]
        if derived:
            hints = ", ".join(f"{k} -> shift {_NOT_SHIFTABLE[k]}" for k in derived)
            raise DataSourceError(f"cannot shift derived columns ({hints})")
        unknown = [k for k in spec if k not in _SHIFTABLE and k not in _SHIFT_SPECIAL]
        if unknown:
            raise DataSourceError(
                f"cannot shift {unknown}: shift keys must be numeric pre-decision columns, "
                "'loan_amnt', 'grade_mix' or 'latent'"
            )
        for k, v in spec.items():
            if k == "grade_mix":
                if not (isinstance(v, (int, float)) or v in _GRADE_MIX):
                    raise DataSourceError("grade_mix must be 'worse', 'better' or a number")
            elif not isinstance(v, (int, float)) or not np.isfinite(v):
                raise DataSourceError(f"shift[{k!r}] must be a finite number")
        return spec, ts

    @staticmethod
    def _grade_offset(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, str):
            return _GRADE_MIX[value]
        return float(value)

    @staticmethod
    def _centred_effect(
        values: np.ndarray, effects: Mapping[str, float], mix: Mapping[str, float]
    ) -> np.ndarray:
        """Category effect minus its mean under the *nominal* mix (sample-independent)."""
        total = float(sum(mix.values()))
        centre = sum(effects.get(k, 0.0) * w for k, w in mix.items()) / total
        eff = pd.Series(values, dtype=object).map(effects).fillna(0.0).to_numpy(dtype=float)
        return eff - centre

    @staticmethod
    def _calibrate_intercept_normal(slope: float, mean: float, sd: float, target: float) -> float:
        """Intercept a with E[expit(a + slope X)] == target for X ~ N(mean, sd^2).

        Deterministic (Gauss-Hermite quadrature + bisection), hence sample-independent.
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(64)
        weights = weights / weights.sum()
        x = mean + sd * nodes
        lo, hi = -20.0, 20.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(weights @ expit(mid + slope * x)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def generate(n: int, seed: int = 0, **kwargs: Any) -> pd.DataFrame:
    """Module-level convenience: ``SyntheticLendingClub(seed).generate(n, **kwargs)``."""
    return SyntheticLendingClub(seed=seed).generate(n, **kwargs)


def generate_table(n: int, seed: int = 0, **kwargs: Any) -> pd.DataFrame:
    """Alias of :func:`generate`."""
    return generate(n, seed=seed, **kwargs)
