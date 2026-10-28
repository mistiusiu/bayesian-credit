"""Arm space A_k = (L_k, r_k, term) with the null (decline) arm (SPEC §1.5, §3.3)."""

import numpy as np
import pytest

from bayesian_credit.config import ArmSpaceConfig
from bayesian_credit.data.arms import NULL_POLICY, ArmSpace, LendingPolicy, annuity_installment
from bayesian_credit.exceptions import ConfigurationError


class TestGrid:
    def test_default_grid_has_128_lending_arms_plus_null(self, arm_space):
        assert len(arm_space) == arm_space.n_arms == 129
        assert arm_space.null_arm == 0
        assert arm_space[0] == NULL_POLICY and arm_space[0].is_null
        assert arm_space[0].label() == "decline"
        assert len(arm_space.lending_arms) == 128
        assert not any(arm_space[i].is_null for i in arm_space.lending_arms)

    def test_without_null(self):
        cfg = ArmSpaceConfig(include_null=False)
        space = ArmSpace.from_config(cfg)
        assert len(space) == 128 and space.null_arm is None

    def test_grid_order_is_term_major_then_amount_then_rate(self, arm_space):
        frame = arm_space.to_frame().iloc[1:]
        keys = list(zip(frame.term, frame.amount, frame.rate, strict=True))
        assert keys == sorted(keys)
        assert arm_space[1] == LendingPolicy(5_000.0, 0.07, 36)
        assert arm_space[128] == LendingPolicy(40_000.0, 0.28, 60)
        assert arm_space.amount_grid == ArmSpaceConfig().amounts
        assert arm_space.term_grid == (36, 60)

    def test_to_frame_columns(self, arm_space):
        frame = arm_space.to_frame()
        assert list(frame.columns) == ["arm", "amount", "rate", "term", "is_null", "installment", "label"]
        assert frame.loc[0, "installment"] == 0.0 and bool(frame.loc[0, "is_null"])

    def test_invalid_spaces(self):
        with pytest.raises(ConfigurationError):
            ArmSpace([])
        with pytest.raises(ConfigurationError):
            ArmSpace([LendingPolicy(5_000, 0.1, 36), NULL_POLICY])
        with pytest.raises(ConfigurationError):
            ArmSpace([NULL_POLICY, NULL_POLICY])
        with pytest.raises(ConfigurationError):
            ArmSpace.from_config(ArmSpaceConfig(rates=(7.0, 10.0)))  # percents, not fractions


class TestNearest:
    def test_exact_grid_points_map_to_themselves(self, arm_space):
        for arm in arm_space.lending_arms:
            p = arm_space[arm]
            assert arm_space.nearest(p.amount, p.rate, p.term) == arm

    def test_nearest_many_matches_scalar(self, arm_space, rng):
        n = 300
        amounts = rng.uniform(1_000, 40_000, n)
        rates = rng.uniform(0.05, 0.31, n)
        terms = rng.choice([36, 60], n)
        many = arm_space.nearest_many(amounts, rates, terms)
        assert [arm_space.nearest(a, r, t) for a, r, t in zip(amounts, rates, terms, strict=True)] == list(many)

    def test_term_mismatch_is_respected(self, arm_space):
        # a 60-month loan whose (amount, rate) is a 36-month grid point still maps to a 60m arm
        arm = arm_space.nearest(10_000, 0.13, 60)
        assert arm_space[arm] == LendingPolicy(10_000.0, 0.13, 60)
        arms = arm_space.nearest_many(np.array([5_000.0] * 2), np.array([0.07] * 2), np.array([36, 60]))
        assert [arm_space[a].term for a in arms] == [36, 60]

    def test_term_outside_grid_falls_back_to_nearest_amount_rate(self, small_arm_space):
        arm = small_arm_space.nearest(15_000, 0.20, 60)  # only 36m arms exist
        assert small_arm_space[arm] == LendingPolicy(15_000.0, 0.20, 36)

    def test_nearest_uses_grid_normalised_distance(self, arm_space):
        p = arm_space[arm_space.nearest(12_400, 0.139, 36)]
        assert (p.amount, p.rate, p.term) == (10_000.0, 0.13, 36)

    def test_missing_inputs_map_to_minus_one(self, arm_space):
        out = arm_space.nearest_many(np.array([np.nan, 10_000]), np.array([0.1, np.nan]), np.array([36, 36]))
        assert list(out) == [-1, -1]


PTI_ONLY = {"rate_corridor": None, "max_amount_to_income": None, "max_total_dti": None}


class TestEligibility:
    def test_unaffordable_arms_excluded_null_kept(self):
        space = ArmSpace.from_config(ArmSpaceConfig(max_payment_to_income=0.4, **PTI_ONLY))
        annual_inc = 30_000.0
        mask = space.eligible(annual_inc)
        cap = 0.4 * annual_inc / 12
        assert mask[0]
        lend = space.lending_arms
        assert np.array_equal(mask[lend], space.installments[lend] <= cap)
        assert mask.sum() < len(space)
        assert not mask[128]  # $40k @ 28% / 60m costs ~$1,250 a month

    def test_default_guardrails_follow_underwriting_conventions(self, arm_space):
        # QM 43 % DTI (12 CFR 1026.43(e)(2)(vi)), 20 % payment-to-income, 50 % amount-to-income
        assert arm_space.guardrails == {
            "max_payment_to_income": 0.20, "rate_corridor": 0.05, "max_amount_to_income": 0.50, "max_total_dti": 0.43
        }
        annual_inc = 30_000.0
        mask = arm_space.eligible(annual_inc)  # no dti / reference rate: those guardrails are skipped
        lend = arm_space.lending_arms
        ok = (arm_space.installments[lend] <= 0.20 * annual_inc / 12) & (arm_space.amounts[lend] <= 0.50 * annual_inc)
        assert np.array_equal(mask[lend], ok) and mask[0]
        assert arm_space.amounts[mask].max() == 15_000.0
        assert not mask[arm_space.nearest(15_000, 0.28, 36)]  # $15k @ 28 % / 36m costs ~$620 > $500 a month

    def test_rate_corridor_around_the_reference_rate(self, arm_space):
        mask = arm_space.eligible(1e7, reference_rate=0.13)
        rates = sorted(set(arm_space.rates[mask & arm_space.is_lending].round(4)))
        assert rates == [0.10, 0.13, 0.16]  # |r - 0.13| <= 0.05
        edge = arm_space.eligible(1e7, reference_rate=0.12)  # 0.07 sits exactly on the edge (float tolerance)
        assert sorted(set(arm_space.rates[edge & arm_space.is_lending].round(4))) == [0.07, 0.10, 0.13, 0.16]
        assert arm_space.eligible(1e7, reference_rate=float("nan")).all()  # unknown reference: no corridor
        off = ArmSpace.from_config(ArmSpaceConfig(rate_corridor=None))
        assert off.eligible(1e7, reference_rate=0.13).all()

    def test_amount_to_income_and_total_dti(self, arm_space):
        inc = 50_000.0
        a2i = arm_space.eligible(inc)
        lend = arm_space.is_lending
        assert (arm_space.amounts[a2i & lend] <= 25_000).all() and a2i[lend & (arm_space.amounts == 25_000) & (arm_space.terms == 60)].any()
        dti = 35.0  # percent, as in Lending Club
        with_dti = arm_space.eligible(inc, dti=dti)
        total = dti / 100 + arm_space.installments * 12 / inc
        assert np.array_equal(with_dti[lend], a2i[lend] & (total[lend] <= 0.43))
        assert with_dti.sum() < a2i.sum()
        assert np.array_equal(arm_space.eligible(inc, dti=float("nan")), a2i)  # unknown dti: skipped
        assert np.array_equal(arm_space.eligible(inc, dti=-1.0), a2i)
        assert arm_space.eligible(None, dti=80.0, reference_rate=None).all()  # unknown income skips income guardrails

    def test_fallback_keeps_the_cheapest_arm_closest_to_the_reference_rate(self, arm_space):
        mask = arm_space.eligible(8_000.0, dti=49.0, reference_rate=0.20)  # nothing lending is affordable
        assert mask.sum() == 2 and mask[0]
        arm = int(np.flatnonzero(mask)[1])
        assert arm_space[arm] == LendingPolicy(5_000.0, 0.19, 60)  # closest rate, smallest installment
        no_ref = arm_space.eligible(8_000.0, dti=49.0)
        assert arm_space[int(np.flatnonzero(no_ref)[1])] == LendingPolicy(5_000.0, 0.07, 60)
        assert list(arm_space.fallback_arms([0.20, np.nan, 0.30])) == [arm, arm_space.nearest(5_000, 0.07, 60), arm_space.nearest(5_000, 0.28, 60)]

    def test_eligible_many_equals_row_wise_eligible(self, arm_space, rng):
        n = 400
        inc = rng.choice([np.nan, 0.0, 8_000.0, 30_000.0, 55_000.0, 90_000.0, 250_000.0], n)
        dti = np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 60, n))
        ref = np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0.05, 0.31, n))
        many = arm_space.eligible_many(inc, dti, ref)
        assert many.shape == (n, arm_space.n_arms) and many[:, 0].all()
        for i in range(n):
            np.testing.assert_array_equal(
                many[i], arm_space.eligible(None if np.isnan(inc[i]) else inc[i], dti=dti[i], reference_rate=ref[i])
            )
        assert (many & arm_space.is_lending).any(axis=1).all()  # always at least one lending option
        np.testing.assert_array_equal(arm_space.eligible_many(inc), np.stack([arm_space.eligible(x) for x in inc]))
        with pytest.raises(ValueError):
            arm_space.eligible_many(inc, dti[:3])

    def test_unknown_income_leaves_everything_eligible(self, arm_space):
        for inc in (None, float("nan"), 0.0, -5.0):
            assert arm_space.eligible(inc).all()

    def test_high_income_everything_eligible(self, arm_space):
        assert arm_space.eligible(1_000_000.0).all()

    def test_degenerate_space_keeps_cheapest_arm(self):
        space = ArmSpace([LendingPolicy(40_000, 0.28, 36), LendingPolicy(20_000, 0.10, 60)])
        mask = space.eligible(1_000.0)
        assert mask.tolist() == [False, True]


class TestPolicyAndSerialisation:
    def test_annuity_installment(self):
        assert annuity_installment(10_000, 0.12, 36) == pytest.approx(332.14, abs=0.01)
        assert annuity_installment(12_000, 0.0, 36) == pytest.approx(333.33, abs=0.01)
        assert annuity_installment(0, 0.1, 36) == 0.0
        assert annuity_installment(1_000, 0.1, 0) == 0.0
        assert LendingPolicy(10_000, 0.12, 36).installment == pytest.approx(332.14, abs=0.01)

    def test_policy_round_trip_and_label(self):
        p = LendingPolicy(15_000.0, 0.13, 60)
        assert LendingPolicy.from_dict(p.to_dict()) == p
        assert p.label() == "$15k @ 13% / 60m"

    def test_arm_space_round_trip_equality_hash(self, arm_space):
        clone = ArmSpace.from_dict(arm_space.to_dict())
        assert clone == arm_space and hash(clone) == hash(arm_space)
        assert clone.max_payment_to_income == arm_space.max_payment_to_income
        assert clone.guardrails == arm_space.guardrails
        assert ArmSpace.from_grid([5_000], [0.1], [36]) != arm_space

    def test_guardrails_round_trip_and_old_dicts_get_the_defaults(self):
        cfg = ArmSpaceConfig(rate_corridor=0.03, max_amount_to_income=None, max_total_dti=0.45)
        space = ArmSpace.from_config(cfg)
        assert space.guardrails == {"max_payment_to_income": 0.20, "rate_corridor": 0.03, "max_amount_to_income": None,
                                    "max_total_dti": 0.45}
        assert ArmSpace.from_dict(space.to_dict()).guardrails == space.guardrails
        old = {"policies": [p.to_dict() for p in space], "max_payment_to_income": 0.35}
        assert ArmSpace.from_dict(old).guardrails == {"max_payment_to_income": 0.35, "rate_corridor": 0.05,
                                                     "max_amount_to_income": 0.50, "max_total_dti": 0.43}
        assert ArmSpaceConfig.from_dict(cfg.to_dict()) == cfg
        with pytest.raises(ConfigurationError):
            ArmSpace.from_config(ArmSpaceConfig(rate_corridor=-0.01))
        with pytest.raises(ConfigurationError):
            ArmSpace.from_grid([5_000], [0.1], [36], max_total_dti=0.0)


class TestReferenceRates:
    def table(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "sub_grade": pd.array(["A1", "A1", "A2", "B3", "C1", None], dtype="string"),
                "grade": pd.array(["A", "A", "A", "B", "C", "D"], dtype="string"),
                "int_rate": [0.05, 0.06, 0.07, 0.11, 0.14, 0.18],
            }
        )

    def test_sub_grade_then_grade_then_global(self):
        from bayesian_credit.data.reference_rates import ReferenceRates

        rr = ReferenceRates.fit(self.table())
        assert rr.lookup("A1", "A") == pytest.approx(0.055)
        assert rr.lookup("A5", "A") == pytest.approx(0.06)  # unseen sub-grade -> grade mean
        assert rr.lookup("A5") == pytest.approx(0.06)  # grade taken from the sub-grade
        assert rr.lookup("Z1", "Z") == pytest.approx(rr.global_rate) == pytest.approx(0.61 / 6)
        assert rr.for_context({"sub_grade": "B3", "grade": "B", "annual_inc": 5e4}) == pytest.approx(0.11)
        assert rr.for_context({"grade": "D"}) == pytest.approx(0.18) and rr.for_context({}) == rr.global_rate
        assert rr.n_fit == 6 and rr.counts["sub_grade:A1"] == 2

    def test_for_frame_matches_for_context_and_round_trips(self):
        import json

        import pandas as pd

        from bayesian_credit.data.reference_rates import ReferenceRates
        from bayesian_credit.data.schema import BorrowerContext

        rr = ReferenceRates.fit(self.table())
        q = pd.DataFrame(
            {
                "sub_grade": pd.array(["A1", "A5", "Z9", None, "C1", "B4"], dtype="string"),
                "grade": pd.array(["A", "A", "Z", "D", None, None], dtype="string"),
            }
        )
        vec = rr.for_frame(q)
        ctx = [rr.for_context(BorrowerContext(i, {k: v for k, v in r.items() if not pd.isna(v)})) for i, r in enumerate(q.to_dict("records"))]
        np.testing.assert_allclose(vec, np.array(ctx, dtype=float))
        back = ReferenceRates.from_dict(json.loads(json.dumps(rr.to_dict())))
        assert back == rr and np.array_equal(back.for_frame(q), vec)
        empty = ReferenceRates()
        assert empty.is_empty and empty.lookup("A1") is None and np.isnan(empty.for_frame(q)).all()
