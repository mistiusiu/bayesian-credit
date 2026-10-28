"""MarkovChainTransitionMatrix (SPEC §2.5, proposal §3.3)."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bayesian_credit.config import MCTMConfig
from bayesian_credit.exceptions import ConfigurationError, InvalidTransitionMatrixError
from bayesian_credit.models.mctm import (
    START_CURRENT,
    START_DELINQUENT,
    MarkovChainTransitionMatrix,
    State,
    TransitionStructure,
    allowed_transitions,
)

C, D, R, L = 0, 1, 2, 3
#: realistic monthly transition counts (rows = from-state)
COUNTS = np.array(
    [[9700.0, 300.0, 0.0, 0.0], [120.0, 150.0, 20.0, 110.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
)
STRUCTURES = [TransitionStructure.PROPOSAL, TransitionStructure.EXTENDED]


def fitted(structure=TransitionStructure.PROPOSAL, prior=1.0, counts=COUNTS):
    return MarkovChainTransitionMatrix(structure, prior).fit_counts(counts)


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("prior", [0.0, 0.5, 1.0, 10.0])
def test_rows_sum_to_one_and_absorbing_rows_are_fixed(structure, prior):
    counts = COUNTS.copy()
    counts[C, R] = 400.0
    P = fitted(structure, prior, counts).matrix
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(P >= 0)
    np.testing.assert_array_equal(P[R], [0, 0, 1, 0])
    np.testing.assert_array_equal(P[L], [0, 0, 0, 1])
    assert not P.flags.writeable


def test_dirichlet_posterior_mean_and_mle():
    P = fitted(prior=1.0).matrix
    assert P[C, D] == pytest.approx((300 + 1) / (9700 + 300 + 2))
    assert P[D, L] == pytest.approx((110 + 1) / (400 + 4))
    mle = fitted(prior=0.0).matrix
    assert mle[C, D] == pytest.approx(300 / 10_000)
    np.testing.assert_allclose(mle[D], COUNTS[D] / COUNTS[D].sum())


def test_structural_zeros_respected_for_proposal_and_relaxed_for_extended():
    counts = COUNTS.copy()
    counts[C, R] = 500.0  # payoffs of never-delinquent loans
    prop = fitted(TransitionStructure.PROPOSAL, 1.0, counts)
    ext = fitted(TransitionStructure.EXTENDED, 1.0, counts)
    assert prop.matrix[C, R] == 0.0 and prop.matrix[C, L] == 0.0
    assert prop.diagnostics["ignored_structural_counts"] == 500.0
    assert ext.matrix[C, R] == pytest.approx((500 + 1) / (9700 + 300 + 500 + 3))
    assert ext.matrix[C, L] == 0.0
    assert ext.diagnostics["ignored_structural_counts"] == 0.0
    assert not allowed_transitions("proposal")[C, R] and allowed_transitions("extended")[C, R]
    with pytest.raises(InvalidTransitionMatrixError):
        MarkovChainTransitionMatrix.from_matrix(ext.matrix, TransitionStructure.PROPOSAL)
    assert MarkovChainTransitionMatrix.from_matrix(ext.matrix, "extended").matrix[C, R] > 0


def test_empty_row_under_mle_stays_put_and_is_flagged():
    counts = COUNTS.copy()
    counts[D] = 0.0
    m = fitted(prior=0.0, counts=counts)
    np.testing.assert_array_equal(m.matrix[D], [0, 1, 0, 0])
    assert m.diagnostics["empty_rows"] == ["DELINQUENT"]
    with pytest.raises(InvalidTransitionMatrixError):
        m.absorption_probabilities()


@pytest.mark.parametrize("structure", STRUCTURES)
def test_absorption_probabilities_equal_limit_of_matrix_power(structure):
    counts = COUNTS.copy()
    counts[C, R] = 250.0
    m = fitted(structure, 1.0, counts)
    B = m.absorption_probabilities()
    limit = np.linalg.matrix_power(np.asarray(m.matrix), 20_000)
    np.testing.assert_allclose(B, limit[:2, 2:], atol=1e-10)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(m.stationary(), [0, 0, B[0, 0], B[0, 1]], atol=1e-12)
    np.testing.assert_allclose(m.stationary(), m.propagate(START_CURRENT, 20_000), atol=1e-10)
    np.testing.assert_allclose(m.stationary(START_DELINQUENT), [0, 0, B[1, 0], B[1, 1]], atol=1e-12)
    assert m.p_recover_given_delinquent() == pytest.approx(B[1, 0])
    assert m.p_loss_given_delinquent() == pytest.approx(B[1, 1])


def test_expected_time_to_absorption_matches_survival_sum():
    m = fitted()
    t = m.expected_time_to_absorption()
    traj = m.trajectory(START_CURRENT, 5000)
    survival = traj[:, C] + traj[:, D]  # P(not absorbed after k steps)
    assert t[0] == pytest.approx(survival[:-1].sum(), rel=1e-8)
    np.testing.assert_allclose(t, m.fundamental_matrix().sum(axis=1))


@pytest.mark.parametrize("structure", STRUCTURES)
def test_p_default_is_non_decreasing_from_current(structure):
    counts = COUNTS.copy()
    counts[C, R] = 200.0
    m = fitted(structure, 1.0, counts)
    curve = np.array([m.p_default(t) for t in range(0, 121)])
    assert curve[0] == 0.0
    assert np.all(np.diff(curve) >= -1e-15)
    np.testing.assert_allclose(curve[:61], m.p_default_curve(60), atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    rows=st.lists(st.floats(0.01, 10.0), min_size=6, max_size=6),
    horizon=st.integers(1, 72),
)
def test_loss_mass_is_non_decreasing_and_timing_cdf_is_valid(rows, horizon):
    counts = np.zeros((4, 4))
    counts[C, [C, D]] = rows[:2]
    counts[D, :] = rows[2:]
    m = MarkovChainTransitionMatrix(prior_strength=0.0).fit_counts(counts)
    traj = m.trajectory(START_CURRENT, horizon)
    assert np.all(np.diff(traj[:, L]) >= -1e-15)
    F = m.default_timing_cdf(horizon)
    assert F.shape == (horizon,)
    assert F[-1] == 1.0
    assert np.all(np.diff(F) >= 0) and np.all((F >= 0) & (F <= 1))
    assert m.default_timing_pmf(horizon).sum() == pytest.approx(1.0)


def test_default_timing_cdf_definition():
    m = fitted()
    T = 36
    F = m.default_timing_cdf(T)
    assert F[-1] == 1.0
    traj = m.trajectory(START_CURRENT, T)
    final = m.maturity_collapse(traj[T])[L]
    np.testing.assert_allclose(F[:-1], traj[1:T, L] / final, rtol=1e-12)
    assert F[0] == 0.0  # C cannot reach L in one month (structural zeros)
    assert m.default_timing_cdf(T) is F  # cached
    Fd = m.default_timing_cdf(12, State.DELINQUENT)
    assert Fd[0] == pytest.approx(m.matrix[D, L] / m.p_loss_given_state(State.DELINQUENT, 12))


def test_default_timing_cdf_is_uniform_without_loss_mass():
    P = np.array([[0.9, 0.1, 0, 0], [0.3, 0.3, 0.4, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]])
    m = MarkovChainTransitionMatrix.from_matrix(P)
    np.testing.assert_allclose(m.default_timing_cdf(24), np.arange(1, 25) / 24)


def test_maturity_collapse():
    m = fitted()
    P = m.matrix
    pi = np.array([0.5, 0.2, 0.1, 0.2])
    w_l = P[D, L] / (P[D, R] + P[D, L])
    out = m.maturity_collapse(pi)
    np.testing.assert_allclose(out, [0, 0, 0.1 + 0.5 + 0.2 * (1 - w_l), 0.2 + 0.2 * w_l])
    np.testing.assert_allclose(m.maturity_collapse(np.vstack([pi, pi])), np.vstack([out, out]))
    no_exit = MarkovChainTransitionMatrix.from_matrix(
        [[0.9, 0.1, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    np.testing.assert_allclose(no_exit.maturity_collapse([0, 1, 0, 0]), [0, 0, 0.5, 0.5])


def test_p_loss_given_state_includes_maturity_collapse():
    m = fitted()
    P = m.matrix
    assert m.p_loss_given_state(State.CURRENT, 0) == 0.0
    assert m.p_loss_given_state(State.DELINQUENT, 0) == pytest.approx(P[D, L] / (P[D, R] + P[D, L]))
    assert m.p_loss_given_state(State.LOSS, 10) == 1.0
    assert m.p_loss_given_state(State.RECOVERED, 10) == 0.0
    for s, k in [(State.CURRENT, 36), (State.DELINQUENT, 7)]:
        expected = m.maturity_collapse(m.propagate(s, k))[L]
        assert m.p_loss_given_state(s, k) == pytest.approx(expected)
    assert m.p_loss_given_state(State.DELINQUENT, 12) > m.p_loss_given_state(State.CURRENT, 12)
    with pytest.raises(ValueError):
        m.p_loss_given_state(State.CURRENT, -1)


def test_propagate_and_trajectory():
    m = fitted()
    traj = m.trajectory(START_CURRENT, 24)
    assert traj.shape == (25, 4)
    np.testing.assert_allclose(traj.sum(axis=1), 1.0)
    np.testing.assert_allclose(traj[24], m.propagate(START_CURRENT, 24), atol=1e-14)
    np.testing.assert_allclose(m.propagate(State.DELINQUENT, 3), m.propagate(START_DELINQUENT, 3))
    batch = m.propagate(np.vstack([START_CURRENT, START_DELINQUENT]), 5)
    np.testing.assert_allclose(batch[1], m.propagate(START_DELINQUENT, 5))
    assert m.p_default(5) == pytest.approx(traj[5, D] + traj[5, L])
    assert m.p_recover_given_delinquent(6) == pytest.approx(m.propagate(START_DELINQUENT, 6)[R])
    assert m.p_loss_given_delinquent(6) == pytest.approx(m.propagate(START_DELINQUENT, 6)[L])
    with pytest.raises(ValueError):
        m.propagate([0.5, 0.6, 0, 0], 1)
    with pytest.raises(ValueError):
        m.propagate(START_CURRENT, -1)


def test_log_likelihood_is_maximised_at_the_mle():
    mle = fitted(prior=0.0)
    ll = mle.log_likelihood(COUNTS)
    mask = COUNTS > 0
    assert ll == pytest.approx(float(np.sum(COUNTS[mask] * np.log(np.asarray(mle.matrix)[mask]))))
    rng = np.random.default_rng(0)
    for _ in range(25):
        P = np.array(mle.matrix)
        for i in (C, D):
            ok = mle.allowed[i]
            P[i, ok] *= np.exp(rng.normal(0.0, 0.05, ok.sum()))
            P[i] /= P[i].sum()
        assert MarkovChainTransitionMatrix.from_matrix(P).log_likelihood(COUNTS) < ll
    assert fitted(prior=1.0).log_likelihood(COUNTS) < ll
    P = np.array(mle.matrix)
    P[D] = [0.5, 0.5, 0.0, 0.0]  # observed D->R / D->L transitions get probability 0
    assert MarkovChainTransitionMatrix.from_matrix(P).log_likelihood(COUNTS) == float("-inf")
    assert mle.log_likelihood(np.zeros((4, 4))) == 0.0


def test_row_prediction_error():
    mle = fitted(prior=0.0)
    np.testing.assert_allclose(mle.row_prediction_error(COUNTS), [0.0, 0.0], atol=1e-15)
    err = fitted(prior=50.0).row_prediction_error(COUNTS)
    assert np.all(err > 0)
    counts = COUNTS.copy()
    counts[D] = 0
    out = mle.row_prediction_error(counts)
    assert out[0] == pytest.approx(0.0, abs=1e-15) and np.isnan(out[1])


def test_singular_fundamental_matrix_raises():
    P = np.array([[0.9, 0.1, 0, 0], [0.3, 0.7, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    m = MarkovChainTransitionMatrix.from_matrix(P)
    for call in (m.absorption_probabilities, m.expected_time_to_absorption, m.stationary, m.fundamental_matrix):
        with pytest.raises(InvalidTransitionMatrixError):
            call()


@pytest.mark.parametrize(
    "P",
    [
        np.eye(3),
        [[0.9, 0.2, 0, 0], [0.3, 0.3, 0.2, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]],  # row sum
        [[1.1, -0.1, 0, 0], [0.3, 0.3, 0.2, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]],  # negative
        [[0.9, 0.1, 0, 0], [0.3, 0.3, 0.2, 0.2], [0.1, 0, 0.9, 0], [0, 0, 0, 1]],  # R not absorbing
        [[0.9, 0.05, 0, 0.05], [0.3, 0.3, 0.2, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]],  # C->L
        [[np.nan, 1, 0, 0], [0.3, 0.3, 0.2, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]],
    ],
)
def test_invalid_matrices_raise(P):
    with pytest.raises(InvalidTransitionMatrixError):
        MarkovChainTransitionMatrix.from_matrix(P, TransitionStructure.EXTENDED)


def test_invalid_counts_and_config():
    with pytest.raises(InvalidTransitionMatrixError):
        MarkovChainTransitionMatrix().fit_counts(-COUNTS)
    with pytest.raises(ValueError):
        MarkovChainTransitionMatrix().fit_counts(np.zeros((3, 3)))
    with pytest.raises(ConfigurationError):
        MarkovChainTransitionMatrix(prior_strength=-1.0)


def test_sample_matrix_respects_structure_and_centres_on_posterior_mean():
    m = fitted(TransitionStructure.PROPOSAL, 1.0, COUNTS / 50.0)  # few counts -> visible spread
    rng = np.random.default_rng(11)
    draws = np.stack([m.sample_matrix(rng) for _ in range(4000)])
    np.testing.assert_allclose(draws.sum(axis=2), 1.0, atol=1e-12)
    assert np.all(draws[:, C, R] == 0) and np.all(draws[:, C, L] == 0)
    assert np.all(draws[:, R] == [0, 0, 1, 0]) and np.all(draws[:, L] == [0, 0, 0, 1])
    np.testing.assert_allclose(draws.mean(axis=0), m.matrix, atol=0.01)
    assert draws[:, D, L].std() > 0.01
    a = m.sample_matrix(np.random.default_rng(3))
    b = m.sample_matrix(np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    known = MarkovChainTransitionMatrix.from_matrix(m.matrix)
    np.testing.assert_array_equal(known.sample_matrix(rng), m.matrix)


def test_fingerprint_tracks_the_matrix():
    m = fitted()
    fp = m.fingerprint
    assert fitted().fingerprint == fp
    m.fit_counts(COUNTS * 2)
    assert m.fingerprint != fp


def test_summary_and_config():
    m = MarkovChainTransitionMatrix.from_config(MCTMConfig(structure="extended", prior_strength=2.0))
    assert m.structure is TransitionStructure.EXTENDED and m.prior_strength == 2.0
    m.fit_counts(COUNTS)
    s = m.summary()
    assert len(s) == 16
    assert list(s.columns) == ["from_state", "to_state", "probability", "count", "prior", "allowed"]
    np.testing.assert_allclose(s["probability"].to_numpy().reshape(4, 4), m.matrix)
    np.testing.assert_allclose(s["count"].to_numpy().reshape(4, 4), COUNTS)
    assert s.loc[(s["from_state"] == "Current") & (s["to_state"] == "Recovered"), "prior"].item() == 2.0
    assert s.loc[(s["from_state"] == "Current") & (s["to_state"] == "Absorbed Loss"), "allowed"].item() is False


@pytest.mark.parametrize("structure", STRUCTURES)
def test_to_dict_round_trip(structure):
    m = fitted(structure, 0.5)
    state = m.to_dict()
    restored = MarkovChainTransitionMatrix.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    np.testing.assert_array_equal(restored.matrix, m.matrix)
    assert restored.fingerprint == m.fingerprint
    known = MarkovChainTransitionMatrix.from_matrix(m.matrix, structure)
    back = MarkovChainTransitionMatrix.from_dict(json.loads(json.dumps(known.to_dict())))
    np.testing.assert_array_equal(back.matrix, known.matrix)
    assert back.dirichlet_alpha is None
