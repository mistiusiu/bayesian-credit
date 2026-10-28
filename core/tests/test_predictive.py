"""Predictive metrics (SPEC §2.12, Table 11)."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from bayesian_credit.evaluation.predictive import (
    brier,
    expected_calibration_error,
    log_loss,
    mae,
    reliability_table,
    rmse,
    roc_auc,
    roc_curve,
    transition_log_likelihood,
    transition_row_error,
)
from bayesian_credit.models.mctm import MarkovChainTransitionMatrix

COUNTS = np.array(
    [[9700.0, 300.0, 0.0, 0.0], [120.0, 150.0, 20.0, 110.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    p = np.round(np.clip(rng.normal(0.35 + 0.25 * y, 0.2), 0, 1), 2)  # informative, many ties
    return y, p


def test_auc_matches_sklearn_with_ties(scored):
    y, p = scored
    assert roc_auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p), rel=1e-12)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == 0.5
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.4, 0.6]))


def test_roc_curve_matches_sklearn(scored):
    y, p = scored
    fpr, tpr, thr = roc_curve(y, p)
    ref_fpr, ref_tpr, ref_thr = metrics.roc_curve(y, p, drop_intermediate=False)
    np.testing.assert_allclose(fpr, ref_fpr)
    np.testing.assert_allclose(tpr, ref_tpr)
    np.testing.assert_allclose(thr[1:], ref_thr[1:])
    assert thr[0] == np.inf
    assert np.trapezoid(tpr, fpr) == pytest.approx(roc_auc(y, p), rel=1e-12)


def test_ece_of_perfect_calibration_is_zero():
    preds = [0.05, 0.25, 0.5, 0.75, 0.95]
    p = np.repeat(preds, 20)
    y = np.concatenate([np.r_[np.ones(round(q * 20)), np.zeros(20 - round(q * 20))] for q in preds])
    ece, table = expected_calibration_error(y, p, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert list(table.columns) == ["bin_low", "bin_high", "n", "mean_pred", "frac_pos"]
    assert len(table) == 10 and table["n"].sum() == 100
    assert table["mean_pred"].isna().sum() == 5
    rng = np.random.default_rng(1)
    q = rng.random(200_000)
    assert expected_calibration_error(rng.random(200_000) < q, q)[0] < 0.01


def test_ece_of_miscalibrated_predictions():
    ece, table = expected_calibration_error([0, 1] * 50, [0.9] * 100, n_bins=10)
    assert ece == pytest.approx(0.4)
    assert table.loc[9, "n"] == 100 and table.loc[9, "frac_pos"] == 0.5
    assert reliability_table([1.0], [1.0], n_bins=4)["n"].tolist() == [0, 0, 0, 1]


def test_brier_constant_half_is_a_quarter(scored):
    assert brier([0, 1, 1, 0, 1], [0.5] * 5) == 0.25
    y, p = scored
    assert brier(y, p) == pytest.approx(metrics.brier_score_loss(y, p), rel=1e-12)


def test_log_loss_matches_sklearn(scored):
    y, p = scored
    assert log_loss(y, p) == pytest.approx(metrics.log_loss(y, np.clip(p, 1e-12, 1 - 1e-12)), rel=1e-10)
    assert log_loss([1, 0], [1.0, 0.0]) == pytest.approx(1e-12, abs=1e-11)


def test_mae_and_rmse():
    assert mae([0.0, 1.0, 2.0], [1.0, 1.0, 4.0]) == pytest.approx(1.0)
    assert rmse([0.0, 1.0, 2.0], [1.0, 1.0, 4.0]) == pytest.approx(math.sqrt(5 / 3))


def test_empty_inputs_return_nan():
    for fn in (brier, log_loss, roc_auc, mae, rmse):
        assert math.isnan(fn([], []))
    ece, table = expected_calibration_error([], [])
    assert math.isnan(ece) and table["n"].sum() == 0
    assert all(arr.size == 0 for arr in roc_curve([], []))
    assert math.isnan(transition_log_likelihood(np.zeros((0, 0)), np.zeros((0, 0))))
    P = MarkovChainTransitionMatrix().matrix
    assert math.isnan(transition_log_likelihood(P, np.zeros((4, 4))))
    assert np.all(np.isnan(transition_row_error(P, np.zeros((4, 4)))))
    assert transition_row_error(np.zeros(0), np.zeros(0)).size == 0


def test_missing_values_are_dropped():
    y = pd.Series([1.0, 0.0, None, 1.0], dtype="Float64")
    p = [0.9, 0.2, 0.5, np.nan]
    assert brier(y, p) == pytest.approx(((0.9 - 1) ** 2 + 0.2**2) / 2)
    assert mae(np.array([1.0, None, 3.0], dtype=object), [1.5, 2.0, 2.0]) == pytest.approx(0.75)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        brier([0, 1], [0.5])
    with pytest.raises(ValueError):
        brier([0, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        log_loss([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        transition_log_likelihood(np.eye(4), np.ones((3, 3)))


def test_transition_metrics_agree_with_mctm():
    m = MarkovChainTransitionMatrix().fit_counts(COUNTS)
    holdout = np.round(COUNTS / 10)
    ll = transition_log_likelihood(m.matrix, holdout)
    assert ll == pytest.approx(m.log_likelihood(holdout))
    assert transition_log_likelihood(m.matrix, holdout, normalize=True) == pytest.approx(ll / holdout.sum())
    err = transition_row_error(m.matrix, holdout)
    np.testing.assert_allclose(err[:2], m.row_prediction_error(holdout))
    assert np.isnan(err[2:]).all()
    P = np.array(m.matrix)
    P[1] = [0.5, 0.5, 0.0, 0.0]
    assert transition_log_likelihood(P, holdout) == float("-inf")
