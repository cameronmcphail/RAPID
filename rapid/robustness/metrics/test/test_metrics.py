"""Tests the Robustness metrics"""

import numpy as np
from .. import common_metrics


def test_maximin():
    """Tests the maximin fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.maximin(f, maximise=True)
    expected = np.asarray(
        [0.5, 0.6])
    assert np.allclose(R, expected)
    R = common_metrics.maximin(f, maximise=False)
    expected = np.asarray(
        [-1.0, -0.69])
    assert np.allclose(R, expected)


def test_maximax():
    """Tests the maximax fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.maximax(f, maximise=True)
    expected = np.asarray(
        [1.0, 0.69])
    assert np.allclose(R, expected)
    R = common_metrics.maximax(f, maximise=False)
    expected = np.asarray(
        [-0.5, -0.6])
    assert np.allclose(R, expected)


def test_hurwicz():
    """Tests the Hurwicz fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    alpha = 0.75
    R = common_metrics.hurwicz(f, maximise=True, alpha=alpha)
    expected = np.asarray(
        [0.625, 0.6225])
    assert np.allclose(R, expected)
    R = common_metrics.hurwicz(f, maximise=False, alpha=alpha)
    expected = np.asarray(
        [-0.875, -0.6675])
    assert np.allclose(R, expected)
    R = common_metrics.hurwicz(f, maximise=False)
    expected = np.asarray(
        [-0.75, -0.645])
    assert np.allclose(R, expected)


def test_laplace():
    """Tests the Laplace fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.laplace(f, maximise=True)
    expected = np.asarray(
        [0.83, 0.63])
    assert np.allclose(R, expected)
    R = common_metrics.laplace(f, maximise=False)
    expected = np.asarray(
        [-0.83, -0.63])
    assert np.allclose(R, expected)


def test_minimax_regret():
    """Tests the minimax_regret fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.minimax_regret(f, maximise=True)
    expected = np.asarray(
        [-0.1, -0.4])
    assert np.allclose(R, expected)
    R = common_metrics.minimax_regret(f, maximise=False)
    expected = np.asarray(
        [-0.4, -0.1])
    assert np.allclose(R, expected)


def test_percentile_regret():
    """Tests the percentile_regret fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.percentile_regret(f, maximise=True, percentile=0.5)
    expected = np.asarray(
        [0.0, -0.3])
    assert np.allclose(R, expected)
    R = common_metrics.percentile_regret(f, maximise=False, percentile=0.5)
    expected = np.asarray(
        [-0.3, 0.0])
    assert np.allclose(R, expected)


def test_mean_variance():
    """Tests the mean_variance fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.mean_variance(f, maximise=True)
    expected = np.asarray(
        [1.42320289996384, 1.54948632859709])
    assert np.allclose(R, expected)
    R = common_metrics.mean_variance(f, maximise=False)
    expected = np.asarray(
        [0.132210105461122, 0.351723890540445])
    assert np.allclose(R, expected)


def test_undesirable_deviations():
    """Tests the undesirable_deviations fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = common_metrics.undesirable_deviations(f, maximise=True)
    expected = np.asarray(
        [-0.245, 0.0])
    assert np.allclose(R, expected)
    R = common_metrics.undesirable_deviations(f, maximise=False)
    expected = np.asarray(
        [-0.005, -0.045])
    assert np.allclose(R, expected)


def test_percentile_skew():
    """Tests the percentile skew fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.61]])
    R = common_metrics.percentile_skew(f, maximise=True)
    expected = np.asarray(
        [0.96, -0.777777777777779])
    assert np.allclose(R, expected)
    R = common_metrics.percentile_skew(f, maximise=False)
    expected = np.asarray(
        [-0.96, 0.777777777777779])
    assert np.allclose(R, expected)


def test_percentile_kurtosis():
    """Tests the percentile kurtosis fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5, 0.52],
        [0.69, 0.6, 0.61, 1.0]])
    R = common_metrics.percentile_kurtosis(f, maximise=True)
    expected = np.asarray(
        [1.06382979, 5.0])
    assert np.allclose(R, expected)


def test_starrs_domain():
    """Tests the starrs_domain fn"""
    f = np.asarray([
        [0.5, 0.99, 1.0],
        [0.6, 0.65, 0.69]])
    threshold = 0.65
    R = common_metrics.starrs_domain(
        f,
        maximise=True,
        threshold=threshold,
        accept_equal=True)
    expected = np.asarray(
        [2./3., 2./3.])
    assert np.allclose(R, expected)
    R = common_metrics.starrs_domain(
        f,
        maximise=True,
        threshold=threshold,
        accept_equal=False)
    expected = np.asarray(
        [2./3., 1./3.])
    assert np.allclose(R, expected)
    R = common_metrics.starrs_domain(
        f,
        maximise=False,
        threshold=threshold,
        accept_equal=True)
    expected = np.asarray(
        [1./3., 2./3.])
    assert np.allclose(R, expected)
    R = common_metrics.starrs_domain(
        f,
        maximise=False,
        threshold=threshold,
        accept_equal=False)
    expected = np.asarray(
        [1./3., 1./3.])
    assert np.allclose(R, expected)
