"""Tests the T1 transformations"""

import numpy as np
from .. import t1


def test_identity():
    """Tests the regret_from_best_da fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    expected = np.copy(f)
    _f = t1.identity(f, maximise=True)
    assert np.allclose(_f, expected)
    _f = t1.identity(f, maximise=False)
    expected = -np.copy(f)
    assert np.allclose(_f, expected)


def test_regret_from_best_da():
    """Tests the regret_from_best_da fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    regret = t1.regret_from_best_da(f, maximise=True)
    expected = np.asarray([
        [0.0, 0.0, -0.1],
        [-0.3, -0.4, 0.0]])
    assert np.allclose(regret, expected)
    regret = t1.regret_from_best_da(f, maximise=False)
    expected = np.asarray([
        [-0.3, -0.4, 0.0],
        [0.0, 0.0, -0.1]])
    assert np.allclose(regret, expected)


def test_regret_from_values():
    """Tests the regret_from_values fn"""
    # Test for array of values
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    values = np.asarray(
        [0.79, 1.0, 1.0])
    regret = t1.regret_from_values(f, values, maximise=True)
    expected = np.asarray([
        [0.2, 0.0, -0.5],
        [-0.1, -0.4, -0.4]])
    assert np.allclose(regret, expected)
    regret = t1.regret_from_values(f, values, maximise=False)
    expected = np.asarray([
        [-0.2, 0.0, 0.5],
        [0.1, 0.4, 0.4]])
    assert np.allclose(regret, expected)
    # Test for a single value
    f = np.asarray([
        [0.9, 1.0, 0.5],
        [1.4, 0.6, 0.6]])
    values = 0.9
    regret = t1.regret_from_values(f, values, maximise=True)
    expected = np.asarray([
        [0.0, 0.1, -0.4],
        [0.5, -0.3, -0.3]])
    assert np.allclose(regret, expected)
    regret = t1.regret_from_values(f, values, maximise=False)
    expected = np.asarray([
        [0.0, -0.1, 0.4],
        [-0.5, 0.3, 0.3]])
    assert np.allclose(regret, expected)


def test_satisficing_regret():
    """Tests the satisficing_regret fn"""
    # Test for array of thresholds
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    thresholds = np.asarray(
        [0.79, 1.0, 1.0])
    regret = t1.satisficing_regret(f, thresholds, maximise=True)
    expected = np.asarray([
        [0.0, 0.0, -0.5],
        [-0.1, -0.4, -0.4]])
    assert np.allclose(regret, expected)
    regret = t1.satisficing_regret(f, thresholds, maximise=False)
    expected = np.asarray([
        [-0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0]])
    assert np.allclose(regret, expected)
    # Test for a single threshold
    f = np.asarray([
        [0.9, 1.0, 0.5],
        [1.4, 0.6, 0.6]])
    thresholds = 0.9
    regret = t1.satisficing_regret(f, thresholds, maximise=True)
    expected = np.asarray([
        [0.0, 0.0, -0.4],
        [0.0, -0.3, -0.3]])
    assert np.allclose(regret, expected)
    regret = t1.satisficing_regret(f, thresholds, maximise=False)
    expected = np.asarray([
        [0.0, -0.1, 0.0],
        [-0.5, 0.0, 0.0]])
    assert np.allclose(regret, expected)


def test_regret_from_median():
    """Tests the regret_from_median fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    regret = t1.regret_from_median(f, maximise=True)
    expected = np.asarray([
        [0.0, 0.01, -0.49],
        [0.09, 0.0, 0.0]])
    assert np.allclose(regret, expected)
    regret = t1.regret_from_median(f, maximise=False)
    expected = np.asarray([
        [0.0, -0.01, 0.49],
        [-0.09, 0.0, 0.0]])
    assert np.allclose(regret, expected)


def test_satisfice():
    """Tests the f_satisfice fn"""
    f = np.asarray([
        [0.5, 0.99, 1.0],
        [0.6, 0.65, 0.69]])
    threshold = 0.65
    _f = t1.satisfice(
        f,
        maximise=True,
        threshold=threshold,
        accept_equal=True)
    expected = np.asarray([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]])
    assert np.allclose(_f, expected)
    _f = t1.satisfice(
        f,
        maximise=True,
        threshold=threshold,
        accept_equal=False)
    expected = np.asarray([
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]])
    assert np.allclose(_f, expected)
    _f = t1.satisfice(
        f,
        maximise=False,
        threshold=threshold,
        accept_equal=True)
    expected = np.asarray([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0]])
    assert np.allclose(_f, expected)
    _f = t1.satisfice(
        f,
        maximise=False,
        threshold=threshold,
        accept_equal=False)
    expected = np.asarray([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]])
    assert np.allclose(_f, expected)
