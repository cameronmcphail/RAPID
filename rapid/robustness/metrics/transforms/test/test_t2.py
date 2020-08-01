"""Tests the T2 transformations."""

import numpy as np
from .. import t2


def test_all_scenarios():
    """Tests the all_scenarios fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    expected = np.copy(f)
    _f = t2.all_scenarios(f)
    assert np.allclose(_f, expected)


def test_worst_case():
    """Tests the worst_case fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    _f = t2.worst_case(f)
    expected = np.asarray([
        [0.5],
        [0.6]])
    assert np.allclose(_f, expected)


def test_best_case():
    """Tests the best_case fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    _f = t2.best_case(f)
    expected = np.asarray([
        [1.0],
        [0.69]])
    assert np.allclose(_f, expected)


def test_worst_and_best_cases():
    """Tests the worst_and_best_cases fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    _f = t2.worst_and_best_cases(f)
    expected = np.asarray([
        [0.5, 1.0],
        [0.6, 0.69]])
    assert np.allclose(_f, expected)


def test_worst_half():
    """Tests the worst_half fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    _f = t2.worst_half(f)
    expected = np.asarray([
        [0.5, 0.99],
        [0.6, 0.6]])
    assert np.allclose(_f, expected)


def percentiles():
    """Tests the worst_half fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5, 0.2],
        [0.69, 0.6, 0.6, 0.2]])
    percentiles = np.asarray(
        [0.25, 0.99])
    _f = t2.select_percentiles(f, percentiles)
    expected = np.asarray([
        [0.5, 1.0],
        [0.6, 0.69]])
    assert np.allclose(_f, expected)
