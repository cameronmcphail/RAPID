"""Tests the T3 transformations."""

import numpy as np
from .. import t3


def test_f_mean():
    """Tests the f_mean fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = t3.f_mean(f)
    expected = np.asarray(
        [0.83, 0.63])
    assert np.allclose(R, expected)


def test_f_range():
    """Tests the f_range fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = t3.f_range(f)
    expected = np.asarray(
        [0.5, 0.09])
    assert np.allclose(R, expected)


def test_f_sum():
    """Tests the f_sum fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = t3.f_sum(f)
    expected = np.asarray(
        [0.83, 0.63])
    assert np.allclose(R, expected)


def test_f_w_sum():
    """Tests the f_w_sum fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    weights = np.asarray(
        [0.5, 0.25, 0.25])
    R = t3.f_w_sum(f, weights)
    expected = np.asarray(
        [0.87, 0.645])
    assert np.allclose(R, expected)


def test_f_variance():
    """Tests the f_variance fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = t3.f_variance(f)
    expected = np.asarray(
        [0.0817, 0.0027])
    assert np.allclose(R, expected)


def test_f_mean_variance():
    """Tests the f_mean_variance fn"""
    f = np.asarray([
        [0.99, 1.0, 0.5],
        [0.69, 0.6, 0.6]])
    R = t3.f_mean_variance(f)
    expected = np.asarray(
        [1.42320289996384, 1.54948632859709])
    assert np.allclose(R, expected)


def test_f_skew():
    """Tests the f_skew fn"""
    f = np.asarray([
        [0.5, 0.99, 1.0],
        [0.6, 0.61, 0.69]])
    R = t3.f_skew(f, reverse=False)
    expected = np.asarray(
        [0.96, -0.777777777777779])
    assert np.allclose(R, expected)
    R = t3.f_skew(f, reverse=True)
    expected = np.asarray(
        [-0.96, 0.777777777777779])
    assert np.allclose(R, expected)


def test_f_kurtosis():
    """Tests the f_skew fn"""
    f = np.asarray([
        [0.5, 0.52, 0.99, 1.0],
        [0.6, 0.61, 0.69, 1.0]])
    R = t3.f_kurtosis(f)
    expected = np.asarray(
        [1.06382978723404, 5.0])
    assert np.allclose(R, expected)
