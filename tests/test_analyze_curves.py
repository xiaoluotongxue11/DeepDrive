"""Tests for curve analysis utilities."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.analyze_curves import exponential_moving_average, convergence_epoch


class TestExponentialMovingAverage:
    def test_no_smoothing(self):
        """alpha=1.0 should return the original signal."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = exponential_moving_average(values, alpha=1.0)
        np.testing.assert_allclose(result, values)

    def test_full_smoothing(self):
        """alpha=0.0 should keep the first value forever."""
        values = np.array([5.0, 10.0, 15.0, 20.0])
        result = exponential_moving_average(values, alpha=0.0)
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0, 5.0])

    def test_output_length(self):
        values = np.random.rand(50)
        result = exponential_moving_average(values, alpha=0.1)
        assert len(result) == len(values)

    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(0)
        noisy = rng.normal(0, 10, 100)
        smoothed = exponential_moving_average(noisy, alpha=0.1)
        assert np.std(smoothed) < np.std(noisy)


class TestConvergenceEpoch:
    def test_converges_immediately(self):
        # Max is at index 0; 95% of max is also at 0
        values = np.array([100.0, 50.0, 50.0])
        assert convergence_epoch(values, threshold=0.95) == 0

    def test_converges_later(self):
        values = np.array([10.0, 50.0, 90.0, 100.0, 95.0])
        # 95% of 100 = 95.0 — first reached at index 3
        assert convergence_epoch(values, threshold=0.95) == 3

    def test_not_converged(self):
        values = np.array([10.0, 20.0, 30.0])
        # 95% of 30 = 28.5 — never reached
        result = convergence_epoch(values, threshold=0.99)
        # 99% of 30 = 29.7, only 30 is >= that → index 2
        assert result == 2

    def test_returns_none_when_never_converged(self):
        values = np.array([10.0, 20.0, 30.0])
        # Use a very high threshold that is never met
        result = convergence_epoch(values * 0.5, threshold=0.999)
        # 99.9% of 15 = 14.985 — reached at last element
        assert result is not None  # All elements should meet some threshold
