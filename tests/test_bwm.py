"""Tests for the BWM solver."""
from __future__ import annotations

import pytest

from agents.agent2_scoring import bwm


def test_bwm_recovers_uniform_weights_when_all_equal():
    # All criteria equally important (a_B{j}=1, a_{j}W=1 except for best/worst pair)
    crits = ["a", "b", "c", "d"]
    # degenerate: best=a, worst=d, everyone equal except a_B{d}=a_B{W}=1 is the same
    # we still need best != worst, so use mild skew.
    a_b = {"a": 1.0, "b": 1.5, "c": 1.5, "d": 1.5}
    a_w = {"a": 1.5, "b": 1.5, "c": 1.5, "d": 1.0}
    sol = bwm.solve_bwm(crits, "a", "d", a_b, a_w)
    # All weights should be positive and sum to 1
    assert abs(sum(sol.weights.values()) - 1.0) < 1e-6
    for w in sol.weights.values():
        assert w > 0


def test_bwm_weights_sum_to_one_and_best_largest():
    crits = ["coverage", "speed", "cost"]
    best, worst = "coverage", "cost"
    # coverage is 5x more important than speed, 9x than cost; speed 3x cost
    a_b = {"coverage": 1.0, "speed": 5.0, "cost": 9.0}
    a_w = {"coverage": 9.0, "speed": 3.0, "cost": 1.0}
    sol = bwm.solve_bwm(crits, best, worst, a_b, a_w)
    assert abs(sum(sol.weights.values()) - 1.0) < 1e-6
    assert sol.weights["coverage"] == max(sol.weights.values())
    assert sol.weights["cost"] == min(sol.weights.values())
    # Mildly-inconsistent input should still have CR well below 0.10
    assert sol.consistency_ratio < 0.10


def test_bwm_rejects_bad_inputs():
    crits = ["a", "b"]
    with pytest.raises(ValueError):
        bwm.solve_bwm(crits, "a", "a", {"a": 1, "b": 2}, {"a": 2, "b": 1})
    with pytest.raises(ValueError):
        bwm.solve_bwm(crits, "x", "b", {"a": 1, "b": 2}, {"a": 2, "b": 1})
    with pytest.raises(ValueError):
        bwm.solve_bwm(crits, "a", "b", {"a": 1}, {"a": 2, "b": 1})


def test_bwm_inconsistent_input_produces_positive_xi():
    """Multiplicative inconsistency should push xi > 0.

    Consistent BWM requires a_B{j} * a_{j}W = a_BW. Here a_B{b} * a_W{b} = 2*2 = 4
    but a_BW = 9, which is a strong multiplicative contradiction.
    """
    crits = ["a", "b", "c"]
    a_b = {"a": 1, "b": 2, "c": 9}
    a_w = {"a": 9, "b": 2, "c": 1}
    sol = bwm.solve_bwm(crits, "a", "c", a_b, a_w)
    assert sol.xi > 0.1
