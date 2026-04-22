"""Best-Worst Method (Rezaei 2015) weight solver.

Implements the linear-program formulation (Rezaei, Omega 53: 49-57):

    minimize xi
    s.t.
        | w_B / w_j - a_Bj | <= xi     for all j
        | w_j / w_W - a_jW | <= xi     for all j
        sum_j w_j = 1
        w_j >= 0

We linearize by rewriting |w_B - a_Bj * w_j| <= xi * w_j. In practice this is
solved via the min-max formulation by variable substitution. The reference
non-linear formulation is clearer; we solve it with scipy.optimize.minimize
(SLSQP) which is plenty for n <= 10.

Consistency ratio (CR) uses the table from the BWM paper:

    CR = xi_star / xi_max(a_BW)

where xi_max depends on a_BW (the scalar best-to-worst comparison).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# Table of xi_max values from Rezaei 2015 (approximate, interpolated for
# fractional a_BW via lookup). Key: integer a_BW on 1-9 Saaty-like scale.
_CR_TABLE = {
    1: 0.00,
    2: 0.44,
    3: 1.00,
    4: 1.63,
    5: 2.30,
    6: 3.00,
    7: 3.73,
    8: 4.47,
    9: 5.23,
}


@dataclass
class BWMSolution:
    criteria: list[str]
    weights: dict[str, float]
    xi: float                 # objective value (consistency proxy)
    consistency_ratio: float  # CR = xi / xi_max(a_BW)
    best: str
    worst: str


def solve_bwm(
    criteria: list[str],
    best: str,
    worst: str,
    a_b: dict[str, float],    # A_B: a_B{j} for each j in criteria (a_B{best}=1)
    a_w: dict[str, float],    # A_W: a_{j}W for each j in criteria (a_worst{W}=1)
) -> BWMSolution:
    """Solve the BWM nonlinear program for weights and xi*."""
    n = len(criteria)
    if best not in criteria or worst not in criteria:
        raise ValueError("best/worst must be members of criteria")
    if best == worst:
        raise ValueError("best and worst must differ")
    # Canonicalize input
    for c in criteria:
        if c not in a_b or c not in a_w:
            raise ValueError(f"missing comparisons for criterion {c!r}")

    bi = criteria.index(best)
    wi = criteria.index(worst)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float]:
        return x[:n], float(x[n])

    def objective(x: np.ndarray) -> float:
        _w, xi = unpack(x)
        return xi

    def constraints() -> list[dict]:
        cons: list[dict] = [
            {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0},
        ]
        for j, c in enumerate(criteria):
            # |w_B - a_Bj * w_j| <= xi  =>  xi - (w_B - a_Bj w_j) >= 0 and reverse
            cons.append({
                "type": "ineq",
                "fun": (lambda x, j=j, abj=a_b[c]: x[n] - (x[bi] - abj * x[j])),
            })
            cons.append({
                "type": "ineq",
                "fun": (lambda x, j=j, abj=a_b[c]: x[n] - (abj * x[j] - x[bi])),
            })
            cons.append({
                "type": "ineq",
                "fun": (lambda x, j=j, ajw=a_w[c]: x[n] - (x[j] - ajw * x[wi])),
            })
            cons.append({
                "type": "ineq",
                "fun": (lambda x, j=j, ajw=a_w[c]: x[n] - (ajw * x[wi] - x[j])),
            })
        return cons

    bounds = [(1e-6, 1.0)] * n + [(0.0, 10.0)]
    x0 = np.concatenate([np.full(n, 1.0 / n), [0.1]])
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints(),
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"BWM solve failed: {res.message}")
    w_arr, xi = unpack(res.x)
    w_arr = w_arr / w_arr.sum()
    weights = {c: float(w_arr[i]) for i, c in enumerate(criteria)}

    a_bw_scalar = float(a_b[worst])                 # aB to W
    xi_max = _CR_TABLE.get(int(round(a_bw_scalar)), 5.23)
    cr = xi / xi_max if xi_max > 0 else 0.0

    return BWMSolution(
        criteria=list(criteria),
        weights=weights,
        xi=float(xi),
        consistency_ratio=float(cr),
        best=best,
        worst=worst,
    )
