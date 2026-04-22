"""Tiny CLI wrapping the BWM solver.

Usage:
    python -m agents.agent2_scoring.bwm_cli --capability coding

Walks the PO through:
  1. pick best criterion
  2. pick worst criterion
  3. enter a_B{j} for each j
  4. enter a_{j}W for each j
Persists the result to scoring_profiles iff CR <= 0.10.
"""
from __future__ import annotations

import argparse
import sys

from agents.agent2_scoring import bwm, capabilities, profiles
from core.database import get_conn, init_db


def _prompt_float(prompt: str) -> float:
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if 1 <= v <= 9:
                return v
        except ValueError:
            pass
        print("  please enter a number in [1, 9]")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capability", required=True, choices=list(capabilities.CAPABILITY_CRITERIA))
    args = p.parse_args()

    init_db()
    crits = [c.name for c in capabilities.CAPABILITY_CRITERIA[args.capability]]
    print(f"Criteria for {args.capability}: {crits}")

    best = input("Best (most important) criterion: ").strip()
    worst = input("Worst (least important) criterion: ").strip()
    if best not in crits or worst not in crits or best == worst:
        print("invalid best/worst", file=sys.stderr)
        return 2

    a_b = {best: 1.0}
    print("\nA_B: how much more important is Best than each criterion? (1-9)")
    for c in crits:
        if c == best:
            continue
        a_b[c] = _prompt_float(f"  a_B[{c}] = ")

    a_w = {worst: 1.0}
    print("\nA_W: how much more important is each criterion than Worst? (1-9)")
    for c in crits:
        if c == worst:
            continue
        a_w[c] = _prompt_float(f"  a_W[{c}] = ")

    sol = bwm.solve_bwm(crits, best, worst, a_b, a_w)
    print("\nWeights:")
    for c, w in sol.weights.items():
        print(f"  {c:>20s}: {w:.4f}")
    print(f"\n xi*  = {sol.xi:.4f}")
    print(f" CR   = {sol.consistency_ratio:.4f}")

    if sol.consistency_ratio > 0.10:
        print("\nCR > 0.10 — not saved. Please revise comparisons.", file=sys.stderr)
        return 1

    with get_conn() as conn:
        profiles.save_profile_from_bwm(conn, args.capability, sol)
    print(f"\nSaved scoring_profile for {args.capability}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
