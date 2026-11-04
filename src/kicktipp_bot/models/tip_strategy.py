"""Expected-points tip strategy based on bookmaker quotes.

Idea
----
1. Convert the 1/X/2 quotes into fair outcome probabilities (remove the
   bookmaker margin).
2. Fit a Dixon-Coles corrected Poisson model (expected goals for home and
   away) so that its 1/X/2 probabilities match the market.
3. Compute, for every plausible score line, the expected Kicktipp points
   (Tendenz / Tordifferenz / Ergebnis) and pick the maximum.

Everything is pure Python (no numpy/scipy) so the bot image stays small.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

Score = Tuple[int, int]

# Maximum goals per team considered in the probability grid.
MAX_GOALS = 8
# Highest score line we are willing to tip (nobody wants a 6:0 tip).
MAX_TIP_GOALS = 4
# Dixon-Coles correlation parameter; slightly negative values are the
# standard empirical finding for top leagues (more 0:0/1:1, fewer 1:0/0:1).
DEFAULT_RHO = -0.08


def poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def implied_probabilities(quotes: Sequence[float]) -> Tuple[float, float, float]:
    """Turn decimal quotes [1, X, 2] into fair probabilities (margin removed)."""
    inverse = [1.0 / q for q in quotes]
    total = sum(inverse)
    p1, px, p2 = (x / total for x in inverse)
    return p1, px, p2


def score_matrix(lambda_home: float, lambda_away: float, rho: float = DEFAULT_RHO) -> Dict[Score, float]:
    """Probability of every score line under a Dixon-Coles Poisson model."""
    matrix: Dict[Score, float] = {}
    for home in range(MAX_GOALS + 1):
        for away in range(MAX_GOALS + 1):
            p = poisson_pmf(lambda_home, home) * poisson_pmf(lambda_away, away)
            if home == 0 and away == 0:
                p *= 1 - lambda_home * lambda_away * rho
            elif home == 0 and away == 1:
                p *= 1 + lambda_home * rho
            elif home == 1 and away == 0:
                p *= 1 + lambda_away * rho
            elif home == 1 and away == 1:
                p *= 1 - rho
            matrix[(home, away)] = max(p, 0.0)
    total = sum(matrix.values())
    return {score: p / total for score, p in matrix.items()}


def outcome_probabilities(matrix: Dict[Score, float]) -> Tuple[float, float, float]:
    p_home = sum(p for (h, a), p in matrix.items() if h > a)
    p_draw = sum(p for (h, a), p in matrix.items() if h == a)
    return p_home, p_draw, 1.0 - p_home - p_draw


@lru_cache(maxsize=4096)
def fit_expected_goals(p_home: float, p_draw: float, rho: float = DEFAULT_RHO) -> Tuple[float, float]:
    """Find expected goals (home, away) whose model 1/X/2 matches the market.

    Coarse-to-fine grid search; deterministic and dependency-free.
    """
    best = (1.4, 1.1)
    best_err = float("inf")
    for step in (0.25, 0.05, 0.01):
        center_home, center_away = best
        for i in range(-5, 6):
            for j in range(-5, 6):
                lam_home = center_home + i * step
                lam_away = center_away + j * step
                if lam_home < 0.15 or lam_away < 0.15:
                    continue
                mh, md, _ = outcome_probabilities(score_matrix(lam_home, lam_away, rho))
                err = (mh - p_home) ** 2 + (md - p_draw) ** 2
                if err < best_err:
                    best, best_err = (lam_home, lam_away), err
    return best


def kicktipp_points(tip: Score, result: Score, scheme: Sequence[int]) -> int:
    """Points for `tip` given `result` under a (Tendenz, Differenz, Ergebnis) scheme."""
    tendency, difference, exact = scheme
    th, ta = tip
    rh, ra = result
    if (th, ta) == (rh, ra):
        return exact
    if th - ta == rh - ra:
        # Also covers "draw tipped, other draw happened".
        return difference
    if (th > ta) == (rh > ra) and (th == ta) == (rh == ra):
        return tendency
    return 0


def expected_points_table(
    matrix: Dict[Score, float],
    scheme: Sequence[int],
    allow_draw: bool = True,
    max_tip_goals: int = MAX_TIP_GOALS,
) -> Dict[Score, float]:
    """Expected points for every candidate tip."""
    candidates = [
        (h, a)
        for h in range(max_tip_goals + 1)
        for a in range(max_tip_goals + 1)
        if allow_draw or h != a
    ]
    return {
        tip: sum(p * kicktipp_points(tip, result, scheme) for result, p in matrix.items())
        for tip in candidates
    }


def best_tip(
    quotes: Sequence[float],
    scheme: Sequence[int] = (2, 3, 4),
    allow_draw: bool = True,
    rho: float = DEFAULT_RHO,
) -> Tuple[Score, float]:
    """Return the tip with the highest expected Kicktipp points and that value."""
    p_home, p_draw, _ = implied_probabilities(quotes)
    lam_home, lam_away = fit_expected_goals(round(p_home, 4), round(p_draw, 4), rho)
    matrix = score_matrix(lam_home, lam_away, rho)
    table = expected_points_table(matrix, scheme, allow_draw)
    # Deterministic tie-break: prefer lower total goals, then home goals.
    tip = max(table, key=lambda t: (table[t], -(t[0] + t[1]), -t[0]))
    return tip, table[tip]


def parse_points_scheme(raw: Optional[str], default: Sequence[int] = (2, 3, 4)) -> Tuple[int, int, int]:
    """Parse 'tendenz,differenz,ergebnis' from the environment."""
    if not raw:
        return tuple(default)  # type: ignore[return-value]
    parts: List[int] = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Points scheme must have 3 values, got: {raw!r}")
    return parts[0], parts[1], parts[2]
