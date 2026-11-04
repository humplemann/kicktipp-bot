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
import re
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


def tendency_of(score: Score) -> int:
    """0 = home win, 1 = draw, 2 = away win (the 1 / X / 2 order)."""
    h, a = score
    return 0 if h > a else (1 if h == a else 2)


def kicktipp_points(
    tip: Score,
    result: Score,
    scheme: Sequence[int],
    tendency_points: Optional[Sequence[float]] = None,
    draw_difference_bonus: bool = True,
) -> float:
    """Points for `tip` given `result`.

    scheme = (Tendenz, Tordifferenz, Ergebnis) as fixed points, e.g. (2, 3, 4).
    tendency_points = optional Kicktipp "Quotenpunkte" per outcome in the
    order 1 / X / 2 (shown as "Heim-Remis-Gast" on the tipping page; computed
    from the league's tips and changing until kickoff). They are ADDED to the
    fixed scheme: with 2/3/4 and points 1-2-4 a correct away tendency earns
    4+2, the right difference 4+3 and the exact result 4+4 ("3 - 8 Punkte").
    draw_difference_bonus = False for leagues whose rule table shows "-" for
    Tordifferenz at Unentschieden: a draw tip with another draw as result then
    earns only the tendency points.
    """
    tendency, difference, exact = scheme
    if tendency_of(tip) != tendency_of(result):
        return 0.0
    extra = tendency_points[tendency_of(result)] if tendency_points else 0.0
    if tip == result:
        return extra + exact
    if tip[0] - tip[1] == result[0] - result[1]:
        # Also covers "draw tipped, other draw happened" (unless disabled).
        if tip[0] == tip[1] and not draw_difference_bonus:
            return extra + tendency
        return extra + difference
    return extra + tendency


def quotenpunkte(share: float, minimum: int = 1, maximum: int = 4) -> int:
    """Kicktipp Quotenpunkte for a tendency tipped by `share` of the league.

    Kicktipp: Punkte = MAX / (10 * T/N) - MAX/10 + MIN, rounded, clamped.
    """
    if share <= 0:
        return maximum
    raw = maximum / (10.0 * share) - maximum / 10.0 + minimum
    return int(min(maximum, max(minimum, round(raw))))


def adjust_for_own_tip(
    tendency_points: Sequence[float],
    outcome: int,
    group_size: int,
    minimum: int = 1,
    maximum: int = 4,
    already_tipped: bool = False,
) -> List[float]:
    """Estimate the Quotenpunkte after our own tip on `outcome` is counted.

    The page only shows the rounded points, so the current tip counts are
    estimated from them: share ~= MAX / (10 * (points - MIN + MAX/10)).
    Returns a new list; outcomes we do not tip are left unchanged (a slightly
    optimistic simplification: N grows by one for them, too).
    """
    pts = list(tendency_points)
    q = pts[outcome]
    if q <= minimum:
        return pts  # already at the floor, can only stay there
    share = maximum / (10.0 * (q - minimum + maximum / 10.0))
    share = min(1.0, max(1.0 / max(group_size, 1), share))
    tips_on_outcome = max(1, round(share * group_size))
    total = group_size if already_tipped else group_size + 1
    own = 0 if already_tipped else 1
    pts[outcome] = quotenpunkte((tips_on_outcome + own) / total, minimum, maximum)
    return pts


def expected_points_table(
    matrix: Dict[Score, float],
    scheme: Sequence[int],
    allow_draw: bool = True,
    max_tip_goals: int = MAX_TIP_GOALS,
    tendency_points: Optional[Sequence[float]] = None,
    draw_difference_bonus: bool = True,
    group_size: Optional[int] = None,
) -> Dict[Score, float]:
    """Expected points for every candidate tip."""
    candidates = [
        (h, a)
        for h in range(max_tip_goals + 1)
        for a in range(max_tip_goals + 1)
        if allow_draw or h != a
    ]
    table: Dict[Score, float] = {}
    for tip in candidates:
        points = tendency_points
        if tendency_points and group_size:
            points = adjust_for_own_tip(tendency_points, tendency_of(tip), group_size)
        table[tip] = sum(
            p * kicktipp_points(tip, result, scheme, points, draw_difference_bonus)
            for result, p in matrix.items())
    return table


def best_tip(
    quotes: Sequence[float],
    scheme: Sequence[int] = (2, 3, 4),
    allow_draw: bool = True,
    rho: float = DEFAULT_RHO,
    tendency_points: Optional[Sequence[float]] = None,
    draw_difference_bonus: bool = True,
    group_size: Optional[int] = None,
) -> Tuple[Score, float]:
    """Return the tip with the highest expected Kicktipp points and that value.

    `quotes` are always the bookmaker quotes (they define the probabilities).
    `tendency_points` are the league's current points per outcome when the
    league uses Quotenpunkte; they only change the payoff, never the odds.
    """
    p_home, p_draw, _ = implied_probabilities(quotes)
    lam_home, lam_away = fit_expected_goals(round(p_home, 4), round(p_draw, 4), rho)
    matrix = score_matrix(lam_home, lam_away, rho)
    table = expected_points_table(
        matrix, scheme, allow_draw, tendency_points=tendency_points,
        draw_difference_bonus=draw_difference_bonus, group_size=group_size)
    # Deterministic tie-break: prefer lower total goals, then home goals.
    tip = max(table, key=lambda t: (table[t], -(t[0] + t[1]), -t[0]))
    return tip, table[tip]


TENDENCY_POINTS_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$")


def parse_tendency_points(text: Optional[str]) -> Optional[List[float]]:
    """Parse Kicktipp's Quotenpunkte cell, e.g. '1 - 2 - 4' (Heim-Remis-Gast)."""
    if not text:
        return None
    match = TENDENCY_POINTS_PATTERN.match(text)
    if not match:
        return None
    return [float(match.group(i)) for i in (1, 2, 3)]


def parse_points_scheme(raw: Optional[str], default: Sequence[int] = (2, 3, 4)) -> Tuple[int, int, int]:
    """Parse 'tendenz,differenz,ergebnis' from the environment."""
    if not raw:
        return tuple(default)  # type: ignore[return-value]
    parts: List[int] = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Points scheme must have 3 values, got: {raw!r}")
    return parts[0], parts[1], parts[2]
