"""Game model for representing football matches and calculating betting tips."""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from . import tip_strategy


@dataclass
class Game:
    """Represents a football game with teams, betting quotes, and tip calculation logic."""

    home_team: str
    away_team: str
    quotes: List[str]
    game_time: datetime
    _validated_quotes: List[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate and process data after initialization."""
        self.home_team = self.home_team.strip()
        self.away_team = self.away_team.strip()
        self._validated_quotes = self._validate_quotes(self.quotes)

    def _validate_quotes(self, quotes: List[str]) -> List[float]:
        """
        Validate and convert quotes to float values.

        Args:
            quotes: List of quote strings

        Returns:
            List of float quotes

        Raises:
            ValueError: If quotes are invalid
        """
        if len(quotes) != 3:
            raise ValueError(f"Expected 3 quotes, got {len(quotes)}")

        try:
            # Kicktipp may render quotes with a decimal comma ("2,10").
            return [float(str(quote).strip().replace(",", ".")) for quote in quotes]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid quote values: {quotes}") from e

    def calculate_tip(
        self,
        home_quote: Optional[float] = None,
        away_quote: Optional[float] = None,
        allow_draw: bool = True,
        strategy: str = "ev",
        points_scheme: Sequence[int] = (2, 3, 4)
    ) -> Tuple[int, int]:
        """
        Calculate betting tip based on the quotes.

        Args:
            home_quote: Quote for home team win (uses self._validated_quotes[0] if None)
            away_quote: Quote for away team win (uses self._validated_quotes[2] if None)
            allow_draw: Whether equal-score predictions are allowed
            strategy: "ev" (maximise expected Kicktipp points, default) or
                "legacy" (original quote-difference heuristic)
            points_scheme: (Tendenz, Tordifferenz, Ergebnis) points of the league

        Returns:
            Tuple of (home_goals, away_goals) prediction
        """
        if strategy == "ev":
            return self.calculate_ev_tip(allow_draw=allow_draw, points_scheme=points_scheme)

        home_quote, away_quote = self._get_effective_win_quotes(
            home_quote, away_quote)
        tip = self._calculate_base_tip(home_quote, away_quote)

        if not allow_draw:
            return self.resolve_draw_tip(tip, home_quote, away_quote)

        return tip

    def calculate_ev_tip(
        self,
        allow_draw: bool = True,
        points_scheme: Sequence[int] = (2, 3, 4)
    ) -> Tuple[int, int]:
        """Tip that maximises the expected Kicktipp points under a Poisson model."""
        tip, _ = tip_strategy.best_tip(
            self._validated_quotes, scheme=points_scheme, allow_draw=allow_draw)
        return tip

    def expected_points(
        self,
        tip: Tuple[int, int],
        points_scheme: Sequence[int] = (2, 3, 4)
    ) -> float:
        """Expected Kicktipp points of an arbitrary tip for this game."""
        p_home, p_draw, _ = tip_strategy.implied_probabilities(self._validated_quotes)
        lam_home, lam_away = tip_strategy.fit_expected_goals(round(p_home, 4), round(p_draw, 4))
        matrix = tip_strategy.score_matrix(lam_home, lam_away)
        return sum(p * tip_strategy.kicktipp_points(tip, result, points_scheme)
                   for result, p in matrix.items())

    def resolve_draw_tip(
        self,
        tip: Tuple[int, int],
        home_quote: Optional[float] = None,
        away_quote: Optional[float] = None
    ) -> Tuple[int, int]:
        """Resolve a draw prediction into a one-goal win based on win quotes."""
        if tip[0] != tip[1]:
            return tip

        home_quote, away_quote = self._get_effective_win_quotes(
            home_quote, away_quote)

        if home_quote <= away_quote:
            return tip[0] + 1, tip[1]

        return tip[0], tip[1] + 1

    def _get_effective_win_quotes(
        self,
        home_quote: Optional[float],
        away_quote: Optional[float]
    ) -> Tuple[float, float]:
        """Return explicit home/away win quotes, falling back to validated values."""
        if home_quote is None:
            home_quote = self._validated_quotes[0]
        if away_quote is None:
            away_quote = self._validated_quotes[2]
        return home_quote, away_quote

    @staticmethod
    def _calculate_base_tip(home_quote: float, away_quote: float) -> Tuple[int, int]:
        """Create a raw tip from home/away quotes before draw-rule adjustments."""
        # Calculate quote difference (negative = home team more likely to win)
        quote_difference = home_quote - away_quote

        # Add randomness for more realistic scores
        random_goal = random.randint(0, 1)

        # Lower coefficient for very unequal games to avoid extreme scores
        coefficient = 0.3 if abs(quote_difference) > 7 else 0.75

        if abs(quote_difference) < 0.25:
            # Very close match - predict draw-like result
            return random_goal, random_goal

        if quote_difference < 0:
            # Home team favored
            home_goals = max(0, round(-quote_difference * coefficient)) + random_goal
            return home_goals, random_goal

        # Away team favored
        away_goals = max(0, round(quote_difference * coefficient)) + random_goal
        return random_goal, away_goals

    def __str__(self) -> str:
        """String representation of the game."""
        return f"{self.home_team} vs {self.away_team} at {self.game_time.strftime('%d.%m.%y %H:%M')}"
