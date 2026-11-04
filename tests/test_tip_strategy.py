"""Unit tests for the expected-points tip strategy (run: python -m unittest discover tests)."""

import importlib.util
import pathlib
import sys
import unittest
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "kicktipp_bot" / "models"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the modules directly so the tests do not need selenium installed.
tip_strategy = _load("kicktipp_bot.models.tip_strategy", "tip_strategy.py")
sys.modules["kicktipp_bot.models"] = type(sys)("kicktipp_bot.models")
sys.modules["kicktipp_bot.models"].tip_strategy = tip_strategy
game_module = _load("kicktipp_bot.models.game", "game.py")
Game = game_module.Game


class KicktippPointsTest(unittest.TestCase):
    def test_default_scheme(self):
        pts = tip_strategy.kicktipp_points
        self.assertEqual(pts((2, 1), (2, 1), (2, 3, 4)), 4)   # exact
        self.assertEqual(pts((2, 1), (3, 2), (2, 3, 4)), 3)   # same difference
        self.assertEqual(pts((2, 1), (1, 0), (2, 3, 4)), 3)
        self.assertEqual(pts((2, 1), (3, 0), (2, 3, 4)), 2)   # tendency only
        self.assertEqual(pts((1, 1), (0, 0), (2, 3, 4)), 3)   # draw vs other draw
        self.assertEqual(pts((1, 1), (2, 1), (2, 3, 4)), 0)
        self.assertEqual(pts((2, 1), (1, 2), (2, 3, 4)), 0)

    def test_custom_scheme(self):
        self.assertEqual(tip_strategy.kicktipp_points((1, 0), (1, 0), (1, 2, 3)), 3)
        self.assertEqual(tip_strategy.kicktipp_points((1, 0), (2, 0), (1, 2, 3)), 1)


class ModelTest(unittest.TestCase):
    def test_implied_probabilities_sum_to_one(self):
        p = tip_strategy.implied_probabilities([1.5, 4.2, 5.5])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[0], p[2])

    def test_fit_reproduces_market(self):
        p_home, p_draw, _ = tip_strategy.implied_probabilities([1.85, 3.6, 4.0])
        lh, la = tip_strategy.fit_expected_goals(round(p_home, 4), round(p_draw, 4))
        mh, md, _ = tip_strategy.outcome_probabilities(tip_strategy.score_matrix(lh, la))
        self.assertAlmostEqual(mh, p_home, delta=0.01)
        self.assertAlmostEqual(md, p_draw, delta=0.01)
        self.assertGreater(lh, la)


class BestTipTest(unittest.TestCase):
    def test_heavy_home_favourite(self):
        tip, _ = tip_strategy.best_tip([1.25, 6.0, 11.0])
        self.assertGreater(tip[0], tip[1])
        self.assertLessEqual(tip[0], 3)

    def test_heavy_away_favourite(self):
        tip, _ = tip_strategy.best_tip([9.5, 6.0, 1.28])
        self.assertLess(tip[0], tip[1])

    def test_even_game_prefers_low_scores(self):
        tip, _ = tip_strategy.best_tip([2.7, 3.3, 2.6])
        self.assertLessEqual(tip[0] + tip[1], 2)

    def test_no_draw_when_not_allowed(self):
        tip, _ = tip_strategy.best_tip([2.8, 3.0, 2.8], allow_draw=False)
        self.assertNotEqual(tip[0], tip[1])

    def test_deterministic(self):
        tips = {tip_strategy.best_tip([2.1, 3.4, 3.4])[0] for _ in range(5)}
        self.assertEqual(len(tips), 1)

    def test_points_scheme_parse(self):
        self.assertEqual(tip_strategy.parse_points_scheme(None), (2, 3, 4))
        self.assertEqual(tip_strategy.parse_points_scheme("1, 2, 3"), (1, 2, 3))
        with self.assertRaises(ValueError):
            tip_strategy.parse_points_scheme("1,2")


class GameIntegrationTest(unittest.TestCase):
    def test_comma_quotes_are_accepted(self):
        game = Game("A", "B", ["1,85", "3,60", "4,00"], datetime.now())
        self.assertEqual(game._validated_quotes, [1.85, 3.6, 4.0])

    def test_default_strategy_is_ev(self):
        game = Game("A", "B", ["1.55", "4.20", "5.50"], datetime.now())
        self.assertEqual(game.calculate_tip(), (2, 1))

    def test_legacy_strategy_still_available(self):
        game = Game("A", "B", ["1.55", "4.20", "5.50"], datetime.now())
        home, away = game.calculate_tip(strategy="legacy")
        self.assertGreater(home, away)

    def test_expected_points_of_best_tip_is_max(self):
        game = Game("A", "B", ["2.3", "3.4", "3.0"], datetime.now())
        best = game.calculate_tip()
        for other in ((3, 0), (2, 2), (0, 3), (4, 1)):
            self.assertGreaterEqual(game.expected_points(best), game.expected_points(other))


if __name__ == "__main__":
    unittest.main()
