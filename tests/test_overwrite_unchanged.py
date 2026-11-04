import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kicktipp_bot.core.game_tipper import GameTipper  # noqa: E402


def _field(value):
    field = MagicMock()
    field.get_attribute.return_value = value
    return field


def test_same_tip_detected():
    assert GameTipper._has_same_tip(_field("2"), _field("1"), (2, 1))


def test_different_tip_detected():
    assert not GameTipper._has_same_tip(_field("1"), _field("1"), (2, 1))


def test_empty_fields_are_not_same():
    assert not GameTipper._has_same_tip(_field(""), _field(""), (2, 1))
