"""Configuration module for Kicktipp Bot."""

import os
from datetime import timedelta
from typing import Optional


class Config:
    """Configuration class containing all environment variables and constants."""

    # Base URLs
    BASE_URL = "https://www.kicktipp.de/"
    LOGIN_URL = "https://www.kicktipp.de/info/profil/login/"

    # Required environment variables
    EMAIL: Optional[str] = os.getenv("KICKTIPP_EMAIL")
    PASSWORD: Optional[str] = os.getenv("KICKTIPP_PASSWORD")
    NAME_OF_COMPETITION: Optional[str] = os.getenv(
        "KICKTIPP_NAME_OF_COMPETITION")

    # Optional environment variables with defaults
    RUN_EVERY_X_MINUTES: Optional[int] = int(
        os.getenv("KICKTIPP_RUN_EVERY_X_MINUTES", "60"))
    OVERWRITE_TIPS: Optional[bool] = os.getenv("OVERWRITE_TIPS", "false").lower() == "true"

    # Tip strategy: "ev" (expected Kicktipp points, default) or "legacy"
    TIP_STRATEGY: str = os.getenv("KICKTIPP_TIP_STRATEGY", "ev").lower()
    # Points of the league for Tendenz, Tordifferenz, Ergebnis (default Kicktipp: 2,3,4)
    POINTS_SCHEME: tuple = tuple(
        int(x) for x in os.getenv("KICKTIPP_POINTS_SCHEME", "2,3,4").split(","))

    # Quotenpunkte leagues: "auto" reads the current points per tendency
    # (Heim-Remis-Gast, e.g. "1 - 2 - 4") from the tipping table when present,
    # "true" requires them, "false" ignores them.
    QUOTENPUNKTE: str = os.getenv("KICKTIPP_QUOTENPUNKTE", "auto").lower()
    # Does a draw tip earn Tordifferenz points for another draw? Rule tables
    # that show "-" in the Unentschieden/Tordifferenz cell mean "false".
    DRAW_DIFFERENCE_BONUS: bool = os.getenv(
        "KICKTIPP_DRAW_DIFFERENCE_BONUS", "true").lower() == "true"
    # Number of active tippers in the league (optional). When set, the bot
    # accounts for its own tip lowering the Quotenpunkte of that tendency.
    GROUP_SIZE: Optional[int] = (
        int(os.getenv("KICKTIPP_GROUP_SIZE")) if os.getenv("KICKTIPP_GROUP_SIZE") else None)

    # Only tip against the bookmaker favourite when the expected points beat
    # the best favourite tip by this margin (0 = pure expected value).
    UNDERDOG_MARGIN: float = float(os.getenv("KICKTIPP_UNDERDOG_MARGIN", "0.3"))

    # Chrome driver path
    CHROMEDRIVER_PATH: Optional[str] = os.getenv("CHROMEDRIVER_PATH")

    # Time configuration
    HOURS_UNTIL_GAME: Optional[int] = int(
        os.getenv("KICKTIPP_HOURS_UNTIL_GAME", "2"))
    TIME_UNTIL_GAME: timedelta = timedelta(hours=HOURS_UNTIL_GAME)

    # Notification URLs
    ZAPIER_URL: Optional[str] = os.getenv("ZAPIER_URL")
    NTFY_URL: Optional[str] = os.getenv("NTFY_URL")
    NTFY_USERNAME: Optional[str] = os.getenv("NTFY_USERNAME")
    NTFY_PASSWORD: Optional[str] = os.getenv("NTFY_PASSWORD")
    WEBHOOK_URL: Optional[str] = os.getenv("WEBHOOK_URL")
    GROUP_NOTIFICATIONS: bool = os.getenv("GROUP_NOTIFICATIONS", "false").lower() == "true"

    @classmethod
    def validate_required_config(cls) -> bool:
        """Validate that all required configuration is present."""
        required_vars = [cls.EMAIL, cls.PASSWORD, cls.NAME_OF_COMPETITION]
        return all(var is not None for var in required_vars)

    @classmethod
    def get_tipp_url(cls) -> str:
        """Get the URL for the tipping page."""
        return f"https://www.kicktipp.de/{cls.NAME_OF_COMPETITION}/tippabgabe"
