"""LeRobot BYOH Hardware Plugin for TuinaDex 22-DOF Robot."""

from .config_tuinadex import TuinaRobotConfig
from .tuinadex import TuinaRobot

__all__ = ["TuinaRobot", "TuinaRobotConfig"]
