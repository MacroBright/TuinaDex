import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in [REPO_ROOT, REPO_ROOT / "Arm-robot_VLA", REPO_ROOT / "Leap_Hand" / "python", REPO_ROOT / "Co_Teleop"]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
