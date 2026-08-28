"""Co_Teleop 穴位感知与检测子包 (AcuPoint Perception Sub-Package)."""

from .base import AcupointDetectorProtocol
from .mock_backend import MockAcupointBackend
from .observation import AcupointObservation
from .visualization import draw_acupoints_on_image
from .worker import AcuPointWorker

try:
    from .ascend_backend import AscendAcupointBackend
except ImportError:
    AscendAcupointBackend = None  # type: ignore

__all__ = [
    "AcupointDetectorProtocol",
    "AcupointObservation",
    "AscendAcupointBackend",
    "MockAcupointBackend",
    "AcuPointWorker",
    "draw_acupoints_on_image",
]
