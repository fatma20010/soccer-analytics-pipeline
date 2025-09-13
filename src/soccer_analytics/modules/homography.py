from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

PITCH_WIDTH_M = 105
PITCH_HEIGHT_M = 68

class HomographyEstimator:
    """Stub for estimating homography from broadcast frame to metric pitch coordinates.
    Initially manual; can be extended with line detection & template matching."""
    def __init__(self):
        self.H: Optional[np.ndarray] = None

    def set_manual_points(self, frame_points: list[tuple[int,int]], pitch_points: list[tuple[float,float]]):
        if len(frame_points) != 4 or len(pitch_points) != 4:
            raise ValueError('Need exactly 4 points for now (corners or known landmarks).')
        src = np.array(frame_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst, method=0)
        return self.H

    def warp_point(self, x: float, y: float) -> Optional[Tuple[float,float]]:
        if self.H is None:
            return None
        pt = np.array([[x, y, 1.0]], dtype=np.float32).T
        mapped = self.H @ pt
        mapped /= mapped[2,0]
        return float(mapped[0,0]), float(mapped[1,0])
