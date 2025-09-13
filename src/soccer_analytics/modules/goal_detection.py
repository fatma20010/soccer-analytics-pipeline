from __future__ import annotations
from typing import Optional, List

class GoalDetector:
    def __init__(self, frame_width: int | None = None, frame_height: int | None = None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.events: List[dict] = []

    def update(self, ball_xy: Optional[tuple[float,float]]):
        if ball_xy is None or self.frame_width is None or self.frame_height is None:
            return
        x,y = ball_xy
        # naive heuristic: ball crosses near left or right edge in vertical central band
        band_top = self.frame_height*0.25
        band_bottom = self.frame_height*0.75
        if band_top < y < band_bottom:
            if x < self.frame_width*0.02:
                self.events.append({'side':'left'})
            elif x > self.frame_width*0.98:
                self.events.append({'side':'right'})

    def goals(self):
        return len(self.events)
