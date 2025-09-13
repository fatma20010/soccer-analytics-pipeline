from __future__ import annotations
from typing import Optional, Dict
import math
import time

class PossessionTracker:
    def __init__(self):
        self.last_touch_team: Optional[int] = None
        self.team_time: Dict[int,float] = {}
        self._last_switch_timestamp = time.time()

    def update(self, ball_xy, tracks, team_lookup):
        # Find nearest track to ball
        if ball_xy is None:
            return
        bx, by = ball_xy
        min_d = 1e9
        closest_tid = None
        for t in tracks:
            x1,y1,x2,y2 = t['bbox']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            d = math.hypot(cx-bx, cy-by)
            if d < min_d:
                min_d = d
                closest_tid = t['track_id']
        if closest_tid is None:
            return
        team = team_lookup(closest_tid)
        if team is None:
            return
        now = time.time()
        if self.last_touch_team is None:
            self.last_touch_team = team
            self._last_switch_timestamp = now
        elif team != self.last_touch_team and min_d < 80:  # heuristic distance threshold for control change
            # allocate time to previous team
            dt = now - self._last_switch_timestamp
            self.team_time[self.last_touch_team] = self.team_time.get(self.last_touch_team,0.0)+dt
            self.last_touch_team = team
            self._last_switch_timestamp = now

    def get_possession(self):
        total = sum(self.team_time.values())
        if self.last_touch_team is not None:
            total += time.time() - self._last_switch_timestamp
        if total == 0:
            return {}
        pct = {team: (t / total)*100 for team, t in self.team_time.items()}
        if self.last_touch_team is not None:
            current = self.team_time.get(self.last_touch_team,0.0) + (time.time() - self._last_switch_timestamp)
            pct[self.last_touch_team] = (current/total)*100
        return pct
