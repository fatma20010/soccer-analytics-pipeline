from __future__ import annotations
from typing import List, Dict, Optional
import time

class PassDetector:
    def __init__(self):
        self.events: List[dict] = []
        self._last_team = None
        self._last_track = None
        self.interceptions: List[dict] = []  # team gaining possession

    def update(self, possession_team: Optional[int], possessor_track: Optional[int]):
        now = time.time()
        if possession_team is None:
            return
        if self._last_team is None:
            self._last_team = possession_team
            self._last_track = possessor_track
            return
        if possession_team == self._last_team and possessor_track != self._last_track and possessor_track is not None and self._last_track is not None:
            # same team, different player -> pass
            self.events.append({'timestamp': now, 'team': possession_team, 'from': self._last_track, 'to': possessor_track})
        elif possession_team != self._last_team:
            # change of team possession not captured as pass -> interception
            self.interceptions.append({'timestamp': now, 'team': possession_team})
        self._last_team = possession_team
        self._last_track = possessor_track

    def stats(self):
        counts = {}
        for e in self.events:
            counts[e['team']] = counts.get(e['team'],0)+1
        inter_counts = {}
        for i in self.interceptions:
            inter_counts[i['team']] = inter_counts.get(i['team'],0)+1
        return {'passes': counts, 'interceptions': inter_counts}

    def new_passes_since(self, last_index: int):
        """Return new pass events and new index pointer."""
        if last_index < 0:
            last_index = 0
        return self.events[last_index:], len(self.events)
