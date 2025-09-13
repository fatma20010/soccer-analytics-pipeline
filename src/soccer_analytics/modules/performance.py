from __future__ import annotations
from typing import Dict
import math
import time

class PerformanceMetrics:
    def __init__(self):
        self.last_positions = {}  # track_id -> (x,y,time)
        self.distance = {}  # track_id -> pixels
        self.speeds = {}  # track_id -> last speed
        self.max_speed = {}  # track_id -> max speed
        self.touches = {}  # track_id -> count
        self.pass_graph = {}  # (from,to) -> count
        self.pass_count = {}  # per player (as passer)
        self.first_seen_time = {}  # track_id -> timestamp

    def update_positions(self, tracks):
        now = time.time()
        for t in tracks:
            tid = t['track_id']
            x1,y1,x2,y2 = t['bbox']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            if tid in self.last_positions:
                px,py,pt = self.last_positions[tid]
                dt = max(now-pt, 1e-6)
                d = math.hypot(cx-px, cy-py)
                self.distance[tid] = self.distance.get(tid,0.0)+d
                self.speeds[tid] = d/dt
                if tid not in self.max_speed or self.speeds[tid] > self.max_speed[tid]:
                    self.max_speed[tid] = self.speeds[tid]
            self.last_positions[tid] = (cx,cy,now)
            if tid not in self.first_seen_time:
                self.first_seen_time[tid] = now

    def register_touch(self, track_id: int):
        self.touches[track_id] = self.touches.get(track_id,0)+1

    def register_pass(self, from_id: int, to_id: int):
        key = (from_id, to_id)
        self.pass_graph[key] = self.pass_graph.get(key,0)+1
        self.pass_count[from_id] = self.pass_count.get(from_id,0)+1

    def snapshot(self):
        return {
            'distance_px': dict(self.distance),
            'speeds_px_per_s': dict(self.speeds),
            'max_speed_px_per_s': dict(self.max_speed),
            'touches': dict(self.touches),
            'pass_graph': dict(self.pass_graph),
            'passes_per_player': dict(self.pass_count),
            'first_seen_time': dict(self.first_seen_time)
        }

    def avg_speed(self, track_id: int) -> float:
        if track_id not in self.distance or track_id not in self.first_seen_time:
            return 0.0
        span = max(time.time() - self.first_seen_time[track_id], 1e-6)
        return self.distance[track_id] / span
