from __future__ import annotations
import cv2
import numpy as np
import time
from typing import Dict

PANEL_BG = (25, 40, 30)
PANEL_ALPHA = 0.55
TITLE_COLOR = (20, 200, 255)
TEXT_COLOR = (230, 230, 230)
ACCENT_A = (0, 200, 255)
ACCENT_B = (255, 180, 0)
TEAM_COLORS = [(40,180,60),(60,80,220)]  # BGR
WARNING = (0,215,255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

class StatsRenderer:
    def __init__(self, max_players_display: int = 8):
        self.max_players_display = max_players_display

    def _draw_panel(self, frame, x, y, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x,y), (x+w, y+h), PANEL_BG, -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1-PANEL_ALPHA, 0, frame)

    def _outlined_text(self, img, text, org, scale=0.5, color=TEXT_COLOR, thickness=1):
        cv2.putText(img, text, org, FONT, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

    def render(self, frame, possession_pct: Dict[int,float], pass_stats: Dict, goals: int, events_summary: Dict,
               performance_snapshot: Dict, performance_obj):
        h, w = frame.shape[:2]
        # compute dynamic height
        panel_w = int(w*0.32)
        header_h = 110
        player_rows = self.max_players_display
        panel_h = header_h + player_rows*18 + 20
        x0, y0 = 8, 8
        self._draw_panel(frame, x0, y0, panel_w, panel_h)
        cursor_y = y0 + 18
        self._outlined_text(frame, 'LIVE MATCH STATS', (x0+10, cursor_y), 0.6, TITLE_COLOR, 2)
        cursor_y += 14
        # possession line
        t0 = possession_pct.get(0,0.0)
        t1 = possession_pct.get(1,0.0)
        self._outlined_text(frame, f'POSS {t0:.1f}% - {t1:.1f}%', (x0+10, cursor_y), 0.5, TEXT_COLOR)
        cursor_y += 8
        # possession bar
        bar_margin = 10
        bar_x1 = x0 + bar_margin
        bar_x2 = x0 + panel_w - bar_margin
        bar_y = cursor_y + 10
        bar_h = 14
        total = t0 + t1 if (t0+t1) > 0 else 1
        w0 = int((t0/total) * (bar_x2-bar_x1))
        cv2.rectangle(frame,(bar_x1, bar_y),(bar_x2, bar_y+bar_h),(60,60,60),1)
        cv2.rectangle(frame,(bar_x1, bar_y),(bar_x1+w0, bar_y+bar_h),TEAM_COLORS[0],-1)
        cv2.rectangle(frame,(bar_x1+w0, bar_y),(bar_x2, bar_y+bar_h),TEAM_COLORS[1],-1)
        cursor_y = bar_y + bar_h + 18
        # team pass / interception stats
        passes = pass_stats.get('passes',{})
        inters = pass_stats.get('interceptions',{})
        self._outlined_text(frame, f'PASS  T0 {passes.get(0,0)}  T1 {passes.get(1,0)}', (x0+10,cursor_y),0.5,ACCENT_A)
        cursor_y += 18
        self._outlined_text(frame, f'INTER T0 {inters.get(0,0)}  T1 {inters.get(1,0)}', (x0+10,cursor_y),0.5,ACCENT_B)
        cursor_y += 18
        self._outlined_text(frame, f"GOALS {goals}  YC {events_summary['yellow_cards']}  RC {events_summary['red_cards']}", (x0+10,cursor_y),0.5,WARNING)
        cursor_y += 24
        # player performance section title
        self._outlined_text(frame, 'PLAYER PERFORMANCE', (x0+10,cursor_y),0.55,TITLE_COLOR,2)
        cursor_y += 18
        # gather player metrics
        dist_map = performance_snapshot['distance_px']
        vmax_map = performance_snapshot['max_speed_px_per_s']
        passes_player = performance_snapshot['passes_per_player']
        first_seen = performance_snapshot['first_seen_time']
        # sort by distance
        ordered = sorted(dist_map.items(), key=lambda x: x[1], reverse=True)[:self.max_players_display]
        now = time.time()
        for tid, dist_px in ordered:
            vmax = vmax_map.get(tid,0.0)
            span = max(now - first_seen.get(tid, now), 1e-6)
            avg_speed = dist_px / span
            pcount = passes_player.get(tid,0)
            line = f'P{tid:02d} D:{dist_px/100:.1f}m Av:{avg_speed:.1f} Max:{vmax:.1f} P:{pcount}'
            self._outlined_text(frame, line, (x0+10,cursor_y),0.45,TEXT_COLOR)
            cursor_y += 16
        return frame
