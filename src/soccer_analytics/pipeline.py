from __future__ import annotations
import cv2
import time
from typing import Optional
from .config import CONFIG
from .modules.detection import Detector
from .modules.tracking import TrackManager
from .modules.team_classifier import TeamClassifier
from .modules.possession import PossessionTracker
from .modules.passes import PassDetector
from .modules.performance import PerformanceMetrics
from .modules.goal_detection import GoalDetector
from .modules.events import MatchEvents
from .modules.stats_overlay import StatsRenderer

class SoccerPipeline:
    def __init__(self):
        self.detector = Detector()
        self.tracker = TrackManager()
        self.team_classifier = TeamClassifier()
        self.possession = PossessionTracker()
        self.passes = PassDetector()
        self.performance = PerformanceMetrics()
        self.goal_detector: Optional[GoalDetector] = None
        self.events = MatchEvents()
        self._pass_index = 0  # pointer for new passes
        self.renderer = StatsRenderer()
        self._no_ball_frames = 0

    def process_frame(self, frame):
        h,w = frame.shape[:2]
        if self.goal_detector is None:
            self.goal_detector = GoalDetector(w,h)
        if CONFIG.frame_resize_width and w != CONFIG.frame_resize_width:
            scale = CONFIG.frame_resize_width / w
            frame = cv2.resize(frame, (CONFIG.frame_resize_width, int(h*scale)))
            h,w = frame.shape[:2]
        result = self.detector.infer(frame)
        entities = self.detector.extract_entities(result)
        tracks = self.tracker.update(entities['players'])
        # team assignment update occasionally
        if len(tracks) >= 4:  # rough threshold
            self.team_classifier.fit_update(frame, tracks)
        # extract ball center
        ball_xy = None
        if entities['ball']:
            b = entities['ball'][0]['bbox']
            ball_xy = ((b[0]+b[2])/2, (b[1]+b[3])/2)
            self._no_ball_frames = 0
        else:
            self._no_ball_frames += 1
        # possession update
        self.possession.update(ball_xy, tracks, self.team_classifier.get_team)
        # performance
        self.performance.update_positions(tracks)
        # pass detection + touches heuristic (if distance < threshold)
        poss_track = None
        if ball_xy is not None:
            # find which track controlling (reuse logic quickly)
            min_d=1e9
            for t in tracks:
                x1,y1,x2,y2=t['bbox']
                cx=(x1+x2)/2; cy=(y1+y2)/2
                d=((cx-ball_xy[0])**2 + (cy-ball_xy[1])**2)**0.5
                if d<min_d:
                    min_d=d; poss_track=t['track_id']
            if poss_track is not None and min_d<70:
                self.performance.register_touch(poss_track)
        current_team = self.possession.last_touch_team
        self.passes.update(current_team, poss_track)
        # update performance with any new pass events
        new_passes, self._pass_index = self.passes.new_passes_since(self._pass_index)
        for ev in new_passes:
            self.performance.register_pass(ev['from'], ev['to'])
        # goal detection
        self.goal_detector.update(ball_xy)
        overlay = frame.copy()
        if CONFIG.debug and self._no_ball_frames > 60:
            cv2.putText(overlay,f'No ball detected {self._no_ball_frames} frames',(10,overlay.shape[0]-40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,200,255),1)
        # draw tracks basic (outline only to not clutter panel)
        for t in tracks:
            x1,y1,x2,y2 = map(int, t['bbox'])
            team = self.team_classifier.get_team(t['track_id'])
            color = (0,255,0) if team == 0 else (0,0,255) if team == 1 else (200,200,200)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
        if ball_xy is not None:
            cv2.circle(overlay,(int(ball_xy[0]), int(ball_xy[1])),6,(0,255,255),-1)
        perf_snapshot = self.performance.snapshot()
        pass_stats = self.passes.stats()
        es = self.events.summary()
        poss = self.possession.get_possession()
        overlay = self.renderer.render(overlay, poss, pass_stats, self.goal_detector.goals(), es, perf_snapshot, self.performance)
        return overlay
