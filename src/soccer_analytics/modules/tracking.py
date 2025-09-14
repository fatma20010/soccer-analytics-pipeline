from __future__ import annotations
from typing import List, Dict
import numpy as np
import math

try:  # optional import (not strictly needed for logic now)
    from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
except Exception:  # pragma: no cover
    BYTETracker = None  # type: ignore

from ..config import CONFIG

class _ArgNamespace:
    """Simple attribute container to satisfy BYTETracker's expected args interface."""
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class SimpleTracker:
    """Fallback lightweight centroid/IoU tracker if BYTETracker not available or fails.
    NOT production-grade—keeps IDs stable using nearest-neighbour + IoU threshold."""
    def __init__(self, max_age: int = 30, iou_thresh: float = 0.2):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, detections: List[dict]):
        updated_ids = set()
        # match existing tracks
        for det in detections:
            db = det['bbox']
            best_iou = 0
            best_id = None
            for tid, info in self.tracks.items():
                iou = self._iou(db, info['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_iou >= self.iou_thresh and best_id is not None:
                self.tracks[best_id]['bbox'] = db
                self.tracks[best_id]['age'] = 0
                updated_ids.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': db, 'age': 0}
                updated_ids.add(tid)
        # age and remove stale
        remove = []
        for tid, info in self.tracks.items():
            if tid not in updated_ids:
                info['age'] += 1
                if info['age'] > self.max_age:
                    remove.append(tid)
        for tid in remove:
            self.tracks.pop(tid, None)
        # return list
        return [{'track_id': tid, 'bbox': info['bbox']} for tid, info in self.tracks.items()]

class TrackManager:
    def __init__(self):
        self.use_fallback = False
        if BYTETracker is not None:
            try:
                args = _ArgNamespace(
                    track_high_thresh=0.5,
                    track_low_thresh=0.1,
                    new_track_thresh=0.6,
                    track_buffer=CONFIG.tracking.track_max_age,
                    match_thresh=CONFIG.tracking.track_iou_threshold,
                    aspect_ratio_thresh=10.0,
                    min_box_area=0,
                    mot20=False,
                )
                self.tracker = BYTETracker(args, frame_rate=25)
            except Exception:  # fallback if API mismatch
                self.use_fallback = True
                self.tracker = SimpleTracker(CONFIG.tracking.track_max_age, CONFIG.tracking.track_iou_threshold)
        else:
            self.use_fallback = True
            self.tracker = SimpleTracker(CONFIG.tracking.track_max_age, CONFIG.tracking.track_iou_threshold)

    def update(self, player_detections: List[dict]):
        if not player_detections:
            return []
        if self.use_fallback:
            return self.tracker.update(player_detections)
        # BYTETracker path
        tlwhs = []
        scores = []
        for det in player_detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            tlwhs.append([x1, y1, w, h])
            scores.append(det['conf'])
        
        # Handle empty detections
        if not tlwhs:
            return []
            
        tlwhs = np.asarray(tlwhs, dtype=float)
        scores = np.asarray(scores, dtype=float)
        try:
            # Newer Ultralytics BYTETracker expects 'dets' shaped (N,6) [x1,y1,x2,y2,score,cls]
            xyxy = np.zeros((len(tlwhs), 4))
            # convert tlwh -> xyxy
            # tlwh: x,y,w,h
            xyxy[:,0] = tlwhs[:,0]  # x1
            xyxy[:,1] = tlwhs[:,1]  # y1
            xyxy[:,2] = tlwhs[:,0] + tlwhs[:,2]  # x2
            xyxy[:,3] = tlwhs[:,1] + tlwhs[:,3]  # y2
            dets = np.concatenate([xyxy, scores.reshape(-1,1), np.zeros((len(scores),1))], axis=1)
            online_targets = self.tracker.update(dets)
        except (TypeError, AttributeError) as e:
            # Fallback to older API or SimpleTracker
            try:
                # Try older internal API variant (custom) may accept tlwhs, scores, classes
                online_targets = self.tracker.update(tlwhs, scores, np.zeros(len(scores)))
            except Exception:
                # Final fallback: switch to SimpleTracker mid-run
                print(f"⚠️  BYTETracker failed ({e}), switching to SimpleTracker")
                self.use_fallback = True
                self.tracker = SimpleTracker(CONFIG.tracking.track_max_age, CONFIG.tracking.track_iou_threshold)
                return self.tracker.update(player_detections)
        tracks = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            tracks.append({'track_id': int(t.track_id), 'bbox': [float(x1), float(y1), float(x1+w), float(y1+h)]})
        return tracks
