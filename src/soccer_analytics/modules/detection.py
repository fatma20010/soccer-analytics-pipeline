from __future__ import annotations
from typing import List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # graceful import
    YOLO = None  # type: ignore

from ..config import CONFIG

PLAYER_LABELS = {0: 'person'}  # YOLOv8 default class 0
BALL_KEYWORDS = {'ball'}  # we will also allow substring 'ball'
GOALKEEPER_ROLE_HINT = {'keeper','goalkeeper','goalie'}  # placeholder for future role model

class Detector:
    def __init__(self, model_path: str | None = None):
        if YOLO is None:
            raise RuntimeError('ultralytics not installed. Install per requirements.txt')
        self.model = YOLO(model_path or CONFIG.model.yolo_model)
        self.names = self.model.model.names  # type: ignore

    def infer(self, frame: np.ndarray):
        """Run inference on a single frame and return raw results."""
        results = self.model.predict(frame, verbose=False, conf=CONFIG.model.conf_threshold, iou=CONFIG.model.iou_threshold, device=CONFIG.model.device)
        return results[0]

    def extract_entities(self, result) -> dict:
        """Extract players, ball, referee candidates from YOLO result.
        Returns dict with keys: players, ball, others (list of (bbox, cls, conf))."""
        players = []
        ball = []
        others = []
        if not hasattr(result, 'boxes'):
            return {'players': players, 'ball': ball, 'others': others}
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0].tolist()
            label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else str(cls_id)
            record = {'bbox': xyxy, 'cls': cls_id, 'label': label, 'conf': conf}
            low_label = label.lower()
            if low_label in BALL_KEYWORDS or 'ball' in low_label:
                ball.append(record)
            elif label == 'person':
                players.append(record)
            else:
                others.append(record)
        return {'players': players, 'ball': ball, 'others': others}
