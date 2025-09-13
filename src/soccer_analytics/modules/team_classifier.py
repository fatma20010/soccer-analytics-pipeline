from __future__ import annotations
from typing import Dict, List
import numpy as np
import cv2
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self, k: int = 2):
        self.k = k
        self.track_team: dict[int,int] = {}
        self._fitted = False

    def _dominant_color(self, frame, bbox):
        x1,y1,x2,y2 = map(int, bbox)
        crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0:
            return None
        # focus on upper body area (reduce grass influence)
        h = crop.shape[0]
        upper = crop[:int(h*0.6)]
        lab = cv2.cvtColor(upper, cv2.COLOR_BGR2LAB)
        lab_reshaped = lab.reshape(-1,3)
        # sample subset for speed
        if lab_reshaped.shape[0] > 400:
            idx = np.random.choice(lab_reshaped.shape[0], 400, replace=False)
            lab_reshaped = lab_reshaped[idx]
        mean = lab_reshaped.mean(axis=0)
        return mean

    def fit_update(self, frame, tracks: List[dict]):
        features = []
        ids = []
        for t in tracks:
            col = self._dominant_color(frame, t['bbox'])
            if col is not None:
                features.append(col)
                ids.append(t['track_id'])
        if len(features) < self.k:
            return
        X = np.vstack(features)
        kmeans = KMeans(n_clusters=self.k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(X)
        # assign cluster indices; optionally reorder by average L channel
        self._fitted = True
        for tid, lab in zip(ids, labels):
            self.track_team[tid] = int(lab)

    def get_team(self, track_id: int) -> int | None:
        return self.track_team.get(track_id)
