import numpy as np
import cv2
from soccer_analytics.modules.team_classifier import TeamClassifier

def test_team_classifier_no_crash():
    frame = np.zeros((200,400,3), dtype=np.uint8)
    # create two colored regions resembling players
    cv2.rectangle(frame,(10,10),(60,160),(0,0,255),-1)  # red
    cv2.rectangle(frame,(100,10),(150,160),(255,0,0),-1)  # blue
    tracks = [
        {'track_id':1,'bbox':[10,10,60,160]},
        {'track_id':2,'bbox':[100,10,150,160]},
    ]
    tc = TeamClassifier()
    tc.fit_update(frame, tracks)
    t1 = tc.get_team(1)
    t2 = tc.get_team(2)
    assert t1 is not None and t2 is not None and t1 != t2
