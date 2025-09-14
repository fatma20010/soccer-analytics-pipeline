import cv2
import argparse
import time
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from soccer_analytics.pipeline import SoccerPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video path or camera index (as string)')
    args = parser.parse_args()
    source = args.source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit('Cannot open video source')
    pipe = SoccerPipeline()
    last_t = time.time()
    print('[Key Help] R=Red card, Y=Yellow card, F=Free kick, P=Penalty, Q=Quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = pipe.process_frame(frame)
        now = time.time()
        fps = 1.0/(now-last_t)
        last_t = now
        cv2.putText(out,f'FPS {fps:.1f}',(10,out.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.imshow('Soccer Analytics', out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key in (ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            # Red card: choose a currently nearest track to screen center if available
            if pipe.performance.last_positions:
                center = (out.shape[1]/2, out.shape[0]/2)
                closest = min(pipe.performance.last_positions.items(), key=lambda kv: ( (kv[1][0]-center[0])**2 + (kv[1][1]-center[1])**2 ))
                tid = closest[0]
                team = pipe.team_classifier.get_team(tid)
                pipe.events.add_card(tid, team, 'R')
        elif key in (ord('y'), ord('Y')):
            if pipe.performance.last_positions:
                center = (out.shape[1]/2, out.shape[0]/2)
                closest = min(pipe.performance.last_positions.items(), key=lambda kv: ( (kv[1][0]-center[0])**2 + (kv[1][1]-center[1])**2 ))
                tid = closest[0]
                team = pipe.team_classifier.get_team(tid)
                pipe.events.add_card(tid, team, 'Y')
        elif key in (ord('f'), ord('F')):
            pipe.events.add_free_kick(pipe.possession.last_touch_team)
        elif key in (ord('p'), ord('P')):
            pipe.events.add_penalty(pipe.possession.last_touch_team)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
