# Real-Time Football (Soccer) Analytics Pipeline (CPU-Friendly)

> Ready for GitHub: includes modular code, config-driven optimizations (`detect_interval`, `team_update_interval`, `frame_resize_width`), and MIT license.

This project provides a modular pipeline to process a broadcast (or webcam) football video and extract **live statistics**:

- Player / ball detection (YOLOv8) + tracking (ByteTrack)
- Team assignment via jersey color clustering
- Ball possession percentages
- Pass detection (heuristic)
- Player touches, distance, speed (pixel scale), pass network
- Naive goal detection heuristic
- Manual event tagging: red/yellow cards, free kicks, penalties (keyboard)
- Homography stub for mapping to pitch coordinates (extendable)

> Designed to run **without CUDA** (CPU only). For better speed choose a lightweight YOLO model (`yolov8n.pt`). Expect limited FPS on pure CPU; optimize later by quantization or model pruning.

## 1. Environment Setup (Windows, CPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip wheel setuptools
# Install torch CPU (adjust if newer version available)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Optional speed-ups (see section Performance Optimization):
- Set `frame_resize_width` (default 960) to 832 or 640
- Increase `detect_interval` (e.g. 2–3) to run YOLO less frequently
- Increase `team_update_interval` (e.g. 30) to reduce KMeans clustering
- ONNX export with `onnxruntime` (future enhancement)

## 2. Run Real-Time / File

```powershell
python scripts/run_realtime.py --source 0              # webcam
python scripts/run_realtime.py --source .\sample.mp4  # file
```

Press `ESC` to exit.

### Keyboard Controls (Manual Event Annotation)
While the window is focused:

| Key | Action |
|-----|--------|
| R   | Red card to nearest player to frame center |
| Y   | Yellow card to nearest player to frame center |
| F   | Free kick (at current possession team) |
| P   | Penalty (at current possession team) |
| Q / ESC | Quit |

Events summary appears in overlay: `YC`, `RC`, `FK`, `PEN`.

## 3. Download Sample Data

Register at [SoccerNet](https://www.soccer-net.org/) and download a small clip OR capture a short TV segment for experimentation (respect rights). For a trivial test (no real gameplay), download a single frame:

```powershell
python scripts/dataset_download.py
```

## 4. Architecture Overview

```
src/soccer_analytics/
  config.py              # Central configuration dataclasses
  pipeline.py            # Orchestrates modules per frame
  modules/
    detection.py         # YOLO inference + entity extraction
    tracking.py          # ByteTrack wrapper
    team_classifier.py   # KMeans color clustering
    possession.py        # Possession timeline
    passes.py            # Pass event detection
    performance.py       # Distance, speed, touches, pass graph
    goal_detection.py    # Goal heuristics
    homography.py        # Pitch homography stub
```

## 5. Configuration Keys

Edit `config.py` (`PipelineConfig`) or override via environment variables if you adapt code:

| Field | Purpose | Typical Values |
|-------|---------|---------------|
| `frame_resize_width` | Downscale width before inference | 640 / 832 / 960 / None |
| `detect_interval` | Run YOLO every N frames | 1–3 |
| `team_update_interval` | KMeans jersey clustering every N frames | 15–60 |
| `model.conf_threshold` | YOLO confidence filter | 0.4–0.6 |
| `possession.ball_player_max_distance` | Radius (px) to attribute touch | 50–80 |

## 6. Current Heuristics & Limitations
- Ball detection depends on YOLO class (model may miss small ball at distance).
- Team color clustering naive; improve with jersey segmentation / histogram features and temporal smoothing.
- Possession uses nearest player to ball; can enhance with velocity alignment & intersection.
- Pass detection is simplistic: change of possessing track within same team.
- Goal detection naive: ball near frame edge in center vertical band.
- Distance & speed are in **pixels** until homography & scale calibration implemented.
- Card / free kick / penalty detection is manual (keyboard) for now.

## 7. Performance Optimization

Baseline on pure CPU can be low FPS. Combine these:

1. Downscale frames: set `frame_resize_width=640`.
2. Frame skipping for detection: `detect_interval=2` (tracker interpolates between detections).
3. Throttle team clustering: raise `team_update_interval` to 30 or 45.
4. Turn off debug overlay (`debug=False`).
5. Comment out heavy table rows in `stats_overlay.py` if profiling shows draw time issues.
6. (Future) Export model: `yolo export model=yolov8n.pt format=onnx` then integrate ONNXRuntime.

## 8. Planned Enhancements
- Automatic pitch line detection to compute metric distances.
- Improved ball tracking with dedicated small-object model or Kalman filter.
- Advanced event detection (shots, interceptions, tackles).
- Pass quality metrics (progressive distance, pressure zones).
- GPU / TensorRT / ONNX acceleration path.

## 9. Homography (Manual)
Use `HomographyEstimator.set_manual_points(frame_points, pitch_points)` with 4 correspondences (e.g. corners of penalty box). Pitch reference coordinates: (0,0) top-left, (105,68) meters.

## 10. Testing
Run minimal tests (coming soon):
```powershell
pytest -q
```

## 11. Troubleshooting
| Issue | Tip |
|-------|-----|
| ImportError ultralytics | `pip install ultralytics` and ensure Python >=3.8 |
| Slow FPS | Reduce frame width, switch to `yolov8n.pt`, skip visualization |
| No ball detected | Consider fine-tuning w/ ball-focused dataset or use a separate small-object model |

## 12. Attribution

## 13. Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes with clear messages
4. Open a Pull Request describing motivation & screenshots/gifs

Please run tests (`pytest -q`) before submitting PRs.

## 14. License

MIT License (see `LICENSE`).
Uses YOLOv8 (Ultralytics), ByteTrack concept, OpenCV, scikit-learn.

---
Contributions and improvements welcome. This is a starting framework to expand into richer analytics.
