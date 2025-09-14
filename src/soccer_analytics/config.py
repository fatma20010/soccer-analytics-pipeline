from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

# NOTE: Previously PipelineConfig inherited from pydantic.BaseModel while using
# dataclasses.field for default_factory. That combination caused pydantic to
# treat the fields as required (no defaults), raising a TypeError when
# instantiating PipelineConfig(). Since we only need simple containers and no
# validation logic here, we convert PipelineConfig itself into a dataclass for
# consistency with the other *Config classes.

@dataclass
class ModelConfig:
    yolo_model: str = os.getenv('YOLO_MODEL', 'yolov8n.pt')  # nano model for CPU speed; can upgrade later
    device: str = os.getenv('DEVICE', 'cpu')  # force cpu (user has no CUDA)
    conf_threshold: float = 0.5
    iou_threshold: float = 0.4

@dataclass
class TrackingConfig:
    tracker_type: str = 'bytetrack'
    track_max_age: int = 30  # frames
    track_iou_threshold: float = 0.2

@dataclass
class TeamAssignmentConfig:
    update_interval_frames: int = 150
    kmeans_clusters: int = 2
    referee_separation: bool = True

@dataclass
class PossessionConfig:
    ball_player_max_distance: int = 60  # pixels radius to attribute possession
    smoothing_window: int = 5

@dataclass
class PassConfig:
    min_ball_speed: float = 2.5  # pixels/frame (heuristic)
    max_pass_time: float = 3.0  # seconds window for a pass event

@dataclass
class PerformanceConfig:
    speed_smoothing: int = 5

@dataclass
class GoalDetectionConfig:
    goal_line_margin: int = 12

@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    teams: TeamAssignmentConfig = field(default_factory=TeamAssignmentConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    passes: PassConfig = field(default_factory=PassConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    goals: GoalDetectionConfig = field(default_factory=GoalDetectionConfig)
    frame_resize_width: int | None = 960  # resize frames to this width maintaining aspect
    show_visualization: bool = True
    debug: bool = False
    detect_interval: int = 3  # run full detection every N frames (tracking in between)
    team_update_interval: int = 15  # run KMeans re-clustering every N frames

CONFIG = PipelineConfig()
