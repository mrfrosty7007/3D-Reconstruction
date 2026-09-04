"""
GeoRecon AI - Pipeline Stage Definitions
Defines the 6 studio pipeline stages, status enums, and data models for 3D reconstruction.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional


class StageType(Enum):
    """The 6 sequential reconstruction pipeline stages."""
    FRAME_EXTRACTION = "frames"
    COLMAP_FEATURES = "colmap_features"
    COLMAP_MATCHING = "colmap_matching"
    COLMAP_MAPPER = "colmap_mapper"
    GAUSSIAN_SPLATTING = "gsplat"
    EXPORT = "export"

    @property
    def display_name(self) -> str:
        names = {
            StageType.FRAME_EXTRACTION: "1. Extract Frames",
            StageType.COLMAP_FEATURES: "2. COLMAP Features",
            StageType.COLMAP_MATCHING: "3. Feature Matching",
            StageType.COLMAP_MAPPER: "4. Sparse Reconstruction",
            StageType.GAUSSIAN_SPLATTING: "5. Training Gaussian Splatting",
            StageType.EXPORT: "6. Exporting Model",
        }
        return names.get(self, self.value)

    @property
    def stage_number(self) -> int:
        numbers = {
            StageType.FRAME_EXTRACTION: 1,
            StageType.COLMAP_FEATURES: 2,
            StageType.COLMAP_MATCHING: 3,
            StageType.COLMAP_MAPPER: 4,
            StageType.GAUSSIAN_SPLATTING: 5,
            StageType.EXPORT: 6,
        }
        return numbers.get(self, 1)


class StageStatus(Enum):
    """Status states for each pipeline stage."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StageResult:
    """Outcome and artifacts produced by a single pipeline stage."""
    stage: StageType
    status: StageStatus = StageStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    message: str = "Ready"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return max(0.0, end - self.start_time)


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution for thread-safe UI updates."""
    stage: StageType
    status: StageStatus
    progress: float  # 0.0 to 1.0
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    global_progress: float = 0.0
    eta_seconds: Optional[float] = None
    total_cameras: int = 0
    registered_cameras: int = 0
    sparse_points: int = 0
    quality_score: int = 0
