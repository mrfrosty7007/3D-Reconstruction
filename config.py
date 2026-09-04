"""
TerraSweep - Configuration Module
SIH-26158: Drone & Mobile Video 3D Reconstruction Platform
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"
TEMP_DIR = BASE_DIR / "data" / "temp"

# Ensure runtime directories exist
for directory in [DATA_DIR, OUTPUTS_DIR, LOGS_DIR, ASSETS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessConfig:
    """Configuration for AI-assisted video preprocessing, blur filtering, and duplicate removal."""
    # Target frame extraction rate (used when fps is not exactly 30 or 60)
    target_extraction_fps: float = 12.0
    
    # Blur Detection (Variance of Laplacian)
    # Frames with variance below this threshold are discarded as blurry/motion-degraded
    blur_threshold: float = 80.0
    
    # Duplicate Removal (SSIM - Structural Similarity Index)
    # Consecutive frames with SSIM above this value are evaluated for redundancy (0.0 to 1.0)
    ssim_threshold: float = 0.90
    
    # ORB Feature Matcher Similarity Threshold
    # Proportion of matching keypoints considered redundant (0.0 to 1.0)
    orb_match_threshold: float = 0.85
    orb_max_features: int = 1500

    # Max frames cap (safety limit, 0 = unlimited)
    max_retained_frames: int = 800


@dataclass
class ColmapConfig:
    """Configuration for COLMAP Structure-from-Motion (SfM) pipeline."""
    # Path to colmap executable (can be configured or auto-detected on PATH)
    executable_path: str = os.getenv("COLMAP_PATH", "colmap")
    camera_model: str = "OPENCV"  # OPENCV, PINHOLE, RADIAL, SIMPLE_RADIAL
    matcher_type: str = "exhaustive"  # exhaustive, sequential, spatial
    single_camera: bool = True
    max_num_features: int = 8192
    quality_preset: str = "high"  # low, medium, high, extreme

    def is_available(self) -> bool:
        """Check if COLMAP executable exists on system PATH or specified path."""
        if shutil.which(self.executable_path):
            return True
        if Path(self.executable_path).is_file():
            return True
        return False


@dataclass
class GSplatConfig:
    """Configuration for 3D Gaussian Splatting training."""
    iterations: int = 30_000
    learning_rate: float = 0.00016
    learning_rate_position: float = 0.00016
    densify_interval: int = 100
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    prune_interval: int = 200
    sh_degree: int = 3
    export_format: str = "ply"  # ply, splat, glb, obj
    device: str = "cuda"  # cuda, cpu


@dataclass
class VideoConfig:
    """Configuration for video input and extraction."""
    supported_extensions: tuple = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
    default_frame_format: str = "png"


@dataclass
class AppConfig:
    """Main application configuration container."""
    app_name: str = "TerraSweep"
    app_subtitle: str = "Drone & Mobile 3D Reconstruction Platform [SIH-26158]"
    version: str = "1.1.0"

    # UI Theme Settings
    ui_appearance_mode: str = "Dark"  # "System", "Dark", "Light"
    ui_color_theme: str = "blue"  # "blue", "green", "dark-blue"
    window_width: int = 1240
    window_height: int = 840
    min_window_width: int = 1020
    min_window_height: int = 720

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR
    logs_dir: Path = LOGS_DIR
    temp_dir: Path = TEMP_DIR

    # Subsystem configs
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    gsplat: GSplatConfig = field(default_factory=GSplatConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
