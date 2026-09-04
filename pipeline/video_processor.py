"""
GeoRecon AI - Video Processor Module
Handles video loading, metadata extraction, and adaptive frame extraction using OpenCV.
"""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import threading
from typing import Callable, Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("GeoRecon.VideoProcessor")


@dataclass
class VideoMetadata:
    """Metadata container for ingested video files."""
    filename: str
    filepath: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    total_frames: int
    size_mb: float
    bitrate_kbps: float = 0.0

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def duration_formatted(self) -> str:
        mins, secs = divmod(int(self.duration_seconds), 60)
        return f"{mins:02d}:{secs:02d}"


def calculate_sampling_step(
    fps: float,
    duration_seconds: float = 0.0,
    total_frames: int = 0,
) -> int:
    """
    Determines the optimal frame sampling interval based on video duration and FPS:
    - Short (< 6.0s or < 120 frames): Every frame (step = 1; or step = 2 if fps >= 55)
    - Medium (6.0s to 20.0s or 120 to 500 frames): Every 2nd frame (step = 2)
    - Long (20.0s to 45.0s or 500 to 1200 frames): Every 3rd frame (step = 3)
    - Very long (> 45.0s or > 1200 frames): Adaptive targeting ~90-120 high-fidelity keyframes.
    """
    if total_frames <= 0 and duration_seconds > 0 and fps > 0:
        total_frames = int(duration_seconds * fps)
    elif duration_seconds <= 0 and total_frames > 0 and fps > 0:
        duration_seconds = total_frames / fps

    # 1. Very Short Video (< 6 sec) -> Every frame to ensure enough multi-view parallax
    if duration_seconds < 6.0 or total_frames < 120:
        step = 2 if fps >= 55.0 else 1
        length_class = "Short (<6s)"

    # 2. Medium Video (6 to 20 sec) -> Every 2nd frame
    elif duration_seconds <= 20.0 or total_frames <= 500:
        step = 3 if fps >= 55.0 else 2
        length_class = "Medium (6-20s)"

    # 3. Long Video (20 to 45 sec) -> Every 3rd frame
    elif duration_seconds <= 45.0 or total_frames <= 1200:
        step = 4 if fps >= 55.0 else 3
        length_class = "Long (20-45s)"

    # 4. Very Long Video (> 45 sec) -> Adaptive targeting ~90-120 keyframes
    else:
        target_keyframes = 100
        step = max(3, round(total_frames / target_keyframes))
        length_class = f"Very long ({duration_seconds:.1f}s)"

    effective_fps = fps / max(1, step)
    est_frames = total_frames // max(1, step) if total_frames > 0 else 0
    logger.info(
        f"Adaptive Frame Sampling [{length_class}]: original={fps:.1f} FPS, {total_frames} frames -> "
        f"step={step} (effective ~{effective_fps:.1f} FPS, ~{est_frames} keyframes)"
    )
    return step


def select_quality_keyframes(
    video_path: Path,
    target_count: Optional[int] = None,
    min_keyframes: int = 24,
    max_keyframes: int = 300,
    stop_event: Optional[threading.Event] = None,
) -> List[int]:
    """
    Quality-Aware Keyframe Selection:
    Combines:
    1. Laplacian blur score (sharpness).
    2. Optical-flow motion score (DISOpticalFlow inter-frame displacement / parallax).
    3. Redundancy filtering.
    Targets approximately 80-110 high-quality keyframes for 10-second handheld videos
    while preserving sufficient overlap.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0.0 or math.isnan(fps):
        fps = 30.0
    duration_s = total_frames / fps if fps > 0 else 0.0

    if target_count is None:
        if duration_s >= 3.0:
            target_count = int(np.clip(round(duration_s * 9.5), min_keyframes, max_keyframes))
        else:
            target_count = max(min_keyframes, total_frames // 2)

    step_est = max(1.5, total_frames / max(1, target_count))
    logger.info(
        f"Quality-Aware Keyframe Selection: {total_frames} frames ({duration_s:.1f}s) -> "
        f"Targeting ~{target_count} keyframes (adaptive window ~{step_est:.2f} frames)..."
    )

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    selected_indices: List[int] = []
    current_window: List[Tuple[int, float, float]] = []
    prev_gray_small = None
    frame_idx = 0

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break

            # Fast evaluation on downscaled grayscale (320x180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_LINEAR)
            blur_score = float(cv2.Laplacian(small, cv2.CV_64F).var())

            motion_score = 0.0
            if prev_gray_small is not None:
                flow = dis.calc(prev_gray_small, small, None)
                motion_score = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            prev_gray_small = small

            # Composite quality metric: high sharpness + adequate parallax
            current_window.append((frame_idx, blur_score, motion_score))

            if len(current_window) >= round(step_est):
                best = max(current_window, key=lambda x: x[1] * (0.6 + min(2.0, x[2])))
                selected_indices.append(best[0])
                current_window = []

            frame_idx += 1

        if current_window:
            best = max(current_window, key=lambda x: x[1] * (0.6 + min(2.0, x[2])))
            selected_indices.append(best[0])

    finally:
        cap.release()

    if not selected_indices:
        step = max(1, round(step_est))
        selected_indices = list(range(0, total_frames, step))

    logger.info(
        f"Quality-Aware Selection complete: {len(selected_indices)} keyframes selected "
        f"from {total_frames} frames ({duration_s:.1f}s)."
    )
    return sorted(selected_indices)


class VideoProcessor:
    """Production-grade OpenCV video ingestion with adaptive sampling and metadata inspection."""

    def __init__(self, target_fps: float = 12.0, target_resolution: Optional[Tuple[int, int]] = None):
        self.target_fps = target_fps
        self.target_resolution = target_resolution

    def inspect_video(self, video_path: Path) -> VideoMetadata:
        """Inspects and returns deep video metadata (resolution, duration, fps, bitrate)."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or math.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0

            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            bitrate_kbps = (file_size_mb * 8 * 1024) / duration if duration > 0 else 0.0

            meta = VideoMetadata(
                filepath=video_path,
                filename=video_path.name,
                width=width,
                height=height,
                fps=round(fps, 2),
                total_frames=total_frames,
                duration_seconds=round(duration, 2),
                size_mb=round(file_size_mb, 2),
                bitrate_kbps=round(bitrate_kbps, 2),
            )
            logger.info(
                f"Inspected '{meta.filename}': {meta.resolution_str} @ {meta.fps} FPS, "
                f"{meta.duration_formatted} ({meta.total_frames} frames, {meta.size_mb} MB)"
            )
            return meta
        finally:
            cap.release()

    def calculate_sampling_step(
        self,
        fps: float,
        duration_seconds: float = 0.0,
        total_frames: int = 0,
    ) -> int:
        return calculate_sampling_step(fps, duration_seconds, total_frames)

    def select_quality_keyframes(
        self,
        video_path: Path,
        target_count: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[int]:
        return select_quality_keyframes(video_path, target_count=target_count, stop_event=stop_event)

    def extract_frames(
        self,
        video_path: Path,
        output_frames_dir: Path,
        on_progress: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
        use_quality_selection: bool = True,
    ) -> Tuple[List[Path], VideoMetadata]:
        """
        Extracts frames using Quality-Aware Keyframe Selection (or adaptive step fallback)
        and saves them as: frame_000001.png, frame_000002.png, ...

        Returns a tuple of (list of saved frame paths, VideoMetadata).
        """
        meta = self.inspect_video(video_path)
        output_frames_dir.mkdir(parents=True, exist_ok=True)

        if use_quality_selection:
            selected_indices = set(select_quality_keyframes(video_path, stop_event=stop_event))
        else:
            step = self.calculate_sampling_step(meta.fps, meta.duration_seconds, meta.total_frames)
            selected_indices = set(range(0, meta.total_frames, step))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for extraction: {video_path}")

        extracted_paths: List[Path] = []
        frame_idx = 0
        saved_count = 0
        total_frames = meta.total_frames
        total_selected = len(selected_indices)

        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.warning("Frame extraction cancelled by stop event.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Extract if selected by quality-aware analyzer
                if frame_idx in selected_indices:
                    saved_count += 1
                    frame_filename = f"frame_{saved_count:06d}.png"
                    frame_dest = output_frames_dir / frame_filename
                    
                    # Write frame using PNG compression
                    cv2.imwrite(str(frame_dest), frame)
                    extracted_paths.append(frame_dest)

                frame_idx += 1

                # Emit progress every 15 frames or on completion
                if frame_idx % 15 == 0 or frame_idx == total_frames:
                    progress = min(1.0, frame_idx / max(1, total_frames))
                    msg = f"Extracted {saved_count}/{total_selected} quality keyframes (scanned {frame_idx}/{total_frames})..."
                    if on_progress:
                        on_progress(
                            progress,
                            msg,
                            {
                                "Scanned": f"{frame_idx}/{total_frames}",
                                "Selected": f"{saved_count}/{total_selected} keyframes",
                                "Strategy": "Quality-Aware (Blur + Flow)",
                            },
                        )

            logger.info(f"Frame extraction completed: {saved_count} quality keyframes saved to {output_frames_dir}")
            return extracted_paths, meta

        finally:
            cap.release()
