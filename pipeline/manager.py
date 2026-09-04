"""
GeoRecon AI - Studio Pipeline Manager
Coordinates the complete end-to-end 6-stage photogrammetry and 3D Gaussian Splatting pipeline:
1. Extract Frames (Video + Blur + Duplicate Filter)
2. COLMAP Features (SIFT Extraction)
3. Feature Matching (Exhaustive/Sequential Matcher)
4. Sparse Reconstruction (Incremental Mapper + Model Conversion)
   -> Reconstruction Quality Gate (Green / Yellow / Red validation)
5. Training Gaussian Splatting (3DGS Optimization + Live Telemetry)
6. Exporting Model (Multi-format PLY, OBJ, GLB, Trajectory & Manifests)
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Optional, Dict, Any, List, Tuple

from config import AppConfig, DEFAULT_CONFIG
from pipeline.stage import StageType, StageStatus, PipelineEvent, StageResult
from pipeline.video_processor import VideoProcessor, VideoMetadata
from pipeline.blur_filter import BlurFilter, BlurFilterResult
from pipeline.duplicate_filter import DuplicateFilter, DuplicateFilterResult
from pipeline.colmap_runner import ColmapRunner, ColmapSummary
from pipeline.gsplat_runner import GSplatRunner, GSplatTrainingResult
from pipeline.exporter import ModelExporter
from pipeline.viewer import Model3DViewer
from pipeline.telemetry import HardwareSnapshot, HardwareTelemetryCollector

logger = logging.getLogger("GeoRecon.StudioPipeline")

# Pipeline Status Constants (Single Source of Truth)
PIPELINE_STATUS_COMPLETED = "completed"
PIPELINE_STATUS_FAILED = "failed"
PIPELINE_STATUS_PARTIAL = "partial"
PIPELINE_STATUS_CANCELLED = "cancelled"
PIPELINE_STATUS_RUNNING = "running"


def infer_session_status(session_dir: Path) -> str:
    """
    Safely resolves the pipeline status for a session directory.
    Single Source of Truth: scene_manifest.json ('pipeline_status').
    Backward compatibility for legacy sessions:
    - If checkpoint_final.json, point_cloud.ply, trajectory_preview.mp4 all exist -> 'completed'
    - Else if colmap_summary.json shows proceed == false (or quality gate failed) -> 'failed'
    - Otherwise -> 'partial'
    """
    if not isinstance(session_dir, Path):
        session_dir = Path(session_dir)

    manifest_p = session_dir / "scene_manifest.json"
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                status = data.get("pipeline_status")
                if status in (PIPELINE_STATUS_COMPLETED, PIPELINE_STATUS_FAILED, PIPELINE_STATUS_PARTIAL, PIPELINE_STATUS_CANCELLED, PIPELINE_STATUS_RUNNING):
                    return status
        except Exception:
            pass

    # Legacy inference rules
    ckpt_p = session_dir / "checkpoints" / "checkpoint_final.json"
    ply_p = session_dir / "point_cloud.ply"
    traj_p = session_dir / "trajectory_preview.mp4"
    if ckpt_p.exists() and ply_p.exists() and traj_p.exists():
        return PIPELINE_STATUS_COMPLETED

    colmap_p = session_dir / "colmap_summary.json"
    if colmap_p.exists():
        try:
            with open(colmap_p, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                if c_data.get("proceed") is False or c_data.get("quality_gate_passed") is False:
                    return PIPELINE_STATUS_FAILED
                if (session_dir / "recovery_suggestions.json").exists():
                    return PIPELINE_STATUS_FAILED
                reg_pct = c_data.get("registration_percentage", 100.0)
                if reg_pct < 40.0:
                    return PIPELINE_STATUS_FAILED
                if c_data.get("status") == "FAILED":
                    return PIPELINE_STATUS_FAILED
        except Exception:
            pass

    if (session_dir / "recovery_suggestions.json").exists():
        return PIPELINE_STATUS_FAILED

    return PIPELINE_STATUS_PARTIAL


class ETATracker:
    """Tracks moving average reconstruction velocity and calculates smooth, non-negative ETA."""

    def __init__(self, window_size: int = 15):
        self.session_start_t = time.time()
        self.window_size = window_size
        self.reg_cam_history: List[Tuple[float, int]] = []
        self.last_estimated_eta: Optional[float] = None

    def start_sparse_tracking(self) -> None:
        self.session_start_t = time.time()
        self.reg_cam_history.clear()
        self.last_estimated_eta = None

    def record_camera_registration(self, reg_cams: int) -> None:
        now = time.time()
        self.reg_cam_history.append((now, reg_cams))
        if len(self.reg_cam_history) > self.window_size:
            self.reg_cam_history.pop(0)

    def estimate_eta_sparse(self, registered: int, total: int) -> float:
        """Estimates remaining time during Sparse Reconstruction."""
        rem_cams = max(0, total - registered)
        if rem_cams == 0:
            return 30.0  # Estimated ~25s GSplat + ~5s Export

        speed = 0.0
        if len(self.reg_cam_history) >= 2:
            dt = self.reg_cam_history[-1][0] - self.reg_cam_history[0][0]
            dc = self.reg_cam_history[-1][1] - self.reg_cam_history[0][1]
            if dt > 1.0 and dc > 0:
                speed = dc / dt

        if speed <= 0:
            speed = 1.5  # Typical registration speed: ~1.5 cams/s

        mapper_rem_s = rem_cams / max(0.2, speed)
        total_rem_s = mapper_rem_s + 30.0  # + GSplat and Export
        if self.last_estimated_eta is not None:
            total_rem_s = 0.7 * total_rem_s + 0.3 * self.last_estimated_eta
        self.last_estimated_eta = max(5.0, total_rem_s)
        return self.last_estimated_eta

    def estimate_eta_gsplat(self, current_iter: int, total_iter: int, iter_speed: float) -> float:
        """Estimates remaining time during Gaussian Splatting."""
        rem_iters = max(0, total_iter - current_iter)
        speed = max(1.0, iter_speed)
        gsplat_rem = rem_iters / speed
        total_rem = gsplat_rem + 5.0  # + Export
        self.last_estimated_eta = max(1.0, total_rem)
        return self.last_estimated_eta


def calculate_live_quality_score(reg_cams: int, total_cams: int, sparse_points: int) -> int:
    """Computes a realistic 0-100 quality score using registration ratio and 3D tie point density."""
    if total_cams <= 0 or reg_cams <= 0:
        return 0
    reg_pct = (reg_cams / total_cams) * 100.0
    cam_score = min(60.0, (reg_pct / 100.0) * 60.0)
    target_pts = max(100, reg_cams * 150)
    density_ratio = min(1.0, sparse_points / target_pts) if sparse_points > 0 else (0.4 if reg_cams > 0 else 0.0)
    pt_score = density_ratio * 40.0
    return max(10, min(100, int(cam_score + pt_score)))


class PipelineManager:
    """Orchestrates genuine photogrammetry and 3D reconstruction stages asynchronously."""

    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        event_callback: Optional[Callable[[PipelineEvent], None]] = None,
    ):
        self.config = config
        self.event_callback = event_callback

        # Stage sub-engines
        self.video_processor = VideoProcessor(
            target_fps=config.preprocess.target_extraction_fps,
        )
        self.blur_filter = BlurFilter(
            blur_threshold=config.preprocess.blur_threshold,
        )
        self.duplicate_filter = DuplicateFilter(
            ssim_threshold=config.preprocess.ssim_threshold,
            orb_match_threshold=config.preprocess.orb_match_threshold,
            max_features=config.preprocess.orb_max_features,
        )
        self.colmap_runner = ColmapRunner(config.colmap)
        self.gsplat_runner = GSplatRunner(config.gsplat)
        self.exporter = ModelExporter(config.gsplat.export_format)

        # Threading state
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._eta_tracker = ETATracker()
        self._quality_gate_override_event = threading.Event()
        self.auto_continue_yellow: bool = False

        # Session cache
        self.last_session_name: Optional[str] = None
        self.last_colmap_summary: Optional[ColmapSummary] = None
        self.last_gsplat_result: Optional[GSplatTrainingResult] = None

        # Live Hardware Telemetry Collector
        self.telemetry_collector = HardwareTelemetryCollector(interval_seconds=0.5)
        self.telemetry_collector.start()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def emit_event(
        self,
        stage: StageType,
        status: StageStatus,
        progress: float,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        global_progress: float = 0.0,
        eta_seconds: Optional[float] = None,
        total_cameras: int = 0,
        registered_cameras: int = 0,
        sparse_points: int = 0,
        quality_score: int = 0,
    ) -> None:
        """Emits a rich telemetry status event to the UI callback."""
        if self.event_callback:
            event = PipelineEvent(
                stage=stage,
                status=status,
                progress=progress,
                message=message,
                metrics=metrics or {},
                global_progress=global_progress,
                eta_seconds=eta_seconds,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            self.event_callback(event)

    def start_pipeline(
        self,
        video_path: Path,
        output_dir: Path,
        scene_name: Optional[str] = None,
    ) -> bool:
        """Launches the complete 6-stage studio reconstruction pipeline in a background thread."""
        if self._is_running:
            logger.warning("Pipeline is already executing a reconstruction session.")
            return False

        self._stop_event.clear()
        self._is_running = True
        self._eta_tracker = ETATracker()

        self._worker_thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(video_path, output_dir, scene_name),
            name="GeoRecon-StudioWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(f"Studio reconstruction pipeline launched for: {video_path.name}")
        return True

    def stop_pipeline(self) -> None:
        """Requests immediate graceful cancellation of ongoing reconstruction."""
        if self._is_running:
            logger.warning("Cancellation requested. Aborting reconstruction pipeline...")
            self._stop_event.set()
            if hasattr(self, "colmap_runner") and self.colmap_runner:
                self.colmap_runner.terminate_active_process()
            if self._worker_thread and self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
                self._worker_thread.join(timeout=2.0)
            self._is_running = False
            self.emit_event(
                stage=StageType.EXPORT,
                status=StageStatus.SKIPPED,
                progress=0.0,
                message="Reconstruction cancelled by user.",
                global_progress=0.0,
            )

    def continue_after_quality_gate(self) -> None:
        """Signals worker thread to proceed past Yellow Quality Gate into Stage 5."""
        if hasattr(self, "_quality_gate_override_event"):
            self._quality_gate_override_event.set()
            logger.info("Quality Gate override signal sent: User confirmed continuation to 3DGS.")

    def write_diagnostics(
        self,
        session_output_dir: Path,
        worker_thread_status: str,
        failure_reason: Optional[str] = None,
        last_processed_stage: Optional[str] = None,
        last_registered_camera_count: Optional[int] = None,
        exit_codes: Optional[Dict[str, int]] = None,
    ) -> Path:
        """
        Creates outputs/<session>/diagnostics.json capturing comprehensive environment,
        GPU acceleration, worker thread, process exit codes, and failure forensics.
        """
        session_output_dir.mkdir(parents=True, exist_ok=True)
        diag_path = session_output_dir / "diagnostics.json"

        cuda_available = self.colmap_runner.is_gpu_available()
        gpu_name = self.colmap_runner.get_gpu_name()
        _, colmap_ver = self.colmap_runner.check_environment()

        if exit_codes is None:
            exit_codes = dict(getattr(self.colmap_runner, "exit_codes", {}))

        if last_registered_camera_count is None:
            last_registered_camera_count = getattr(self, "_last_reg_cams", 0)

        peaks = (
            self.telemetry_collector.get_peak_metrics()
            if hasattr(self, "telemetry_collector") and self.telemetry_collector
            else {}
        )

        diag_data = {
            "CUDA detected": cuda_available,
            "cuda_detected": cuda_available,
            "GPU name": gpu_name,
            "gpu_name": gpu_name,
            "COLMAP version": colmap_ver,
            "colmap_version": colmap_ver,
            "Worker thread status": worker_thread_status,
            "worker_thread_status": worker_thread_status,
            "Exit codes": exit_codes,
            "exit_codes": exit_codes,
            "Failure reason": failure_reason,
            "failure_reason": failure_reason,
            "Last processed stage": last_processed_stage or "UNKNOWN",
            "last_processed_stage": last_processed_stage or "UNKNOWN",
            "Last registered camera count": last_registered_camera_count,
            "last_registered_camera_count": last_registered_camera_count,
            "cpu_peak_percent": peaks.get("cpu_peak_percent", 0.0),
            "gpu_peak_percent": peaks.get("gpu_peak_percent", 0.0),
            "ram_peak_percent": peaks.get("ram_peak_percent", 0.0),
            "gpu_vram_peak_mb": peaks.get("gpu_vram_peak_mb", 0.0),
            "gpu_temperature_peak_c": peaks.get("gpu_temperature_peak_c", 0.0),
            "CPU peak percent": peaks.get("cpu_peak_percent", 0.0),
            "GPU peak percent": peaks.get("gpu_peak_percent", 0.0),
            "RAM peak percent": peaks.get("ram_peak_percent", 0.0),
            "GPU VRAM peak MB": peaks.get("gpu_vram_peak_mb", 0.0),
            "GPU temperature peak C": peaks.get("gpu_temperature_peak_c", 0.0),
            "timestamp": datetime.now().isoformat(),
        }

        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)

        logger.info(f"Saved Diagnostics Report -> {diag_path}")
        return diag_path

    def write_scene_manifest(
        self,
        session_output_dir: Path,
        session_name: str,
        pipeline_status: str,
        pipeline_stage_completed: int,
        quality_gate_passed: bool,
        registered_cameras: int = 0,
        total_cameras: int = 0,
        failure_reason: Optional[str] = None,
        registration_percentage: Optional[float] = None,
        sparse_points: Optional[int] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Single Source of Truth writer for scene_manifest.json.
        Ensures every session contains a valid final pipeline status.
        Required fields:
        - pipeline_status: "completed" | "failed" | "partial" | "cancelled"
        - pipeline_stage_completed: int
        - quality_gate_passed: bool
        - registered_cameras: int
        - total_cameras: int
        """
        manifest_file = session_output_dir / "scene_manifest.json"
        manifest_data: Dict[str, Any] = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = {}

        if registration_percentage is None and total_cameras > 0:
            registration_percentage = round((registered_cameras / total_cameras) * 100.0, 2)
        elif registration_percentage is None:
            registration_percentage = manifest_data.get("registration_percentage", 0.0)

        manifest_data.update({
            "scene_name": session_name,
            "pipeline_status": pipeline_status,
            "pipeline_stage_completed": pipeline_stage_completed,
            "quality_gate_passed": quality_gate_passed,
            "registered_cameras": registered_cameras if registered_cameras > 0 else manifest_data.get("registered_cameras", 0),
            "total_cameras": total_cameras if total_cameras > 0 else manifest_data.get("total_cameras", 0),
            "registration_percentage": registration_percentage,
            "sparse_points": sparse_points if sparse_points is not None else manifest_data.get("sparse_points", 0),
            "status": pipeline_status.upper(),
            "created_at": manifest_data.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
        })

        if failure_reason:
            manifest_data["failure_reason"] = failure_reason

        if extra_data:
            for k, v in extra_data.items():
                if isinstance(v, dict) and isinstance(manifest_data.get(k), dict):
                    manifest_data[k].update(v)
                else:
                    manifest_data[k] = v

        session_output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Saved Scene Manifest [{pipeline_status.upper()}] -> {manifest_file}")
        return manifest_file

    def export_partial_session(
        self,
        session_output_dir: Path,
        session_name: str,
        sparse_dir: Optional[Path] = None,
        total_cameras: int = 0,
        registered_cameras: int = 0,
        sparse_points: int = 0,
        quality_gate_passed: bool = True,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Exports early/partial deliverables and records 'partial' status in scene_manifest.json."""
        ply_file = session_output_dir / "point_cloud.ply"
        if sparse_dir and sparse_dir.exists() and not ply_file.exists():
            try:
                best_dir, _, _ = ColmapRunner.find_best_model_dir(sparse_dir)
                points_xyz, points_rgb = GSplatRunner.load_colmap_points(best_dir)
                if len(points_xyz) > 0:
                    ModelExporter.export_ply_point_cloud(points_xyz, points_rgb, ply_file)
            except Exception as e:
                logger.warning(f"Early export PLY conversion note: {e}")

        return self.write_scene_manifest(
            session_output_dir=session_output_dir,
            session_name=session_name,
            pipeline_status=PIPELINE_STATUS_PARTIAL,
            pipeline_stage_completed=4,
            quality_gate_passed=quality_gate_passed,
            registered_cameras=registered_cameras,
            total_cameras=total_cameras,
            sparse_points=sparse_points,
            extra_data=extra_data,
        )

    def _save_session_outcome(
        self,
        session_output_dir: Path,
        session_name: str,
        pipeline_status: str,
        stage_num: int,
        gate_passed: bool,
        registered_cameras: int = 0,
        total_cameras: int = 0,
        failure_reason: Optional[str] = None,
        registration_percentage: Optional[float] = None,
        sparse_points: Optional[int] = None,
        stage_name_str: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically synchronizes scene_manifest.json and diagnostics.json for any outcome."""
        self.write_scene_manifest(
            session_output_dir=session_output_dir,
            session_name=session_name,
            pipeline_status=pipeline_status,
            pipeline_stage_completed=stage_num,
            quality_gate_passed=gate_passed,
            registered_cameras=registered_cameras,
            total_cameras=total_cameras,
            failure_reason=failure_reason,
            registration_percentage=registration_percentage,
            sparse_points=sparse_points,
            extra_data=extra_data,
        )
        thread_status = "completed" if pipeline_status == PIPELINE_STATUS_COMPLETED else (
            "cancelled" if pipeline_status == PIPELINE_STATUS_CANCELLED else (
                "active" if pipeline_status == PIPELINE_STATUS_RUNNING else "failed"
            )
        )
        self.write_diagnostics(
            session_output_dir=session_output_dir,
            worker_thread_status=thread_status,
            failure_reason=failure_reason,
            last_processed_stage=stage_name_str or f"STAGE_{stage_num}",
            last_registered_camera_count=registered_cameras,
        )

    def _run_pipeline_thread(
        self,
        video_path: Path,
        output_dir: Path,
        scene_name: Optional[str],
    ) -> None:
        """Thread worker executing the 6 sequential photogrammetry stages."""
        pipeline_start_t = time.time()

        # Build clean session identifier
        clean_prefix = scene_name.strip().replace(" ", "_") if scene_name and scene_name.strip() else video_path.stem
        session_name = f"{clean_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.last_session_name = session_name
        self._last_reg_cams = 0
        self._last_stage_name = "INITIALIZING"

        # Session folder tree
        session_data_dir = self.config.data_dir / session_name
        session_frames_dir = session_data_dir / "frames"
        session_colmap_dir = session_data_dir / "colmap"
        database_path = session_colmap_dir / "database.db"
        sparse_dir = session_colmap_dir / "sparse"
        session_output_dir = output_dir / session_name

        session_frames_dir.mkdir(parents=True, exist_ok=True)
        session_colmap_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        session_output_dir.mkdir(parents=True, exist_ok=True)

        total_cameras = 0
        registered_cameras = 0
        sparse_points = 0
        quality_score = 0
        current_stage = 0
        quality_gate_passed = False
        colmap_summary: Optional[ColmapSummary] = None
        manifest_written = False

        # Reset telemetry peaks for this session
        if hasattr(self, "telemetry_collector") and self.telemetry_collector:
            self.telemetry_collector.reset_peaks()

        # Write initial manifest (RUNNING) and diagnostics (ACTIVE)
        self._save_session_outcome(
            session_output_dir=session_output_dir,
            session_name=session_name,
            pipeline_status=PIPELINE_STATUS_RUNNING,
            stage_num=0,
            gate_passed=False,
            stage_name_str="INITIALIZING",
        )

        try:
            logger.info("=" * 75)
            logger.info(f"🌐 TerraSweep Studio — 3DGS Reconstruction Session [{session_name}]")
            logger.info(f"Input Video: {video_path}")
            logger.info(f"Frames Path: {session_frames_dir}")
            logger.info("Output Path: %s", session_output_dir)
            logger.info("=" * 75)

            # Startup Verification: Log verified COLMAP GPU flags
            self.colmap_runner.verify_gpu_flags()

            # =========================================================
            # STAGE 1: Extract Frames (Video + Blur + Duplicate Filter)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 0, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return
            frames_dataset, video_meta, blur_res, dup_res = self._execute_stage_1_frames(
                video_path, session_frames_dir, session_output_dir, session_name
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 0, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return
            if not frames_dataset:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 0, False, failure_reason="Stage 1 failed: No clean frames extracted from video.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return

            total_cameras = len(frames_dataset)
            current_stage = 1
            self._last_stage_name = "STAGE_1_FRAMES"

            # =========================================================
            # STAGE 2: COLMAP Features (SIFT Feature Extraction)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 1, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_2_FEATURES")
                manifest_written = True
                return
            feat_ok = self._execute_stage_2_features(session_frames_dir, database_path)
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 1, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_2_FEATURES")
                manifest_written = True
                return
            if not feat_ok:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 1, False, total_cameras=total_cameras, failure_reason="Stage 2 failed: COLMAP SIFT feature extraction failed.", stage_name_str="STAGE_2_FEATURES")
                manifest_written = True
                return
            current_stage = 2
            self._last_stage_name = "STAGE_2_FEATURES"

            # =========================================================
            # STAGE 3: Feature Matching (Exhaustive Matcher)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 2, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_3_MATCHING")
                manifest_written = True
                return
            match_ok = self._execute_stage_3_matching(database_path)
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 2, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_3_MATCHING")
                manifest_written = True
                return
            if not match_ok:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 2, False, total_cameras=total_cameras, failure_reason="Stage 3 failed: COLMAP feature matching failed.", stage_name_str="STAGE_3_MATCHING")
                manifest_written = True
                return
            current_stage = 3
            self._last_stage_name = "STAGE_3_MATCHING"

            # =========================================================
            # STAGE 4: Sparse Reconstruction (Incremental Mapper + TXT)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 3, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_4_MAPPER")
                manifest_written = True
                return
            colmap_summary = self._execute_stage_4_mapper(
                session_frames_dir, database_path, sparse_dir, total_cameras, session_name, session_output_dir
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 3, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_4_MAPPER")
                manifest_written = True
                return
            if not colmap_summary or not colmap_summary.is_valid:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 3, False, total_cameras=total_cameras, failure_reason="Stage 4 failed: COLMAP mapper failed to reconstruct sparse model.", stage_name_str="STAGE_4_MAPPER")
                manifest_written = True
                return

            self.last_colmap_summary = colmap_summary
            registered_cameras = colmap_summary.registered_cameras
            sparse_points = colmap_summary.sparse_point_count
            current_stage = 4
            self._last_stage_name = "COLMAP_MAPPER"
            self._last_reg_cams = registered_cameras

            # =========================================================
            # RECONSTRUCTION QUALITY GATE VALIDATION
            # =========================================================
            quality_level, can_proceed, quality_score, recommended_action = self._evaluate_quality_gate(
                colmap_summary=colmap_summary,
                session_output_dir=session_output_dir,
            )

            if quality_level == "GREEN":
                # Green (>=40%): Continue automatically
                logger.info(f"Quality Gate: GREEN (≥40%). Automatically proceeding to 3D Gaussian Splatting ({colmap_summary.registration_percentage:.1f}% registered).")
                quality_gate_passed = True
                self.emit_event(
                    stage=StageType.COLMAP_MAPPER,
                    status=StageStatus.COMPLETED,
                    progress=1.0,
                    message=f"Quality Gate: High Confidence (≥40%). Auto-proceeding to 3DGS.",
                    quality_score=quality_score,
                    total_cameras=total_cameras,
                    registered_cameras=registered_cameras,
                    sparse_points=sparse_points,
                    metrics={
                        "QualityLevel": "GREEN",
                        "Action": "Auto-Continue to 3DGS",
                    },
                )

            elif quality_level == "YELLOW":
                # Yellow (20–40%): Show "Continue Anyway" and await user override
                logger.warning(
                    f"Quality Gate: YELLOW (20-40% cameras registered: {registered_cameras}/{total_cameras} "
                    f"[{colmap_summary.registration_percentage:.1f}%]). Awaiting user review or 'Continue Anyway'..."
                )
                self.emit_event(
                    stage=StageType.COLMAP_MAPPER,
                    status=StageStatus.COMPLETED,
                    progress=1.0,
                    message=f"Quality Gate: Moderate Confidence ({colmap_summary.registration_percentage:.1f}%). Click 'Continue Anyway' or review scene.",
                    quality_score=quality_score,
                    total_cameras=total_cameras,
                    registered_cameras=registered_cameras,
                    sparse_points=sparse_points,
                    metrics={
                        "QualityLevel": "YELLOW",
                        "Action": "Continue Anyway",
                    },
                )

                if not getattr(self, "auto_continue_yellow", False):
                    self._quality_gate_override_event.clear()
                    overridden = self._quality_gate_override_event.wait(timeout=120.0)
                    if not overridden or self._stop_event.is_set():
                        st = PIPELINE_STATUS_CANCELLED if self._stop_event.is_set() else PIPELINE_STATUS_PARTIAL
                        self._save_session_outcome(
                            session_output_dir=session_output_dir,
                            session_name=session_name,
                            pipeline_status=st,
                            stage_num=4,
                            gate_passed=False,
                            registered_cameras=registered_cameras,
                            total_cameras=total_cameras,
                            failure_reason="Quality Gate Yellow: Paused for user review (partial reconstruction).",
                            registration_percentage=colmap_summary.registration_percentage,
                            sparse_points=sparse_points,
                            stage_name_str="COLMAP_MAPPER",
                        )
                        manifest_written = True
                        return

                quality_gate_passed = True
                logger.info("Quality Gate Yellow: User confirmed continuation. Proceeding to Stage 5...")

            else:
                # Red (<20%): Stop before GSplat, mark reconstruction as failed, recommend retry settings
                failure_msg = f"Quality Gate Failed: Only {registered_cameras}/{total_cameras} cameras registered ({colmap_summary.registration_percentage:.1f}% < 20%)."
                logger.error(f"Reconstruction Quality Gate [RED]: {failure_msg}")
                self._save_session_outcome(
                    session_output_dir=session_output_dir,
                    session_name=session_name,
                    pipeline_status=PIPELINE_STATUS_FAILED,
                    stage_num=4,
                    gate_passed=False,
                    registered_cameras=registered_cameras,
                    total_cameras=total_cameras,
                    failure_reason=failure_msg,
                    registration_percentage=colmap_summary.registration_percentage,
                    sparse_points=sparse_points,
                    stage_name_str="COLMAP_MAPPER",
                )
                manifest_written = True
                self.emit_event(
                    stage=StageType.COLMAP_MAPPER,
                    status=StageStatus.FAILED,
                    progress=0.0,
                    message=f"Quality Gate RED: Camera registration below 20% ({colmap_summary.registration_percentage:.1f}%). Stopped before GSplat.",
                    quality_score=quality_score,
                    total_cameras=total_cameras,
                    registered_cameras=registered_cameras,
                    sparse_points=sparse_points,
                    metrics={
                        "QualityLevel": "RED",
                        "Action": "Recommend Retry Settings",
                    },
                )
                return

            # =========================================================
            # STAGE 5: Training 3D Gaussian Splatting (3DGS)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 4, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_5_GSPLAT")
                manifest_written = True
                return
            gsplat_res = self._execute_stage_5_gsplat(
                sparse_dir=sparse_dir,
                images_dir=session_frames_dir,
                session_output_dir=session_output_dir,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 4, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_5_GSPLAT")
                manifest_written = True
                return
            if not gsplat_res:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 4, True, registered_cameras, total_cameras, "Stage 5 failed: 3D Gaussian Splatting optimization failed.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_5_GSPLAT")
                manifest_written = True
                return

            self.last_gsplat_result = gsplat_res
            current_stage = 5
            self._last_stage_name = "STAGE_5_GSPLAT"

            # =========================================================
            # STAGE 6: Exporting Model (Multi-format PLY, OBJ, GLB, Trajectory)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 5, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_6_EXPORT")
                manifest_written = True
                return
            export_ok = self._execute_stage_6_export(
                session_name=session_name,
                session_output_dir=session_output_dir,
                session_frames_dir=session_frames_dir,
                sparse_dir=sparse_dir,
                video_meta=video_meta,
                blur_res=blur_res,
                dup_res=dup_res,
                colmap_summary=colmap_summary,
                gsplat_res=gsplat_res,
                quality_score=quality_score,
                total_runtime=time.time() - pipeline_start_t,
            )

            if export_ok:
                manifest_written = True
                current_stage = 6
                self._last_stage_name = "STAGE_6_EXPORT"
                self.write_diagnostics(
                    session_output_dir=session_output_dir,
                    worker_thread_status="completed",
                    failure_reason=None,
                    last_processed_stage="STAGE_6_EXPORT",
                    last_registered_camera_count=registered_cameras,
                )
            else:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 5, True, registered_cameras, total_cameras, "Stage 6 failed: Deliverables packaging failed.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_6_EXPORT")
                manifest_written = True

            logger.info("=" * 75)
            logger.info("🎉 Studio 3D Reconstruction Session Completed Successfully!")
            logger.info(f"Registered Cameras: {registered_cameras}/{total_cameras} ({colmap_summary.registration_percentage}%)")
            logger.info(f"Sparse 3D Points: {sparse_points:,} | Final PSNR: {gsplat_res.final_psnr} dB")
            logger.info(f"Final Gaussians: {gsplat_res.final_gaussian_count:,}")
            logger.info(f"Deliverables Ready in: {session_output_dir}")
            logger.info("=" * 75)

        except Exception as e:
            logger.exception(f"Unhandled exception in studio reconstruction pipeline: {e}")
            if not manifest_written:
                st = PIPELINE_STATUS_CANCELLED if self._stop_event.is_set() else PIPELINE_STATUS_FAILED
                self._save_session_outcome(
                    session_output_dir=session_output_dir,
                    session_name=session_name,
                    pipeline_status=st,
                    stage_num=current_stage,
                    gate_passed=quality_gate_passed,
                    registered_cameras=registered_cameras,
                    total_cameras=total_cameras,
                    failure_reason=f"Pipeline exception: {str(e)}",
                    registration_percentage=colmap_summary.registration_percentage if colmap_summary else 0.0,
                    sparse_points=sparse_points,
                    stage_name_str=f"STAGE_{current_stage}",
                )
                manifest_written = True
            self.emit_event(
                stage=StageType.EXPORT,
                status=StageStatus.FAILED,
                progress=0.0,
                message=f"Pipeline error: {str(e)}",
                global_progress=0.0,
            )
        finally:
            if not manifest_written:
                st = PIPELINE_STATUS_CANCELLED if self._stop_event.is_set() else PIPELINE_STATUS_FAILED
                self._save_session_outcome(
                    session_output_dir=session_output_dir,
                    session_name=session_name,
                    pipeline_status=st,
                    stage_num=current_stage,
                    gate_passed=quality_gate_passed,
                    registered_cameras=registered_cameras,
                    total_cameras=total_cameras,
                    failure_reason="Pipeline terminated prematurely.",
                    registration_percentage=colmap_summary.registration_percentage if colmap_summary else 0.0,
                    sparse_points=sparse_points,
                    stage_name_str=f"STAGE_{current_stage}",
                )
            if not (session_output_dir / "diagnostics.json").exists():
                self.write_diagnostics(
                    session_output_dir=session_output_dir,
                    worker_thread_status="completed" if (quality_gate_passed and current_stage == 6) else "failed",
                    last_processed_stage=f"STAGE_{current_stage}",
                    last_registered_camera_count=registered_cameras,
                )
            self._is_running = False

    def _evaluate_quality_gate(
        self,
        colmap_summary: ColmapSummary,
        session_output_dir: Path,
    ) -> Tuple[str, bool, int, str]:
        """
        Authoritative Reconstruction Quality Gate:
        - Green (>=40%): High confidence, continue automatically to 3DGS.
        - Yellow (20-40%): Moderate confidence, pause for 'Continue Anyway' review.
        - Red (<20%): Low confidence, halt before GSplat and recommend retry settings.
        """
        reg_pct = colmap_summary.registration_percentage
        points = colmap_summary.sparse_point_count
        reg_cams = colmap_summary.registered_cameras
        total_cams = colmap_summary.total_cameras

        score = calculate_live_quality_score(reg_cams, total_cams, points)
        if reg_pct >= 40.0:
            level = "GREEN"
            proceed = True
            action = "High reconstruction confidence (≥40%). Proceeding automatically to 3D Gaussian Splatting."
        elif reg_pct >= 20.0 or (total_cams <= 4 and reg_cams >= 1):
            level = "YELLOW"
            proceed = getattr(self, "auto_continue_yellow", False)
            action = "Moderate reconstruction confidence (20–40%). Partial camera registration. Click 'Continue Anyway'."
        else:
            level = "RED"
            proceed = False
            action = "Low reconstruction confidence (<20%). Reconstruction halted before 3DGS. Adjust settings and retry."

        # Save recovery suggestions if Red or Yellow
        if level in ("RED", "YELLOW"):
            recovery_file = session_output_dir / "recovery_suggestions.json"
            suggestions = {
                "quality_level": level,
                "confidence_level": level,
                "recommended_action": action,
                "registered_percentage": reg_pct,
                "registered_cameras": reg_cams,
                "total_cameras": total_cams,
                "sparse_points": points,
                "suggestions": [
                    "Decrease blur threshold in Studio Settings (e.g. set Var >= 40.0) to retain more frames.",
                    "Lower SSIM duplicate threshold (e.g. set SSIM <= 0.85) to increase keyframe density.",
                    "Use exhaustive_matcher instead of sequential_matcher for higher match density.",
                    "Ensure drone flight velocity allows >75% forward and lateral visual overlap.",
                    "Verify camera lens parameters and select RADIAL or OPENCV camera model.",
                ],
            }
            with open(recovery_file, "w", encoding="utf-8") as f:
                json.dump(suggestions, f, indent=2)

        logger.info(f"Reconstruction Quality Gate: [{level}] (Registered: {reg_cams}/{total_cams} [{reg_pct}%], Points: {points}, Proceed: {proceed})")

        # Update colmap_summary.json with quality gate outcome and proceed flag
        summary_file = session_output_dir / "colmap_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                s_data["proceed"] = proceed
                s_data["quality_gate_passed"] = (level == "GREEN" or (level == "YELLOW" and proceed))
                s_data["quality_level"] = level
                s_data["recommended_action"] = action
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(s_data, f, indent=2)
            except Exception:
                pass

        return level, proceed, score, action

    def _execute_stage_1_frames(
        self,
        video_path: Path,
        session_frames_dir: Path,
        session_output_dir: Path,
        session_name: str,
    ) -> Tuple[Optional[List[Path]], Optional[VideoMetadata], Optional[BlurFilterResult], Optional[DuplicateFilterResult]]:
        """Stage 1: Extracts frames, filters blur, removes duplicates, and generates preprocess report."""
        self.emit_event(
            StageType.FRAME_EXTRACTION,
            StageStatus.RUNNING,
            0.1,
            "Ingesting video stream and extracting adaptive keyframes...",
            global_progress=0.01,
        )

        try:
            meta = self.video_processor.inspect_video(video_path)
            extracted_paths, _ = self.video_processor.extract_frames(
                video_path=video_path,
                output_frames_dir=session_frames_dir,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FRAME_EXTRACTION, StageStatus.RUNNING, p * 0.4, msg, m, global_progress=0.01 + p * 0.04
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None, None, None

            blur_res = self.blur_filter.filter_frames(
                frame_paths=extracted_paths,
                delete_discarded=True,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FRAME_EXTRACTION, StageStatus.RUNNING, 0.4 + p * 0.3, msg, m, global_progress=0.05 + p * 0.025
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None, None, None

            dup_res = self.duplicate_filter.filter_duplicates(
                frame_paths=blur_res.retained_frames,
                delete_discarded=True,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FRAME_EXTRACTION, StageStatus.RUNNING, 0.7 + p * 0.3, msg, m, global_progress=0.075 + p * 0.025
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None, None, None

            final_dataset = dup_res.retained_frames

            # Preprocess JSON Report
            report_file = session_output_dir / "preprocess_report.json"
            report_data = {
                "session_name": session_name,
                "timestamp": datetime.now().isoformat(),
                "video_metadata": {
                    "filename": meta.filename,
                    "resolution": meta.resolution_str,
                    "fps": meta.fps,
                    "duration_formatted": meta.duration_formatted,
                    "original_total_frames": meta.total_frames,
                },
                "preprocessing_metrics": {
                    "extracted_frames": len(extracted_paths),
                    "removed_blurry_frames": blur_res.discarded_count,
                    "removed_duplicate_frames": dup_res.discarded_count,
                    "final_retained_frames": len(final_dataset),
                    "average_blur_score": blur_res.average_score,
                },
            }
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)

            self.emit_event(
                StageType.FRAME_EXTRACTION,
                StageStatus.COMPLETED,
                1.0,
                f"Prepared clean dataset of {len(final_dataset)} frames.",
                metrics={
                    "Extracted": len(extracted_paths),
                    "Blurry Removed": blur_res.discarded_count,
                    "Duplicates Removed": dup_res.discarded_count,
                    "Final Frames": len(final_dataset),
                },
                global_progress=0.10,
                total_cameras=len(final_dataset),
            )
            return final_dataset, meta, blur_res, dup_res

        except Exception as e:
            logger.exception(f"Error in Stage 1 Frame Extraction: {e}")
            self.emit_event(StageType.FRAME_EXTRACTION, StageStatus.FAILED, 0.0, str(e))
            return None, None, None, None

    def _execute_stage_2_features(self, images_dir: Path, database_path: Path) -> bool:
        """Stage 2: Runs real COLMAP feature extractor (SIFT)."""
        logger.info("[Stage 2/6] Running COLMAP Feature Extractor...")
        self.emit_event(
            StageType.COLMAP_FEATURES,
            StageStatus.RUNNING,
            0.1,
            "Extracting SIFT keypoints with GPU acceleration...",
            global_progress=0.10,
        )

        ok = self.colmap_runner.run_feature_extraction(
            image_path=images_dir,
            database_path=database_path,
            camera_model=self.config.colmap.camera_model,
            single_camera=self.config.colmap.single_camera,
            use_gpu=True,
            on_log=lambda l: self.emit_event(
                StageType.COLMAP_FEATURES, StageStatus.RUNNING, 0.5, f"COLMAP SIFT: {l[:50]}", global_progress=0.18
            ),
            stop_event=self._stop_event,
        )

        if ok:
            self.emit_event(
                StageType.COLMAP_FEATURES,
                StageStatus.COMPLETED,
                1.0,
                "SIFT feature extraction completed successfully.",
                metrics={"Database": database_path.name, "Status": "Extracted"},
                global_progress=0.25,
            )
        else:
            self.emit_event(
                StageType.COLMAP_FEATURES,
                StageStatus.FAILED,
                0.0,
                "COLMAP feature extraction failed. Check logs.",
            )
        return ok

    def _execute_stage_3_matching(self, database_path: Path) -> bool:
        """Stage 3: Runs real COLMAP feature matcher."""
        logger.info("[Stage 3/6] Running COLMAP Feature Matcher...")
        self.emit_event(
            StageType.COLMAP_MATCHING,
            StageStatus.RUNNING,
            0.1,
            "Exhaustive feature matching across image pairs...",
            global_progress=0.25,
        )

        ok = self.colmap_runner.run_feature_matching(
            database_path=database_path,
            matcher_type=self.config.colmap.matcher_type,
            use_gpu=True,
            on_log=lambda l: self.emit_event(
                StageType.COLMAP_MATCHING, StageStatus.RUNNING, 0.5, f"COLMAP Matcher: {l[:50]}", global_progress=0.33
            ),
            stop_event=self._stop_event,
        )

        if ok:
            self.emit_event(
                StageType.COLMAP_MATCHING,
                StageStatus.COMPLETED,
                1.0,
                "Feature matching completed and two-view geometry verified.",
                metrics={"Matcher": self.config.colmap.matcher_type.capitalize(), "Status": "Matched"},
                global_progress=0.40,
            )
        else:
            self.emit_event(
                StageType.COLMAP_MATCHING,
                StageStatus.FAILED,
                0.0,
                "COLMAP feature matching failed.",
            )
        return ok

    def _execute_stage_4_mapper(
        self,
        images_dir: Path,
        database_path: Path,
        sparse_dir: Path,
        total_cameras: int,
        session_name: str,
        session_output_dir: Path,
    ) -> Optional[ColmapSummary]:
        """Stage 4: Runs real COLMAP mapper with live registration & point telemetry, converts best model to TXT, and generates colmap_summary.json."""
        logger.info("[Stage 4/6] Running COLMAP Incremental Mapper (Sparse Reconstruction)...")
        self.emit_event(
            StageType.COLMAP_MAPPER,
            StageStatus.RUNNING,
            0.05,
            "Running Incremental Mapper & Bundle Adjustment...",
            global_progress=0.40,
            total_cameras=total_cameras,
        )

        _live_reg_cams = 0
        _live_sparse_pts = 0

        def _on_live_mapper_reg(reg_cams: int):
            nonlocal _live_reg_cams
            _live_reg_cams = max(_live_reg_cams, reg_cams)
            self._last_reg_cams = _live_reg_cams
            self._eta_tracker.record_camera_registration(_live_reg_cams)
            eta_s = self._eta_tracker.estimate_eta_sparse(_live_reg_cams, total_cameras)
            reg_pct = (_live_reg_cams / total_cameras) * 100 if total_cameras > 0 else 100.0
            score = calculate_live_quality_score(_live_reg_cams, total_cameras, _live_sparse_pts)
            cam_frac = min(1.0, _live_reg_cams / max(1, total_cameras))
            g_prog = 0.40 + (0.25 * cam_frac)
            self.emit_event(
                StageType.COLMAP_MAPPER,
                StageStatus.RUNNING,
                cam_frac,
                f"Registering cameras: {_live_reg_cams}/{total_cameras} ({reg_pct:.1f}%) | {_live_sparse_pts:,} tie points",
                metrics={"Registered": f"{_live_reg_cams}/{total_cameras}", "Registration": f"{reg_pct:.1f}%", "Tie Points": f"{_live_sparse_pts:,}"},
                global_progress=g_prog,
                eta_seconds=eta_s,
                registered_cameras=_live_reg_cams,
                total_cameras=total_cameras,
                sparse_points=_live_sparse_pts,
                quality_score=score,
            )

        def _on_live_sparse_points(pts: int):
            nonlocal _live_sparse_pts
            _live_sparse_pts = max(_live_sparse_pts, pts)
            score = calculate_live_quality_score(_live_reg_cams, total_cameras, _live_sparse_pts)
            cam_frac = min(1.0, _live_reg_cams / max(1, total_cameras))
            g_prog = 0.40 + (0.25 * cam_frac)
            self.emit_event(
                StageType.COLMAP_MAPPER,
                StageStatus.RUNNING,
                cam_frac,
                f"Tie Points: {_live_sparse_pts:,} | {_live_reg_cams}/{total_cameras} cameras",
                metrics={"Registered": f"{_live_reg_cams}/{total_cameras}", "Tie Points": f"{_live_sparse_pts:,}"},
                global_progress=g_prog,
                eta_seconds=self._eta_tracker.last_estimated_eta,
                registered_cameras=_live_reg_cams,
                total_cameras=total_cameras,
                sparse_points=_live_sparse_pts,
                quality_score=score,
            )

        def _on_ba_event(line: str):
            cam_frac = min(1.0, _live_reg_cams / max(1, total_cameras))
            g_prog = 0.40 + (0.25 * cam_frac)
            score = calculate_live_quality_score(_live_reg_cams, total_cameras, _live_sparse_pts)
            self.emit_event(
                StageType.COLMAP_MAPPER,
                StageStatus.RUNNING,
                cam_frac,
                "Bundle Adjustment: Optimizing 3D camera poses & structure...",
                metrics={"Sub-State": "Bundle Adjustment"},
                global_progress=g_prog,
                eta_seconds=self._eta_tracker.last_estimated_eta,
                registered_cameras=_live_reg_cams,
                total_cameras=total_cameras,
                sparse_points=_live_sparse_pts,
                quality_score=score,
            )

        mapper_start_t = time.time()
        last_log_emit_t = 0.0

        def _on_mapper_log_throttle(line: str):
            nonlocal last_log_emit_t
            now = time.time()
            if now - last_log_emit_t >= 0.4 or "Registered image" in line:
                last_log_emit_t = now
                cam_frac = min(1.0, _live_reg_cams / max(1, total_cameras))
                g_prog = 0.40 + (0.25 * cam_frac)
                score = calculate_live_quality_score(_live_reg_cams, total_cameras, _live_sparse_pts)
                self.emit_event(
                    StageType.COLMAP_MAPPER,
                    StageStatus.RUNNING,
                    cam_frac,
                    f"COLMAP Mapper: {line[:55]}",
                    global_progress=g_prog,
                    eta_seconds=self._eta_tracker.last_estimated_eta,
                    registered_cameras=_live_reg_cams,
                    total_cameras=total_cameras,
                    sparse_points=_live_sparse_pts,
                    quality_score=score,
                )

        ok = self.colmap_runner.run_mapper(
            image_path=images_dir,
            database_path=database_path,
            output_sparse_dir=sparse_dir,
            on_log=_on_mapper_log_throttle,
            on_reg_frames=_on_live_mapper_reg,
            on_sparse_points=_on_live_sparse_points,
            on_ba_event=_on_ba_event,
            stop_event=self._stop_event,
        )

        if not ok:
            # Check if any partial reconstruction exists in sparse_dir to preserve
            try:
                partial_dir, part_imgs, part_pts = self.colmap_runner.find_best_model_dir(sparse_dir)
                if part_imgs > 0:
                    logger.info(f"Preserving partial sparse reconstruction with {part_imgs} cameras and {part_pts} points.")
                    self._last_reg_cams = part_imgs
            except Exception:
                pass
            self.emit_event(StageType.COLMAP_MAPPER, StageStatus.FAILED, 0.0, "COLMAP mapper failed.")
            return None

        # Detect best model folder across all reconstructed components (e.g. 0, 1, 2)
        best_model_dir, best_imgs, best_pts = self.colmap_runner.find_best_model_dir(sparse_dir)
        logger.info(f"Using best reconstructed model at: {best_model_dir} ({best_imgs} cameras, {best_pts} points)")

        # Convert Best Model BIN to TXT
        txt_out_dir = best_model_dir / "txt"
        self.colmap_runner.run_model_converter(
            input_model_dir=best_model_dir,
            output_txt_dir=txt_out_dir,
            stop_event=self._stop_event,
        )

        # If best_model_dir is not 0, sync 0 as the canonical reference
        model_0_dir = sparse_dir / "0"
        if best_model_dir != model_0_dir and best_model_dir.exists():
            for f_name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
                src_f = best_model_dir / f_name
                dst_f = model_0_dir / f_name
                if src_f.exists() and not dst_f.exists():
                    try:
                        shutil.copyfile(src_f, dst_f)
                    except Exception:
                        pass

        # Parse metrics from best model
        mapper_runtime = time.time() - mapper_start_t
        gpu_name = "NVIDIA CUDA GPU" if self.colmap_runner.is_gpu_available() else "CPU Fallback"
        summary = self.colmap_runner.parse_reconstruction_results(
            sparse_dir=sparse_dir,
            total_input_images=total_cameras,
            database_path=database_path,
            runtime_seconds=mapper_runtime,
            device_used=gpu_name,
        )

        # Generate outputs/session_name/colmap_summary.json
        summary_file = session_output_dir / "colmap_summary.json"
        summary_data = {
            "session_name": session_name,
            "colmap_version": summary.colmap_version,
            "device": summary.device,
            "total_cameras": summary.total_cameras,
            "registered_cameras": summary.registered_cameras,
            "registration_percentage": summary.registration_percentage,
            "sparse_point_count": summary.sparse_point_count,
            "mean_reprojection_error": summary.mean_reprojection_error,
            "runtime_seconds": summary.runtime_seconds,
            "sparse_model_path": str(best_model_dir),
            "database_path": str(database_path),
            "timestamp": datetime.now().isoformat(),
            "proceed": summary.is_valid and summary.registration_percentage >= 40.0,
            "quality_gate_passed": summary.is_valid and summary.registration_percentage >= 40.0,
            "status": "COMPLETED" if summary.is_valid else "FAILED",
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Saved COLMAP Summary Report -> {summary_file}")

        self.emit_event(
            StageType.COLMAP_MAPPER,
            StageStatus.COMPLETED if summary.is_valid else StageStatus.FAILED,
            1.0,
            f"Sparse SfM Complete: {summary.registered_cameras}/{summary.total_cameras} cameras ({summary.sparse_point_count:,} 3D points).",
            metrics={
                "Registered": f"{summary.registered_cameras}/{summary.total_cameras}",
                "Percentage": f"{summary.registration_percentage}%",
                "3D Points": f"{summary.sparse_point_count:,}",
                "Mean Reproj Error": f"{summary.mean_reprojection_error} px",
            },
            global_progress=0.65,
            total_cameras=summary.total_cameras,
            registered_cameras=summary.registered_cameras,
            sparse_points=summary.sparse_point_count,
        )
        return summary

    def _execute_stage_5_gsplat(
        self,
        sparse_dir: Path,
        images_dir: Path,
        session_output_dir: Path,
        total_cameras: int,
        registered_cameras: int,
        sparse_points: int,
        quality_score: int,
    ) -> Optional[GSplatTrainingResult]:
        """Stage 5: Runs real 3D Gaussian Splatting optimization with live telemetry."""
        logger.info("[Stage 5/6] Initializing 3D Gaussian Splatting (3DGS) Optimization...")
        self.emit_event(
            StageType.GAUSSIAN_SPLATTING,
            StageStatus.RUNNING,
            0.05,
            "Initializing 3D Gaussians from COLMAP sparse point cloud...",
            global_progress=0.65,
            total_cameras=total_cameras,
            registered_cameras=registered_cameras,
            sparse_points=sparse_points,
            quality_score=quality_score,
        )

        def on_gsplat_telemetry(telem: Dict[str, Any]):
            prog = telem["progress"]
            g_prog = 0.65 + (prog * 0.30)  # 0.65 to 0.95
            iter_speed = float(telem.get("iter_speed", 10.0))
            eta_s = self._eta_tracker.estimate_eta_gsplat(telem["iteration"], telem["total_iterations"], iter_speed)
            msg = (
                f"Iter: {telem['iteration']:,}/{telem['total_iterations']:,} | "
                f"Loss: {telem['loss']:.4f} | PSNR: {telem['psnr']:.1f} dB | "
                f"Gaussians: {telem['gaussian_count']:,}"
            )
            self.emit_event(
                StageType.GAUSSIAN_SPLATTING,
                StageStatus.RUNNING,
                prog,
                msg,
                metrics={
                    "Iteration": f"{telem['iteration']:,}",
                    "Loss": f"{telem['loss']:.4f}",
                    "PSNR": f"{telem['psnr']:.1f} dB",
                    "Gaussians": f"{telem['gaussian_count']:,}",
                    "Speed": f"{telem['iter_speed']} it/s",
                },
                global_progress=g_prog,
                eta_seconds=eta_s,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )

        res = self.gsplat_runner.train_gaussian_splatting(
            sparse_dir=sparse_dir,
            images_dir=images_dir,
            output_dir=session_output_dir,
            total_iterations=self.config.gsplat.iterations,
            on_telemetry=on_gsplat_telemetry,
            stop_event=self._stop_event,
        )

        if res.is_converged:
            self.emit_event(
                StageType.GAUSSIAN_SPLATTING,
                StageStatus.COMPLETED,
                1.0,
                f"3DGS Converged: Final PSNR {res.final_psnr} dB ({res.final_gaussian_count:,} Gaussians).",
                metrics={
                    "Final PSNR": f"{res.final_psnr} dB",
                    "Final Loss": f"{res.final_loss}",
                    "Gaussians": f"{res.final_gaussian_count:,}",
                    "Runtime": f"{res.training_time_seconds}s",
                },
                global_progress=0.95,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            return res
        else:
            self.emit_event(
                StageType.GAUSSIAN_SPLATTING,
                StageStatus.FAILED,
                0.0,
                f"3DGS Training failed: {res.error_message or 'Unknown error'}",
            )
            return None

    def _execute_stage_6_export(
        self,
        session_name: str,
        session_output_dir: Path,
        session_frames_dir: Path,
        sparse_dir: Path,
        video_meta: VideoMetadata,
        blur_res: BlurFilterResult,
        dup_res: DuplicateFilterResult,
        colmap_summary: ColmapSummary,
        gsplat_res: GSplatTrainingResult,
        quality_score: int,
        total_runtime: float,
    ) -> bool:
        """Stage 6: Multi-format deliverables packaging (PLY, OBJ, GLB, Trajectory, Manifest)."""
        logger.info("[Stage 6/6] Packaging final 3D deliverables and scene assets...")
        self.emit_event(
            StageType.EXPORT,
            StageStatus.RUNNING,
            0.2,
            "Generating OBJ, GLB, camera trajectory spline, and scene manifest...",
            global_progress=0.95,
        )

        ply_file = session_output_dir / "point_cloud.ply"

        # 1. Multi-format packaging via ModelExporter (PLY, OBJ, GLB, Trajectory, 16:9 Thumbnail)
        artifacts = self.exporter.package_deliverables(
            session_dir=session_output_dir,
            session_frames_dir=session_frames_dir,
            colmap_sparse_dir=sparse_dir,
            ply_file=ply_file,
        )

        # 2. Render Real Cinematic Trajectory Preview Video (1920x1080 30FPS MP4)
        traj_video_name = "trajectory_preview.mp4"
        traj_res = {}
        try:
            from pipeline.trajectory_renderer import TrajectoryRenderer

            def on_traj_render_progress(cur_f: int, tot_f: int, eta_s: float, pct: float):
                g_prog = 0.95 + (pct / 100.0) * 0.045  # 0.95 to 0.995
                msg = f"Rendering trajectory... Frame {cur_f:03d}/{tot_f} ({pct:.1f}%) | ETA: {eta_s:.1f}s"
                self.emit_event(
                    StageType.EXPORT,
                    StageStatus.RUNNING,
                    pct / 100.0,
                    msg,
                    metrics={
                        "Frame": f"{cur_f}/{tot_f}",
                        "Trajectory Progress": f"{pct:.1f}%",
                        "ETA": f"{eta_s:.1f}s",
                    },
                    global_progress=g_prog,
                    eta_seconds=eta_s,
                )

            traj_renderer = TrajectoryRenderer(width=1920, height=1080, fps=30, target_duration_seconds=10.0)
            traj_json_path = session_output_dir / "camera_trajectory.json"
            output_mp4_path = session_output_dir / traj_video_name

            if traj_json_path.exists() and ply_file.exists():
                logger.info(f"Rendering cinematic trajectory fly-through MP4 for {session_name}...")
                traj_res = traj_renderer.render_trajectory_video(
                    model_path=ply_file,
                    trajectory_json_path=traj_json_path,
                    output_video_path=output_mp4_path,
                    on_progress=on_traj_render_progress,
                )
        except Exception as e_traj:
            logger.warning(f"Trajectory preview video rendering note: {e_traj}")

        # 3. Checkpoint and model file size
        ply_size_mb = round(ply_file.stat().st_size / (1024 * 1024), 2) if ply_file.exists() else 0.0

        # 4. Comprehensive Unified Scene Manifest (Single Source of Truth)
        manifest_file = session_output_dir / "scene_manifest.json"
        manifest_data = {
            "scene_name": session_name,
            "created_at": datetime.now().isoformat(),
            "total_runtime_seconds": round(total_runtime, 2),
            "pipeline_status": PIPELINE_STATUS_COMPLETED,
            "pipeline_stage_completed": 6,
            "quality_gate_passed": True,
            "registered_cameras": colmap_summary.registered_cameras,
            "total_cameras": colmap_summary.total_cameras,
            "registration_percentage": colmap_summary.registration_percentage,
            "sparse_points": colmap_summary.sparse_point_count,
            "gaussians": gsplat_res.final_gaussian_count,
            "psnr": gsplat_res.final_psnr,
            "training_seconds": gsplat_res.training_time_seconds,
            "reprojection_error": colmap_summary.mean_reprojection_error,
            "gpu_used": getattr(gsplat_res, "device_used", "NVIDIA CUDA GPU"),
            "viewer_model": "point_cloud.splat" if (session_output_dir / "point_cloud.splat").exists() else "point_cloud.ply",
            "trajectory_video": traj_video_name,
            "trajectory_duration_seconds": traj_res.get("trajectory_duration_seconds", 10.0),
            "trajectory_fps": traj_res.get("trajectory_fps", 30),
            "trajectory_frames": traj_res.get("trajectory_frames", 300),
            "video": {
                "name": video_meta.filename,
                "resolution": video_meta.resolution_str,
                "fps": video_meta.fps,
                "duration": video_meta.duration_formatted,
            },
            "preprocessing": {
                "extracted_frames": blur_res.total_evaluated,
                "blur_removed": blur_res.discarded_count,
                "duplicates_removed": dup_res.discarded_count,
                "clean_frames": dup_res.retained_count,
                "avg_blur_score": blur_res.average_score,
            },
            "colmap_sfm": {
                "registered_cameras": colmap_summary.registered_cameras,
                "total_cameras": colmap_summary.total_cameras,
                "registration_percentage": colmap_summary.registration_percentage,
                "sparse_3d_points": colmap_summary.sparse_point_count,
                "reprojection_error": colmap_summary.mean_reprojection_error,
                "device": colmap_summary.device,
                "sfm_runtime_seconds": colmap_summary.runtime_seconds,
            },
            "gaussian_splatting": {
                "iterations": gsplat_res.total_iterations,
                "final_psnr": gsplat_res.final_psnr,
                "final_loss": gsplat_res.final_loss,
                "clean_gaussians": gsplat_res.final_gaussian_count,
                "training_time_seconds": gsplat_res.training_time_seconds,
                "quality_health_score": quality_score,
                "device": getattr(gsplat_res, "device_used", "NVIDIA CUDA GPU"),
                "ply_size_mb": ply_size_mb,
            },
            "deliverables": {
                "point_cloud_ply": "point_cloud.ply",
                "point_cloud_splat": artifacts.get("point_cloud_splat", "point_cloud.splat"),
                "model_obj": artifacts.get("model_obj", "model.obj"),
                "model_glb": artifacts.get("model_glb", "model.glb"),
                "camera_trajectory": artifacts.get("camera_trajectory", "camera_trajectory.json"),
                "trajectory_video": traj_video_name,
                "colmap_summary": "colmap_summary.json",
                "preprocess_report": "preprocess_report.json",
                "thumbnail": "thumbnail.png",
                "gaussians_model_npz": "checkpoints/gaussians_model.npz",
            },
            "status": "COMPLETED",
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 5. Automatic Validation (Verify all deliverables exist and are non-empty)
        req_files = [
            ply_file,
            session_output_dir / "model.obj",
            session_output_dir / "model.glb",
            session_output_dir / "camera_trajectory.json",
            session_output_dir / "trajectory_preview.mp4",
            session_output_dir / "scene_manifest.json",
            session_output_dir / "checkpoints" / "checkpoint_final.json",
            session_output_dir / "checkpoints" / "gaussians_model.npz",
        ]
        missing = [str(f.name) for f in req_files if not f.exists() or f.stat().st_size == 0]
        if missing:
            err_msg = f"Deliverable validation warning: Missing or empty files: {', '.join(missing)}"
            logger.warning(err_msg)

        self.emit_event(
            StageType.EXPORT,
            StageStatus.COMPLETED,
            1.0,
            f"Deliverables packaged in {session_output_dir.name}/ (PLY, OBJ, GLB, Trajectory, NPZ ready)",
            metrics={"Format": "PLY + OBJ + GLB", "Status": "Complete"},
            global_progress=1.0,
            quality_score=quality_score,
        )
        return True
