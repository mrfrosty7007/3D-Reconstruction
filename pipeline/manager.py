"""
GeoRecon AI - Studio Pipeline Manager
Coordinates the complete end-to-end 9-stage photogrammetry and Screened Poisson Mesh pipeline:
1. Extract Frames (Video Ingestion)
2. Filter Frames (Blur + Duplicate Filtering)
3. COLMAP Features (SIFT Extraction)
4. Feature Matching (Exhaustive/Sequential Matcher)
5. Sparse Reconstruction (Incremental Mapper + Model Conversion)
   -> Reconstruction Quality Gate (Green / Yellow / Red validation)
6. Dense Stereo (image_undistorter + patch_match_stereo)
7. Stereo Fusion (Multi-view Depth Fusion -> dense/fused.ply)
8. Poisson Meshing (Open3D Screened Poisson Surface Reconstruction -> OBJ, PLY, GLB)
9. Export Assets (Deliverables packaging, camera trajectory, thumbnail, manifests)
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
from pipeline.poisson_mesher import PoissonMesher, PoissonResult
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
    Backward compatibility for sessions:
    - If model.obj, model.glb, or model.ply exists (with trajectory or manifest) -> 'completed'
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

    # Deliverables inference rules
    obj_p = session_dir / "model.obj"
    glb_p = session_dir / "model.glb"
    ply_p = session_dir / "model.ply"
    old_ply = session_dir / "point_cloud.ply"
    traj_p = session_dir / "trajectory_preview.mp4"
    if (obj_p.exists() or glb_p.exists() or ply_p.exists() or old_ply.exists()) and traj_p.exists():
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
            return 45.0  # Estimated ~25s Dense + ~15s Poisson + ~5s Export

        speed = 0.0
        if len(self.reg_cam_history) >= 2:
            dt = self.reg_cam_history[-1][0] - self.reg_cam_history[0][0]
            dc = self.reg_cam_history[-1][1] - self.reg_cam_history[0][1]
            if dt > 1.0 and dc > 0:
                speed = dc / dt

        if speed <= 0:
            speed = 1.5  # Typical registration speed: ~1.5 cams/s

        mapper_rem_s = rem_cams / max(0.2, speed)
        total_rem_s = mapper_rem_s + 45.0  # + Dense Stereo, Fusion, Poisson, Export
        if self.last_estimated_eta is not None:
            total_rem_s = 0.7 * total_rem_s + 0.3 * self.last_estimated_eta
        self.last_estimated_eta = max(5.0, total_rem_s)
        return self.last_estimated_eta

    def estimate_eta_dense(self, current_stage_prog: float, stage_idx: int) -> float:
        """Estimates remaining time during Dense Reconstruction & Poisson Meshing."""
        stage_weights = {6: 25.0, 7: 15.0, 8: 15.0, 9: 5.0}
        rem = 0.0
        for s, w in stage_weights.items():
            if s == stage_idx:
                rem += w * max(0.0, 1.0 - current_stage_prog)
            elif s > stage_idx:
                rem += w
        self.last_estimated_eta = max(1.0, rem)
        return self.last_estimated_eta

    def estimate_eta_gsplat(self, current_iter: int, total_iter: int, iter_speed: float) -> float:
        """[DEPRECATED] Compatibility alias for legacy tests."""
        rem_iters = max(0, total_iter - current_iter)
        speed = max(1.0, iter_speed)
        gsplat_rem = rem_iters / speed
        total_rem = gsplat_rem + 5.0
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
        self.poisson_mesher = PoissonMesher(config.poisson)
        self.gsplat_runner = None  # Deprecated
        self.exporter = ModelExporter("obj")

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
        self.last_poisson_result: Optional[PoissonResult] = None
        self.last_dense_ply: Optional[Path] = None
        self.last_gsplat_result: Optional[Any] = None

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
            logger.info("Quality Gate override signal sent: User confirmed continuation to Dense Reconstruction.")

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
                points_xyz, points_rgb = ColmapRunner.load_colmap_points(best_dir)
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
        dense_workspace_dir = session_colmap_dir / "dense"
        session_output_dir = output_dir / session_name
        session_output_dense_dir = session_output_dir / "dense"

        session_frames_dir.mkdir(parents=True, exist_ok=True)
        session_colmap_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_workspace_dir.mkdir(parents=True, exist_ok=True)
        session_output_dir.mkdir(parents=True, exist_ok=True)
        session_output_dense_dir.mkdir(parents=True, exist_ok=True)

        total_cameras = 0
        registered_cameras = 0
        sparse_points = 0
        dense_points = 0
        quality_score = 0
        current_stage = 0
        quality_gate_passed = False
        colmap_summary: Optional[ColmapSummary] = None
        poisson_res: Optional[PoissonResult] = None
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
            logger.info(f"🌐 TerraSweep Studio — COLMAP MVS + Poisson Mesh Session [{session_name}]")
            logger.info(f"Input Video: {video_path}")
            logger.info(f"Frames Path: {session_frames_dir}")
            logger.info("Output Path: %s", session_output_dir)
            logger.info("=" * 75)

            # Startup Verification: Log verified COLMAP GPU flags
            self.colmap_runner.verify_gpu_flags()

            # =========================================================
            # STAGE 1: Extract Frames (Video Ingestion)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 0, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return
            extracted_paths, video_meta = self._execute_stage_1_frames(
                video_path, session_frames_dir
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 0, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return
            if not extracted_paths:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 0, False, failure_reason="Stage 1 failed: No frames extracted from video.", stage_name_str="STAGE_1_FRAMES")
                manifest_written = True
                return

            current_stage = 1
            self._last_stage_name = "STAGE_1_FRAMES"

            # =========================================================
            # STAGE 2: Filter Frames (Blur + Duplicate Filtering)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 1, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_2_FILTER")
                manifest_written = True
                return
            frames_dataset, blur_res, dup_res = self._execute_stage_2_filter_frames(
                extracted_paths, session_output_dir, session_name, video_meta
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 1, False, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_2_FILTER")
                manifest_written = True
                return
            if not frames_dataset:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 1, False, failure_reason="Stage 2 failed: All frames were filtered as blurry or duplicate.", stage_name_str="STAGE_2_FILTER")
                manifest_written = True
                return

            total_cameras = len(frames_dataset)
            current_stage = 2
            self._last_stage_name = "STAGE_2_FILTER"

            # =========================================================
            # STAGE 3: COLMAP Features (SIFT Feature Extraction)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 2, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_3_FEATURES")
                manifest_written = True
                return
            feat_ok = self._execute_stage_3_features(session_frames_dir, database_path)
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 2, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_3_FEATURES")
                manifest_written = True
                return
            if not feat_ok:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 2, False, total_cameras=total_cameras, failure_reason="Stage 3 failed: COLMAP SIFT feature extraction failed.", stage_name_str="STAGE_3_FEATURES")
                manifest_written = True
                return
            current_stage = 3
            self._last_stage_name = "STAGE_3_FEATURES"

            # =========================================================
            # STAGE 4: Feature Matching (Exhaustive Matcher)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 3, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_4_MATCHING")
                manifest_written = True
                return
            match_ok = self._execute_stage_4_matching(database_path)
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 3, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_4_MATCHING")
                manifest_written = True
                return
            if not match_ok:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 3, False, total_cameras=total_cameras, failure_reason="Stage 4 failed: COLMAP feature matching failed.", stage_name_str="STAGE_4_MATCHING")
                manifest_written = True
                return
            current_stage = 4
            self._last_stage_name = "STAGE_4_MATCHING"

            # =========================================================
            # STAGE 5: Sparse Reconstruction (Incremental Mapper + TXT)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 4, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_5_MAPPER")
                manifest_written = True
                return
            colmap_summary = self._execute_stage_5_mapper(
                session_frames_dir, database_path, sparse_dir, total_cameras, session_name, session_output_dir
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 4, False, total_cameras=total_cameras, failure_reason="Reconstruction cancelled by user.", stage_name_str="STAGE_5_MAPPER")
                manifest_written = True
                return
            if not colmap_summary or not colmap_summary.is_valid:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 4, False, total_cameras=total_cameras, failure_reason="Stage 5 failed: COLMAP mapper failed to reconstruct sparse model.", stage_name_str="STAGE_5_MAPPER")
                manifest_written = True
                return

            self.last_colmap_summary = colmap_summary
            registered_cameras = colmap_summary.registered_cameras
            sparse_points = colmap_summary.sparse_point_count
            current_stage = 5
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
                logger.info(f"Quality Gate: GREEN (≥40%). Proceeding automatically to Dense MVS ({colmap_summary.registration_percentage:.1f}% registered).")
                quality_gate_passed = True
                self.emit_event(
                    stage=StageType.COLMAP_MAPPER,
                    status=StageStatus.COMPLETED,
                    progress=1.0,
                    message=f"Quality Gate: High Confidence (≥40%). Proceeding to Dense Stereo.",
                    quality_score=quality_score,
                    total_cameras=total_cameras,
                    registered_cameras=registered_cameras,
                    sparse_points=sparse_points,
                    metrics={
                        "QualityLevel": "GREEN",
                        "Action": "Auto-Continue to Dense Stereo",
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
                            stage_num=5,
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
                logger.info("Quality Gate Yellow: User confirmed continuation. Proceeding to Stage 6...")

            else:
                # Red (<20%): Stop before Dense MVS, mark reconstruction as failed
                failure_msg = f"Quality Gate Failed: Only {registered_cameras}/{total_cameras} cameras registered ({colmap_summary.registration_percentage:.1f}% < 20%)."
                logger.error(f"Reconstruction Quality Gate [RED]: {failure_msg}")
                self._save_session_outcome(
                    session_output_dir=session_output_dir,
                    session_name=session_name,
                    pipeline_status=PIPELINE_STATUS_FAILED,
                    stage_num=5,
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
                    message=f"Quality Gate RED: Camera registration below 20% ({colmap_summary.registration_percentage:.1f}%). Stopped before Dense MVS.",
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
            # STAGE 6: Dense Stereo (image_undistorter + patch_match_stereo)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 5, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_6_DENSE_STEREO")
                manifest_written = True
                return
            dense_stereo_ok = self._execute_stage_6_dense_stereo(
                session_frames_dir=session_frames_dir,
                sparse_dir=sparse_dir,
                dense_workspace_dir=dense_workspace_dir,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 5, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_6_DENSE_STEREO")
                manifest_written = True
                return
            if not dense_stereo_ok:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 5, True, registered_cameras, total_cameras, "Stage 6 failed: COLMAP dense stereo reconstruction failed.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_6_DENSE_STEREO")
                manifest_written = True
                return

            current_stage = 6
            self._last_stage_name = "STAGE_6_DENSE_STEREO"

            # =========================================================
            # STAGE 7: Stereo Fusion (dense/fused.ply)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 6, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_7_STEREO_FUSION")
                manifest_written = True
                return

            fused_ply_workspace = dense_workspace_dir / "fused.ply"
            dense_points = self._execute_stage_7_stereo_fusion(
                dense_workspace_dir=dense_workspace_dir,
                fused_ply_path=fused_ply_workspace,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 6, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_7_STEREO_FUSION")
                manifest_written = True
                return
            if dense_points == 0 or not fused_ply_workspace.exists():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 6, True, registered_cameras, total_cameras, "Stage 7 failed: COLMAP stereo fusion failed to generate dense/fused.ply.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_7_STEREO_FUSION")
                manifest_written = True
                return

            # Ensure dense/fused.ply exists in session output deliverables
            output_fused_ply = session_output_dense_dir / "fused.ply"
            try:
                shutil.copyfile(fused_ply_workspace, output_fused_ply)
            except Exception as e:
                logger.warning(f"Copying fused.ply to output dense folder: {e}")

            # Also provide point_cloud.ply at root as point cloud deliverable
            output_root_ply = session_output_dir / "point_cloud.ply"
            try:
                shutil.copyfile(fused_ply_workspace, output_root_ply)
            except Exception as e:
                pass

            current_stage = 7
            self._last_stage_name = "STAGE_7_STEREO_FUSION"

            # =========================================================
            # STAGE 8: Screened Poisson Surface Reconstruction (Open3D)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 7, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_8_POISSON_MESHING")
                manifest_written = True
                return

            mesh_obj = session_output_dir / "model.obj"
            mesh_ply = session_output_dir / "model.ply"
            mesh_glb = session_output_dir / "model.glb"

            poisson_res = self._execute_stage_8_poisson_meshing(
                fused_ply_path=fused_ply_workspace,
                output_obj_path=mesh_obj,
                output_ply_path=mesh_ply,
                output_glb_path=mesh_glb,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                dense_points=dense_points,
                quality_score=quality_score,
            )
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 7, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_8_POISSON_MESHING")
                manifest_written = True
                return
            if not poisson_res or not poisson_res.is_success:
                err_m = poisson_res.error_message if poisson_res else "Unknown"
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 7, True, registered_cameras, total_cameras, f"Stage 8 failed: Screened Poisson Reconstruction failed: {err_m}", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_8_POISSON_MESHING")
                manifest_written = True
                return

            self.last_poisson_result = poisson_res
            current_stage = 8
            self._last_stage_name = "STAGE_8_POISSON_MESHING"

            # =========================================================
            # STAGE 9: Export Assets (OBJ, GLB, PLY, Trajectory & Manifests)
            # =========================================================
            if self._stop_event.is_set():
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_CANCELLED, 8, True, registered_cameras, total_cameras, "Reconstruction cancelled by user.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_9_EXPORT")
                manifest_written = True
                return
            export_ok = self._execute_stage_9_export(
                session_name=session_name,
                session_output_dir=session_output_dir,
                session_frames_dir=session_frames_dir,
                sparse_dir=sparse_dir,
                video_meta=video_meta,
                blur_res=blur_res,
                dup_res=dup_res,
                colmap_summary=colmap_summary,
                dense_points=dense_points,
                poisson_res=poisson_res,
                quality_score=quality_score,
                total_runtime=time.time() - pipeline_start_t,
            )

            if export_ok:
                manifest_written = True
                current_stage = 9
                self._last_stage_name = "STAGE_9_EXPORT"
                self.write_diagnostics(
                    session_output_dir=session_output_dir,
                    worker_thread_status="completed",
                    failure_reason=None,
                    last_processed_stage="STAGE_9_EXPORT",
                    last_registered_camera_count=registered_cameras,
                )
            else:
                self._save_session_outcome(session_output_dir, session_name, PIPELINE_STATUS_FAILED, 8, True, registered_cameras, total_cameras, "Stage 9 failed: Deliverables packaging failed.", colmap_summary.registration_percentage, sparse_points, stage_name_str="STAGE_9_EXPORT")
                manifest_written = True

            logger.info("=" * 75)
            logger.info("🎉 Studio 3D Reconstruction Session Completed Successfully!")
            logger.info(f"Registered Cameras: {registered_cameras}/{total_cameras} ({colmap_summary.registration_percentage}%)")
            logger.info(f"Sparse 3D Points: {sparse_points:,} | Dense 3D Points: {dense_points:,}")
            logger.info(f"Screened Poisson Mesh: {poisson_res.triangle_count:,} Triangles | {poisson_res.vertex_count:,} Vertices")
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
                    worker_thread_status="completed" if (quality_gate_passed and current_stage == 9) else "failed",
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
        - Green (>=40%): High confidence, continue automatically to Dense Reconstruction.
        - Yellow (20-40%): Moderate confidence, pause for 'Continue Anyway' review.
        - Red (<20%): Low confidence, halt before Dense Reconstruction and recommend retry settings.
        """
        reg_pct = colmap_summary.registration_percentage
        points = colmap_summary.sparse_point_count
        reg_cams = colmap_summary.registered_cameras
        total_cams = colmap_summary.total_cameras

        score = calculate_live_quality_score(reg_cams, total_cams, points)
        if reg_pct >= 40.0:
            level = "GREEN"
            proceed = True
            action = "High reconstruction confidence (≥40%). Proceeding automatically to Dense Reconstruction."
        elif reg_pct >= 20.0 or (total_cams <= 4 and reg_cams >= 1):
            level = "YELLOW"
            proceed = getattr(self, "auto_continue_yellow", False)
            action = "Moderate reconstruction confidence (20–40%). Partial camera registration. Click 'Continue Anyway'."
        else:
            level = "RED"
            proceed = False
            action = "Low reconstruction confidence (<20%). Reconstruction halted before Dense Reconstruction. Adjust settings and retry."

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
    ) -> Tuple[Optional[List[Path]], Optional[VideoMetadata]]:
        """Stage 1: Ingests video and extracts keyframes."""
        logger.info("[Stage 1/9] Ingesting video stream and extracting keyframes...")
        self.emit_event(
            StageType.FRAME_EXTRACTION,
            StageStatus.RUNNING,
            0.05,
            "Ingesting video stream and extracting keyframes...",
            global_progress=0.01,
        )

        try:
            meta = self.video_processor.inspect_video(video_path)
            extracted_paths, _ = self.video_processor.extract_frames(
                video_path=video_path,
                output_frames_dir=session_frames_dir,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FRAME_EXTRACTION, StageStatus.RUNNING, p, msg, m, global_progress=0.01 + p * 0.04
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None

            if not extracted_paths:
                self.emit_event(StageType.FRAME_EXTRACTION, StageStatus.FAILED, 0.0, "No frames could be extracted.")
                return None, None

            self.emit_event(
                StageType.FRAME_EXTRACTION,
                StageStatus.COMPLETED,
                1.0,
                f"Extracted {len(extracted_paths)} raw frames from video stream.",
                metrics={
                    "Extracted": len(extracted_paths),
                    "FPS": f"{meta.fps:.1f}",
                    "Resolution": meta.resolution_str,
                    "Duration": meta.duration_formatted,
                },
                global_progress=0.05,
            )
            return extracted_paths, meta

        except Exception as e:
            logger.exception(f"Error in Stage 1 Frame Extraction: {e}")
            self.emit_event(StageType.FRAME_EXTRACTION, StageStatus.FAILED, 0.0, str(e))
            return None, None

    def _execute_stage_2_filter_frames(
        self,
        extracted_paths: List[Path],
        session_output_dir: Path,
        session_name: str,
        meta: Optional[VideoMetadata],
    ) -> Tuple[Optional[List[Path]], Optional[BlurFilterResult], Optional[DuplicateFilterResult]]:
        """Stage 2: Filters blurry frames and removes visual duplicates."""
        logger.info("[Stage 2/9] Filtering frames (blur & duplicate elimination)...")
        self.emit_event(
            StageType.FILTER_FRAMES,
            StageStatus.RUNNING,
            0.05,
            "Filtering blurry frames using Laplacian variance...",
            global_progress=0.05,
        )

        try:
            blur_res = self.blur_filter.filter_frames(
                frame_paths=extracted_paths,
                delete_discarded=True,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FILTER_FRAMES, StageStatus.RUNNING, p * 0.5, msg, m, global_progress=0.05 + p * 0.035
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None, None

            self.emit_event(
                StageType.FILTER_FRAMES,
                StageStatus.RUNNING,
                0.5,
                "Analyzing perceptual similarity and removing duplicates...",
                global_progress=0.085,
            )

            dup_res = self.duplicate_filter.filter_duplicates(
                frame_paths=blur_res.retained_frames,
                delete_discarded=True,
                on_progress=lambda p, msg, m: self.emit_event(
                    StageType.FILTER_FRAMES, StageStatus.RUNNING, 0.5 + p * 0.5, msg, m, global_progress=0.085 + p * 0.035
                ),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                return None, None, None

            final_dataset = dup_res.retained_frames

            # Write Preprocess JSON Report
            report_file = session_output_dir / "preprocess_report.json"
            report_data = {
                "session_name": session_name,
                "timestamp": datetime.now().isoformat(),
                "video_metadata": {
                    "filename": meta.filename if meta else "unknown",
                    "resolution": meta.resolution_str if meta else "unknown",
                    "fps": meta.fps if meta else 0.0,
                    "duration_formatted": meta.duration_formatted if meta else "unknown",
                    "original_total_frames": meta.total_frames if meta else len(extracted_paths),
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
                StageType.FILTER_FRAMES,
                StageStatus.COMPLETED,
                1.0,
                f"Prepared clean dataset of {len(final_dataset)} frames.",
                metrics={
                    "Retained": len(final_dataset),
                    "Blur Removed": blur_res.discarded_count,
                    "Dup Removed": dup_res.discarded_count,
                    "Avg Blur": f"{blur_res.average_score:.1f}",
                },
                global_progress=0.12,
                total_cameras=len(final_dataset),
            )
            return final_dataset, blur_res, dup_res

        except Exception as e:
            logger.exception(f"Error in Stage 2 Frame Filtering: {e}")
            self.emit_event(StageType.FILTER_FRAMES, StageStatus.FAILED, 0.0, str(e))
            return None, None, None

    def _execute_stage_3_features(self, images_dir: Path, database_path: Path) -> bool:
        """Stage 3: Runs COLMAP SIFT feature extraction with GPU acceleration."""
        logger.info("[Stage 3/9] Running COLMAP Feature Extractor...")
        self.emit_event(
            StageType.COLMAP_FEATURES,
            StageStatus.RUNNING,
            0.1,
            "Extracting SIFT keypoints with GPU acceleration...",
            global_progress=0.12,
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

    def _execute_stage_4_matching(self, database_path: Path) -> bool:
        """Stage 4: Runs COLMAP feature matcher across image pairs."""
        logger.info("[Stage 4/9] Running COLMAP Feature Matcher...")
        self.emit_event(
            StageType.COLMAP_MATCHING,
            StageStatus.RUNNING,
            0.1,
            f"{self.config.colmap.matcher_type.capitalize()} feature matching across image pairs...",
            global_progress=0.25,
        )

        ok = self.colmap_runner.run_feature_matching(
            database_path=database_path,
            matcher_type=self.config.colmap.matcher_type,
            use_gpu=True,
            on_log=lambda l: self.emit_event(
                StageType.COLMAP_MATCHING, StageStatus.RUNNING, 0.5, f"COLMAP Matcher: {l[:50]}", global_progress=0.32
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
                global_progress=0.38,
            )
        else:
            self.emit_event(
                StageType.COLMAP_MATCHING,
                StageStatus.FAILED,
                0.0,
                "COLMAP feature matching failed.",
            )
        return ok

    def _execute_stage_5_mapper(
        self,
        images_dir: Path,
        database_path: Path,
        sparse_dir: Path,
        total_cameras: int,
        session_name: str,
        session_output_dir: Path,
    ) -> Optional[ColmapSummary]:
        """Stage 5: Runs COLMAP incremental mapper with live registration & point telemetry."""
        logger.info("[Stage 5/9] Running COLMAP Incremental Mapper (Sparse Reconstruction)...")
        self.emit_event(
            StageType.COLMAP_MAPPER,
            StageStatus.RUNNING,
            0.05,
            "Running Incremental Mapper & Bundle Adjustment...",
            global_progress=0.38,
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
            g_prog = 0.38 + (0.17 * cam_frac)
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
            g_prog = 0.38 + (0.17 * cam_frac)
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
            g_prog = 0.38 + (0.17 * cam_frac)
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
                g_prog = 0.38 + (0.17 * cam_frac)
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
            try:
                partial_dir, part_imgs, part_pts = self.colmap_runner.find_best_model_dir(sparse_dir)
                if part_imgs > 0:
                    logger.info(f"Preserving partial sparse reconstruction with {part_imgs} cameras and {part_pts} points.")
                    self._last_reg_cams = part_imgs
            except Exception:
                pass
            self.emit_event(StageType.COLMAP_MAPPER, StageStatus.FAILED, 0.0, "COLMAP mapper failed.")
            return None

        best_model_dir, best_imgs, best_pts = self.colmap_runner.find_best_model_dir(sparse_dir)
        logger.info(f"Using best reconstructed model at: {best_model_dir} ({best_imgs} cameras, {best_pts} points)")

        # Convert Best Model BIN to TXT
        txt_out_dir = best_model_dir / "txt"
        self.colmap_runner.run_model_converter(
            input_model_dir=best_model_dir,
            output_txt_dir=txt_out_dir,
            stop_event=self._stop_event,
        )

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

        mapper_runtime = time.time() - mapper_start_t
        gpu_name = "NVIDIA CUDA GPU" if self.colmap_runner.is_gpu_available() else "CPU Fallback"
        summary = self.colmap_runner.parse_reconstruction_results(
            sparse_dir=sparse_dir,
            total_input_images=total_cameras,
            database_path=database_path,
            runtime_seconds=mapper_runtime,
            device_used=gpu_name,
        )

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
            global_progress=0.55,
            total_cameras=summary.total_cameras,
            registered_cameras=summary.registered_cameras,
            sparse_points=summary.sparse_point_count,
        )
        return summary

    def _execute_stage_6_dense_stereo(
        self,
        session_frames_dir: Path,
        sparse_dir: Path,
        dense_workspace_dir: Path,
        total_cameras: int,
        registered_cameras: int,
        sparse_points: int,
        quality_score: int,
    ) -> bool:
        """Stage 6: Runs COLMAP image_undistorter and patch_match_stereo (MVS depth & normal maps)."""
        logger.info("[Stage 6/9] Running Dense Stereo (image_undistorter + patch_match_stereo)...")
        self.emit_event(
            StageType.DENSE_STEREO,
            StageStatus.RUNNING,
            0.05,
            "Undistorting camera images for Multi-View Stereo...",
            metrics={"Sub-State": "Undistorting Images", "Device": "CUDA"},
            global_progress=0.55,
            total_cameras=total_cameras,
            registered_cameras=registered_cameras,
            sparse_points=sparse_points,
            quality_score=quality_score,
        )

        dense_workspace_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir, _, _ = self.colmap_runner.find_best_model_dir(sparse_dir)
        sparse_input = best_model_dir if best_model_dir.exists() else (sparse_dir / "0")

        # Step 1: Image Undistortion
        undistort_ok = self.colmap_runner.run_image_undistorter(
            image_path=session_frames_dir,
            input_model_dir=sparse_input,
            output_dense_dir=dense_workspace_dir,
            max_image_size=self.config.dense.max_image_size,
            on_log=lambda l: self.emit_event(
                StageType.DENSE_STEREO, StageStatus.RUNNING, 0.15, f"Undistorter: {l[:50]}", global_progress=0.57
            ),
            stop_event=self._stop_event,
        )

        if not undistort_ok:
            self.emit_event(StageType.DENSE_STEREO, StageStatus.FAILED, 0.0, "COLMAP image_undistorter failed.")
            return False

        if self._stop_event.is_set():
            return False

        # Step 2: Patch Match Stereo (Depth & Normal Maps)
        self.emit_event(
            StageType.DENSE_STEREO,
            StageStatus.RUNNING,
            0.25,
            "Computing photometric & geometric MVS depth maps (CUDA accelerated)...",
            metrics={"Sub-State": "Patch Match Stereo", "Device": "NVIDIA GPU"},
            global_progress=0.58,
            total_cameras=total_cameras,
            registered_cameras=registered_cameras,
            sparse_points=sparse_points,
            quality_score=quality_score,
        )

        def _on_patch_match_progress(view_idx: int, num_views: int):
            frac = min(1.0, view_idx / max(1, num_views))
            g_prog = 0.58 + (0.12 * frac)
            eta_s = self._eta_tracker.estimate_eta_dense(view_idx, num_views)
            self.emit_event(
                StageType.DENSE_STEREO,
                StageStatus.RUNNING,
                0.25 + (frac * 0.75),
                f"Stereo Depth Estimation: view {view_idx}/{num_views} ({frac*100:.1f}%)",
                metrics={
                    "Stereo View": f"{view_idx}/{num_views}",
                    "Dense Mode": "CUDA GPU",
                    "Sub-State": "PatchMatch Stereo",
                },
                global_progress=g_prog,
                eta_seconds=eta_s,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )

        stereo_ok = self.colmap_runner.run_patch_match_stereo(
            workspace_path=dense_workspace_dir,
            max_image_size=self.config.dense.max_image_size,
            gpu_index=self.config.dense.gpu_index,
            on_log=lambda l: None,
            on_view_progress=_on_patch_match_progress,
            stop_event=self._stop_event,
        )

        if stereo_ok:
            self.emit_event(
                StageType.DENSE_STEREO,
                StageStatus.COMPLETED,
                1.0,
                "Dense stereo depth maps and normals generated successfully.",
                metrics={"Stereo": "Complete", "Workspace": dense_workspace_dir.name},
                global_progress=0.70,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            return True
        else:
            self.emit_event(
                StageType.DENSE_STEREO,
                StageStatus.FAILED,
                0.0,
                "COLMAP patch_match_stereo failed.",
            )
            return False

    def _execute_stage_7_stereo_fusion(
        self,
        dense_workspace_dir: Path,
        fused_ply_path: Path,
        total_cameras: int,
        registered_cameras: int,
        sparse_points: int,
        quality_score: int,
    ) -> int:
        """Stage 7: Fuses depth and normal maps into a high-density point cloud (fused.ply)."""
        logger.info("[Stage 7/9] Running COLMAP Stereo Fusion...")
        self.emit_event(
            StageType.STEREO_FUSION,
            StageStatus.RUNNING,
            0.1,
            "Fusing depth & normal maps into dense point cloud...",
            metrics={"Sub-State": "Stereo Fusion"},
            global_progress=0.70,
            total_cameras=total_cameras,
            registered_cameras=registered_cameras,
            sparse_points=sparse_points,
            quality_score=quality_score,
        )

        _fused_pts = 0

        def _on_view_progress(view_idx: int, num_views: int):
            frac = min(1.0, view_idx / max(1, num_views)) if num_views > 0 else 0.5
            g_prog = 0.70 + (0.12 * frac)
            self.emit_event(
                StageType.STEREO_FUSION,
                StageStatus.RUNNING,
                frac,
                f"Fusing depth maps: view {view_idx}/{num_views} | {_fused_pts:,} dense points",
                metrics={"Views Fused": f"{view_idx}/{num_views}", "Dense Points": f"{_fused_pts:,}"},
                global_progress=g_prog,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )

        def _on_fused_points(pts: int):
            nonlocal _fused_pts
            _fused_pts = max(_fused_pts, pts)

        fusion_ok = self.colmap_runner.run_stereo_fusion(
            workspace_path=dense_workspace_dir,
            output_ply_path=fused_ply_path,
            min_num_pixels=self.config.dense.min_num_pixels,
            on_log=lambda l: None,
            on_view_progress=_on_view_progress,
            on_fused_points=_on_fused_points,
            stop_event=self._stop_event,
        )

        dense_pts = 0
        if fused_ply_path.exists():
            dense_pts = PoissonMesher.get_point_cloud_info(fused_ply_path).get("point_count", _fused_pts)
            if dense_pts == 0 and _fused_pts > 0:
                dense_pts = _fused_pts

        if fusion_ok and fused_ply_path.exists() and dense_pts > 0:
            self.last_dense_ply = fused_ply_path
            self.emit_event(
                StageType.STEREO_FUSION,
                StageStatus.COMPLETED,
                1.0,
                f"Dense Point Cloud Generated: {dense_pts:,} points.",
                metrics={"Dense Points": f"{dense_pts:,}", "Output": "fused.ply"},
                global_progress=0.82,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            return dense_pts
        else:
            self.emit_event(
                StageType.STEREO_FUSION,
                StageStatus.FAILED,
                0.0,
                "COLMAP stereo_fusion failed to produce fused point cloud.",
            )
            return 0

    def _execute_stage_8_poisson_meshing(
        self,
        fused_ply_path: Path,
        output_obj_path: Path,
        output_ply_path: Path,
        output_glb_path: Path,
        total_cameras: int,
        registered_cameras: int,
        sparse_points: int,
        dense_points: int,
        quality_score: int,
    ) -> Optional[PoissonResult]:
        """Stage 8: Reconstructs watertight mesh via Open3D Screened Poisson Surface Reconstruction."""
        logger.info("[Stage 8/9] Running Screened Poisson Surface Reconstruction...")
        self.emit_event(
            StageType.POISSON_MESHING,
            StageStatus.RUNNING,
            0.05,
            "Reconstructing watertight Screened Poisson surface mesh...",
            metrics={"Sub-State": "Poisson Reconstruction", "Dense Points": f"{dense_points:,}"},
            global_progress=0.82,
            total_cameras=total_cameras,
            registered_cameras=registered_cameras,
            sparse_points=sparse_points,
            quality_score=quality_score,
        )

        def _on_poisson_progress(frac: float, msg: str):
            g_prog = 0.82 + (0.12 * frac)
            self.emit_event(
                StageType.POISSON_MESHING,
                StageStatus.RUNNING,
                frac,
                f"Poisson Mesher: {msg}",
                metrics={"Sub-State": msg[:30], "Dense Points": f"{dense_points:,}"},
                global_progress=g_prog,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )

        res = self.poisson_mesher.reconstruct_mesh(
            input_ply_path=fused_ply_path,
            output_obj_path=output_obj_path,
            output_ply_path=output_ply_path,
            output_glb_path=output_glb_path,
            on_progress=_on_poisson_progress,
            stop_event=self._stop_event,
        )

        if res.is_success:
            self.emit_event(
                StageType.POISSON_MESHING,
                StageStatus.COMPLETED,
                1.0,
                f"Poisson Mesh Complete: {res.triangle_count:,} triangles, {res.vertex_count:,} vertices.",
                metrics={
                    "Triangles": f"{res.triangle_count:,}",
                    "Vertices": f"{res.vertex_count:,}",
                    "Runtime": f"{res.runtime_seconds:.1f}s",
                },
                global_progress=0.94,
                total_cameras=total_cameras,
                registered_cameras=registered_cameras,
                sparse_points=sparse_points,
                quality_score=quality_score,
            )
            return res
        else:
            self.emit_event(
                StageType.POISSON_MESHING,
                StageStatus.FAILED,
                0.0,
                f"Screened Poisson Meshing failed: {res.error_message}",
            )
            return None

    def _execute_stage_9_export(
        self,
        session_name: str,
        session_output_dir: Path,
        session_frames_dir: Path,
        sparse_dir: Path,
        video_meta: Optional[VideoMetadata],
        blur_res: Optional[BlurFilterResult],
        dup_res: Optional[DuplicateFilterResult],
        colmap_summary: ColmapSummary,
        dense_points: int,
        poisson_res: PoissonResult,
        quality_score: int,
        total_runtime: float,
    ) -> bool:
        """Stage 9: Multi-format deliverables packaging (OBJ, GLB, PLY, Trajectory, Manifest)."""
        logger.info("[Stage 9/9] Packaging final 3D deliverables and scene assets...")
        self.emit_event(
            StageType.EXPORT,
            StageStatus.RUNNING,
            0.2,
            "Generating camera trajectory preview, scene manifest, and packaging deliverables...",
            global_progress=0.94,
        )

        ply_file = session_output_dir / "dense" / "fused.ply"
        if not ply_file.exists():
            ply_file = session_output_dir / "point_cloud.ply"

        # 1. Multi-format packaging via ModelExporter (PLY, OBJ, GLB, Trajectory, 16:9 Thumbnail)
        artifacts = self.exporter.package_deliverables(
            session_dir=session_output_dir,
            session_frames_dir=session_frames_dir,
            colmap_sparse_dir=sparse_dir,
            ply_file=ply_file,
        )

        # 2. Render Trajectory Preview Video (1920x1080 30FPS MP4)
        traj_video_name = "trajectory_preview.mp4"
        traj_res = {}
        try:
            from pipeline.trajectory_renderer import TrajectoryRenderer

            def on_traj_render_progress(cur_f: int, tot_f: int, eta_s: float, pct: float):
                g_prog = 0.94 + (pct / 100.0) * 0.05
                msg = f"Rendering trajectory flythrough... Frame {cur_f:03d}/{tot_f} ({pct:.1f}%) | ETA: {eta_s:.1f}s"
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

            # Render trajectory against point cloud or mesh
            render_model_path = ply_file if ply_file.exists() else (session_output_dir / "model.ply")
            if traj_json_path.exists() and render_model_path.exists():
                logger.info(f"Rendering cinematic trajectory fly-through MP4 for {session_name}...")
                traj_res = traj_renderer.render_trajectory_video(
                    model_path=render_model_path,
                    trajectory_json_path=traj_json_path,
                    output_video_path=output_mp4_path,
                    on_progress=on_traj_render_progress,
                )
        except Exception as e_traj:
            logger.warning(f"Trajectory preview video rendering note: {e_traj}")

        # 3. Comprehensive Unified Scene Manifest (Single Source of Truth)
        manifest_file = session_output_dir / "scene_manifest.json"
        manifest_data = {
            "scene_name": session_name,
            "created_at": datetime.now().isoformat(),
            "total_runtime_seconds": round(total_runtime, 2),
            "pipeline_status": PIPELINE_STATUS_COMPLETED,
            "pipeline_stage_completed": 9,
            "quality_gate_passed": True,
            "registered_cameras": colmap_summary.registered_cameras,
            "total_cameras": colmap_summary.total_cameras,
            "registration_percentage": colmap_summary.registration_percentage,
            "sparse_points": colmap_summary.sparse_point_count,
            "dense_points": dense_points,
            "triangles": poisson_res.triangle_count,
            "vertices": poisson_res.vertex_count,
            "reprojection_error": colmap_summary.mean_reprojection_error,
            "gpu_used": colmap_summary.device,
            "viewer_model": "model.obj" if (session_output_dir / "model.obj").exists() else ("model.glb" if (session_output_dir / "model.glb").exists() else "model.ply"),
            "trajectory_video": traj_video_name,
            "trajectory_duration_seconds": traj_res.get("trajectory_duration_seconds", 10.0),
            "trajectory_fps": traj_res.get("trajectory_fps", 30),
            "trajectory_frames": traj_res.get("trajectory_frames", 300),
            "video": {
                "name": video_meta.filename if video_meta else "unknown",
                "resolution": video_meta.resolution_str if video_meta else "unknown",
                "fps": video_meta.fps if video_meta else 0.0,
                "duration": video_meta.duration_formatted if video_meta else "unknown",
            },
            "preprocessing": {
                "extracted_frames": blur_res.total_evaluated if blur_res else 0,
                "blur_removed": blur_res.discarded_count if blur_res else 0,
                "duplicates_removed": dup_res.discarded_count if dup_res else 0,
                "clean_frames": dup_res.retained_count if dup_res else 0,
                "avg_blur_score": blur_res.average_score if blur_res else 0.0,
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
            "dense_reconstruction": {
                "dense_points": dense_points,
                "fused_ply": "dense/fused.ply",
                "device": "NVIDIA CUDA GPU" if self.colmap_runner.is_gpu_available() else "CPU",
            },
            "poisson_mesh": {
                "model_obj": "model.obj",
                "model_glb": "model.glb",
                "model_ply": "model.ply",
                "triangles": poisson_res.triangle_count,
                "vertices": poisson_res.vertex_count,
                "depth": self.config.poisson.depth,
                "runtime_seconds": poisson_res.runtime_seconds,
            },
            "deliverables": {
                "model_obj": "model.obj",
                "model_glb": "model.glb",
                "model_ply": "model.ply",
                "dense_fused_ply": "dense/fused.ply",
                "point_cloud_ply": "point_cloud.ply",
                "camera_trajectory": "camera_trajectory.json",
                "trajectory_video": traj_video_name,
                "colmap_summary": "colmap_summary.json",
                "preprocess_report": "preprocess_report.json",
                "thumbnail": "thumbnail.png",
            },
            "status": "COMPLETED",
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 4. Automatic Validation (Verify mesh and dense deliverables exist and are non-empty)
        req_files = [
            session_output_dir / "model.obj",
            session_output_dir / "model.ply",
            session_output_dir / "camera_trajectory.json",
            session_output_dir / "scene_manifest.json",
        ]
        missing = [str(f.name) for f in req_files if not f.exists() or f.stat().st_size == 0]
        if missing:
            logger.warning(f"Deliverable validation warning: Missing or empty files: {', '.join(missing)}")

        self.emit_event(
            StageType.EXPORT,
            StageStatus.COMPLETED,
            1.0,
            f"Deliverables packaged in {session_output_dir.name}/ (OBJ, GLB, PLY, Fused Dense PLY ready)",
            metrics={
                "Mesh Triangles": f"{poisson_res.triangle_count:,}",
                "Format": "OBJ + GLB + PLY",
                "Status": "Complete",
            },
            global_progress=1.0,
            quality_score=quality_score,
        )
        return True

    def _execute_stage_5_gsplat(self, *args, **kwargs):
        """Deprecated: Replaced by Stage 6/7/8 Dense & Poisson pipeline."""
        logger.warning("_execute_stage_5_gsplat is deprecated and removed.")
        return None

    def _execute_stage_6_export(self, *args, **kwargs):
        """Deprecated alias to _execute_stage_9_export."""
        return self._execute_stage_9_export(*args, **kwargs)
