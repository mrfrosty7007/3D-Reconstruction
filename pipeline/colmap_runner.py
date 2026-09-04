"""
GeoRecon AI - COLMAP Runner Module
Executes real COLMAP Structure-from-Motion (SfM) via subprocess with live streaming,
GPU acceleration, model conversion (BIN -> TXT), and summary generation.
"""

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
from typing import Callable, Dict, Any, List, Optional, Tuple

from config import ColmapConfig

logger = logging.getLogger("GeoRecon.COLMAP")


@dataclass
class ColmapSummary:
    """Summary of COLMAP Structure-from-Motion results."""
    database_path: str
    sparse_model_path: str
    total_cameras: int = 0
    registered_cameras: int = 0
    registration_percentage: float = 0.0
    sparse_point_count: int = 0
    mean_reprojection_error: Optional[float] = None
    device: str = "CUDA GPU"
    colmap_version: str = "4.1"
    runtime_seconds: float = 0.0
    is_valid: bool = False
    error_message: Optional[str] = None


class ColmapRunner:
    """Orchestrates genuine COLMAP SfM feature extraction, matching, mapping, and export."""

    def __init__(self, config: Optional[ColmapConfig] = None):
        self.config = config or ColmapConfig()
        self.resolved_executable = self._find_executable()
        self._active_process: Optional[subprocess.Popen] = None
        self.exit_codes: Dict[str, int] = {}
        self.last_exit_code: int = 0
        self.is_cancelled: bool = False

    def get_gpu_name(self) -> str:
        """Retrieves hardware GPU name if available."""
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip().splitlines()[0]
            except Exception:
                pass
        return "Unknown GPU" if self.is_gpu_available() else "CPU Fallback"

    def get_optimal_thread_count(self) -> int:
        """Returns bounded CPU thread count to prevent laptop CPU thermal throttling and thread starvation."""
        return min(8, max(2, (os.cpu_count() or 4) - 1))

    def terminate_active_process(self) -> None:
        """Immediately and safely terminates the running COLMAP subprocess and all child processes."""
        proc = self._active_process
        if proc and proc.poll() is None:
            pid = getattr(proc, "pid", None)
            logger.warning(f"Terminating COLMAP process tree (PID: {pid})...")
            if os.name == "nt":
                # 1. Graceful signal (CTRL_BREAK_EVENT)
                try:
                    if hasattr(signal, "CTRL_BREAK_EVENT"):
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    pass
                # 2. Standard terminate
                try:
                    proc.terminate()
                except Exception:
                    pass
                # 3. Wait briefly for clean exit
                try:
                    proc.wait(timeout=0.8)
                    logger.info(f"COLMAP process {pid} exited gracefully.")
                    self._active_process = None
                    return
                except (subprocess.TimeoutExpired, Exception):
                    pass
                # 4. Force kill entire process tree via Windows taskkill if pid is available
                if pid:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            timeout=3,
                            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                        )
                        logger.info(f"Terminated COLMAP process tree {pid} via taskkill.")
                    except Exception as e:
                        logger.debug(f"taskkill fallback: {e}")
                # 5. Final fallback
                try:
                    proc.kill()
                except Exception:
                    pass
            else:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._active_process = None

    def _find_executable(self) -> str:
        """Finds the absolute path or command for the COLMAP executable."""
        # Check configured path or environment variable
        env_colmap = os.environ.get("COLMAP_PATH")
        if env_colmap and Path(env_colmap).is_file():
            return env_colmap

        # Check common Windows install locations
        windows_paths = [
            r"C:\COLMAP\colmap-x64-windows-cuda\bin\colmap.exe",
            r"C:\COLMAP\colmap-x64-windows-cuda\colmap.exe",
            r"C:\Program Files\COLMAP\colmap.exe",
            r"C:\COLMAP\colmap.exe",
        ]
        for p in windows_paths:
            if Path(p).is_file():
                return p

        # Check system PATH
        which_path = shutil.which(self.config.executable_path)
        if which_path:
            return which_path

        which_colmap = shutil.which("colmap")
        if which_colmap:
            return which_colmap

        return self.config.executable_path

    def check_environment(self) -> Tuple[bool, str]:
        """Verifies COLMAP executable and retrieves version info."""
        try:
            cmd = [self.resolved_executable, "help"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            version_line = res.stdout.splitlines()[0] if res.stdout else "COLMAP Available"
            logger.info(f"COLMAP detected: {version_line} ({self.resolved_executable})")
            return True, version_line
        except Exception as e:
            logger.warning(f"COLMAP binary check failed for '{self.resolved_executable}': {e}")
            return False, str(e)

    def is_gpu_available(self) -> bool:
        """Determines if NVIDIA CUDA GPU is available for COLMAP SIFT."""
        if hasattr(self, "_is_cuda_supported"):
            return self._is_cuda_supported
        if not shutil.which("nvidia-smi"):
            self._is_cuda_supported = False
            return False
        has_colmap, ver_line = self.check_environment()
        if not has_colmap:
            self._is_cuda_supported = False
            return False
        # COLMAP banner displays 'with CUDA' when compiled with CUDA support
        if "CUDA" in ver_line.upper():
            self._is_cuda_supported = True
            return True
        # Probing fallback
        try:
            cmd = [self.resolved_executable, "feature_extractor", "--help"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self._is_cuda_supported = "use_gpu" in (res.stdout + res.stderr)
        except Exception:
            self._is_cuda_supported = False
        return self._is_cuda_supported

    def get_feature_extraction_gpu_flag_name(self) -> str:
        """Determines version-appropriate GPU flag for feature extraction."""
        if hasattr(self, "_feat_gpu_flag"):
            return self._feat_gpu_flag
        try:
            res = subprocess.run(
                [self.resolved_executable, "feature_extractor", "-h"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            help_text = res.stdout + res.stderr
            if "--SiftExtraction.use_gpu" in help_text:
                self._feat_gpu_flag = "--SiftExtraction.use_gpu"
            else:
                self._feat_gpu_flag = "--FeatureExtraction.use_gpu"
        except Exception:
            self._feat_gpu_flag = "--FeatureExtraction.use_gpu"
        return self._feat_gpu_flag

    def get_feature_matching_gpu_flag_name(self) -> str:
        """Determines version-appropriate GPU flag for feature matching."""
        if hasattr(self, "_match_gpu_flag"):
            return self._match_gpu_flag
        try:
            res = subprocess.run(
                [self.resolved_executable, "exhaustive_matcher", "-h"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            help_text = res.stdout + res.stderr
            if "--SiftMatching.use_gpu" in help_text:
                self._match_gpu_flag = "--SiftMatching.use_gpu"
            else:
                self._match_gpu_flag = "--FeatureMatching.use_gpu"
        except Exception:
            self._match_gpu_flag = "--FeatureMatching.use_gpu"
        return self._match_gpu_flag

    def verify_gpu_flags(self, on_log: Optional[Callable[[str], None]] = None) -> List[str]:
        """Startup verification logging for COLMAP GPU flags."""
        _, ver_str = self.check_environment()
        ver_match = re.search(r"COLMAP\s+([\d\.]+)", ver_str)
        ver_display = ver_match.group(1) if ver_match else "4.1.1"

        feat_gpu = "ENABLED" if self.is_gpu_available() else "DISABLED"
        match_gpu = "ENABLED" if self.is_gpu_available() else "DISABLED"

        msgs = [
            f"Verified COLMAP GPU flags for version {ver_display}",
            f"Feature Extraction GPU: {feat_gpu}",
            f"Feature Matching GPU: {match_gpu}",
        ]
        for m in msgs:
            logger.info(m)
            if on_log:
                on_log(f"[INFO] {m}")
        return msgs

    def _run_colmap_command(
        self,
        args: List[str],
        stage_name: str,
        on_log: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
        sparse_poll_dir: Optional[Path] = None,
        on_sparse_points: Optional[Callable[[int], None]] = None,
        on_reg_frames: Optional[Callable[[int], None]] = None,
    ) -> bool:
        """Executes a COLMAP command with live line-by-line stdout streaming and non-blocking watchdog."""
        cmd = [self.resolved_executable] + args
        cmd_str = " ".join(cmd)
        logger.info(f"[{stage_name}] Running: {cmd_str}")

        cmd_name = args[0]
        if on_log:
            on_log(f"[{stage_name}] Executing: {cmd[0]} {cmd_name}")

        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )
        self._active_process = process
        self.is_cancelled = False

        # Non-blocking reader thread continuously consumes the OS pipe so it NEVER blocks or deadlocks
        out_q: queue.Queue = queue.Queue()

        def _drain_stdout(pipe, q):
            try:
                for line in iter(pipe.readline, ""):
                    q.put(line)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        reader_thread = threading.Thread(
            target=_drain_stdout,
            args=(process.stdout, out_q),
            daemon=True,
            name=f"COLMAP-Drain-{cmd_name}",
        )
        reader_thread.start()

        last_disk_poll_t = 0.0

        try:
            while True:
                # 1. Instant check for cancellation
                if stop_event and stop_event.is_set():
                    self.is_cancelled = True
                    logger.warning(f"[{stage_name}] Cancellation received. Terminating COLMAP process tree...")
                    self.terminate_active_process()
                    self.exit_codes[cmd_name] = -1
                    self.last_exit_code = -1
                    return False

                # 2. Drain lines from queue
                had_lines = False
                while True:
                    try:
                        line = out_q.get_nowait()
                        had_lines = True
                        clean = line.strip()
                        if clean:
                            logger.info(f"[{stage_name}] {clean}")
                            if on_log:
                                on_log(clean)
                    except queue.Empty:
                        break

                # 3. Periodic disk inspection for live point cloud & registered camera metrics
                if sparse_poll_dir and (time.time() - last_disk_poll_t >= 1.2):
                    last_disk_poll_t = time.time()
                    try:
                        for cand in [sparse_poll_dir / "0", sparse_poll_dir]:
                            p_bin = cand / "points3D.bin"
                            if p_bin.exists() and p_bin.stat().st_size >= 8:
                                with open(p_bin, "rb") as f:
                                    p_cnt = struct.unpack("<Q", f.read(8))[0]
                                    if p_cnt > 0 and on_sparse_points:
                                        on_sparse_points(p_cnt)
                            i_bin = cand / "images.bin"
                            if i_bin.exists() and i_bin.stat().st_size >= 8:
                                with open(i_bin, "rb") as f:
                                    i_cnt = struct.unpack("<Q", f.read(8))[0]
                                    if i_cnt > 0 and on_reg_frames:
                                        on_reg_frames(i_cnt)
                    except Exception:
                        pass

                # 4. Check process exit
                rc = process.poll()
                if rc is not None:
                    # Final drain of any remaining lines
                    while not out_q.empty():
                        try:
                            line = out_q.get_nowait()
                            clean = line.strip()
                            if clean:
                                logger.info(f"[{stage_name}] {clean}")
                                if on_log:
                                    on_log(clean)
                        except queue.Empty:
                            break

                    self.exit_codes[cmd_name] = rc
                    self.last_exit_code = rc
                    if rc != 0:
                        logger.error(f"[{stage_name}] COLMAP exited with return code {rc}")
                        return False
                    return True

                if not had_lines:
                    time.sleep(0.02)

        except Exception as e:
            logger.exception(f"[{stage_name}] Error executing COLMAP: {e}")
            self.terminate_active_process()
            self.exit_codes[cmd_name] = -1
            self.last_exit_code = -1
            return False
        finally:
            self._active_process = None

    def run_feature_extraction(
        self,
        image_path: Path,
        database_path: Path,
        camera_model: str = "OPENCV",
        single_camera: bool = True,
        use_gpu: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Stage 1: Feature Extraction using CUDA accelerated SIFT."""
        database_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_enabled = use_gpu and self.is_gpu_available()
        gpu_flag = "1" if gpu_enabled else "0"

        gpu_name = self.get_gpu_name()
        if gpu_enabled:
            cuda_msg = f"CUDA SIFT enabled on {gpu_name}"
            logger.info(cuda_msg)
            if on_log:
                on_log(cuda_msg)

        opt_threads = str(min(8, max(2, (os.cpu_count() or 4) - 1)))
        feat_gpu_flag = self.get_feature_extraction_gpu_flag_name()
        args = [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            feat_gpu_flag, gpu_flag,
            "--FeatureExtraction.num_threads", opt_threads,
            "--SiftExtraction.max_num_features", str(self.config.max_num_features),
        ]
        return self._run_colmap_command(args, "COLMAP-Features", on_log, stop_event)

    def run_feature_matching(
        self,
        database_path: Path,
        matcher_type: str = "exhaustive",
        use_gpu: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Stage 2: Feature Matching using CUDA accelerated SIFT."""
        gpu_enabled = use_gpu and self.is_gpu_available()
        gpu_flag = "1" if gpu_enabled else "0"
        matcher_cmd = f"{matcher_type}_matcher" if matcher_type in ("exhaustive", "sequential", "spatial") else "exhaustive_matcher"

        gpu_name = self.get_gpu_name()
        if gpu_enabled:
            cuda_msg = f"CUDA SIFT Matching enabled on {gpu_name}"
            logger.info(cuda_msg)
            if on_log:
                on_log(cuda_msg)

        opt_threads = str(min(8, max(2, (os.cpu_count() or 4) - 1)))
        match_gpu_flag = self.get_feature_matching_gpu_flag_name()
        args = [
            matcher_cmd,
            "--database_path", str(database_path),
            match_gpu_flag, gpu_flag,
            "--FeatureMatching.num_threads", opt_threads,
        ]
        return self._run_colmap_command(args, "COLMAP-Matching", on_log, stop_event)

    def run_mapper(
        self,
        image_path: Path,
        database_path: Path,
        output_sparse_dir: Path,
        on_log: Optional[Callable[[str], None]] = None,
        on_reg_frames: Optional[Callable[[int], None]] = None,
        on_sparse_points: Optional[Callable[[int], None]] = None,
        on_ba_event: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Stage 3: Sparse Reconstruction (Incremental Mapper) with tuned parameters and live metrics parsing."""
        output_sparse_dir.mkdir(parents=True, exist_ok=True)

        opt_threads = str(min(8, max(2, (os.cpu_count() or 4) - 1)))
        args = [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_sparse_dir),
            # Bundle Adjustment Optimization (drastically reduces SfM runtime)
            "--Mapper.ba_global_frames_ratio", "1.35",
            "--Mapper.ba_global_points_ratio", "1.35",
            "--Mapper.ba_global_max_num_iterations", "25",
            "--Mapper.ba_local_max_num_iterations", "15",
            "--Mapper.ba_global_max_refinements", "2",
            "--Mapper.ba_local_max_refinements", "2",
            # Initialization & inlier tuning (fixes "Could not register, trying another image")
            "--Mapper.min_model_size", "3",
            "--Mapper.init_min_tri_angle", "4.0",
            "--Mapper.init_min_num_inliers", "40",
            "--Mapper.init_max_error", "4.0",
            "--Mapper.init_num_trials", "300",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
            "--Mapper.max_reg_trials", "3",
            "--Mapper.max_num_models", "3",
            "--Mapper.num_threads", opt_threads,
        ]

        def _log_with_reg_detection(line: str):
            if on_log:
                on_log(line)

            # 1. Detect live registration progress
            match_cams = (
                re.search(r"num_reg_frames=(\d+)", line)
                or re.search(r"num_reg_images=(\d+)", line)
                or re.search(r"Registering image #(\d+)", line)
                or re.search(r"Registered image #(\d+)", line)
            )
            if match_cams and on_reg_frames:
                try:
                    reg_cams = int(match_cams.group(1))
                    on_reg_frames(reg_cams)
                except ValueError:
                    pass

            # 2. Detect live sparse tie points
            match_pts = (
                re.search(r"num_reg_points=(\d+)", line)
                or re.search(r"num_points3D=(\d+)", line)
                or re.search(r"Triangulated (\d+) points", line)
                or re.search(r"Image registered with (\d+) points", line)
            )
            if match_pts and on_sparse_points:
                try:
                    pts = int(match_pts.group(1))
                    on_sparse_points(pts)
                except ValueError:
                    pass

            # 3. Detect Bundle Adjustment notifications
            if "Bundle adjustment" in line or "bundle adjustment" in line:
                if on_ba_event:
                    on_ba_event(line)

        return self._run_colmap_command(
            args=args,
            stage_name="COLMAP-Mapper",
            on_log=_log_with_reg_detection,
            stop_event=stop_event,
            sparse_poll_dir=output_sparse_dir,
            on_sparse_points=on_sparse_points,
            on_reg_frames=on_reg_frames,
        )

    def run_model_converter(
        self,
        input_model_dir: Path,
        output_txt_dir: Path,
        on_log: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> bool:
        """Stage 4: Converts binary COLMAP model (BIN) to readable text format (TXT)."""
        output_txt_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "model_converter",
            "--input_path", str(input_model_dir),
            "--output_path", str(output_txt_dir),
            "--output_type", "TXT",
        ]
        return self._run_colmap_command(args, "COLMAP-ModelConverter", on_log, stop_event)

    @staticmethod
    def find_best_model_dir(sparse_dir: Path) -> Tuple[Path, int, int]:
        """
        Scans all reconstructed model folders in sparse_dir (e.g. sparse/0, sparse/1, sparse/2)
        and selects the candidate with the highest number of registered images and 3D points.
        Returns (best_model_dir, num_registered_images, num_points3D).
        """
        candidate_dirs = []
        if (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists():
            candidate_dirs.append(sparse_dir)

        if sparse_dir.exists():
            for sub in sparse_dir.iterdir():
                if sub.is_dir() and ((sub / "images.bin").exists() or (sub / "images.txt").exists()):
                    candidate_dirs.append(sub)

        best_dir = sparse_dir / "0" if (sparse_dir / "0").exists() else sparse_dir
        best_imgs = 0
        best_pts = 0

        for c_dir in candidate_dirs:
            num_imgs = 0
            num_pts = 0

            # 1. Read binary header (exact uint64 count)
            img_bin = c_dir / "images.bin"
            if img_bin.exists():
                try:
                    with open(img_bin, "rb") as f:
                        num_imgs = struct.unpack("<Q", f.read(8))[0]
                except Exception:
                    pass

            pts_bin = c_dir / "points3D.bin"
            if pts_bin.exists():
                try:
                    with open(pts_bin, "rb") as f:
                        num_pts = struct.unpack("<Q", f.read(8))[0]
                except Exception:
                    pass

            # 2. If binary header is 0, check text format
            if num_imgs == 0:
                txt_images = c_dir / "txt" / "images.txt" if (c_dir / "txt" / "images.txt").exists() else c_dir / "images.txt"
                if txt_images.exists():
                    with open(txt_images, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("# Number of images:"):
                                m = re.search(r"Number of images:\s*(\d+)", line)
                                if m:
                                    num_imgs = int(m.group(1))
                                    break
                            elif line and not line.startswith("#"):
                                parts = line.split()
                                if len(parts) >= 9 and ("." in parts[-1] or parts[-1].endswith(".png") or parts[-1].endswith(".jpg")):
                                    num_imgs += 1

            if num_pts == 0:
                txt_pts = c_dir / "txt" / "points3D.txt" if (c_dir / "txt" / "points3D.txt").exists() else c_dir / "points3D.txt"
                if txt_pts.exists():
                    with open(txt_pts, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("# Number of points:"):
                                m = re.search(r"Number of points:\s*(\d+)", line)
                                if m:
                                    num_pts = int(m.group(1))
                                    break
                            elif line and not line.startswith("#"):
                                num_pts += 1

            logger.info(f"COLMAP Model Candidate [{c_dir.name}]: {num_imgs} registered images, {num_pts} 3D points")
            if num_imgs > best_imgs or (num_imgs == best_imgs and num_pts > best_pts):
                best_imgs = num_imgs
                best_pts = num_pts
                best_dir = c_dir

        logger.info(f"Selected Best COLMAP Model: {best_dir} ({best_imgs} images, {best_pts} points)")
        return best_dir, best_imgs, best_pts

    def parse_reconstruction_results(
        self,
        sparse_dir: Path,
        total_input_images: int,
        database_path: Path,
        runtime_seconds: float,
        device_used: str,
    ) -> ColmapSummary:
        """Parses COLMAP reconstruction metrics by locating the best model and unpacking binary/text data."""
        best_model_dir, best_imgs, best_pts = self.find_best_model_dir(sparse_dir)

        summary = ColmapSummary(
            database_path=str(database_path),
            sparse_model_path=str(best_model_dir),
            total_cameras=total_input_images,
            registered_cameras=best_imgs,
            sparse_point_count=best_pts,
            device=device_used,
            runtime_seconds=round(runtime_seconds, 2),
        )

        # Check required files in best model directory
        bin_cameras = best_model_dir / "cameras.bin"
        bin_images = best_model_dir / "images.bin"
        bin_points = best_model_dir / "points3D.bin"

        txt_cameras = best_model_dir / "txt" / "cameras.txt" if (best_model_dir / "txt" / "cameras.txt").exists() else best_model_dir / "cameras.txt"
        txt_images = best_model_dir / "txt" / "images.txt" if (best_model_dir / "txt" / "images.txt").exists() else best_model_dir / "images.txt"
        txt_points = best_model_dir / "txt" / "points3D.txt" if (best_model_dir / "txt" / "points3D.txt").exists() else best_model_dir / "points3D.txt"

        has_bin = bin_cameras.exists() and bin_images.exists() and bin_points.exists()
        has_txt = txt_cameras.exists() and txt_images.exists() and txt_points.exists()

        if not database_path.exists() or (not has_bin and not has_txt):
            summary.is_valid = False
            summary.error_message = "COLMAP sparse model files or database.db missing."
            return summary

        # Compute Mean Reprojection Error from points3D.txt if available
        total_error = 0.0
        parsed_pts = 0
        if txt_points.exists():
            with open(txt_points, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 8:
                            parsed_pts += 1
                            try:
                                total_error += float(parts[7])
                            except ValueError:
                                pass
            if parsed_pts > 0:
                summary.mean_reprojection_error = round(total_error / parsed_pts, 3)
            else:
                summary.mean_reprojection_error = None
        else:
            summary.mean_reprojection_error = None

        # Compute accurate registration percentage
        if total_input_images > 0:
            summary.registration_percentage = round((summary.registered_cameras / total_input_images) * 100, 1)
        else:
            summary.registration_percentage = 100.0 if summary.registered_cameras > 0 else 0.0

        summary.is_valid = (summary.registered_cameras > 0 or has_bin)
        return summary
