"""
GeoRecon AI - Real Cinematic Trajectory Renderer Module
SIH-26158: Drone & Mobile Video 3D Reconstruction Platform
Renders high-fidelity cinematic fly-through videos following reconstructed COLMAP camera trajectories.
Performs Catmull-Rom/Cubic Spline position interpolation and Quaternion SLERP orientation smoothing.
Uses Open3D offscreen hardware acceleration with #0F121C Dark Studio theme and encodes 1920x1080 30FPS MP4.
"""

import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("GeoRecon.Trajectory")


class TrajectoryRenderer:
    """Renders 8–12 second smooth cinematic trajectory fly-throughs from real COLMAP poses."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        target_duration_seconds: float = 10.0,
        point_size: float = 4.0,
        background_color: Tuple[float, float, float] = (15 / 255.0, 18 / 255.0, 28 / 255.0),  # #0F121C
        fov_degrees: float = 60.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.target_duration_seconds = target_duration_seconds
        self.total_frames = int(target_duration_seconds * fps)  # Default: 300 frames (10.0s)
        self.point_size = point_size
        self.background_color = np.array(background_color, dtype=np.float64)
        self.fov_degrees = fov_degrees

    @staticmethod
    def _natural_sort_key(filename: str):
        """Natural alphanumeric sort key for frame filenames (e.g. frame_1, frame_2, frame_10)."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]

    def interpolate_camera_trajectory(
        self,
        raw_poses: List[Dict],
        total_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Interpolates raw COLMAP camera poses using Cubic Spline for 3D positions and Quaternion SLERP for rotations.
        Applies smooth cosine ease-in/ease-out to remove sudden jumps and create cinematic motion.
        Returns a list of 4x4 extrinsic matrices (world-to-camera).
        """
        if not raw_poses:
            raise ValueError("No camera poses provided for trajectory interpolation.")

        if total_frames is None:
            total_frames = self.total_frames

        # 1. Sort poses chronologically by image name
        poses = sorted(raw_poses, key=lambda p: self._natural_sort_key(p.get("image", "")))

        # 2. Extract camera centers in world coordinates C = -R_cw^T * t_cw and rotations
        centers = []
        rot_matrices = []
        quats = []  # [x, y, z, w] for scipy

        for pose in poses:
            qw, qx, qy, qz = pose["rotation"]
            tx, ty, tz = pose["translation"]
            t_cw = np.array([tx, ty, tz], dtype=np.float64)

            # R_cw rotation matrix (world to camera)
            r_cw = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # Camera center in world coordinates: C = -R_cw^T * t_cw
            c_world = -r_cw.T @ t_cw

            # World-to-camera rotation in world coordinates: R_wc = R_cw^T
            r_wc = r_cw.T
            q_wc = R.from_matrix(r_wc).as_quat()  # [x, y, z, w]

            # Ensure quaternion hemisphere continuity: q_i . q_{i-1} >= 0
            if quats:
                dot_prod = np.dot(q_wc, quats[-1])
                if dot_prod < 0.0:
                    q_wc = -q_wc

            centers.append(c_world)
            rot_matrices.append(r_wc)
            quats.append(q_wc)

        centers = np.array(centers, dtype=np.float64)
        num_keyframes = len(centers)
        logger.info(f"Interpolating {num_keyframes} keyframe camera poses into {total_frames} cinematic frames (30 FPS)...")

        # Edge case: Very few keyframes
        if num_keyframes == 1:
            ext = np.eye(4)
            ext[:3, :3] = R.from_quat(quats[0]).as_matrix().T
            ext[:3, 3] = -ext[:3, :3] @ centers[0]
            return [ext for _ in range(total_frames)]

        # 3. Setup keyframe timestamps t_k in [0.0, 1.0]
        t_keyframes = np.linspace(0.0, 1.0, num_keyframes)

        # 4. Position interpolation with Natural Cubic Spline
        spline_x = CubicSpline(t_keyframes, centers[:, 0], bc_type="natural")
        spline_y = CubicSpline(t_keyframes, centers[:, 1], bc_type="natural")
        spline_z = CubicSpline(t_keyframes, centers[:, 2], bc_type="natural")

        # 5. Rotation interpolation with Quaternion SLERP
        rot_objects = R.from_quat(quats)
        slerp = Slerp(t_keyframes, rot_objects)

        # 6. Generate smooth query times with subtle ease-in / ease-out
        raw_u = np.linspace(0.0, 1.0, total_frames)
        # Smooth cosine easing: u_eased = (1 - cos(pi * u)) / 2 * 0.15 + u * 0.85 (balanced smooth progression)
        query_times = 0.5 * (1.0 - np.cos(np.pi * raw_u)) * 0.20 + raw_u * 0.80
        query_times = np.clip(query_times, 0.0, 1.0)

        # 7. Evaluate interpolated poses
        interp_x = spline_x(query_times)
        interp_y = spline_y(query_times)
        interp_z = spline_z(query_times)
        interp_centers = np.column_stack([interp_x, interp_y, interp_z])

        interp_rots = slerp(query_times)
        interp_rot_matrices = interp_rots.as_matrix()

        # 8. Convert back to Open3D Extrinsic matrices E = [R_cw | t_cw]
        extrinsics = []
        for i in range(total_frames):
            c_world_i = interp_centers[i]
            r_wc_i = interp_rot_matrices[i]

            # R_cw = R_wc^T
            r_cw_i = r_wc_i.T
            # t_cw = -R_cw * C
            t_cw_i = -r_cw_i @ c_world_i

            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :3] = r_cw_i
            ext_4x4[:3, 3] = t_cw_i
            extrinsics.append(ext_4x4)

        return extrinsics

    def load_geometry(self, model_path: Path) -> o3d.geometry.Geometry3D:
        """Loads PointCloud or TriangleMesh from PLY, OBJ, GLB, or NPZ into Open3D."""
        ext = model_path.suffix.lower()

        if ext == ".npz":
            data = np.load(model_path)
            pts = data.get("positions")
            if pts is None:
                pts = data.get("points")
            if pts is None:
                pts = data.get("means")
            if pts is None:
                raise ValueError("NPZ file does not contain positions array.")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

            if "colors" in data:
                cols = data["colors"]
                if cols.max() > 1.0:
                    cols = cols / 255.0
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            elif "sh_coefficients" in data or "sh" in data:
                sh_dc = data.get("sh_coefficients", data.get("sh"))
                c0 = 0.28209479177387814
                if sh_dc.ndim >= 2:
                    rgb = np.clip((sh_dc[:, :3] if sh_dc.shape[1] >= 3 else sh_dc) * c0 + 0.5, 0.0, 1.0)
                    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            return pcd

        elif ext in (".ply", ".pcd", ".xyz"):
            pcd = o3d.io.read_point_cloud(str(model_path))
            if len(pcd.points) > 0:
                return pcd
            mesh = o3d.io.read_triangle_mesh(str(model_path))
            if len(mesh.vertices) > 0:
                mesh.compute_vertex_normals()
                return mesh

        elif ext in (".obj", ".glb", ".gltf", ".stl"):
            mesh = o3d.io.read_triangle_mesh(str(model_path))
            if len(mesh.vertices) > 0:
                mesh.compute_vertex_normals()
                return mesh
            pcd = o3d.io.read_point_cloud(str(model_path))
            if len(pcd.points) > 0:
                return pcd

        raise ValueError(f"Unable to load valid geometry from {model_path}")

    def render_trajectory_video(
        self,
        model_path: Path,
        trajectory_json_path: Path,
        output_video_path: Path,
        frames_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict:
        """
        Renders the full camera trajectory into individual PNG frames and encodes an MP4 video.
        
        Args:
            model_path: Path to reconstructed point_cloud.ply or model.obj/glb/npz
            trajectory_json_path: Path to camera_trajectory.json
            output_video_path: Destination path for trajectory_preview.mp4
            frames_dir: Optional directory to save frame_0000.png frames
            on_progress: Callback signature (current_frame, total_frames, eta_seconds, percent)
        """
        model_path = Path(model_path).resolve()
        trajectory_json_path = Path(trajectory_json_path).resolve()
        output_video_path = Path(output_video_path).resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Reconstructed model not found at {model_path}")
        if not trajectory_json_path.exists():
            raise FileNotFoundError(f"Camera trajectory JSON not found at {trajectory_json_path}")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        if frames_dir is None:
            frames_dir = output_video_path.parent / "trajectory_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Camera Trajectory JSON
        with open(trajectory_json_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)

        raw_poses = traj_data.get("camera_trajectory", [])
        if not raw_poses:
            raise ValueError(f"No poses found in {trajectory_json_path}")

        # 2. Interpolate Trajectory into Smooth Extrinsic Matrices (300 frames)
        extrinsics = self.interpolate_camera_trajectory(raw_poses, total_frames=self.total_frames)
        total_frames = len(extrinsics)

        # 3. Load 3D Geometry
        logger.info(f"Loading 3D scene geometry from {model_path.name} for trajectory rendering...")
        geometry = self.load_geometry(model_path)

        # 4. Initialize Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        # Create hidden window for offscreen rendering
        vis.create_window(
            window_name="TerraSweep Trajectory Offscreen Renderer",
            width=self.width,
            height=self.height,
            visible=False,
        )
        vis.add_geometry(geometry)

        # Set Dark Studio Theme & Rendering Options
        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.background_color = self.background_color
            render_opt.point_size = self.point_size
            render_opt.show_coordinate_frame = False

        # Compute focal length from FOV: fx = fy = width / (2 * tan(fov / 2))
        fov_rad = np.deg2rad(self.fov_degrees)
        focal_length = (self.width / 2.0) / np.tan(fov_rad / 2.0)
        cx = self.width / 2.0
        cy = self.height / 2.0

        view_ctrl = vis.get_view_control()

        # 5. Initialize OpenCV VideoWriter
        # FourCC codec priority: mp4v -> avc1 -> H264
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            float(self.fps),
            (self.width, self.height),
            isColor=True,
        )

        if not video_writer.isOpened():
            # Fallback to alternative fourcc if mp4v failed
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                float(self.fps),
                (self.width, self.height),
                isColor=True,
            )

        render_start_t = time.time()
        logger.info(f"Starting cinematic trajectory rendering: {total_frames} frames @ {self.fps} FPS ({self.width}x{self.height})...")

        # 6. Render Frame-by-Frame
        for frame_idx, ext_mat in enumerate(extrinsics):
            t_frame_start = time.time()

            # Set Camera Pinhole Parameters
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.width, self.height, focal_length, focal_length, cx, cy
            )
            cam_params.extrinsic = ext_mat
            view_ctrl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()

            # Capture Frame Buffer
            img_buffer = vis.capture_screen_float_buffer(do_render=True)
            img_rgb = (np.asarray(img_buffer) * 255.0).astype(np.uint8)

            # Ensure exact target dimensions (handling OS window decoration differences)
            if img_rgb.shape[0] != self.height or img_rgb.shape[1] != self.width:
                img_rgb = cv2.resize(img_rgb, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Save frame to disk: trajectory_frames/frame_0000.png
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(frame_path), img_bgr)

            # Write to MP4 Video
            video_writer.write(img_bgr)

            # Telemetry calculation
            elapsed = time.time() - render_start_t
            frames_done = frame_idx + 1
            fps_speed = frames_done / max(0.001, elapsed)
            remaining_frames = total_frames - frames_done
            eta_seconds = remaining_frames / max(0.001, fps_speed)
            percent = (frames_done / total_frames) * 100.0

            if on_progress:
                on_progress(frames_done, total_frames, eta_seconds, percent)

            if frames_done % 30 == 0 or frames_done == total_frames:
                logger.info(
                    f"[Trajectory Render] Frame {frames_done:03d}/{total_frames} ({percent:.1f}%) | "
                    f"Speed: {fps_speed:.1f} fps | ETA: {eta_seconds:.1f}s"
                )

        # Cleanup Visualizer & Video Writer
        video_writer.release()
        vis.destroy_window()

        total_duration = time.time() - render_start_t
        video_duration_seconds = total_frames / float(self.fps)

        logger.info(
            f"Cinematic trajectory preview rendered successfully in {total_duration:.1f}s -> "
            f"{output_video_path.name} ({video_duration_seconds:.1f}s, {output_video_path.stat().st_size / (1024*1024):.2f} MB)"
        )

        return {
            "status": "SUCCESS",
            "trajectory_video": output_video_path.name,
            "trajectory_duration_seconds": round(video_duration_seconds, 2),
            "trajectory_fps": self.fps,
            "trajectory_frames": total_frames,
            "video_path": str(output_video_path),
            "render_time_seconds": round(total_duration, 2),
        }


def render_session_trajectory(
    session_dir: Path,
    on_progress: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict:
    """Convenience helper to render trajectory preview for a given session directory."""
    session_dir = Path(session_dir).resolve()
    model_path = session_dir / "point_cloud.ply"
    if not model_path.exists():
        model_path = session_dir / "model.obj"
    if not model_path.exists():
        model_path = session_dir / "model.glb"
    if not model_path.exists():
        model_path = session_dir / "checkpoints" / "gaussians_model.npz"

    traj_json = session_dir / "camera_trajectory.json"
    output_mp4 = session_dir / "trajectory_preview.mp4"

    renderer = TrajectoryRenderer(width=1920, height=1080, fps=30, target_duration_seconds=10.0)
    return renderer.render_trajectory_video(
        model_path=model_path,
        trajectory_json_path=traj_json,
        output_video_path=output_mp4,
        on_progress=on_progress,
    )
