"""
TerraSweep - Interactive 3D Open3D, Trimesh & CloudCompare Viewer Module
SIH-26158: Drone & Mobile Video 3D Reconstruction Platform
Provides hardware-accelerated 3D point cloud, mesh, and Gaussian visualization with orbit,
pan, zoom, point size controls, camera resets, screenshot capture, and multi-engine fallback.
Guarantees persistent window lifecycles without premature process termination.
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

logger = logging.getLogger("GeoRecon.Viewer")


class Model3DViewer:
    """Launches and manages the interactive 3D reconstruction viewport in an isolated process."""

    @staticmethod
    def launch_viewer_process(
        model_path: Path,
        window_name: str = "TerraSweep — 3D Reconstruction Viewer"
    ) -> Optional[subprocess.Popen]:
        """
        Launches the 3D viewer in an independent non-blocking process so the main Studio GUI never freezes.
        Supports .ply, .obj, .glb, and .npz files.
        Redirects output to viewer_log.txt to ensure the process remains responsive and never blocks on pipes.
        """
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            logger.error(f"Cannot launch 3D viewer: Model file not found at {model_path}")
            return None

        # Determine session directory to store viewer_log.txt
        session_dir = model_path.parent
        if session_dir.name == "checkpoints":
            session_dir = session_dir.parent
        viewer_log_path = session_dir / "viewer_log.txt"

        viewer_script = Path(__file__).resolve()
        cmd = [
            sys.executable,
            str(viewer_script),
            "--model", str(model_path),
            "--title", window_name,
            "--log-file", str(viewer_log_path.resolve()),
        ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        logger.info(f"Spawning 3D viewport process for: {model_path.name}")
        try:
            log_file = open(viewer_log_path, "w", encoding="utf-8", errors="replace", buffering=1)
            # Write initial launch header
            log_file.write(f"=== TerraSweep 3D Viewer Launch Log ===\n")
            log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Model Path: {model_path}\n")
            log_file.write(f"Window Title: {window_name}\n\n")
            log_file.flush()

            # Note: Do NOT use CREATE_NO_WINDOW so OpenGL/GLFW GUI window creates properly with full context
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                close_fds=(os.name != "nt"),
            )
            return process
        except Exception as e:
            logger.exception(f"Failed to spawn 3D viewer process: {e}")
            return None


def _write_viewer_log(log_path: Optional[Path], msg: str):
    """Appends messages to viewer_log.txt and prints to stdout safely."""
    try:
        print(msg, flush=True)
    except Exception:
        try:
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            pass

    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def _load_geometry_data(model_path: Path) -> Tuple[str, any, any, int]:
    """
    Loads 3D geometry points and colors from PLY, OBJ, GLB, or NPZ.
    Returns: (geometry_type, data, colors, vertex_count)
    """
    ext = model_path.suffix.lower()

    if ext == ".splat":
        content = model_path.read_bytes()
        num_splats = len(content) // 32
        if num_splats == 0:
            raise ValueError("Splat file is empty.")
        dtype = np.dtype([
            ("pos", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("color", "u1", (4,)),
            ("rot", "u1", (4,)),
        ])
        arr = np.frombuffer(content[:num_splats * 32], dtype=dtype)
        pts = arr["pos"]
        cols = arr["color"][:, :3].astype(np.float64) / 255.0
        return "splat_points", pts, cols, num_splats

    if ext == ".npz":
        data = np.load(model_path)
        pts = data.get("positions")
        if pts is None:
            pts = data.get("points")
        if pts is None:
            pts = data.get("means")
        if pts is None:
            raise ValueError("NPZ file does not contain positions/points/means array.")

        cols = None
        if "colors" in data:
            cols = data["colors"]
        elif "sh_coefficients" in data or "sh" in data:
            sh_dc = data.get("sh_coefficients", data.get("sh"))
            c0 = 0.28209479177387814
            if sh_dc.ndim >= 2:
                rgb = (sh_dc[:, :3] if sh_dc.shape[1] >= 3 else sh_dc) * c0 + 0.5
                cols = np.clip(rgb, 0.0, 1.0)

        num_verts = len(pts)
        return "npz_points", pts, cols, num_verts

    return "file", str(model_path), None, 0


def run_open3d_viewer(model_path: str, title: str = "TerraSweep — 3D Viewer", log_path: Optional[Path] = None) -> bool:
    """
    Priority 1: Interactive Open3D Visualizer.
    Keeps the visualizer alive with a blocking rendering event loop until closed by user.
    """
    try:
        import open3d as o3d

        path = Path(model_path).resolve()
        if not path.exists():
            _write_viewer_log(log_path, f"[Viewer] Error: Model not found at {model_path}")
            return False

        _write_viewer_log(log_path, f"[Viewer] Loading: {path}")
        _write_viewer_log(log_path, f"[Viewer] Backend: Open3D")

        geom_type, raw_data, cols, npz_verts = _load_geometry_data(path)
        geometry = None
        num_vertices = 0

        if geom_type in ("npz_points", "splat_points"):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_data.astype(np.float64))
            if cols is not None:
                if cols.max() > 1.0:
                    cols = cols / 255.0
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            geometry = pcd
            num_vertices = len(pcd.points)

        elif path.suffix.lower() in (".ply", ".pcd", ".xyz"):
            pcd = o3d.io.read_point_cloud(str(path))
            if len(pcd.points) > 0:
                geometry = pcd
                num_vertices = len(pcd.points)
            else:
                mesh = o3d.io.read_triangle_mesh(str(path))
                if len(mesh.vertices) > 0:
                    mesh.compute_vertex_normals()
                    geometry = mesh
                    num_vertices = len(mesh.vertices)

        elif path.suffix.lower() in (".obj", ".glb", ".gltf", ".stl", ".fbx"):
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.vertices) > 0:
                mesh.compute_vertex_normals()
                geometry = mesh
                num_vertices = len(mesh.vertices)
            else:
                pcd = o3d.io.read_point_cloud(str(path))
                if len(pcd.points) > 0:
                    geometry = pcd
                    num_vertices = len(pcd.points)

        # Fallback direct read with plyfile/trimesh if Open3D parser yielded 0 vertices
        if geometry is None or num_vertices == 0:
            try:
                import trimesh
                t_mesh = trimesh.load(str(path))
                if isinstance(t_mesh, trimesh.Scene):
                    dumped = t_mesh.dump()
                    all_pts = []
                    all_cols = []
                    for m in (dumped if isinstance(dumped, list) else [dumped]):
                        if hasattr(m, "vertices") and len(m.vertices) > 0:
                            all_pts.append(np.asarray(m.vertices, dtype=np.float64))
                            if hasattr(m, "colors") and m.colors is not None and len(m.colors) == len(m.vertices):
                                all_cols.append(np.asarray(m.colors[:, :3], dtype=np.float64) / 255.0)
                            elif hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None and len(m.visual.vertex_colors) == len(m.vertices):
                                all_cols.append(np.asarray(m.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0)
                    if all_pts:
                        pts = np.vstack(all_pts)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        if all_cols:
                            pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
                        geometry = pcd
                        num_vertices = len(pts)
                elif hasattr(t_mesh, "vertices") and len(t_mesh.vertices) > 0:
                    pts = np.asarray(t_mesh.vertices, dtype=np.float64)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if hasattr(t_mesh, "colors") and t_mesh.colors is not None and len(t_mesh.colors) == len(pts):
                        pcd.colors = o3d.utility.Vector3dVector(np.asarray(t_mesh.colors[:, :3], dtype=np.float64) / 255.0)
                    elif hasattr(t_mesh.visual, "vertex_colors") and t_mesh.visual.vertex_colors is not None and len(t_mesh.visual.vertex_colors) == len(pts):
                        pcd.colors = o3d.utility.Vector3dVector(np.asarray(t_mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0)
                    geometry = pcd
                    num_vertices = len(pts)
            except Exception as e_tri:
                _write_viewer_log(log_path, f"[Viewer] Open3D fallback parser note: {e_tri}")

        if geometry is None or num_vertices == 0:
            raise ValueError(f"Model file '{path.name}' contains 0 vertices or is invalid.")

        _write_viewer_log(log_path, f"[Viewer] Vertices: {num_vertices}")
        _write_viewer_log(log_path, f"Vertices loaded: {num_vertices}")
        _write_viewer_log(log_path, f"[Viewer] Status: SUCCESS")

        # Build Open3D Visualizer Window
        vis = o3d.visualization.VisualizerWithKeyCallback()
        win_title = f"{title} [{path.name} — {num_vertices:,} Vertices]"
        created = vis.create_window(window_name=win_title, width=1280, height=800, visible=True)
        if not created:
            # Fallback to standard draw_geometries if custom window creation failed
            _write_viewer_log(log_path, "[Viewer] VisualizerWithKeyCallback window creation fallback -> draw_geometries")
            o3d.visualization.draw_geometries(
                [geometry],
                window_name=win_title,
                width=1280,
                height=800,
                point_show_normal=False,
            )
            return True

        vis.add_geometry(geometry)
        vis.reset_view_point(True)

        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.background_color = np.array([0.06, 0.07, 0.10])  # Studio Dark Theme
            render_option.point_size = 4.0
            render_option.show_coordinate_frame = True

        # Keybindings
        def increase_point_size(v):
            opt = v.get_render_option()
            opt.point_size = min(30.0, opt.point_size + 1.0)
            print(f"[Viewer] Point size: {opt.point_size:.1f}", flush=True)
            return False

        def decrease_point_size(v):
            opt = v.get_render_option()
            opt.point_size = max(1.0, opt.point_size - 1.0)
            print(f"[Viewer] Point size: {opt.point_size:.1f}", flush=True)
            return False

        def take_screenshot(v):
            screenshot_dir = path.parent / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            shot_path = screenshot_dir / f"screenshot_{int(time.time())}.png"
            v.capture_screen_image(str(shot_path), do_render=True)
            print(f"[Viewer] Screenshot saved -> {shot_path}", flush=True)
            return False

        def reset_camera(v):
            v.reset_view_point(True)
            print("[Viewer] Camera view reset.", flush=True)
            return False

        vis.register_key_callback(ord("P"), increase_point_size)
        vis.register_key_callback(ord("="), increase_point_size)
        vis.register_key_callback(ord("+"), increase_point_size)
        vis.register_key_callback(ord("M"), decrease_point_size)
        vis.register_key_callback(ord("-"), decrease_point_size)
        vis.register_key_callback(ord("S"), take_screenshot)
        vis.register_key_callback(ord("R"), reset_camera)

        _write_viewer_log(log_path, "\n" + "=" * 60)
        _write_viewer_log(log_path, "[Viewer] TerraSweep — Interactive 3D Viewport Active")
        _write_viewer_log(log_path, "=" * 60)
        _write_viewer_log(log_path, "* Left Mouse Drag             : Rotate / Orbit Scene")
        _write_viewer_log(log_path, "* Shift + Left Drag / Middle  : Pan Camera")
        _write_viewer_log(log_path, "* Scroll Wheel / Right Drag   : Zoom In / Out")
        _write_viewer_log(log_path, "* Key [R]                     : Reset Camera View")
        _write_viewer_log(log_path, "* Key [P] or [+]              : Increase Point Size")
        _write_viewer_log(log_path, "* Key [M] or [-]              : Decrease Point Size")
        _write_viewer_log(log_path, "* Key [S]                     : Capture High-Res Screenshot")
        _write_viewer_log(log_path, "* Key [Q] or [Esc]            : Exit Viewer")
        _write_viewer_log(log_path, "=" * 60 + "\n")

        # Blocking run loop that stays open until closed by user
        vis.run()
        vis.destroy_window()
        return True

    except Exception as e:
        _write_viewer_log(log_path, f"[Viewer] Failed to open Open3D viewer.")
        _write_viewer_log(log_path, f"[Viewer] Exception: {e}")
        _write_viewer_log(log_path, f"[Viewer] File path: {model_path}")
        _write_viewer_log(log_path, f"[Viewer] Backend attempted: Open3D")
        _write_viewer_log(log_path, traceback.format_exc())
        return False


def run_trimesh_viewer(model_path: str, title: str = "TerraSweep — 3D Viewer", log_path: Optional[Path] = None) -> bool:
    """
    Priority 2: Fallback Trimesh Viewer.
    Uses trimesh.Scene() and scene.show() to maintain an active blocking window.
    """
    try:
        import trimesh

        path = Path(model_path).resolve()
        if not path.exists():
            return False

        _write_viewer_log(log_path, f"[Viewer] Loading: {path}")
        _write_viewer_log(log_path, f"[Viewer] Backend: Trimesh")

        if path.suffix.lower() == ".splat":
            _, pts, cols, num_vertices = _load_geometry_data(path)
            c_uint = (cols * 255.0).astype(np.uint8) if cols is not None else None
            scene = trimesh.points.PointCloud(vertices=pts, colors=c_uint)
        elif path.suffix.lower() == ".npz":
            data = np.load(path)
            pts = data.get("positions") or data.get("points") or data.get("means")
            scene = trimesh.points.PointCloud(vertices=pts)
            num_vertices = len(pts)
        else:
            loaded = trimesh.load(str(path))
            if isinstance(loaded, trimesh.Scene):
                scene = loaded
                num_vertices = sum(len(g.vertices) for g in scene.geometry.values() if hasattr(g, "vertices"))
            else:
                scene = trimesh.Scene(loaded)
                num_vertices = len(loaded.vertices) if hasattr(loaded, "vertices") else 0

        _write_viewer_log(log_path, f"[Viewer] Vertices: {num_vertices}")
        _write_viewer_log(log_path, f"Vertices loaded: {num_vertices}")
        _write_viewer_log(log_path, f"[Viewer] Status: SUCCESS")

        scene.show(title=f"{title} [{path.name}]", flags={"cull": False})
        return True
    except Exception as e:
        _write_viewer_log(log_path, f"[Viewer] Trimesh viewer failed: {e}")
        _write_viewer_log(log_path, f"[Viewer] File path: {model_path}")
        _write_viewer_log(log_path, f"[Viewer] Backend attempted: Trimesh")
        return False


def run_cloudcompare_fallback(model_path: str, log_path: Optional[Path] = None) -> bool:
    """
    Priority 3: Fallback External CloudCompare binary.
    Keeps the CloudCompare application detached.
    """
    cc_paths = [
        r"C:\Program Files\CloudCompare\CloudCompare.exe",
        r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
    ]
    which_cc = shutil.which("CloudCompare")
    if which_cc:
        cc_paths.insert(0, which_cc)

    path = Path(model_path).resolve()
    for cc in cc_paths:
        if Path(cc).is_file():
            try:
                _write_viewer_log(log_path, f"[Viewer] Loading: {path}")
                _write_viewer_log(log_path, f"[Viewer] Backend: CloudCompare ({cc})")
                _write_viewer_log(log_path, f"[Viewer] Status: SUCCESS")
                subprocess.Popen([cc, str(path)])
                return True
            except Exception as e:
                _write_viewer_log(log_path, f"[Viewer] CloudCompare launch error: {e}")
    return False


def launch_viewer_main(model_path: str, title: str = "TerraSweep — 3D Viewer", log_file: Optional[str] = None):
    """
    Orchestrates viewer launch across priority order:
    1. Open3D
    2. Trimesh
    3. CloudCompare
    """
    path = Path(model_path).resolve()
    log_path = Path(log_file).resolve() if log_file else (path.parent / "viewer_log.txt")

    if not path.exists():
        _write_viewer_log(log_path, f"Error: Model file does not exist at {model_path}")
        return

    # Priority 1: Open3D
    if run_open3d_viewer(str(path), title, log_path):
        return

    # Priority 2: Trimesh
    if run_trimesh_viewer(str(path), title, log_path):
        return

    # Priority 3: CloudCompare
    if run_cloudcompare_fallback(str(path), log_path):
        return

    _write_viewer_log(log_path, "Error: No 3D viewer backend could be initialized.")
    _write_viewer_log(log_path, f"Failed to open Open3D viewer on {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraSweep 3D Viewer")
    parser.add_argument("--model", type=str, required=True, help="Path to 3D model (.splat, .ply, .obj, .glb, .npz)")
    parser.add_argument("--title", type=str, default="TerraSweep — 3D Viewer", help="Window title")
    parser.add_argument("--log-file", type=str, default=None, help="Path to write viewer execution log")
    args = parser.parse_args()

    launch_viewer_main(args.model, args.title, args.log_file)
