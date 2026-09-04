"""
TerraSweep (SIH-26158) - Interactive 3D OBJ Mesh Viewer Module
Provides hardware-accelerated interactive 3D OBJ mesh viewing using Trimesh and Plotly.
Supports:
- Left drag -> Rotate (Orbit)
- Mouse wheel -> Zoom
- Right drag -> Pan
- scene.aspectmode = "data"
- Zero padding/margins
- Both Trimesh and Scene objects with automatic geometry merging
- Isolated non-blocking process execution to keep the main Studio UI responsive
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Union
import webbrowser

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import trimesh

logger = logging.getLogger("GeoRecon.OBJViewer")


def find_obj_file(output_dir: Union[str, Path]) -> Optional[Path]:
    """
    Search the reconstruction output directory in this priority order:
    1. model.obj
    2. model.glb
    3. model.ply
    4. mesh.obj
    5. textured.obj
    6. any *.obj file (fallback)
    7. dense/fused.ply or point_cloud.ply

    Returns the first valid, non-empty file found, or None.
    """
    if not output_dir:
        return None

    dir_path = Path(output_dir)
    if not dir_path.is_dir():
        return None

    priority_candidates = [
        dir_path / "model.obj",
        dir_path / "model.glb",
        dir_path / "model.ply",
        dir_path / "mesh.obj",
        dir_path / "textured.obj",
    ]

    for candidate in priority_candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    # Fallback: search for any *.obj in the directory
    obj_files = sorted(dir_path.glob("*.obj"))
    for candidate in obj_files:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    for fallback in [dir_path / "dense" / "fused.ply", dir_path / "point_cloud.ply"]:
        if fallback.is_file() and fallback.stat().st_size > 0:
            return fallback

    return None


def create_obj_figure(obj_path: Union[str, Path]) -> go.Figure:
    """
    Loads an OBJ file and constructs an interactive Plotly 3D Figure
    configured with dark theme, data aspect mode, and zero margins.
    Handles both Trimesh and Scene objects, merging geometries when necessary.
    """
    path = Path(obj_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"OBJ file does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"OBJ file is empty (0 bytes): {path}")

    # Load OBJ using trimesh (handling process=False to avoid losing vertex colors/attributes)
    loaded = trimesh.load(str(path), process=False)

    vertices = None
    faces = None
    colors = None

    # Handle Scene vs Trimesh vs PointCloud
    if isinstance(loaded, trimesh.Scene):
        # Extract all geometries from scene
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        if not geoms:
            raise ValueError(f"Scene in '{path.name}' contains no valid 3D geometry.")

        if len(geoms) == 1:
            merged = geoms[0]
        else:
            # Merge scene geometries
            trimeshes = [g for g in geoms if isinstance(g, trimesh.Trimesh)]
            if len(trimeshes) == len(geoms):
                merged = trimesh.util.concatenate(trimeshes)
            else:
                # Merge general vertex/face lists with index offsets
                v_list = []
                f_list = []
                offset = 0
                for g in geoms:
                    v = np.asarray(g.vertices)
                    v_list.append(v)
                    if hasattr(g, "faces") and g.faces is not None and len(g.faces) > 0:
                        f_list.append(np.asarray(g.faces) + offset)
                    offset += len(v)
                v_all = np.vstack(v_list)
                f_all = np.vstack(f_list) if f_list else np.array([])
                merged = trimesh.Trimesh(vertices=v_all, faces=f_all, process=False)

        vertices = np.asarray(merged.vertices)
        faces = np.asarray(merged.faces) if hasattr(merged, "faces") and merged.faces is not None and len(merged.faces) > 0 else np.array([])
        if hasattr(merged, "visual") and hasattr(merged.visual, "vertex_colors") and merged.visual.vertex_colors is not None:
            colors = merged.visual.vertex_colors

    elif isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices)
        faces = np.asarray(loaded.faces) if loaded.faces is not None and len(loaded.faces) > 0 else np.array([])
        if hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors") and loaded.visual.vertex_colors is not None:
            colors = loaded.visual.vertex_colors

    elif hasattr(loaded, "vertices"):
        vertices = np.asarray(loaded.vertices)
        faces = np.asarray(loaded.faces) if hasattr(loaded, "faces") and loaded.faces is not None and len(loaded.faces) > 0 else np.array([])
        if hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors") and loaded.visual.vertex_colors is not None:
            colors = loaded.visual.vertex_colors

    if vertices is None or len(vertices) == 0:
        raise ValueError(f"OBJ file '{path.name}' contains 0 vertices.")

    # Convert vertex colors if present
    vertex_colors_str = None
    if colors is not None and len(colors) == len(vertices):
        # trimesh visual.vertex_colors are shape (N, 3) or (N, 4) in [0..255]
        vertex_colors_str = [
            f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
            for c in colors[:, :3]
        ]

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    traces = []
    if len(faces) > 0:
        # Standard triangle mesh with explicit face topology
        mesh_trace = go.Mesh3d(
            x=x, y=y, z=z,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            vertexcolor=vertex_colors_str,
            color="#3B82F6" if vertex_colors_str is None else None,
            opacity=1.0,
            lighting=dict(ambient=0.45, diffuse=0.6, specular=0.25, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
            name="OBJ Mesh",
            hoverinfo="skip"
        )
        traces.append(mesh_trace)
    else:
        # Point-cloud OBJ (vertices without explicit face elements)
        # 1. Build Mesh3d surface representation
        if len(vertices) > 15000:
            step = len(vertices) // 15000
            x_m, y_m, z_m = x[::step], y[::step], z[::step]
        else:
            x_m, y_m, z_m = x, y, z

        mesh_trace = go.Mesh3d(
            x=x_m, y=y_m, z=z_m,
            alphahull=0,
            color="#3B82F6",
            opacity=0.65,
            lighting=dict(ambient=0.5, diffuse=0.5),
            name="Surface Mesh",
            hoverinfo="skip"
        )
        traces.append(mesh_trace)

        # 2. Add high-definition vertex points for detail
        scatter_step = max(1, len(vertices) // 35000)
        p_colors = (
            vertex_colors_str[::scatter_step]
            if vertex_colors_str
            else "#60A5FA"
        )
        point_trace = go.Scatter3d(
            x=x[::scatter_step], y=y[::scatter_step], z=z[::scatter_step],
            mode="markers",
            marker=dict(size=2, color=p_colors, opacity=0.85),
            name="Vertices",
            hoverinfo="skip"
        )
        traces.append(point_trace)

    fig = go.Figure(data=traces)

    # Layout: aspectmode="data", dark theme, zero margins, dragmode="orbit"
    # Orbit controls: Left drag -> Rotate, Scroll -> Zoom, Right drag -> Pan
    fig.update_layout(
        title=dict(
            text=f"TerraSweep — {path.name} ({len(vertices):,} vertices)",
            font=dict(size=14, color="#94A3B8"),
            x=0.02,
            y=0.98
        ),
        scene=dict(
            aspectmode="data",
            dragmode="orbit",
            bgcolor="#0E1117",
            xaxis=dict(
                title="",
                showgrid=True,
                gridcolor="#1E2330",
                showbackground=True,
                backgroundcolor="#11141D",
                zerolinecolor="#2E384D",
                showticklabels=False
            ),
            yaxis=dict(
                title="",
                showgrid=True,
                gridcolor="#1E2330",
                showbackground=True,
                backgroundcolor="#11141D",
                zerolinecolor="#2E384D",
                showticklabels=False
            ),
            zaxis=dict(
                title="",
                showgrid=True,
                gridcolor="#1E2330",
                showbackground=True,
                backgroundcolor="#11141D",
                zerolinecolor="#2E384D",
                showticklabels=False
            ),
        ),
        paper_bgcolor="#0F1117",
        plot_bgcolor="#0F1117",
        margin=dict(l=0, r=0, b=0, t=0),
        uirevision="constant",
    )

    return fig


def generate_obj_html(obj_path: Union[str, Path], output_html_path: Optional[Path] = None) -> Path:
    """
    Renders the Plotly figure and writes a standalone, dark-themed HTML file with
    interactive controls bar.
    """
    path = Path(obj_path)
    fig = create_obj_figure(path)

    if output_html_path is None:
        temp_dir = path.parent
        output_html_path = temp_dir / f"{path.stem}_viewer.html"

    # Export Plotly figure to HTML with CDN plotly.js
    raw_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config=dict(
            scrollZoom=True,
            displayModeBar=True,
            displaylogo=False,
            modeBarButtonsToRemove=["resetCameraDefault3d"],
            responsive=True
        )
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TerraSweep — 3D Reconstruction</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background-color: #0E1117;
            color: #E2E8F0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #topbar {{
            position: absolute;
            top: 12px;
            left: 16px;
            z-index: 100;
            background: rgba(19, 23, 34, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid #202738;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 12px;
            pointer-events: none;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            background: #1E293B;
            color: #38BDF8;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        #controls-hint {{
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            background: rgba(19, 23, 34, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid #202738;
            border-radius: 20px;
            padding: 6px 18px;
            font-size: 11px;
            color: #94A3B8;
            pointer-events: none;
        }}
        #plotly-container {{
            width: 100vw;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <div id="topbar">
        <span class="badge">TerraSweep</span>
        <span><b>3D Mesh:</b> {path.name}</span>
        <span style="color:#64748B;">|</span>
        <span style="color:#34D399;">WebGL Hardware Accelerated</span>
    </div>
    <div id="controls-hint">
        🎮 <b>Controls:</b> Left-Click Drag: Rotate &bull; Scroll Wheel: Zoom &bull; Right-Click Drag: Pan
    </div>
    <div id="plotly-container">
        {raw_html}
    </div>
    <script>
        window.addEventListener('resize', function() {{
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (plotDiv) {{
                Plotly.Plots.resize(plotDiv);
            }}
        }});
    </script>
</body>
</html>"""

    output_html_path.write_text(full_html, encoding="utf-8")
    return output_html_path


def launch_obj_viewer_window(obj_path: Union[str, Path], title: str = "TerraSweep — 3D OBJ Mesh Viewer"):
    """
    Opens an interactive hardware-accelerated 3D OBJ viewer window using pywebview (Edge WebView2)
    with automatic fallback to the system default browser.
    """
    path = Path(obj_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    html_file = generate_obj_html(path)

    try:
        import webview
        logger.info(f"Opening pywebview window for {path.name}...")
        window = webview.create_window(
            title=f"{title} — [{path.name}]",
            url=str(html_file.resolve()),
            width=1100,
            height=780,
            background_color="#0E1117",
            easy_drag=False
        )
        webview.start()
    except Exception as e_webview:
        logger.warning(f"pywebview window could not start ({e_webview}). Falling back to browser view.")
        webbrowser.open(html_file.as_uri())


def launch_obj_viewer_process(
    obj_path: Union[str, Path],
    title: str = "TerraSweep — 3D OBJ Mesh Viewer"
) -> Optional[object]:
    """
    Launches the 3D OBJ viewer in an independent non-blocking process so the main
    Studio UI never freezes.
    """
    import subprocess

    path = Path(obj_path).resolve()
    if not path.exists():
        logger.error(f"Cannot launch OBJ viewer: File not found at {path}")
        return None

    cmd = [
        sys.executable,
        "-m", "viewer.obj_viewer",
        "--model", str(path),
        "--title", title
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent)
        )
        logger.info(f"Spawned 3D OBJ viewer process (PID: {proc.pid}) for: {path.name}")
        return proc
    except Exception as e:
        logger.error(f"Failed to launch OBJ viewer process: {e}")
        return None


def view_obj(obj_path: Union[str, Path]):
    """
    Primary API function: Renders the interactive 3D OBJ mesh viewer.
    - Loads the OBJ using trimesh.
    - Handles both Trimesh and Scene objects.
    - Merges scene geometries when necessary.
    - Renders the mesh using plotly.graph_objects.Mesh3d.
    - Uses scene.aspectmode="data".
    - Removes extra margins.
    - Launches the interactive viewer.
    """
    launch_obj_viewer_window(obj_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraSweep 3D OBJ Mesh Viewer")
    parser.add_argument("--model", type=str, required=True, help="Path to OBJ model file")
    parser.add_argument("--title", type=str, default="TerraSweep — 3D OBJ Mesh Viewer", help="Window title")
    args = parser.parse_args()

    try:
        launch_obj_viewer_window(args.model, args.title)
    except Exception as exc:
        print(f"Error in OBJ viewer: {exc}", file=sys.stderr)
        sys.exit(1)
