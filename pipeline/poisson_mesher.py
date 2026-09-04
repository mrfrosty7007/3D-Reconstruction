"""
GeoRecon AI - Screened Poisson Surface Reconstruction Module
Reconstructs watertight 3D triangle meshes from dense COLMAP point clouds using Open3D.
Exports high-fidelity Wavefront OBJ, binary PLY, and Binary glTF (GLB) with vertex colors.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Callable, Dict, Any, Optional, Tuple
import threading

import numpy as np
import open3d as o3d
import trimesh

from config import PoissonConfig

logger = logging.getLogger("GeoRecon.PoissonMesher")


@dataclass
class PoissonResult:
    """Outcomes and geometric metrics produced by Screened Poisson Surface Reconstruction."""
    is_success: bool
    mesh_obj_path: Optional[Path] = None
    mesh_ply_path: Optional[Path] = None
    mesh_glb_path: Optional[Path] = None
    vertex_count: int = 0
    triangle_count: int = 0
    density_trimmed_count: int = 0
    densities_mean: float = 0.0
    densities_std: float = 0.0
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None

    @property
    def runtime_seconds(self) -> float:
        return self.processing_time_seconds

    @property
    def success(self) -> bool:
        return self.is_success

    @property
    def num_triangles(self) -> int:
        return self.triangle_count

    @property
    def num_vertices(self) -> int:
        return self.vertex_count

    @property
    def model_obj(self) -> Optional[Path]:
        return self.mesh_obj_path

    @property
    def model_ply(self) -> Optional[Path]:
        return self.mesh_ply_path

    @property
    def model_glb(self) -> Optional[Path]:
        return self.mesh_glb_path

    @property
    def obj_path(self) -> Optional[Path]:
        return self.mesh_obj_path

    @property
    def ply_path(self) -> Optional[Path]:
        return self.mesh_ply_path

    @property
    def glb_path(self) -> Optional[Path]:
        return self.mesh_glb_path



class PoissonMesher:
    """Orchestrates Open3D Screened Poisson Surface Reconstruction and multi-format export."""

    def __init__(self, config: Optional[PoissonConfig] = None):
        self.config = config or PoissonConfig()

    def run_poisson_reconstruction(
        self,
        fused_ply_path: Optional[Path] = None,
        output_obj_path: Optional[Path] = None,
        output_ply_path: Optional[Path] = None,
        output_glb_path: Optional[Path] = None,
        depth: Optional[int] = None,
        density_trim_quantile: Optional[float] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        stop_event: Optional[threading.Event] = None,
        input_ply_path: Optional[Path] = None,
    ) -> PoissonResult:
        """
        Executes genuine Open3D Screened Poisson Surface Reconstruction on fused.ply:
        1. Loads fused point cloud (XYZ + RGB)
        2. Estimates and consistently orients normals
        3. Reconstructs triangle mesh using Screened Poisson (depth 8-10)
        4. Trims low-density boundary artifacts
        5. Exports model.obj, model.ply, and model.glb with full topology
        """
        start_t = time.time()
        depth = depth or self.config.depth
        depth = max(8, min(10, depth))  # Constrain to 8-10 for balance of speed and quality
        trim_quantile = density_trim_quantile if density_trim_quantile is not None else self.config.density_trim_quantile

        target_ply = fused_ply_path or input_ply_path
        if not target_ply:
            return PoissonResult(is_success=False, error_message="No input PLY path provided.")
        fused_ply_path = Path(target_ply).resolve()
        if not fused_ply_path.exists() or fused_ply_path.stat().st_size == 0:
            err = f"Dense fused point cloud not found or empty at {fused_ply_path}"
            logger.error(err)
            return PoissonResult(is_success=False, error_message=err)

        if on_progress:
            on_progress(0.05, f"Loading dense point cloud: {fused_ply_path.name}...")

        try:
            # 1. Load fused point cloud
            pcd = o3d.io.read_point_cloud(str(fused_ply_path))
            num_pts = len(pcd.points)
            if num_pts == 0:
                err = f"Loaded dense cloud contains 0 points from {fused_ply_path}"
                logger.error(err)
                return PoissonResult(is_success=False, error_message=err)

            logger.info(f"Loaded {num_pts:,} dense points from {fused_ply_path.name}")
            if stop_event and stop_event.is_set():
                return PoissonResult(is_success=False, error_message="Cancelled by user")

            if on_progress:
                on_progress(0.20, f"Estimating normals for {num_pts:,} points...")

            # 2. Normal estimation and consistent orientation
            if not pcd.has_normals() or len(pcd.normals) == 0:
                # Calculate adaptive search radius from nearest neighbors
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.01
                search_radius = max(1e-4, avg_dist * 4.0)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=search_radius,
                        max_nn=30,
                    )
                )

            if on_progress:
                on_progress(0.35, "Orienting surface normals consistently...")

            # Orient normals with consistent tangent planes
            pcd.orient_normals_consistent_tangent_plane(k=15)

            if stop_event and stop_event.is_set():
                return PoissonResult(is_success=False, error_message="Cancelled by user")

            if on_progress:
                on_progress(0.50, f"Running Screened Poisson Reconstruction (octree depth {depth})...")

            # 3. Screened Poisson Surface Reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd=pcd,
                depth=depth,
                linear_fit=self.config.linear_fit,
            )

            raw_triangles = len(mesh.triangles)
            raw_vertices = len(mesh.vertices)
            logger.info(f"Raw Poisson Mesh: {raw_vertices:,} vertices, {raw_triangles:,} triangles")

            if raw_triangles == 0:
                err = "Poisson reconstruction yielded 0 triangles."
                logger.error(err)
                return PoissonResult(is_success=False, error_message=err)

            if stop_event and stop_event.is_set():
                return PoissonResult(is_success=False, error_message="Cancelled by user")

            if on_progress:
                on_progress(0.70, "Trimming low-density surface artifacts...")

            # 4. Remove low-density artifacts (boundary / false hull floaters)
            densities_arr = np.asarray(densities)
            d_mean = float(np.mean(densities_arr)) if len(densities_arr) > 0 else 0.0
            d_std = float(np.std(densities_arr)) if len(densities_arr) > 0 else 0.0
            trimmed_count = 0

            if len(densities_arr) > 0 and trim_quantile > 0.0:
                density_threshold = np.quantile(densities_arr, trim_quantile)
                vertices_to_remove = densities_arr < density_threshold
                trimmed_count = int(np.sum(vertices_to_remove))
                mesh.remove_vertices_by_mask(vertices_to_remove)

            # Clean degenerate & duplicate triangles
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.compute_vertex_normals()

            final_triangles = len(mesh.triangles)
            final_vertices = len(mesh.vertices)
            logger.info(
                f"Cleaned Poisson Mesh: {final_vertices:,} vertices, {final_triangles:,} triangles "
                f"(trimmed {trimmed_count:,} low-density vertices)"
            )

            if final_triangles == 0:
                err = "Mesh cleaning trimmed all triangles. Density threshold too aggressive."
                logger.error(err)
                return PoissonResult(is_success=False, error_message=err)

            if stop_event and stop_event.is_set():
                return PoissonResult(is_success=False, error_message="Cancelled by user")

            # 5. Export Deliverables: model.obj, model.ply, model.glb
            if output_obj_path is None:
                out_dir = fused_ply_path.parent
                resolved_obj = out_dir / "model.obj"
            else:
                resolved_obj = Path(output_obj_path)
                if resolved_obj.is_dir():
                    out_dir = resolved_obj
                    resolved_obj = out_dir / "model.obj"
                else:
                    out_dir = resolved_obj.parent

            if output_ply_path is None:
                resolved_ply = out_dir / "model.ply"
            else:
                resolved_ply = Path(output_ply_path)
                if resolved_ply.is_dir():
                    resolved_ply = resolved_ply / "model.ply"

            if output_glb_path is None:
                resolved_glb = out_dir / "model.glb"
            else:
                resolved_glb = Path(output_glb_path)
                if resolved_glb.is_dir():
                    resolved_glb = resolved_glb / "model.glb"

            output_obj_path = resolved_obj.resolve()
            output_ply_path = resolved_ply.resolve()
            output_glb_path = resolved_glb.resolve()


            for p in [output_obj_path, output_ply_path, output_glb_path]:
                p.parent.mkdir(parents=True, exist_ok=True)

            if on_progress:
                on_progress(0.80, f"Saving Wavefront OBJ mesh: {output_obj_path.name}...")

            # Export model.obj (with vertex normals and vertex colors)
            o3d.io.write_triangle_mesh(
                str(output_obj_path),
                mesh,
                write_ascii=False,
                compressed=False,
                write_vertex_normals=True,
                write_vertex_colors=mesh.has_vertex_colors(),
            )

            if on_progress:
                on_progress(0.88, f"Saving binary PLY mesh: {output_ply_path.name}...")

            # Export model.ply (with vertex colors and triangle face elements)
            o3d.io.write_triangle_mesh(
                str(output_ply_path),
                mesh,
                write_ascii=False,
                compressed=False,
                write_vertex_normals=True,
                write_vertex_colors=mesh.has_vertex_colors(),
            )

            if on_progress:
                on_progress(0.94, f"Saving Binary glTF asset: {output_glb_path.name}...")

            # Export model.glb
            glb_written = False
            try:
                # Attempt Open3D glb export first
                glb_written = o3d.io.write_triangle_mesh(str(output_glb_path), mesh)
            except Exception as e_glb:
                logger.debug(f"Open3D GLB export note: {e_glb}")

            if not glb_written or not output_glb_path.exists() or output_glb_path.stat().st_size == 0:
                # Robust fallback via trimesh
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                v_cols = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
                tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_cols, process=False)
                tm.export(str(output_glb_path), file_type="glb")

            elapsed = time.time() - start_t
            logger.info(f"Screened Poisson Meshing completed in {elapsed:.2f}s -> OBJ, PLY, GLB ready")

            return PoissonResult(
                is_success=True,
                mesh_obj_path=output_obj_path,
                mesh_ply_path=output_ply_path,
                mesh_glb_path=output_glb_path,
                vertex_count=final_vertices,
                triangle_count=final_triangles,
                density_trimmed_count=trimmed_count,
                densities_mean=d_mean,
                densities_std=d_std,
                processing_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            logger.exception(f"Error during Screened Poisson Reconstruction: {e}")
            return PoissonResult(
                is_success=False,
                error_message=str(e),
                processing_time_seconds=round(time.time() - start_t, 2),
            )

    # Method alias for consistency across callers
    reconstruct_mesh = run_poisson_reconstruction

    @staticmethod
    def get_point_cloud_info(ply_path: Path) -> Dict[str, Any]:
        """Inspects a PLY point cloud file and returns point count and attributes."""
        ply_path = Path(ply_path)
        if not ply_path.exists() or ply_path.stat().st_size == 0:
            return {"point_count": 0, "has_colors": False, "has_normals": False}
        try:
            count = 0
            has_colors = False
            has_normals = False
            with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("element vertex"):
                        count = int(line.split()[-1])
                    elif "diffuse_red" in line or "red" in line:
                        has_colors = True
                    elif "nx" in line or "normal_x" in line:
                        has_normals = True
                    elif line.startswith("end_header"):
                        break
            if count > 0:
                return {"point_count": count, "has_colors": has_colors, "has_normals": has_normals}
            pcd = o3d.io.read_point_cloud(str(ply_path))
            return {
                "point_count": len(pcd.points),
                "has_colors": pcd.has_colors(),
                "has_normals": pcd.has_normals(),
            }
        except Exception as e:
            logger.debug(f"Error reading point cloud info: {e}")
            return {"point_count": 0, "has_colors": False, "has_normals": False}
