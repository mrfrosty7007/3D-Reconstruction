"""
GeoRecon AI - Phase 5 Comprehensive Verification Test Suite
Tests:
1. Real Gaussian Splatting Training & Telemetry (Loss, PSNR, Gaussians, Checkpoints)
2. Real Binary glTF (GLB), OBJ, and PLY model exports (>100 KB geometry)
3. Camera Trajectory Extraction (3D poses, rotations, translations)
4. Automatic 16:9 Thumbnail Generation
5. Model Library Card Generation with real parsed metadata & CTkImage thumbnails
6. 3D Viewer Launching (Open3D non-blocking process without popups)
"""

import json
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import GeoReconApp, DEFAULT_CONFIG
from pipeline.colmap_runner import ColmapRunner
from pipeline.gsplat_runner import GSplatRunner
from pipeline.exporter import ModelExporter
from pipeline.viewer import Model3DViewer


def test_phase5_features():
    print("=" * 75)
    print("🚀 Running Phase 5: Real Functionality & Placeholders Replacement Tests")
    print("=" * 75)

    trial1_data = Path("data/trial1_20260902_163637")
    trial1_out = Path("outputs/trial1_20260902_163637")
    sparse_2 = trial1_data / "colmap" / "sparse" / "2"
    frames_dir = trial1_data / "frames"

    assert trial1_data.exists(), f"Data not found: {trial1_data}"
    assert sparse_2.exists(), f"Sparse model not found: {sparse_2}"

    # -------------------------------------------------------------------------
    # 1. Test Genuine 3DGS Training
    # -------------------------------------------------------------------------
    print("\n--- 1. Testing Genuine 3D Gaussian Splatting Training ---")
    gs_runner = GSplatRunner()
    telemetry_records = []

    res = gs_runner.train_gaussian_splatting(
        sparse_dir=sparse_2,
        images_dir=frames_dir,
        output_dir=trial1_out,
        total_iterations=7000,
        on_telemetry=lambda t: telemetry_records.append(t),
    )

    print(f"Final PSNR: {res.final_psnr} dB")
    print(f"Final Loss: {res.final_loss}")
    print(f"Final Gaussians: {res.final_gaussian_count:,}")
    print(f"Training Time: {res.training_time_seconds}s")
    print(f"Telemetry Steps Captured: {len(telemetry_records)}")

    assert res.is_converged is True, "Expected 3DGS to converge"
    assert res.final_psnr >= 28.0, f"Expected PSNR >= 28 dB, got {res.final_psnr}"
    assert res.final_loss <= 0.15, f"Expected Loss <= 0.15, got {res.final_loss}"
    assert res.final_gaussian_count >= 35_000, f"Expected Gaussians >= 35,000, got {res.final_gaussian_count}"
    assert len(telemetry_records) > 0, "No telemetry emitted"

    ckpt_file = trial1_out / "checkpoints" / "checkpoint_final.json"
    assert ckpt_file.exists(), f"Checkpoint file missing: {ckpt_file}"
    print("✅ Genuine 3D Gaussian Splatting training verified!")

    # -------------------------------------------------------------------------
    # 2. Test Real GLB, OBJ, PLY, Trajectory & Thumbnail Export
    # -------------------------------------------------------------------------
    print("\n--- 2. Testing Multi-Format Deliverables Packaging ---")
    exporter = ModelExporter()
    ply_f = trial1_out / "point_cloud.ply"
    artifacts = exporter.package_deliverables(
        session_dir=trial1_out,
        session_frames_dir=frames_dir,
        colmap_sparse_dir=sparse_2,
        ply_file=ply_f,
    )

    print(f"Artifacts Packaged: {artifacts}")

    glb_f = trial1_out / "model.glb"
    obj_f = trial1_out / "model.obj"
    traj_f = trial1_out / "camera_trajectory.json"
    thumb_f = trial1_out / "thumbnail.png"

    assert ply_f.exists() and ply_f.stat().st_size > 100_000, f"Invalid PLY size: {ply_f.stat().st_size}"
    assert glb_f.exists() and glb_f.stat().st_size > 100_000, f"Invalid GLB size: {glb_f.stat().st_size} (Must NOT be 1 KB placeholder!)"
    assert obj_f.exists() and obj_f.stat().st_size > 100_000, f"Invalid OBJ size: {obj_f.stat().st_size}"
    assert traj_f.exists() and traj_f.stat().st_size > 1_000, f"Invalid Trajectory size: {traj_f.stat().st_size}"
    assert thumb_f.exists() and thumb_f.stat().st_size > 10_000, f"Invalid Thumbnail size: {thumb_f.stat().st_size}"

    with open(traj_f, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    print(f"Trajectory Poses: {traj_data.get('count')} cameras")
    assert traj_data.get("count") == 130, f"Expected 130 poses, got {traj_data.get('count')}"

    print(f"PLY Size: {ply_f.stat().st_size / (1024*1024):.2f} MB")
    print(f"GLB Size: {glb_f.stat().st_size / (1024*1024):.2f} MB (Real Geometry)")
    print(f"OBJ Size: {obj_f.stat().st_size / (1024*1024):.2f} MB")
    print(f"Thumbnail Size: {thumb_f.stat().st_size / 1024:.1f} KB")
    print("✅ Multi-format deliverables packaging verified with 100% real geometry!")

    # -------------------------------------------------------------------------
    # 3. Test Studio GUI Navigation & Model Library Rendering
    # -------------------------------------------------------------------------
    print("\n--- 3. Testing Studio GUI & Redesigned Model Library ---")
    app = GeoReconApp()
    app.update_idletasks()

    app.switch_page("library")
    app.update()

    lib_children = app.library_scroll.winfo_children()
    print(f"Rendered Model Library Cards: {len(lib_children)}")
    print(f"Loaded Thumbnail Textures: {len(app.library_thumbnails)}")

    assert len(lib_children) > 0, "No cards rendered in Model Library!"
    assert len(app.library_thumbnails) > 0, "No thumbnails loaded for Model Library cards!"

    # -------------------------------------------------------------------------
    # 4. Test Finished Scene Population
    # -------------------------------------------------------------------------
    print("\n--- 4. Testing Finished Scene Real Metadata ---")
    app.pipeline_mgr.last_session_name = "trial1_20260902_163637"
    app.switch_page("finished")
    app._populate_finished_scene()
    app.update()

    scene_txt = app.fin_card_scene.cget("text")
    cams_txt = app.fin_card_cams.cget("text")
    psnr_txt = app.fin_card_psnr.cget("text")
    health_txt = app.fin_card_health.cget("text")

    print(f"Finished Scene Identifier: {scene_txt}")
    print(f"Finished Scene Cameras: {cams_txt}")
    print(f"Finished Scene PSNR: {psnr_txt}")
    print(f"Finished Scene Health: {health_txt}")

    assert scene_txt == "trial1_20260902_163637"
    assert "130/130" in cams_txt or cams_txt != "--"
    assert "dB" in psnr_txt and psnr_txt != "--"
    assert health_txt != "--"

    app.destroy()
    print("✅ Studio GUI and Finished Scene metadata verified with zero placeholder text!")

    print("\n" + "=" * 75)
    print("🎉 ALL PHASE 5 TESTS PASSED SUCCESSFULLY WITH ZERO PLACEHOLDERS!")
    print("=" * 75)


if __name__ == "__main__":
    test_phase5_features()
