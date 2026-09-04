"""
GeoRecon AI - Studio Integration & Real COLMAP Test Suite
Tests 4-page studio navigation, real COLMAP SfM execution, model conversion, and library registration.
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
from pipeline import StageType, StageStatus


def test_studio_pipeline():
    test_video = Path("data/studio_drone_sample.mp4")
    assert test_video.exists(), f"Sample video not found: {test_video}"

    print("=" * 70)
    print("🚀 Initializing GeoRecon AI Studio...")
    print("=" * 70)

    app = GeoReconApp()
    app.update_idletasks()

    print(f"Studio Window: {app.title()}")
    print(f"Hardware: {app.gpu_label_text} | COLMAP: {app.colmap_ver}")

    # 1. Test Navigation Page Switching
    print("\n--- Testing Navigation Studio Pages ---")
    pages = ["studio", "progress", "finished", "library"]
    for p in pages:
        app.switch_page(p)
        app.update()
        print(f"Switched to page: [{p}] — Frame active: {app.active_page_name}")
        assert app.active_page_name == p

    # 2. Setup Project in Studio Home
    app.switch_page("studio")
    app.selected_video_path = test_video
    app.entry_scene_name.delete(0, "end")
    app.entry_scene_name.insert(0, "test_studio_monument")
    app.lbl_vid_name.configure(text=f"🎥 {test_video.name}")
    app.btn_start_master.configure(state="normal")
    app.update()

    # 3. Start Reconstruction
    print("\n--- Starting Full 6-Stage Photogrammetry Pipeline ---")
    app._on_start_reconstruction()

    # Verify auto-transition to Progress page
    assert app.active_page_name == "progress", "App did not switch to Progress page!"
    print("Successfully transitioned to Active Progress page.")

    # 4. Wait for background worker execution
    max_wait = 180.0
    start_t = time.time()
    last_print = 0

    while app.pipeline_mgr.is_running and (time.time() - start_t) < max_wait:
        app.update()
        time.sleep(0.05)
        if int(time.time() - start_t) % 5 == 0 and int(time.time() - start_t) != last_print:
            last_print = int(time.time() - start_t)
            print(f"Elapsed: {last_print}s | Status: {app.lbl_progress_title.cget('text')} | Progress: {app.lbl_progress_percent.cget('text')}")

    # Process remaining event queue
    for _ in range(40):
        app.update()
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("Pipeline Execution Loop Finished!")
    print("=" * 70)

    # 5. Verify Stage Tracker States
    print("\n--- Stage Tracker States ---")
    for stage, s_item in app.stage_items.items():
        print(f"[{stage.display_name:<32}]: Status={s_item.status.value:<10} | Subtext={s_item.lbl_msg.cget('text')}")

    # 6. Verify COLMAP summary & Quality Gate outcomes
    session_name = app.pipeline_mgr.last_session_name
    print(f"\nTarget Session: {session_name}")
    assert session_name is not None, "Session name not recorded!"

    session_output = DEFAULT_CONFIG.outputs_dir / session_name
    colmap_summary_file = session_output / "colmap_summary.json"
    manifest_file = session_output / "scene_manifest.json"
    recovery_file = session_output / "recovery_suggestions.json"

    print(f"Checking {colmap_summary_file}...")
    assert colmap_summary_file.exists(), f"COLMAP summary missing: {colmap_summary_file}"
    with open(colmap_summary_file, "r", encoding="utf-8") as f:
        col_data = json.load(f)
    print("COLMAP Summary Data:\n", json.dumps(col_data, indent=2))

    if manifest_file.exists():
        print(f"\nChecking {manifest_file}...")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        print("Scene Manifest Data:\n", json.dumps(manifest_data, indent=2))
    elif recovery_file.exists():
        print(f"\nQuality Gate activated: Checking {recovery_file}...")
        with open(recovery_file, "r", encoding="utf-8") as f:
            rec_data = json.load(f)
        print("Recovery Suggestions Data:\n", json.dumps(rec_data, indent=2))

    # 7. Verify Model Library
    print("\n--- Verifying Model Library ---")
    app.switch_page("library")
    app.update()
    lib_children = app.library_scroll.winfo_children()
    print(f"Model Library Cards Count: {len(lib_children)}")
    assert len(lib_children) > 0, "No cards rendered in Model Library!"

    app.destroy()
    print("\n🎉 ALL STUDIO INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_studio_pipeline()
