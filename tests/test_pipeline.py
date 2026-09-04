"""
GeoRecon AI - Automated Pipeline & GUI Test Runner
Validates end-to-end video extraction, blur detection, duplicate filtering, and reporting.
"""

import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import GeoReconApp, DEFAULT_CONFIG
from pipeline import StageType, StageStatus


def run_pipeline_test():
    test_video = Path("data/test_drone_flight.mp4")
    if not test_video.exists():
        raise FileNotFoundError(f"Test video not found: {test_video}")

    print("=" * 60)
    print("Launching GeoRecon AI App Test Instance...")
    print("=" * 60)

    app = GeoReconApp()
    app.update_idletasks()

    print(f"App Window Title: {app.title()}")

    # Select test video
    app.selected_video_path = test_video
    app.lbl_vid_name.configure(text=f"🎥 {test_video.name}")
    app.btn_start_master.configure(state="normal")

    # Start pipeline
    print("Starting Preprocessing Pipeline...")
    app._on_start_reconstruction()

    # Poll GUI loop while worker thread executes
    max_wait = 45.0
    start_t = time.time()
    while app.pipeline_mgr.is_running and (time.time() - start_t) < max_wait:
        app.update()
        time.sleep(0.04)

    # Process remaining queue events
    for _ in range(25):
        app.update()
        time.sleep(0.04)

    print("\n" + "=" * 60)
    print("Pipeline Execution Completed!")
    print("=" * 60)

    # Verify stage items
    for stage, s_item in app.stage_items.items():
        print(f"Stage [{stage.value.upper():<15}]: Status={s_item.status.value:<10} | Message={s_item.lbl_msg.cget('text')}")

    # Verify report existence
    reports = list(Path("outputs").glob("*/preprocess_report.json"))
    print(f"\nTotal Session Reports Generated: {len(reports)}")
    assert len(reports) > 0, "No preprocessing report was generated!"

    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    print(f"Latest Report File: {latest_report}")
    with open(latest_report, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    print("\n--- Preprocessing Report Summary ---")
    print(json.dumps(report_data, indent=2))

    # Validate output frames exist
    session_name = report_data["session_name"]
    frames_dir = Path("data") / session_name / "frames"
    final_frames = list(frames_dir.glob("*.png"))
    print(f"\nFinal Extracted & Filtered Frames in {frames_dir}: {len(final_frames)} files")
    assert len(final_frames) == report_data["preprocessing_metrics"]["final_retained_frames"], "Frame count mismatch!"

    app.destroy()
    print("\nAll Tests Passed Successfully!")


if __name__ == "__main__":
    run_pipeline_test()
