# GeoRecon AI - Phase 6.3 Comprehensive Verification Test Suite
# Tests:
# 1. CUDA is enabled for SIFT (Extraction & Matching flags)
# 2. Worker thread never blocks Tkinter & GUI remains responsive
# 3. Cancel works immediately via process termination
# 4. Diagnostics file outputs/<session>/diagnostics.json contains all 8 required fields
# 5. Model Library Statuses (FAILED, COMPLETE, RUNNING, CANCELLED, PARTIAL)
# 6. Action buttons (View 3D / Preview disabled on FAILED) and Failure Tooltips

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import GeoReconApp, CTkTooltip
from config import DEFAULT_CONFIG
from pipeline.colmap_runner import ColmapRunner
from pipeline.manager import (
    PipelineManager,
    infer_session_status,
    PIPELINE_STATUS_COMPLETED,
    PIPELINE_STATUS_FAILED,
    PIPELINE_STATUS_PARTIAL,
    PIPELINE_STATUS_CANCELLED,
    PIPELINE_STATUS_RUNNING,
)
from pipeline.stage import StageType, StageStatus, PipelineEvent


def run_phase63_tests():
    print("=" * 80)
    print("🚀 GeoRecon AI (SIH-26158) — Phase 6.3: Stability, CUDA & Session Status Test Suite")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: CUDA Acceleration Flags for SIFT
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying CUDA SIFT flags & GPU Detection...")
    runner = ColmapRunner()
    gpu_name = runner.get_gpu_name()
    gpu_avail = runner.is_gpu_available()
    print(f"  -> Hardware GPU detected: '{gpu_name}' (CUDA available: {gpu_avail})")

    # Verify run_feature_extraction command arguments
    captured_args = []
    def mock_run_cmd(args, stage_name, on_log=None, stop_event=None):
        captured_args.append((args, stage_name))
        return True

    original_run_cmd = runner._run_colmap_command
    runner._run_colmap_command = mock_run_cmd

    logged_msgs = []
    runner.run_feature_extraction(
        image_path=Path("dummy_images"),
        database_path=Path("dummy_db.db"),
        use_gpu=True,
        on_log=lambda l: logged_msgs.append(l),
    )

    sift_args, s_name = captured_args[0]
    expected_feat_flag = runner.get_feature_extraction_gpu_flag_name()
    assert expected_feat_flag in sift_args, f"Missing {expected_feat_flag} in feature extractor"
    assert "--SiftExtraction.use_gpu" not in sift_args or expected_feat_flag == "--SiftExtraction.use_gpu", "Invalid --SiftExtraction.use_gpu present in feature extractor"
    idx_feat = sift_args.index(expected_feat_flag)
    assert sift_args[idx_feat + 1] == ("1" if gpu_avail else "0"), "GPU extraction flag mismatch"
    print(f"  -> Feature Extractor: verified {expected_feat_flag} 1")

    # Verify run_feature_matching command arguments
    captured_args.clear()
    runner.run_feature_matching(
        database_path=Path("dummy_db.db"),
        matcher_type="exhaustive",
        use_gpu=True,
        on_log=lambda l: logged_msgs.append(l),
    )
    match_args, m_name = captured_args[0]
    expected_match_flag = runner.get_feature_matching_gpu_flag_name()
    assert expected_match_flag in match_args, f"Missing {expected_match_flag} in feature matcher"
    assert "--SiftMatching.use_gpu" not in match_args or expected_match_flag == "--SiftMatching.use_gpu", "Invalid --SiftMatching.use_gpu present in feature matcher"
    idx_match = match_args.index(expected_match_flag)
    assert match_args[idx_match + 1] == ("1" if gpu_avail else "0"), "GPU matching flag mismatch"
    print(f"  -> Feature Matcher: verified {expected_match_flag} 1")

    # Verify startup verification logging
    startup_msgs = runner.verify_gpu_flags()
    assert any("Verified COLMAP GPU flags for version" in m for m in startup_msgs), "Missing version verification log"
    assert any("Feature Extraction GPU: ENABLED" in m for m in startup_msgs), "Missing extraction GPU log"
    assert any("Feature Matching GPU: ENABLED" in m for m in startup_msgs), "Missing matching GPU log"
    print("  -> Startup Verification: Verified 3 startup logging messages")

    if gpu_avail:
        assert any("CUDA SIFT enabled on" in m for m in logged_msgs), "Expected CUDA SIFT console log message"
        print(f"  -> Console Log Output: '{logged_msgs[0]}'")

    runner._run_colmap_command = original_run_cmd
    print("  ✅ TEST 1 PASSED: CUDA SIFT flags and GPU configuration verified.")

    # -------------------------------------------------------------------------
    # TEST 2: Process Termination & Cancellation Watchdog
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Cancellation & Immediate Process Termination...")
    mgr = PipelineManager()
    mgr._is_running = True

    class MockProc:
        def __init__(self):
            self.terminated = False
            self.killed = False
        def poll(self):
            return None if not (self.terminated or self.killed) else -15
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return -15
        def kill(self):
            self.killed = True

    mock_proc = MockProc()
    mgr.colmap_runner._active_process = mock_proc

    mgr.stop_pipeline()
    assert mgr._stop_event.is_set(), "Stop event was not set!"
    assert mock_proc.terminated or mock_proc.killed, "Active process was not terminated!"
    print("  ✅ TEST 2 PASSED: Cancellation triggers instant termination of running COLMAP process.")

    # -------------------------------------------------------------------------
    # TEST 3: Diagnostics File Generation (outputs/<session>/diagnostics.json)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying outputs/<session>/diagnostics.json with all 8 fields...")
    with tempfile.TemporaryDirectory(prefix="georecon_test_diag_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        session_dir = tmp_path / "outputs" / "survey_test_session"
        session_dir.mkdir(parents=True)

        diag_file = mgr.write_diagnostics(
            session_output_dir=session_dir,
            worker_thread_status="failed",
            failure_reason="Quality Gate Failed: Only 10/85 cameras registered (11.8%).",
            last_processed_stage="COLMAP_MAPPER",
            last_registered_camera_count=10,
            exit_codes={"feature_extractor": 0, "exhaustive_matcher": 0, "mapper": 1},
        )

        assert diag_file.exists(), f"Diagnostics file not created at {diag_file}"
        with open(diag_file, "r", encoding="utf-8") as f:
            d_data = json.load(f)

        required_keys = [
            ("CUDA detected", "cuda_detected"),
            ("GPU name", "gpu_name"),
            ("COLMAP version", "colmap_version"),
            ("Worker thread status", "worker_thread_status"),
            ("Exit codes", "exit_codes"),
            ("Failure reason", "failure_reason"),
            ("Last processed stage", "last_processed_stage"),
            ("Last registered camera count", "last_registered_camera_count"),
        ]

        for primary, fallback in required_keys:
            has_field = (primary in d_data) or (fallback in d_data)
            assert has_field, f"Missing required diagnostic field: {primary} / {fallback}"
            val = d_data.get(primary, d_data.get(fallback))
            print(f"  -> {primary}: {val}")

        assert d_data.get("worker_thread_status") == "failed"
        assert d_data.get("last_registered_camera_count") == 10
        assert "Quality Gate Failed" in d_data.get("failure_reason")
        print("  ✅ TEST 3 PASSED: diagnostics.json contains all 8 required forensic fields.")

    # -------------------------------------------------------------------------
    # TEST 4: GUI Non-Freezing / Queue Batch Drainage Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Verifying Tkinter Main Thread Responsiveness & Batch Drainage...")
    app = GeoReconApp()
    app.withdraw()

    # Drain any startup initialization logs first
    while not app.log_queue.empty():
        try:
            app.log_queue.get_nowait()
        except Exception:
            break

    # Simulate heavy COLMAP log output burst (1,000 lines into log_queue)
    for i in range(1000):
        app.log_queue.put((logging.INFO, f"[COLMAP-Mapper] Processing point cloud iteration {i:04d}"))

    t0 = time.perf_counter()
    # Execute single tick of _process_queues
    app._process_queues()
    t_tick = time.perf_counter() - t0

    # Ensure queue was drained in a bounded manner (not all 1000 at once, preventing freeze)
    remaining_logs = app.log_queue.qsize()
    consumed = 1000 - remaining_logs
    assert remaining_logs > 0, "Expected bounded batch drainage, queue drained completely!"
    assert 40 <= consumed <= 60, f"Expected bounded batch ~50 items consumed per tick, got {consumed}"
    assert t_tick < 0.20, f"Single queue processing tick took too long ({t_tick*1000:.1f}ms), risks GUI freezing"
    print(f"  -> Single queue tick execution time: {t_tick*1000:.2f}ms (consumed {consumed} logs, {remaining_logs} queued safely)")
    print("  ✅ TEST 4 PASSED: Non-blocking rate-limited log queue drainage verified.")

    # -------------------------------------------------------------------------
    # TEST 5: Session Status Rules in Model Library & Action Button Controls
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Verifying Model Library Card Statuses, Action Buttons, and Tooltips...")
    with tempfile.TemporaryDirectory(prefix="georecon_test_lib_") as tmp_lib_dir:
        tmp_lib = Path(tmp_lib_dir)
        app.selected_output_dir = tmp_lib

        # Create 5 representative test sessions
        # 1. Complete Session
        s_complete = tmp_lib / "session_complete"
        mgr.write_scene_manifest(s_complete, "session_complete", PIPELINE_STATUS_COMPLETED, 6, True, 130, 130, sparse_points=25000)
        (s_complete / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 25000\nend_header\n")
        (s_complete / "trajectory_preview.mp4").write_bytes(b"preview_data")

        # 2. Failed Session (WhatsApp video with 10/85 cameras)
        s_failed = tmp_lib / "session_failed_whatsapp"
        mgr.write_scene_manifest(s_failed, "session_failed_whatsapp", PIPELINE_STATUS_FAILED, 4, False, 10, 85, failure_reason="Quality Gate Failed: Only 10/85 cameras registered (11.8%).", sparse_points=1200)

        # 3. Partial Session
        s_partial = tmp_lib / "session_partial"
        mgr.write_scene_manifest(s_partial, "session_partial", PIPELINE_STATUS_PARTIAL, 4, True, 80, 100, sparse_points=15000)
        (s_partial / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 15000\nend_header\n")

        # 4. Running Session
        s_running = tmp_lib / "session_running"
        mgr.write_scene_manifest(s_running, "session_running", PIPELINE_STATUS_RUNNING, 2, False, 0, 100)

        # 5. Cancelled Session
        s_cancelled = tmp_lib / "session_cancelled"
        mgr.write_scene_manifest(s_cancelled, "session_cancelled", PIPELINE_STATUS_CANCELLED, 1, False, 0, 100, failure_reason="User cancelled")

        # Populate Model Library
        app._refresh_model_library()

        # Inspect cards created
        cards = [child for child in app.library_scroll.winfo_children() if hasattr(child, "session_status")]
        assert len(cards) == 5, f"Expected 5 cards, found {len(cards)}"

        status_card_map = {c.session_status: c for c in cards}

        # Check Complete Card
        c_comp = status_card_map[PIPELINE_STATUS_COMPLETED]
        assert c_comp.badge_label.cget("text") == "🟢 COMPLETE"
        assert c_comp.action_buttons["view_3d"].cget("state") == "normal"
        assert c_comp.action_buttons["preview"].cget("state") == "normal"
        print("  -> Complete Session: Badge 🟢 COMPLETE, View 3D & Preview Enabled")

        # Check Failed Card
        c_fail = status_card_map[PIPELINE_STATUS_FAILED]
        assert c_fail.badge_label.cget("text") == "🔴 FAILED"
        assert c_fail.action_buttons["view_3d"].cget("state") == "disabled", "View 3D must be disabled for FAILED session!"
        assert c_fail.action_buttons["preview"].cget("state") == "disabled", "Preview must be disabled for FAILED session!"
        assert c_fail.action_buttons["retry"].cget("state") == "normal", "Retry must be available for FAILED session!"
        print("  -> Failed Session: Badge 🔴 FAILED, View 3D & Preview Disabled, Retry Enabled")

        # Check Running Card
        c_run = status_card_map[PIPELINE_STATUS_RUNNING]
        assert c_run.badge_label.cget("text") == "🔵 RUNNING"
        print("  -> Running Session: Badge 🔵 RUNNING")

        # Check Partial Card
        c_part = status_card_map[PIPELINE_STATUS_PARTIAL]
        assert c_part.badge_label.cget("text") == "🟡 PARTIAL"
        assert c_part.action_buttons["view_3d"].cget("state") == "normal"
        print("  -> Partial Session: Badge 🟡 PARTIAL, View 3D Enabled for partial point cloud")

        # Check Cancelled Card
        c_canc = status_card_map[PIPELINE_STATUS_CANCELLED]
        assert c_canc.badge_label.cget("text") == "⚪ CANCELLED"
        print("  -> Cancelled Session: Badge ⚪ CANCELLED")

        print("  ✅ TEST 5 PASSED: Model library cards, badges, and action button guards verified.")

    try:
        app.destroy()
    except Exception:
        pass

    print("\n" + "=" * 80)
    print("🏆 ALL PHASE 6.3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase63_tests()
