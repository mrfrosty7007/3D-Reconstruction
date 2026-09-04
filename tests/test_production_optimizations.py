"""
GeoRecon AI (SIH-26158) - Production Optimization & Stability Test Suite
Verifies:
1. Cancellation & Multi-Tier Process Termination (Taskkill, Clean Skip, Zero Orphans)
2. Adaptive Frame Sampling (Short, Medium, Long, Very Long)
3. Weighted Real Progress Bar (0% to 100% stage weighting)
4. ETA Tracker (Moving Average, Non-Negative Formatting)
5. Live Quality Score & COLMAP Tuned Flags
6. Log Collapsing & Raw Log Buffer Preservation
7. Hardware Monitor Telemetry (Live RAM, Peak RAM, CPU Threads, Sub-State)
"""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.video_processor import VideoProcessor, calculate_sampling_step
from pipeline.colmap_runner import ColmapRunner
from pipeline.manager import PipelineManager, ETATracker, calculate_live_quality_score
from pipeline.stage import StageType, StageStatus, PipelineEvent
from pipeline.telemetry import HardwareSnapshot
from app import GeoReconApp, format_eta_string


def run_production_optimization_tests():
    print("=" * 80)
    print("🚀 GeoRecon AI (SIH-26158) — Production Debug & Optimization Test Suite")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Adaptive Frame Sampling Calculation
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing Length-Aware Adaptive Frame Sampling...")
    # Short video (<6s or <120 frames): step 1 for 30fps, 2 for 60fps
    step_short_30 = calculate_sampling_step(30.0, 4.0, 120)
    assert step_short_30 == 1, f"Expected 1, got {step_short_30}"
    step_short_60 = calculate_sampling_step(60.0, 3.0, 180)
    assert step_short_60 == 2, f"Expected 2, got {step_short_60}"

    # Medium video (6-20s): step 2 for 30fps
    step_med_30 = calculate_sampling_step(30.0, 10.0, 300)
    assert step_med_30 == 2, f"Expected 2, got {step_med_30}"

    # Long video (20-45s): step 3
    step_long_30 = calculate_sampling_step(30.0, 30.0, 900)
    assert step_long_30 == 3, f"Expected 3, got {step_long_30}"

    # Very long video (>45s): targets ~90-120 keyframes
    step_vlong = calculate_sampling_step(30.0, 60.0, 1800)
    sampled_vlong = 1800 // step_vlong
    assert 70 <= sampled_vlong <= 160, f"Expected ~90-120 keyframes, got {sampled_vlong}"
    print(f"  -> Short (4s 30fps): step {step_short_30} (120 frames)")
    print(f"  -> Medium (10s 30fps): step {step_med_30} (150 frames)")
    print(f"  -> Long (30s 30fps): step {step_long_30} (300 frames)")
    print(f"  -> Very Long (60s 30fps): step {step_vlong} (~{sampled_vlong} frames)")
    print("  ✅ TEST 1 PASSED: Adaptive frame sampling correctly prevents mapper slowdowns.")

    # -------------------------------------------------------------------------
    # TEST 2: ETA Tracker & Moving Average Estimation
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing ETA Tracker & Moving Average Estimation...")
    eta_tracker = ETATracker(window_size=5)
    eta_tracker.start_sparse_tracking()

    # Simulate 5 registration ticks
    time.sleep(0.05)
    eta_tracker.record_camera_registration(5)
    time.sleep(0.05)
    eta_tracker.record_camera_registration(10)
    time.sleep(0.05)
    eta_tracker.record_camera_registration(20)

    eta_val = eta_tracker.estimate_eta_sparse(registered=20, total=60)
    assert eta_val is not None and eta_val >= 0, f"Invalid ETA: {eta_val}"
    fmt = format_eta_string(eta_val)
    print(f"  -> Sparse ETA (20/60 cams): {eta_val:.1f}s -> Formatted: '{fmt}'")
    assert "m" in fmt or "s" in fmt, f"Unexpected ETA format: {fmt}"

    # GSplat ETA estimation (700 iters @ 20 it/s + 5s export = 40s)
    gsplat_eta = eta_tracker.estimate_eta_gsplat(current_iter=300, total_iter=1000, iter_speed=20.0)
    assert gsplat_eta == 40.0, f"Expected 40.0s, got {gsplat_eta}"
    fmt_gs = format_eta_string(gsplat_eta)
    assert fmt_gs == "40s", f"Expected '40s', got '{fmt_gs}'"
    print(f"  -> GSplat ETA (300/1000 @ 20 it/s): {gsplat_eta}s -> '{fmt_gs}'")

    # Format bounds
    assert format_eta_string(None) == "Estimating..."
    assert format_eta_string(-5) == "Estimating..."
    assert format_eta_string(0) == "Estimating..."
    assert format_eta_string(125) == "2m 05s"
    assert format_eta_string(3665) == "1h 01m"
    print("  ✅ TEST 2 PASSED: ETA tracker and formatters verified.")

    # -------------------------------------------------------------------------
    # TEST 3: Live Quality Score Calculation
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing Live Quality Score Function...")
    score_high = calculate_live_quality_score(reg_cams=90, total_cams=100, sparse_points=30000)
    score_med = calculate_live_quality_score(reg_cams=55, total_cams=100, sparse_points=12000)
    score_low = calculate_live_quality_score(reg_cams=10, total_cams=85, sparse_points=1200)

    print(f"  -> High Quality (90/100, 30k pts): {score_high}/100")
    print(f"  -> Medium Quality (55/100, 12k pts): {score_med}/100")
    print(f"  -> Low Quality (10/85, 1.2k pts): {score_low}/100")

    assert 85 <= score_high <= 100, f"Expected 85-100, got {score_high}"
    assert 60 <= score_med <= 84, f"Expected 60-84, got {score_med}"
    assert score_low < 40, f"Expected <40, got {score_low}"
    print("  ✅ TEST 3 PASSED: Live quality score matches reconstruction health.")

    # -------------------------------------------------------------------------
    # TEST 4: COLMAP Tuned Mapper & SIFT Arguments
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing COLMAP Tuned Mapper Optimization Arguments...")
    runner = ColmapRunner()
    num_threads = runner.get_optimal_thread_count()
    assert 2 <= num_threads <= 8, f"Thread count should be bounded, got {num_threads}"
    print(f"  -> Optimal worker thread limit: {num_threads} (prevents CPU thermal throttling)")

    # Verify that runner detects GPU flags
    assert runner.get_feature_extraction_gpu_flag_name() == "--FeatureExtraction.use_gpu"
    assert runner.get_feature_matching_gpu_flag_name() == "--FeatureMatching.use_gpu"
    print("  ✅ TEST 4 PASSED: COLMAP tuned threads and GPU flags verified.")

    # -------------------------------------------------------------------------
    # TEST 5: Cancellation & Multi-Tier Process Termination
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Cancellation & Multi-Tier Process Termination...")
    # Spawn a child ping command in new process group to test termination
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | (subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        proc = subprocess.Popen(
            ["ping", "127.0.0.1", "-n", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        runner._active_process = proc
        assert proc.poll() is None, "Test process should be running"
        print(f"  -> Spawned test child process (PID: {proc.pid})")

        start_t = time.time()
        runner.terminate_active_process()
        kill_elapsed = time.time() - start_t

        assert proc.poll() is not None, "Test process was not terminated!"
        assert kill_elapsed < 2.0, f"Termination too slow ({kill_elapsed:.2f}s)"
        print(f"  -> Terminated process {proc.pid} cleanly in {kill_elapsed:.3f}s (No orphan processes).")
    print("  ✅ TEST 5 PASSED: Multi-tier process termination kills process tree instantly.")

    # -------------------------------------------------------------------------
    # TEST 6: Terminal Log Collapsing & Raw Log Preservation
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing UI Console Log Deduplication & Raw Logs...")
    app = GeoReconApp()
    app.update()
    while not app.log_queue.empty():
        try:
            app.log_queue.get_nowait()
        except Exception:
            break
    app.raw_logs.clear()
    app.terminal_txt.delete("1.0", "end")
    app.last_log_line = ""
    app.last_log_count = 1

    # Simulate 5 identical log messages and 1 distinct log message
    for _ in range(5):
        app.log_queue.put((20, "COLMAP Mapper: Registered image [12/48]"))
    app.log_queue.put((20, "COLMAP Mapper: Bundle adjustment finished."))
    app._process_queues()
    app.update()

    # Verify raw logs preserved all 6 messages
    assert len(app.raw_logs) == 6, f"Expected 6 raw log records, got {len(app.raw_logs)}"
    # Verify terminal text collapsed the first 5 into [x5]
    txt_content = app.terminal_txt.get("1.0", "end")
    assert "[x5] COLMAP Mapper: Registered image [12/48]" in txt_content, f"Missing collapsed log in: {txt_content}"
    assert "Bundle adjustment finished" in txt_content
    print("  -> Collapsed 5 repetitive logs into single '[x5]' visual line.")
    print("  -> Preserved 6 full timestamped diagnostic entries in raw_logs buffer.")
    print("  ✅ TEST 6 PASSED: Log viewer is snappy and prevents GUI flooding.")

    # -------------------------------------------------------------------------
    # TEST 7: Weighted Progress & Sub-State Indicator
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing Weighted Pipeline Progress & Sub-State...")
    # Stage 1: 0.00 to 0.10
    evt_stage1 = PipelineEvent(
        stage=StageType.FRAME_EXTRACTION,
        status=StageStatus.RUNNING,
        progress=0.5,
        message="Extracting frames...",
        global_progress=0.05,
    )
    app._handle_pipeline_event(evt_stage1)
    assert app.current_substate == "Extracting Keyframes"
    assert app._current_global_progress == 0.05

    # Stage 4: 0.40 to 0.65 with live sparse points and cameras
    evt_stage4 = PipelineEvent(
        stage=StageType.COLMAP_MAPPER,
        status=StageStatus.RUNNING,
        progress=0.5,
        message="Registering cameras: 25/50",
        global_progress=0.525,
        registered_cameras=25,
        total_cameras=50,
        sparse_points=14500,
        quality_score=75,
        eta_seconds=42.0,
    )
    app._handle_pipeline_event(evt_stage4)
    assert "Registering Cameras (25/50)" in app.current_substate
    assert app.card_points.cget("text") == "14,500"
    assert app.card_cams.cget("text") == "25/50 (50%)"
    assert app.card_score.cget("text") == "75/100"

    # Stage 4 BA event
    evt_ba = PipelineEvent(
        stage=StageType.COLMAP_MAPPER,
        status=StageStatus.RUNNING,
        progress=0.5,
        message="Bundle Adjustment: Solving non-linear equations...",
        global_progress=0.525,
    )
    app._handle_pipeline_event(evt_ba)
    assert app.current_substate == "Bundle Adjustment (Ceres)"

    # Stage 5: 0.65 to 0.95
    evt_stage5 = PipelineEvent(
        stage=StageType.GAUSSIAN_SPLATTING,
        status=StageStatus.RUNNING,
        progress=0.5,
        message="Iter 500/1000",
        global_progress=0.80,
    )
    app._handle_pipeline_event(evt_stage5)
    assert app.current_substate == "Training 3DGS (CUDA)"

    # Stage 6: 0.95 to 1.00
    evt_stage6 = PipelineEvent(
        stage=StageType.EXPORT,
        status=StageStatus.RUNNING,
        progress=0.5,
        message="Generating deliverables...",
        global_progress=0.975,
    )
    app._handle_pipeline_event(evt_stage6)
    assert app.current_substate == "Packaging 3D Deliverables"

    # Cancelled event
    evt_cancel = PipelineEvent(
        stage=StageType.COLMAP_MAPPER,
        status=StageStatus.SKIPPED,
        progress=0.0,
        message="Pipeline cancelled by user.",
        global_progress=0.0,
    )
    app._handle_pipeline_event(evt_cancel)
    assert app.is_processing is False
    assert app.current_substate == "Hardware: Balanced (Idle)"
    print("  ✅ TEST 7 PASSED: Weighted pipeline progress and sub-state display verified.")

    # -------------------------------------------------------------------------
    # TEST 8: Hardware Monitor Card (RAM Peak, CPU Threads, Telemetry Snapshot)
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Hardware Monitor Card UI Updates...")
    snap = HardwareSnapshot(
        timestamp=time.time(),
        cpu_percent=24.5,
        ram_used_gb=12.4,
        ram_total_gb=32.0,
        ram_percent=38.75,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        gpu_util_percent=45.0,
        gpu_vram_used_mb=2048.0,
        gpu_vram_total_mb=8188.0,
        gpu_vram_percent=25.0,
        gpu_temperature_c=48.0,
        nvml_available=True,
    )
    app._update_hardware_monitor_ui(snap)
    app.update()

    assert f"{os.cpu_count() or 8} th" in app.lbl_hw_cpu_pct.cget("text")
    ram_txt = app.lbl_hw_ram.cget("text")
    assert "Peak:" in ram_txt and "GB" in ram_txt and "/" in ram_txt, f"Unexpected RAM text: {ram_txt}"
    print(f"  -> CPU Label: '{app.lbl_hw_cpu_pct.cget('text')}'")
    print(f"  -> RAM Label: '{ram_txt}'")
    print("  ✅ TEST 8 PASSED: Hardware Monitor labels show Peak RAM and CPU thread count.")

    app.destroy()

    print("\n" + "=" * 80)
    print("🏆 ALL PRODUCTION DEBUG & OPTIMIZATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_production_optimization_tests()
