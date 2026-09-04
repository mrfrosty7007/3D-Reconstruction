"""
GeoRecon AI (SIH-26158) - Phase 6.4: Live Hardware Telemetry Dashboard Test Suite
Verifies:
1. psutil and NVML (pynvml) telemetry collection (CPU, RAM, GPU, VRAM, Temp)
2. Thread-safe background collector daemon and queue delivery
3. Peak metrics tracking and diagnostics.json output
4. Tkinter Hardware Monitor Card responsiveness and Stage-Aware Highlighting
"""

import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.telemetry import HardwareSnapshot, HardwareTelemetryCollector
from pipeline.manager import PipelineManager
from pipeline.stage import StageType, StageStatus, PipelineEvent
from app import GeoReconApp


def run_hardware_monitor_tests():
    print("=" * 80)
    print("🚀 GeoRecon AI (SIH-26158) — Phase 6.4: Live Hardware Telemetry Test Suite")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Live Hardware Telemetry Collection (CPU, RAM, GPU, VRAM, Temp)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying System Telemetry Collection (psutil & NVML)...")
    collector = HardwareTelemetryCollector(interval_seconds=0.2)
    snap = collector.sample_now()

    print(f"  -> CPU Load: {snap.cpu_percent}%")
    print(f"  -> RAM Used: {snap.ram_used_gb} / {snap.ram_total_gb} GB ({snap.ram_percent}%)")
    print(f"  -> GPU Name: {snap.gpu_name}")
    print(f"  -> GPU Utilization: {snap.gpu_util_percent}%")
    print(f"  -> GPU VRAM: {snap.gpu_vram_used_mb} / {snap.gpu_vram_total_mb} MB ({snap.gpu_vram_percent}%)")
    print(f"  -> GPU Temperature: {snap.gpu_temperature_c}°C")
    print(f"  -> NVML Available: {snap.nvml_available}")

    assert snap.ram_total_gb > 0, "Failed to query system RAM"
    assert snap.ram_percent >= 0, "Invalid RAM percentage"
    assert snap.cpu_percent >= 0, "Invalid CPU percentage"
    assert snap.gpu_name, "GPU name should not be empty"
    assert snap.gpu_vram_total_mb > 0, "Failed to query GPU VRAM total"
    assert snap.gpu_temperature_c >= 0, "Invalid GPU temperature"
    print("  ✅ TEST 1 PASSED: Live CPU, RAM, GPU, VRAM, and Temp data collected successfully.")

    # -------------------------------------------------------------------------
    # TEST 2: Daemon Thread & Non-Blocking Queue Delivery
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Background Daemon Thread & Queue Delivery...")
    telemetry_q = queue.Queue(maxsize=10)
    collector.set_queue(telemetry_q)
    collector.start()

    time.sleep(0.7)  # Allow ~3 samples at 0.2s interval
    collector.stop()

    received_snapshots = []
    while not telemetry_q.empty():
        received_snapshots.append(telemetry_q.get_nowait())

    print(f"  -> Daemon dispatched {len(received_snapshots)} snapshots to telemetry queue")
    assert len(received_snapshots) >= 2, "Expected at least 2 snapshots in queue"
    assert isinstance(received_snapshots[0], HardwareSnapshot)
    print("  ✅ TEST 2 PASSED: Daemon thread pushes telemetry asynchronously without blocking.")

    # -------------------------------------------------------------------------
    # TEST 3: Peak Metric Tracking
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Session Peak Metric Tracking...")
    collector.reset_peaks()
    assert collector.cpu_peak_percent == 0.0
    assert collector.gpu_peak_percent == 0.0

    # Collect fresh sample to populate peaks
    collector.sample_now()
    peaks = collector.get_peak_metrics()
    print(f"  -> Peak Metrics: {peaks}")

    required_peak_keys = [
        "cpu_peak_percent",
        "gpu_peak_percent",
        "ram_peak_percent",
        "gpu_vram_peak_mb",
        "gpu_temperature_peak_c",
    ]
    for k in required_peak_keys:
        assert k in peaks, f"Missing required peak metric: {k}"
    print("  ✅ TEST 3 PASSED: Peak metrics tracked and reset successfully.")

    # -------------------------------------------------------------------------
    # TEST 4: Diagnostics.json Output Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Verifying diagnostics.json with Peak Hardware Statistics...")
    with tempfile.TemporaryDirectory(prefix="georecon_diag_hw_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        mgr = PipelineManager()

        diag_p = mgr.write_diagnostics(
            session_output_dir=tmp_p,
            worker_thread_status="completed",
            last_processed_stage="EXPORT",
            last_registered_camera_count=130,
        )

        assert diag_p.exists(), "diagnostics.json was not created"
        with open(diag_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("  -> diagnostics.json Hardware Fields:")
        for k in required_peak_keys:
            assert k in data, f"Missing {k} in diagnostics.json"
            print(f"     * {k}: {data[k]}")

        assert "CUDA detected" in data
        assert "GPU name" in data
        assert "COLMAP version" in data
        mgr.telemetry_collector.stop()
        print("  ✅ TEST 4 PASSED: All 5 peak hardware statistics written to diagnostics.json.")

    # -------------------------------------------------------------------------
    # TEST 5: Tkinter Hardware Monitor Card & Stage-Aware Highlighting
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Verifying Tkinter Hardware Monitor Card & Highlighting...")
    app = GeoReconApp()
    app.update()

    # 1. Verify Card Widgets Exist
    assert hasattr(app, "telemetry_card"), "telemetry_card missing in sidebar"
    assert hasattr(app, "lbl_hw_live"), "lbl_hw_live missing"
    assert hasattr(app, "lbl_hw_gpu_name"), "lbl_hw_gpu_name missing"
    assert hasattr(app, "lbl_hw_gpu_pct"), "lbl_hw_gpu_pct missing"
    assert hasattr(app, "lbl_hw_vram"), "lbl_hw_vram missing"
    assert hasattr(app, "lbl_hw_cpu_pct"), "lbl_hw_cpu_pct missing"
    assert hasattr(app, "lbl_hw_ram"), "lbl_hw_ram missing"
    assert hasattr(app, "lbl_hw_stage_mode"), "lbl_hw_stage_mode missing"
    assert hasattr(app, "canvas_hw_gpu_spark"), "canvas_hw_gpu_spark missing"
    assert hasattr(app, "canvas_hw_cpu_spark"), "canvas_hw_cpu_spark missing"
    assert hasattr(app, "canvas_hw_ram_spark"), "canvas_hw_ram_spark missing"
    assert hasattr(app, "lbl_hw_timer"), "lbl_hw_timer missing"
    assert hasattr(app, "lbl_hw_fps"), "lbl_hw_fps missing"
    assert hasattr(app, "lbl_hw_points"), "lbl_hw_points missing"
    print("  -> All 14 Hardware Monitor Card UI widgets verified.")

    # 2. Simulate Telemetry UI Update with Rolling History
    test_snap = HardwareSnapshot(
        cpu_percent=42.5,
        ram_used_gb=16.4,
        ram_total_gb=32.0,
        ram_percent=51.2,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        gpu_util_percent=88.0,
        gpu_vram_used_mb=4096.0,
        gpu_vram_total_mb=8188.0,
        gpu_vram_percent=50.0,
        gpu_temperature_c=68.0,
        nvml_available=True,
        cpu_history=[10.0, 20.0, 30.0, 42.5],
        gpu_history=[5.0, 15.0, 50.0, 88.0],
        ram_history=[40.0, 45.0, 50.0, 51.2],
    )
    app.is_processing = True
    app.active_session_start_t = time.time() - 75.0  # 1m 15s elapsed
    app._current_global_progress = 0.50             # 50% done -> ~1m 15s remaining
    app.rendered_points = 125400
    app.viewer_fps = 60.0

    app._update_hardware_monitor_ui(test_snap)
    app.update()

    assert "88%" in app.lbl_hw_gpu_pct.cget("text"), "GPU percent UI mismatch"
    assert "42%" in app.lbl_hw_cpu_pct.cget("text"), "CPU percent UI mismatch"
    assert "68°C" in app.lbl_hw_gpu_temp.cget("text"), "GPU temp UI mismatch"
    assert "4.0 / 8.0 GB" in app.lbl_hw_vram.cget("text"), "VRAM UI mismatch"
    assert "01:15" in app.lbl_hw_timer.cget("text"), "Elapsed timer UI mismatch"
    assert "ETA:" in app.lbl_hw_timer.cget("text"), "ETA timer UI mismatch"
    assert "60.0 FPS" in app.lbl_hw_fps.cget("text"), "FPS UI mismatch"
    assert "125,400 pts" in app.lbl_hw_points.cget("text"), "Points UI mismatch"
    print("  -> UI updated successfully with live hardware numbers, sparklines, timers, and viewport metrics.")

    # 3. Test Stage-Aware Highlighting
    # GPU Active Stages (Feature Extraction, Matching, Gaussian Splatting)
    app.update_hardware_highlight(StageType.COLMAP_FEATURES)
    app.update()
    assert "GPU ACTIVE" in app.lbl_hw_stage_mode.cget("text"), "Expected GPU ACTIVE mode"
    assert app.telemetry_card.cget("border_color") == "#0284C7", "Expected cyan GPU border"
    print("  -> Feature Extraction: verified GPU Highlighting (⚡ ACCELERATION: GPU ACTIVE)")

    app.update_hardware_highlight(StageType.COLMAP_MATCHING)
    app.update()
    assert "GPU ACTIVE" in app.lbl_hw_stage_mode.cget("text")
    print("  -> Feature Matching: verified GPU Highlighting")

    app.update_hardware_highlight(StageType.GAUSSIAN_SPLATTING)
    app.update()
    assert "GPU ACTIVE" in app.lbl_hw_stage_mode.cget("text")
    print("  -> Gaussian Splatting: verified GPU Highlighting")

    # CPU Active Stage (Sparse Mapper)
    app.update_hardware_highlight(StageType.COLMAP_MAPPER)
    app.update()
    assert "CPU ACTIVE" in app.lbl_hw_stage_mode.cget("text"), "Expected CPU ACTIVE mode"
    assert app.telemetry_card.cget("border_color") == "#F59E0B", "Expected amber CPU border"
    print("  -> Sparse Mapper: verified CPU Highlighting (🔥 PROCESSING: CPU ACTIVE)")

    # Balanced Stages (Export / Idle)
    app.update_hardware_highlight(StageType.EXPORT)
    app.update()
    assert "BALANCED" in app.lbl_hw_stage_mode.cget("text"), "Expected BALANCED mode"
    print("  -> Export: verified Balanced Highlighting (⚖️ HARDWARE: BALANCED)")

    app.destroy()
    print("  ✅ TEST 5 PASSED: Hardware Monitor Card layout, sparklines, timers, and stage-aware highlighting verified.")

    # -------------------------------------------------------------------------
    # TEST 6: Real GSplat Training Peak Metrics Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Verifying GSplat Real GPU Peaks and diagnostics.json recording...")
    from tests.test_gsplat_gpu_peaks import validate_gsplat_gpu_peaks
    validate_gsplat_gpu_peaks()
    print("  ✅ TEST 6 PASSED: Real GSplat GPU peak increase and diagnostics.json confirmed.")

    print("\n" + "=" * 80)
    print("🏆 ALL PHASE 6.4 HARDWARE MONITOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_hardware_monitor_tests()
