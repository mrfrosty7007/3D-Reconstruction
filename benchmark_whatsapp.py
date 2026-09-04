"""
GeoRecon AI (SIH-26158) - WhatsApp Demo Video Real Benchmark
Runs the complete reconstruction pipeline on the exact WhatsApp demo video:
data/WhatsApp Video 2026-09-03 at 9.30.17 PM.mp4

Measures:
- Original video length
- Original frame count
- Frames kept after adaptive sampling
- Stage 1 time
- Stage 2 time
- Stage 3 time
- Stage 4 (Sparse Reconstruction) time
- Cameras registered
- Sparse points generated
- Stage 5 training speed (iter/s)
- Peak CPU
- Peak GPU
- Peak RAM
- Total reconstruction time
"""

import json
import os
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, AppConfig
from pipeline.manager import PipelineManager
from pipeline.video_processor import VideoProcessor
from pipeline.stage import StageType, StageStatus, PipelineEvent


def run_benchmark():
    video_path = Path("data/WhatsApp Video 2026-09-03 at 9.30.17 PM.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print("=" * 80)
    print("🚀 GeoRecon AI — WhatsApp Video Real Benchmark (SIH-26158)")
    print("=" * 80)

    # 1. Video Inspection
    vp = VideoProcessor()
    meta = vp.inspect_video(video_path)
    print(f"\n[1] Source Video Metadata:")
    print(f"  -> File: {meta.filename}")
    print(f"  -> Resolution: {meta.resolution_str}")
    print(f"  -> Duration: {meta.duration_seconds:.2f} seconds ({meta.duration_formatted})")
    print(f"  -> FPS: {meta.fps:.1f}")
    print(f"  -> Total Frames: {meta.total_frames}")

    # Set up config with realistic GSplat iteration count for benchmark (e.g. 2000 iters for full measurement)
    config = AppConfig(
        app_name=DEFAULT_CONFIG.app_name,
        app_subtitle=DEFAULT_CONFIG.app_subtitle,
        outputs_dir=DEFAULT_CONFIG.outputs_dir,
        data_dir=DEFAULT_CONFIG.data_dir,
    )
    config.gsplat.iterations = 2000  # Genuine CUDA training run measuring accurate it/s

    # Metrics container
    metrics = {
        "video_duration_seconds": meta.duration_seconds,
        "video_total_frames": meta.total_frames,
        "frames_kept": 0,
        "stage1_time_s": 0.0,
        "stage2_time_s": 0.0,
        "stage3_time_s": 0.0,
        "stage4_time_s": 0.0,
        "stage5_time_s": 0.0,
        "stage6_time_s": 0.0,
        "total_time_s": 0.0,
        "cameras_registered": 0,
        "total_cameras": 0,
        "sparse_points": 0,
        "gsplat_speed_iters_per_sec": 0.0,
        "gsplat_final_psnr": 0.0,
        "peak_cpu_percent": 0.0,
        "peak_gpu_percent": 0.0,
        "peak_ram_percent": 0.0,
        "peak_ram_gb": 0.0,
        "peak_vram_mb": 0.0,
    }

    stage_timings = {}
    current_stage = None
    stage_start_t = 0.0

    def on_event(event: PipelineEvent):
        nonlocal current_stage, stage_start_t
        if event.status == StageStatus.RUNNING and event.stage != current_stage:
            if current_stage is not None:
                elapsed = time.perf_counter() - stage_start_t
                stage_timings[current_stage] = elapsed
            current_stage = event.stage
            stage_start_t = time.perf_counter()

        if event.stage == StageType.FRAME_EXTRACTION and event.status == StageStatus.COMPLETED:
            metrics["frames_kept"] = event.total_cameras
            print(f"  [Stage 1 Done] Frames kept after filtering: {event.total_cameras}")

        elif event.stage == StageType.COLMAP_MAPPER and event.status == StageStatus.COMPLETED:
            metrics["cameras_registered"] = event.registered_cameras
            metrics["total_cameras"] = event.total_cameras
            metrics["sparse_points"] = event.sparse_points
            print(f"  [Stage 4 Done] Registered: {event.registered_cameras}/{event.total_cameras}, Points: {event.sparse_points:,}")

        elif event.stage == StageType.GAUSSIAN_SPLATTING and event.metrics and "Speed" in event.metrics:
            spd_str = event.metrics["Speed"].replace(" it/s", "")
            try:
                metrics["gsplat_speed_iters_per_sec"] = float(spd_str)
            except ValueError:
                pass

        if event.status == StageStatus.COMPLETED:
            elapsed = time.perf_counter() - stage_start_t
            stage_timings[event.stage] = elapsed

    mgr = PipelineManager(config=config, event_callback=on_event)
    mgr.auto_continue_yellow = True
    collector = mgr.telemetry_collector
    collector.reset_peaks()

    session_name = f"benchmark_whatsapp_{int(time.time())}"
    output_dir = config.outputs_dir / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2] Executing Benchmark Pipeline -> {session_name}")
    benchmark_start_t = time.perf_counter()

    # Run pipeline synchronously
    mgr._run_pipeline_thread(video_path, output_dir, session_name)
    total_time = time.perf_counter() - benchmark_start_t

    # Harvest final peaks
    peaks = collector.get_peaks()
    snap = collector.sample_now()

    metrics["stage1_time_s"] = stage_timings.get(StageType.FRAME_EXTRACTION, 0.0)
    metrics["stage2_time_s"] = stage_timings.get(StageType.COLMAP_FEATURES, 0.0)
    metrics["stage3_time_s"] = stage_timings.get(StageType.COLMAP_MATCHING, 0.0)
    metrics["stage4_time_s"] = stage_timings.get(StageType.COLMAP_MAPPER, 0.0)
    metrics["stage5_time_s"] = stage_timings.get(StageType.GAUSSIAN_SPLATTING, 0.0)
    metrics["stage6_time_s"] = stage_timings.get(StageType.EXPORT, 0.0)
    metrics["total_time_s"] = total_time

    metrics["peak_cpu_percent"] = peaks.get("cpu_peak_percent", 0.0)
    metrics["peak_gpu_percent"] = peaks.get("gpu_peak_percent", 0.0)
    metrics["peak_ram_percent"] = peaks.get("ram_peak_percent", 0.0)
    metrics["peak_ram_gb"] = (peaks.get("ram_peak_percent", 0.0) / 100.0) * snap.ram_total_gb
    metrics["peak_vram_mb"] = peaks.get("gpu_vram_peak_mb", 0.0)

    # Check colmap summary
    summary_file = output_dir / "colmap_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            s_data = json.load(f)
            metrics["cameras_registered"] = s_data.get("registered_cameras", metrics["cameras_registered"])
            metrics["total_cameras"] = s_data.get("total_cameras", metrics["total_cameras"])
            metrics["sparse_points"] = s_data.get("sparse_point_count", metrics["sparse_points"])

    # Write benchmark metrics to JSON
    benchmark_res_file = output_dir / "benchmark_results.json"
    with open(benchmark_res_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("🏆 WHATSAPP DEMO BENCHMARK COMPLETE — MEASURED RESULTS")
    print("=" * 80)
    print(f"• Original Video Length: {metrics['video_duration_seconds']:.2f}s ({meta.duration_formatted})")
    print(f"• Original Frame Count: {metrics['video_total_frames']} frames")
    print(f"• Frames Kept After Adaptive Sampling: {metrics['frames_kept']} frames")
    print(f"• Stage 1 (Frame Extraction & Filter) Time: {metrics['stage1_time_s']:.2f}s")
    print(f"• Stage 2 (GPU SIFT Extraction) Time: {metrics['stage2_time_s']:.2f}s")
    print(f"• Stage 3 (GPU Feature Matching) Time: {metrics['stage3_time_s']:.2f}s")
    print(f"• Stage 4 (Sparse SfM & Bundle Adjustment) Time: {metrics['stage4_time_s']:.2f}s")
    print(f"• Cameras Registered: {metrics['cameras_registered']}/{metrics['total_cameras']} ({metrics['cameras_registered']/max(1, metrics['total_cameras'])*100:.1f}%)")
    print(f"• Sparse 3D Points Generated: {metrics['sparse_points']:,}")
    print(f"• Stage 5 3DGS Training Speed: {metrics['gsplat_speed_iters_per_sec']:.1f} it/s (Time: {metrics['stage5_time_s']:.2f}s)")
    print(f"• Peak CPU Usage: {metrics['peak_cpu_percent']:.1f}%")
    print(f"• Peak GPU Usage: {metrics['peak_gpu_percent']:.1f}%")
    print(f"• Peak System RAM: {metrics['peak_ram_gb']:.2f} GB ({metrics['peak_ram_percent']:.1f}%)")
    print(f"• Peak GPU VRAM: {metrics['peak_vram_mb']:.1f} MB")
    print(f"• Total Reconstruction Time: {metrics['total_time_s']:.2f}s ({metrics['total_time_s']/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
