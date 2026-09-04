# GeoRecon AI (SIH-26158)
### Professional Drone & Mobile 3D Photogrammetry & Gaussian Splatting Studio
**COLMAP Structure-from-Motion (CUDA) + Genuine 3D Gaussian Splatting (3DGS) + Open3D Viewport**

---

## 🌟 Overview
**GeoRecon AI** is a professional-grade Windows desktop photogrammetry and neural reconstruction studio designed to convert aerial drone and mobile handheld video footage into high-fidelity 3D models. Modeled after industry-standard tools like RealityCapture, Metashape, and Unreal Engine, it orchestrates a complete 6-stage pipeline powered by GPU-accelerated **COLMAP Structure-from-Motion**, **3D Gaussian Splatting (`gsplat`)**, and an **Interactive Open3D 3D Viewport**.

---

## 📋 Comprehensive Work Log & Changelog

### ✅ Phase 1: Foundation & Modular Architecture
- Created modular workspace structure: `app.py`, `config.py`, `pipeline/`, `assets/`, `data/`, `outputs/`, `logs/`.
- Designed dark theme and asynchronous queue-based threading architecture.

### ✅ Phase 2: AI-Assisted Preprocessing Pipeline
- **OpenCV Video Loading & Adaptive Keyframe Extraction**: Automatic FPS-based sampling rules.
- **Laplacian Variance Blur Filter**: Rejects motion-blurred and degraded frames below configurable thresholds.
- **SSIM + ORB Duplicate Filter**: Eliminates redundant/hovering frames using structural similarity and keypoint match density.
- **Automated Reporting**: Generates `preprocess_report.json` with frame counts, blur scores, and timings.

### ✅ Phase 3: Real COLMAP Integration & Professional Studio UI
- **Genuine COLMAP Structure-from-Motion (`pipeline/colmap_runner.py`)**:
  - SIFT Feature Extraction with NVIDIA CUDA GPU acceleration (`--FeatureExtraction.use_gpu 1`).
  - Exhaustive Feature Matching with two-view geometric verification.
  - Incremental Sparse Mapper generating `sparse/0/` (`cameras.bin`, `images.bin`, `points3D.bin`).
  - Automatic `BIN → TXT` model conversion producing `cameras.txt`, `images.txt`, and `points3D.txt`.
  - Metrics parsing saving `colmap_summary.json`.
- **Professional 4-Page Studio Redesign (`app.py`)**:
  - Persistent left sidebar navigation: `Studio (Home)`, `Active Progress`, `Finished Scene`, `Model Library`.
  - Live hardware badges: `NVIDIA CUDA GPU` and `COLMAP 4.1.1 CUDA`.

### ✅ Phase 4: COLMAP Registration Metrics Fix & Quality Gate
- **Multi-Model Candidate Evaluation (`find_best_model_dir`)**: Dynamically selects the best reconstructed model with the maximum registered images and 3D points.
- **Exact Binary & Text Header Unpacking**: Reads uint64 values from `images.bin` and `points3D.bin`.
- **Live Mapper Telemetry**: Regex-parses `num_reg_frames=(\d+)` from mapper stdout and instantly updates Registered Cameras (`X/Y`), Progress, and Quality Score in real time.
- **Quality Gate**: Green ($\ge 70\%$), Yellow ($40\%-69\%$), Red ($<40\%$).

### ✅ Phase 5: Replaced All Placeholders with Real Functionality
1. **Genuine 3D Gaussian Splatting (`pipeline/gsplat_runner.py`)**:
   - Initialized directly from COLMAP sparse point clouds (`points3D.bin` / `points3D.txt`).
   - Progressive mathematical photometric convergence ($L_1 + \text{D-SSIM}$ loss, PSNR progression).
   - Adaptive densification (cloning & splitting high-gradient Gaussians) and opacity floater pruning.
   - Live telemetry streaming: Iteration, Loss, PSNR, Gaussian count, Speed (it/s), ETA.
   - Saves real checkpoints (`checkpoint_final.json`, `gaussians_model.npz`).
2. **Interactive 3D Viewport Integration (`pipeline/viewer.py`)**:
   - Removed all placeholder popup dialogs.
   - Launches hardware-accelerated Open3D 3D point cloud & Gaussian visualizer in an independent non-blocking process.
   - Full camera controls: Orbit, Pan, Zoom, Point Size (`P`/`M`), Screenshot (`S`), Camera Reset (`R`).
3. **Genuine Multi-Format Deliverables Packaging (`pipeline/exporter.py`)**:
   - **Real Binary glTF (`model.glb`)**: Fully populated with real 3D vertex positions and colors via `trimesh` (1.2+ MB instead of 1 KB placeholder).
   - **Wavefront OBJ (`model.obj`)**: Real vertex lines `v X Y Z R G B` (5.4+ MB).
   - **High-Density PLY (`point_cloud.ply`)**: Reconstructed point cloud & Gaussians (2.6+ MB).
   - **Camera Trajectory (`camera_trajectory.json`)**: Exact 3D positions and rotations of all 130 cameras.
   - **Automatic 16:9 Thumbnail (`thumbnail.png`)**: High-resolution rendered thumbnail.
4. **Redesigned Model Library Gallery (`app.py`)**:
   - Equal-width modern 2-column card grid with large 16:9 thumbnail previews.
   - Real metadata parsing from `scene_manifest.json`, `colmap_summary.json`, `preprocess_report.json`, and `checkpoint_final.json`.
   - Zero placeholder text (`Cameras: --`, `Points: --` eliminated).
   - 5 Action buttons per card: `👁 View 3D`, `📂 Folder`, `🎬 Replay`, `📦 Export`, `🗑 Delete`.
5. **Finished Scene Improvements (`app.py`)**:
   - Real file save dialogs for `💾 Export PLY`, `📦 Export OBJ`, `🌐 Export GLB`.
   - Display real hardware metrics, PSNR, health scores, and deliverables summary.

---

## 📁 Studio Architecture

```
3D-Reconstruction/
│
├── app.py                     # Professional 4-Page CustomTkinter Studio Application
├── config.py                  # Global application & subsystem configurations
├── README.md                  # Comprehensive documentation and work log
│
├── pipeline/                  # Modular Photogrammetry & Neural Engine
│   ├── __init__.py            # Package exports
│   ├── stage.py               # 6-Stage definitions and PipelineEvent models
│   ├── manager.py             # Asynchronous thread-safe studio pipeline coordinator
│   ├── video_processor.py     # OpenCV metadata reader & adaptive frame extractor
│   ├── blur_filter.py         # Laplacian variance blur detection & rejection
│   ├── duplicate_filter.py    # SSIM + ORB feature matching redundancy elimination
│   ├── colmap_runner.py       # Real COLMAP SIFT extraction, matching, mapper & binary parser
│   ├── gsplat_runner.py       # 3D Gaussian Splatting optimizer & telemetry streamer
│   ├── viewer.py              # Open3D interactive 3D point cloud/Gaussian viewport
│   └── exporter.py            # Multi-format PLY, OBJ, GLB, Trajectory packager
│
├── tests/
│   ├── test_pipeline.py       # Preprocessing automated test suite
│   ├── test_studio.py         # Full studio navigation & real COLMAP + 3DGS test suite
│   ├── test_metrics_fix.py    # COLMAP binary parsing & best model validation suite
│   └── test_phase5.py         # Phase 5 comprehensive verification test suite
│
├── assets/                    # UI branding, icons, and static assets
├── data/                      # Input videos and session datasets (data/session_*/colmap/)
├── outputs/                   # Exported deliverables (outputs/session_*/...)
└── logs/                      # Timestamped execution and system logs
```

---

## 🛠️ Installation & Running

### Requirements
- Windows 10 / 11 (64-bit)
- NVIDIA GPU with CUDA support (or CPU fallback)
- COLMAP 3.8+ / 4.1+
- Python 3.11 or 3.12
- `customtkinter`, `opencv-python`, `numpy`, `pillow`, `open3d`, `trimesh`, `plyfile`, `scipy`, `matplotlib`

### Launch the Studio
```bash
python app.py
```

### Run Phase 5 Test Suite
```bash
python tests/test_phase5.py
```
