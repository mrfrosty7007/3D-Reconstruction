# Quick real verification of COLMAP 4.1.1 CUDA feature extraction & matching
import os
from pathlib import Path
import shutil
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.colmap_runner import ColmapRunner

def test_real_colmap_cuda():
    print("Testing real COLMAP 4.1.1 CUDA execution on GPU...")
    runner = ColmapRunner()
    runner.verify_gpu_flags()

    # Find an existing frames directory with a few images
    data_dir = Path("data")
    frames_dir = None
    for p in data_dir.glob("*/frames"):
        if p.is_dir() and len(list(p.glob("*.png"))) >= 5:
            frames_dir = p
            break

    if not frames_dir:
        print("No existing frames folder found, skipping real extraction test.")
        return

    print(f"Using source frames from: {frames_dir}")

    with tempfile.TemporaryDirectory(prefix="colmap_cuda_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_images = tmp_path / "images"
        test_images.mkdir()
        db_path = tmp_path / "database.db"

        # Copy 4 frames for quick verification
        for idx, f in enumerate(sorted(frames_dir.glob("*.png"))[:4]):
            shutil.copy(f, test_images / f.name)

        print(f"Copied {len(list(test_images.glob('*.png')))} images for quick CUDA SfM test.")

        # Stage 1: Feature Extraction
        print("Running Feature Extraction with GPU...")
        feat_ok = runner.run_feature_extraction(
            image_path=test_images,
            database_path=db_path,
            use_gpu=True,
            on_log=lambda l: print(f"  [Extraction Log] {l}"),
        )
        assert feat_ok, "Real feature extraction failed!"
        print("✅ Feature Extraction on GPU succeeded!")

        # Stage 2: Feature Matching
        print("Running Feature Matching with GPU...")
        match_ok = runner.run_feature_matching(
            database_path=db_path,
            matcher_type="exhaustive",
            use_gpu=True,
            on_log=lambda l: print(f"  [Matching Log] {l}"),
        )
        assert match_ok, "Real feature matching failed!"
        print("✅ Feature Matching on GPU succeeded!")

    print("\n🎉 Real COLMAP 4.1.1 CUDA verification completed successfully!")

if __name__ == "__main__":
    test_real_colmap_cuda()
