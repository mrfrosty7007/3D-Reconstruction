import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
venv_site = ROOT / ".venv" / "Lib" / "site-packages"
if venv_site.exists():
    sys.path.insert(0, str(venv_site))

from pipeline.gsplat_runner import GSplatRunner


class TestGSplatPipelineReal(unittest.TestCase):
    def test_gsplat_training_and_artifacts(self):
        sparse_dir = ROOT / "data" / "trial2_20260904_094601" / "colmap" / "sparse"
        images_dir = ROOT / "data" / "trial2_20260904_094601" / "frames"
        output_dir = ROOT / "outputs" / "trial2_20260904_094601"

        self.assertTrue(sparse_dir.exists(), f"Sparse dir missing: {sparse_dir}")
        self.assertTrue(images_dir.exists(), f"Frames dir missing: {images_dir}")

        import torch
        self.assertTrue(torch.cuda.is_available(), "CUDA GPU should be available")
        runner = GSplatRunner()

        telemetry_records = []

        def on_telemetry(data):
            telemetry_records.append(data)
            if data["iteration"] % 50 == 0 or data["iteration"] == data["total_iterations"]:
                print(
                    f"[{data['iteration']}/{data['total_iterations']}] "
                    f"Loss: {data['loss']:.4f} | PSNR: {data['psnr']:.2f} dB | "
                    f"Gaussians: {data['gaussian_count']} | Speed: {data['iter_speed']} it/s",
                    flush=True
                )

        t0 = time.time()
        result = runner.train_gaussian_splatting(
            sparse_dir=sparse_dir,
            images_dir=images_dir,
            output_dir=output_dir,
            total_iterations=150,
            on_telemetry=on_telemetry,
        )
        elapsed = time.time() - t0

        print(f"\n[Test] Training completed in {elapsed:.2f}s")
        print(f"[Test] Converged: {result.is_converged}")
        print(f"[Test] Final PSNR: {result.final_psnr} dB")
        print(f"[Test] Final Gaussians: {result.final_gaussian_count}")
        print(f"[Test] Device: {result.device_used}")

        self.assertTrue(result.is_converged)
        self.assertGreater(result.final_psnr, 14.0, "PSNR should improve to >14dB within 150 iterations")
        self.assertGreater(result.final_gaussian_count, 25082, "Gaussians should have grown via densification")

        # Check deliverables exist
        splat_file = output_dir / "point_cloud.splat"
        ply_file = output_dir / "point_cloud.ply"
        npz_file = output_dir / "checkpoints" / "gaussians_model.npz"

        self.assertTrue(splat_file.exists(), "point_cloud.splat must be generated")
        self.assertGreater(splat_file.stat().st_size, 32 * 25082, "point_cloud.splat must contain real binary splats")

        self.assertTrue(ply_file.exists(), "point_cloud.ply must exist")
        self.assertGreater(ply_file.stat().st_size, 100_000)

        self.assertTrue(npz_file.exists(), "checkpoints/gaussians_model.npz must exist")


if __name__ == "__main__":
    unittest.main()
