"""
GeoRecon AI - Duplicate Filter Module
Removes redundant and near-identical consecutive frames using SSIM and ORB feature matching.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger("GeoRecon.DuplicateFilter")


@dataclass
class DuplicateFilterResult:
    """Outcome of duplicate and redundancy removal."""
    retained_frames: List[Path] = field(default_factory=list)
    discarded_duplicates: List[Path] = field(default_factory=list)
    comparison_metrics: List[Dict[str, Any]] = field(default_factory=list)
    total_evaluated: int = 0
    ssim_threshold_used: float = 0.90
    orb_threshold_used: float = 0.85

    @property
    def retained_count(self) -> int:
        return len(self.retained_frames)

    @property
    def discarded_count(self) -> int:
        return len(self.discarded_duplicates)


class DuplicateFilter:
    """Filters consecutive redundant frames using Structural Similarity (SSIM) and ORB feature matching."""

    def __init__(
        self,
        ssim_threshold: float = 0.90,
        orb_match_threshold: float = 0.85,
        max_features: int = 1500,
    ):
        self.ssim_threshold = ssim_threshold
        self.orb_match_threshold = orb_match_threshold
        self.max_features = max_features
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @staticmethod
    def compute_ssim(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
        """
        Computes Structural Similarity Index (SSIM) between two grayscale images.
        Vectorized Gaussian-filtered SSIM calculation (Wang et al. 2004).
        """
        # Ensure identical dimensions
        if img1_gray.shape != img2_gray.shape:
            img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1_gray.astype(np.float64)
        img2 = img2_gray.astype(np.float64)

        kernel_size = 11
        sigma = 1.5

        mu1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), sigma)
        mu2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), sigma)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (kernel_size, kernel_size), sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (kernel_size, kernel_size), sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (kernel_size, kernel_size), sigma) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / (denominator + 1e-10)

        return float(np.clip(ssim_map.mean(), 0.0, 1.0))

    def compute_orb_similarity(
        self,
        kp1: List[cv2.KeyPoint],
        des1: Optional[np.ndarray],
        kp2: List[cv2.KeyPoint],
        des2: Optional[np.ndarray],
    ) -> float:
        """
        Calculates feature match similarity ratio and displacement between two descriptor sets.
        Returns a similarity metric between 0.0 (completely distinct) and 1.0 (identical viewpoints).
        """
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.5  # Inconclusive, fallback to SSIM

        try:
            matches = self.matcher.match(des1, des2)
            if not matches:
                return 0.0

            # Sort matches by hamming distance
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 45.0]

            min_kps = min(len(kp1), len(kp2))
            match_ratio = len(good_matches) / max(1, min_kps)
            return float(np.clip(match_ratio, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"ORB matching error: {e}")
            return 0.5

    def filter_duplicates(
        self,
        frame_paths: List[Path],
        delete_discarded: bool = True,
        on_progress: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> DuplicateFilterResult:
        """
        Sequentially filters near-duplicate frames using SSIM and ORB feature matching.
        Keeps only frames with sufficient viewpoint variation / scene changes.
        """
        total = len(frame_paths)
        if total <= 1:
            return DuplicateFilterResult(
                retained_frames=list(frame_paths),
                total_evaluated=total,
                ssim_threshold_used=self.ssim_threshold,
                orb_threshold_used=self.orb_match_threshold,
            )

        retained: List[Path] = [frame_paths[0]]  # Always keep first keyframe
        discarded: List[Path] = []
        metrics_log: List[Dict[str, Any]] = []

        logger.info(
            f"Starting duplicate detection on {total} frames "
            f"(SSIM Thresh: {self.ssim_threshold:.2f}, ORB Thresh: {self.orb_match_threshold:.2f})..."
        )

        # Load initial reference frame
        ref_img = cv2.imread(str(frame_paths[0]))
        if ref_img is None:
            raise RuntimeError(f"Failed to read initial keyframe: {frame_paths[0]}")

        # Downscale for high-speed SSIM computation
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        eval_w, eval_h = 640, int(640 * (ref_gray.shape[0] / ref_gray.shape[1]))
        ref_gray_small = cv2.resize(ref_gray, (eval_w, eval_h))
        ref_kp, ref_des = self.orb.detectAndCompute(ref_gray, None)

        for idx, candidate_path in enumerate(frame_paths[1:], start=2):
            if stop_event and stop_event.is_set():
                logger.warning("Duplicate filtering cancelled by stop event.")
                break

            cand_img = cv2.imread(str(candidate_path))
            if cand_img is None:
                discarded.append(candidate_path)
                continue

            cand_gray = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY)
            cand_gray_small = cv2.resize(cand_gray, (eval_w, eval_h))

            # 1. Compute SSIM
            ssim_val = self.compute_ssim(ref_gray_small, cand_gray_small)

            # 2. If SSIM is high, compute ORB feature match ratio
            orb_similarity = 0.0
            is_duplicate = False

            if ssim_val >= self.ssim_threshold:
                cand_kp, cand_des = self.orb.detectAndCompute(cand_gray, None)
                orb_similarity = self.compute_orb_similarity(ref_kp, ref_des, cand_kp, cand_des)
                
                # If both structural similarity and feature match indicate negligible viewpoint shift
                if orb_similarity >= self.orb_match_threshold or ssim_val >= 0.96:
                    is_duplicate = True
            else:
                cand_kp, cand_des = None, None

            record = {
                "frame": candidate_path.name,
                "reference": retained[-1].name,
                "ssim": round(ssim_val, 4),
                "orb_similarity": round(orb_similarity, 4),
                "is_duplicate": is_duplicate,
            }
            metrics_log.append(record)

            if is_duplicate:
                discarded.append(candidate_path)
                if delete_discarded:
                    try:
                        candidate_path.unlink(missing_ok=True)
                    except Exception as err:
                        logger.debug(f"Failed to unlink duplicate frame {candidate_path}: {err}")
            else:
                retained.append(candidate_path)
                # Advance reference frame
                ref_gray_small = cand_gray_small
                if cand_kp is None or cand_des is None:
                    ref_kp, ref_des = self.orb.detectAndCompute(cand_gray, None)
                else:
                    ref_kp, ref_des = cand_kp, cand_des

            # Emit progress update
            if idx % 10 == 0 or idx == total:
                progress = idx / total
                msg = f"Duplicate Filter: {idx}/{total} frames (Retained: {len(retained)}, Duplicates: {len(discarded)})"
                if on_progress:
                    on_progress(
                        progress,
                        msg,
                        {
                            "Scanned": f"{idx}/{total}",
                            "Unique Frames": len(retained),
                            "Duplicates Discarded": len(discarded),
                            "Latest SSIM": f"{ssim_val:.3f}",
                        },
                    )

        logger.info(
            f"Duplicate filtering finished: {len(retained)}/{total} unique frames kept, "
            f"{len(discarded)} duplicates removed."
        )

        return DuplicateFilterResult(
            retained_frames=retained,
            discarded_duplicates=discarded,
            comparison_metrics=metrics_log,
            total_evaluated=total,
            ssim_threshold_used=self.ssim_threshold,
            orb_threshold_used=self.orb_match_threshold,
        )
