"""
GeoRecon AI - Blur Filter Module
Implements Variance of Laplacian blur detection to filter out motion-blurred frames.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger("GeoRecon.BlurFilter")


@dataclass
class BlurFilterResult:
    """Summary and artifacts produced by the blur filtering stage."""
    retained_frames: List[Path] = field(default_factory=list)
    discarded_frames: List[Path] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    average_score: float = 0.0
    threshold_used: float = 80.0
    total_evaluated: int = 0

    @property
    def retained_count(self) -> int:
        return len(self.retained_frames)

    @property
    def discarded_count(self) -> int:
        return len(self.discarded_frames)


class BlurFilter:
    """Computes Laplacian variance blur scores and filters motion-blurred frames."""

    def __init__(self, blur_threshold: float = 80.0):
        self.blur_threshold = blur_threshold

    @staticmethod
    def compute_blur_score(image: np.ndarray) -> float:
        """
        Computes the Variance of Laplacian on grayscale image.
        Higher values indicate sharp edges and fine details.
        Lower values indicate blurriness or flat/uniform texture.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return laplacian_var

    def filter_frames(
        self,
        frame_paths: List[Path],
        delete_discarded: bool = True,
        on_progress: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> BlurFilterResult:
        """
        Evaluates a list of frame images against the blur threshold.
        Frames below the threshold are discarded (and optionally removed from disk).
        """
        total = len(frame_paths)
        if total == 0:
            return BlurFilterResult(threshold_used=self.blur_threshold)

        retained: List[Path] = []
        discarded: List[Path] = []
        scores: Dict[str, float] = {}
        total_score_accum = 0.0

        logger.info(f"Starting blur assessment on {total} frames (Threshold: {self.blur_threshold:.1f})...")

        for idx, frame_path in enumerate(frame_paths, start=1):
            if stop_event and stop_event.is_set():
                logger.warning("Blur filtering cancelled by stop event.")
                break

            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning(f"Could not read frame for blur scoring: {frame_path.name}")
                discarded.append(frame_path)
                continue

            score = self.compute_blur_score(img)
            scores[frame_path.name] = round(score, 2)
            total_score_accum += score

            if score >= self.blur_threshold:
                retained.append(frame_path)
            else:
                discarded.append(frame_path)
                if delete_discarded:
                    try:
                        frame_path.unlink(missing_ok=True)
                    except Exception as err:
                        logger.debug(f"Failed to unlink blurry frame {frame_path}: {err}")

            # Emit progress update periodically
            if idx % 10 == 0 or idx == total:
                progress = idx / total
                msg = f"Evaluated {idx}/{total} frames (Retained: {len(retained)}, Blurry Discarded: {len(discarded)})"
                if on_progress:
                    on_progress(
                        progress,
                        msg,
                        {
                            "Evaluated": f"{idx}/{total}",
                            "Sharp Frames": len(retained),
                            "Blurry Filtered": len(discarded),
                            "Mean Score": f"{(total_score_accum / idx):.1f}",
                        },
                    )

        avg_score = (total_score_accum / max(1, len(scores))) if scores else 0.0
        logger.info(
            f"Blur filtering finished: {len(retained)}/{total} frames retained, "
            f"{len(discarded)} blurry frames removed (Avg Blur Score: {avg_score:.2f})"
        )

        return BlurFilterResult(
            retained_frames=retained,
            discarded_frames=discarded,
            scores=scores,
            average_score=round(avg_score, 2),
            threshold_used=self.blur_threshold,
            total_evaluated=total,
        )
