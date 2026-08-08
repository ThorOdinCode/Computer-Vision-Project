from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import data_loader
import feature_extraction as base_extraction


# ============================================================
# Configuration
# ============================================================

DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_bark"
DEFAULT_MAX_POINTS = base_extraction.DEFAULT_MAX_POINTS
DEFAULT_NFEATURES = 5000
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "ORB" / "extraction_samples_max"

DATASET_CANDIDATES = (
    PROJECT_DIR / "data" / "hpatches-sequences-release",
    PROJECT_DIR.parent / "hpatches-benchmark" / "data" / "hpatches-sequences-release",
)

for candidate in DATASET_CANDIDATES:
    if candidate.exists():
        data_loader.SEQUENCE_DATASET_PATH = str(candidate)
        break
else:
    raise FileNotFoundError(
        "Could not find the HPatches sequence dataset. "
        "Checked: " + ", ".join(str(path) for path in DATASET_CANDIDATES)
    )


# ============================================================
# Create ORB
# ============================================================

def create_orb(
    nfeatures=DEFAULT_NFEATURES,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
):
    """
    Create an ORB detector with a much higher feature cap.

    ORB still needs a finite `nfeatures` value, but this script uses a
    much larger default than the original `feature_extraction.py` so you
    can inspect the strongest keypoints without the 100-feature cap.
    """

    if hasattr(cv2, "ORB_create"):
        return cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold,
        )

    raise RuntimeError("OpenCV was built without ORB support.")


# ============================================================
# Dataset Run
# ============================================================

def run_extraction_evaluation(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    max_points=DEFAULT_MAX_POINTS,
    nfeatures=DEFAULT_NFEATURES,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run the original ORB extraction pipeline, but with a higher cap on
    detected keypoints.
    """

    original_create_orb = base_extraction.create_orb

    try:
        # Patch the base module so its existing dataset runner uses this
        # higher-cap ORB factory.
        base_extraction.create_orb = lambda: create_orb(nfeatures=nfeatures)

        return base_extraction.process_dataset(
            lighting_sequence=lighting_sequence,
            viewpoint_sequence=viewpoint_sequence,
            max_points=max_points,
            output_dir=output_dir,
        )
    finally:
        base_extraction.create_orb = original_create_orb


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ORB on the full HPatches sequence dataset using a much "
            "higher feature cap than the default ORB extraction script."
        )
    )
    parser.add_argument(
        "--lighting-sequence",
        default=DEFAULT_LIGHTING_SEQUENCE,
        help="HPatches i_ sequence for illumination samples.",
    )
    parser.add_argument(
        "--viewpoint-sequence",
        default=DEFAULT_VIEWPOINT_SEQUENCE,
        help="HPatches v_ sequence for viewpoint samples.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum keypoints drawn per sample.",
    )
    parser.add_argument(
        "--nfeatures",
        type=int,
        default=DEFAULT_NFEATURES,
        help=(
            "ORB feature cap. Larger values keep more keypoints; "
            "the original script used 100."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    args = parser.parse_args()

    run_extraction_evaluation(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        max_points=args.max_points,
        nfeatures=args.nfeatures,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
