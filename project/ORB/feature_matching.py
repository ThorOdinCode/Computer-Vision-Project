from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loader import list_image_sequences, load_sequence_image


# ============================================================
# Configuration
# ============================================================

DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_adam"

DEFAULT_REFERENCE_IMAGE = 1
DEFAULT_TEST_IMAGE = 2

DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_TOP_K = 10

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "ORB" / "matching"


# ============================================================
# ORB
# ============================================================

def create_orb():
    """Create the ORB detector."""

    if hasattr(cv2, "ORB_create"):
        return cv2.ORB_create()

    raise RuntimeError(
        "ORB is not available in this OpenCV installation."
    )


# ============================================================
# Feature Extraction
# ============================================================

def extract_features(image, orb):
    """
    Extract ORB keypoints and descriptors from an image.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    return keypoints or [], descriptors


# ============================================================
# Feature Matching
# ============================================================

def match_features(
    reference_descriptors,
    test_descriptors,
    ratio_threshold=DEFAULT_RATIO_THRESHOLD
):
    """
    Match descriptors using Hamming distance and Lowe's ratio test.
    """

    if (
        reference_descriptors is None
        or test_descriptors is None
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        reference_descriptors,
        test_descriptors,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    good_matches.sort(
        key=lambda match: match.distance
    )

    return good_matches


# ============================================================
# Mutual Best Matching
# ============================================================

def mutual_best_matches(
    reference_descriptors,
    test_descriptors,
    matches
):
    """
    Keep matches where the reference descriptor and test
    descriptor agree on each other as their best match.
    """

    if not matches:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    mutual = matcher.match(
        reference_descriptors,
        test_descriptors
    )

    mutual_pairs = {
        (
            match.queryIdx,
            match.trainIdx
        )
        for match in mutual
    }

    return [
        match
        for match in matches
        if (
            match.queryIdx,
            match.trainIdx
        ) in mutual_pairs
    ]


# ============================================================
# Draw Matches
# ============================================================

def draw_matches(
    reference_image,
    reference_keypoints,
    test_image,
    test_keypoints,
    matches,
    top_k=DEFAULT_TOP_K
):
    """
    Draw the strongest feature matches.
    """

    selected_matches = sorted(
        matches,
        key=lambda match: match.distance
    )[:top_k]

    reference_bgr = cv2.cvtColor(
        reference_image,
        cv2.COLOR_RGB2BGR
    )

    test_bgr = cv2.cvtColor(
        test_image,
        cv2.COLOR_RGB2BGR
    )

    output = cv2.drawMatches(
        reference_bgr,
        reference_keypoints,
        test_bgr,
        test_keypoints,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# Process One Pair
# ============================================================

def process_pair(
    sequence,
    reference_image_id,
    test_image_id,
    output_dir,
    orb,
    ratio_threshold,
    top_k
):
    """
    Run ORB feature extraction and matching for one
    reference/test image pair.
    """

    reference_image = load_sequence_image(
        sequence,
        reference_image_id
    )

    test_image = load_sequence_image(
        sequence,
        test_image_id
    )

    # Extract ORB features
    reference_keypoints, reference_descriptors = (
        extract_features(
            reference_image,
            orb
        )
    )

    test_keypoints, test_descriptors = (
        extract_features(
            test_image,
            orb
        )
    )

    # Lowe ratio test
    good_matches = match_features(
        reference_descriptors,
        test_descriptors,
        ratio_threshold
    )

    # Mutual-best filtering
    final_matches = mutual_best_matches(
        reference_descriptors,
        test_descriptors,
        good_matches
    )

    # Create visualization
    output_image = draw_matches(
        reference_image,
        reference_keypoints,
        test_image,
        test_keypoints,
        final_matches,
        top_k
    )

    # Save output
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / (
            f"{sequence}_ref"
            f"{reference_image_id}_test"
            f"{test_image_id}_top"
            f"{top_k}.png"
        )
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            output_image,
            cv2.COLOR_RGB2BGR
        )
    )

    print(
        f"{sequence}: "
        f"{len(reference_keypoints)} reference keypoints | "
        f"{len(test_keypoints)} test keypoints | "
        f"{len(final_matches)} matches"
    )

    print(
        f"  Saved: {output_path}"
    )

    return {
        "sequence": sequence,
        "reference_keypoints": len(reference_keypoints),
        "test_keypoints": len(test_keypoints),
        "matches": len(final_matches),
        "output": output_path
    }


# ============================================================
# Run Matching
# ============================================================

def run_matching(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    reference_image_id=DEFAULT_REFERENCE_IMAGE,
    test_image_id=DEFAULT_TEST_IMAGE,
    ratio_threshold=DEFAULT_RATIO_THRESHOLD,
    top_k=DEFAULT_TOP_K,
    output_dir=DEFAULT_OUTPUT_DIR
):

    sequences = list_image_sequences()

    if not sequences:
        raise RuntimeError(
            "No HPatches sequences were found."
        )

    if lighting_sequence not in sequences:
        raise ValueError(
            f"Lighting sequence '{lighting_sequence}' "
            "was not found."
        )

    if viewpoint_sequence not in sequences:
        raise ValueError(
            f"Viewpoint sequence '{viewpoint_sequence}' "
            "was not found."
        )

    orb = create_orb()

    # Create separate output folders
    lighting_output = (
        output_dir / "lighting"
    )

    viewpoint_output = (
        output_dir / "viewpoint"
    )

    lighting_output.mkdir(
        parents=True,
        exist_ok=True
    )

    viewpoint_output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("HPatches ORB Feature Matching")
    print("=" * 70)

    print(
        f"\nTotal sequences available: {len(sequences)}"
    )

    # ========================================================
    # Lighting
    # ========================================================

    print(
        f"\n--- Illumination Sequence: "
        f"{lighting_sequence} ---"
    )

    lighting_result = process_pair(
        lighting_sequence,
        reference_image_id,
        test_image_id,
        lighting_output,
        orb,
        ratio_threshold,
        top_k
    )

    # ========================================================
    # Viewpoint
    # ========================================================

    print(
        f"\n--- Viewpoint Sequence: "
        f"{viewpoint_sequence} ---"
    )

    viewpoint_result = process_pair(
        viewpoint_sequence,
        reference_image_id,
        test_image_id,
        viewpoint_output,
        orb,
        ratio_threshold,
        top_k
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("MATCHING COMPLETE")
    print("=" * 70)

    print(
        f"\nLighting output:"
    )
    print(
        f"  {lighting_output}"
    )

    print(
        f"Viewpoint output:"
    )
    print(
        f"  {viewpoint_output}"
    )

    print()
    print(
        f"Lighting matches: "
        f"{lighting_result['matches']}"
    )

    print(
        f"Viewpoint matches: "
        f"{viewpoint_result['matches']}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run ORB feature matching on HPatches "
            "illumination and viewpoint sequences."
        )
    )

    parser.add_argument(
        "--lighting-sequence",
        default=DEFAULT_LIGHTING_SEQUENCE
    )

    parser.add_argument(
        "--viewpoint-sequence",
        default=DEFAULT_VIEWPOINT_SEQUENCE
    )

    parser.add_argument(
        "--reference-image",
        type=int,
        default=DEFAULT_REFERENCE_IMAGE
    )

    parser.add_argument(
        "--test-image",
        type=int,
        default=DEFAULT_TEST_IMAGE
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )

    args = parser.parse_args()

    run_matching(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        reference_image_id=args.reference_image,
        test_image_id=args.test_image,
        ratio_threshold=args.ratio,
        top_k=args.top_k,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()