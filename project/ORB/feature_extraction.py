from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

#This gives a compiler error but works when you run the code
from data_loader import list_image_sequences, load_sequence_image


# ============================================================
# Configuration
# ============================================================

DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_bark"

DEFAULT_SAMPLE_IMAGE_IDS = (1, 2, 3, 4, 5)

DEFAULT_MAX_POINTS = 150

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "ORB" / "extraction_samples"
)

HPATCHES_IMAGE_IDS = range(1, 7)


# ============================================================
# Create ORB
# ============================================================
"""
You must set nfeatures to some number as it treats 0 as literal 0 features whereas SIFT treats
it as unlimited
"""

def create_orb(
    nfeatures=100,
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
    Create an ORB detector.
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

    raise RuntimeError(
        "OpenCV was built without ORB support."
    )


# ============================================================
# Feature Extraction
# ============================================================

def detect_and_describe(image, orb=None):
    """
    Detect ORB keypoints and compute descriptors.
    """

    if orb is None:
        orb = create_orb()

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    start = cv2.getTickCount()

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    runtime = (
        cv2.getTickCount() - start
    ) / cv2.getTickFrequency()

    #added an explict check for empty keypoints
    if keypoints is None:
        keypoints = []

    return (
        keypoints,
        descriptors,
        runtime
    )


# ============================================================
# Keypoint Visualization
# ============================================================

def visualize_keypoints(
    image,
    keypoints,
    max_points=150
):
    """
    Draw the strongest ORB keypoints.

    Each keypoint shows:
        - location
        - scale
    """

    output = image.copy()

    sorted_keypoints = sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True
    )

    for kp in sorted_keypoints[:max_points]:

        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))

        radius = max(
            2,
            int(round(kp.size / 2.0))
        )

        # Scale circle
        cv2.circle(
            output,
            (x, y),
            radius,
            (0, 255, 0),
            1
        )

        # Keypoint center
        cv2.circle(
            output,
            (x, y),
            2,
            (255, 0, 0),
            -1
        )

    return output


# ============================================================
# Save Image
# ============================================================

def save_rgb_image(
    image_rgb,
    output_path
):
    """
    Save an RGB image using OpenCV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# Image Montage
# ============================================================

def fit_image_to_box(
    image_rgb,
    box_width,
    box_height
):
    """
    Resize image while preserving aspect ratio.
    """

    image_height, image_width = image_rgb.shape[:2]

    scale = min(
        box_width / image_width,
        box_height / image_height
    )

    new_width = max(
        1,
        int(round(image_width * scale))
    )

    new_height = max(
        1,
        int(round(image_height * scale))
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized = cv2.resize(
        image_rgb,
        (new_width, new_height),
        interpolation=interpolation
    )

    canvas = np.full(
        (box_height, box_width, 3),
        255,
        dtype=np.uint8
    )

    y0 = (
        box_height - new_height
    ) // 2

    x0 = (
        box_width - new_width
    ) // 2

    canvas[
        y0:y0 + new_height,
        x0:x0 + new_width
    ] = resized

    return canvas


def make_labeled_tile(
    image_rgb,
    label,
    tile_width=540,
    tile_height=420,
    header_height=44
):

    tile = np.full(
        (tile_height, tile_width, 3),
        245,
        dtype=np.uint8
    )

    tile[:header_height, :] = (
        28,
        28,
        28
    )

    cv2.putText(
        tile,
        label,
        (12, header_height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    tile[
        header_height:,
        :
    ] = fit_image_to_box(
        image_rgb,
        tile_width,
        tile_height - header_height
    )

    return tile


def build_montage(
    sample_results,
    sequence,
    condition,
    grid_path
):

    if not sample_results:
        return

    tile_width = 540
    tile_height = 420
    header_height = 44

    title_bar_height = 68
    margin = 12

    cols = min(
        3,
        len(sample_results)
    )

    rows = math.ceil(
        len(sample_results) / cols
    )

    tiles = []

    for result in sample_results:

        stats = result["statistics"]

        label = (
            f"{result['image_number']}.ppm | "
            f"{stats['number_of_keypoints']} kp | "
            f"{stats['runtime_seconds']:.4f}s"
        )

        tiles.append(
            make_labeled_tile(
                result["overlay_rgb"],
                label,
                tile_width,
                tile_height,
                header_height
            )
        )

    grid_width = (
        cols * tile_width
        + (cols + 1) * margin
    )

    grid_height = (
        title_bar_height
        + (rows + 1) * margin
        + rows * tile_height
    )

    canvas = np.full(
        (grid_height, grid_width, 3),
        255,
        dtype=np.uint8
    )

    canvas[
        :title_bar_height,
        :
    ] = (36, 36, 36)

    cv2.putText(
        canvas,
        f"ORB - {condition}: {sequence}",
        (margin, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Top ORB keypoints shown in green",
        (margin, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA
    )

    for idx, tile in enumerate(tiles):

        row = idx // cols
        col = idx % cols

        y0 = (
            title_bar_height
            + margin
            + row * (tile_height + margin)
        )

        x0 = (
            margin
            + col * (tile_width + margin)
        )

        canvas[
            y0:y0 + tile_height,
            x0:x0 + tile_width
        ] = tile

    grid_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(grid_path),
        cv2.cvtColor(
            canvas,
            cv2.COLOR_RGB2BGR
        )
    )


# ============================================================
# Process One Sample Sequence
# ============================================================

def process_sample_sequence(
    sequence,
    condition,
    orb,
    output_dir,
    max_points
):

    sample_results = []

    print()
    print(
        f"Creating {condition} samples: {sequence}"
    )

    for image_number in DEFAULT_SAMPLE_IMAGE_IDS:

        image_rgb = load_sequence_image(
            sequence,
            image_number
        )

        keypoints, descriptors, runtime = (
            detect_and_describe(
                image_rgb,
                orb
            )
        )

        overlay = visualize_keypoints(
            image_rgb,
            keypoints,
            max_points
        )

        statistics = {
            "number_of_keypoints": len(keypoints),
            "descriptor_shape": (
                None
                if descriptors is None
                else descriptors.shape
            ),
            "runtime_seconds": runtime
        }

        output_path = (
            output_dir
            / f"{sequence}_{image_number}_orb.png"
        )

        save_rgb_image(
            overlay,
            output_path
        )

        sample_results.append(
            {
                "image_number": image_number,
                "overlay_rgb": overlay,
                "statistics": statistics
            }
        )

        print(
            f"  Image {image_number}: "
            f"{len(keypoints)} keypoints | "
            f"{runtime:.4f}s"
        )

    grid_path = (
        output_dir
        / f"{sequence}_orb_grid.png"
    )

    build_montage(
        sample_results,
        sequence,
        condition,
        grid_path
    )

    return sample_results


# ============================================================
# Run ORB on Entire Dataset
# ============================================================

def process_dataset(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    max_points=DEFAULT_MAX_POINTS,
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

    total_images = 0
    total_keypoints = 0
    total_runtime = 0.0

    print("=" * 72)
    print("HPatches ORB Feature Extraction")
    print("=" * 72)

    print(
        f"\nFound {len(sequences)} sequences."
    )

    print(
        "\nRunning ORB on all images..."
    )

    # --------------------------------------------------------
    # Process every HPatches sequence
    # --------------------------------------------------------

    for sequence in sequences:

        sequence_images = 0
        sequence_keypoints = 0
        sequence_runtime = 0.0

        for image_number in HPATCHES_IMAGE_IDS:

            try:
                image_rgb = load_sequence_image(
                    sequence,
                    image_number
                )

            except FileNotFoundError:
                continue

            keypoints, descriptors, runtime = (
                detect_and_describe(
                    image_rgb,
                    orb
                )
            )

            sequence_images += 1
            sequence_keypoints += len(keypoints)
            sequence_runtime += runtime

            total_images += 1
            total_keypoints += len(keypoints)
            total_runtime += runtime

        if sequence_images:

            avg_keypoints = (
                sequence_keypoints
                / sequence_images
            )

            avg_runtime = (
                sequence_runtime
                / sequence_images
            )

            print(
                f"{sequence:20s} | "
                f"{sequence_images} images | "
                f"{avg_keypoints:7.1f} avg keypoints | "
                f"{avg_runtime:.4f}s avg"
            )

    # --------------------------------------------------------
    # Create lighting samples
    # --------------------------------------------------------

    lighting_samples = process_sample_sequence(
        lighting_sequence,
        "Lighting Changes",
        orb,
        lighting_output,
        max_points
    )

    # --------------------------------------------------------
    # Create viewpoint samples
    # --------------------------------------------------------

    viewpoint_samples = process_sample_sequence(
        viewpoint_sequence,
        "Viewpoint Changes",
        orb,
        viewpoint_output,
        max_points
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ORB EXTRACTION COMPLETE")
    print("=" * 72)

    print(
        f"Images processed: {total_images}"
    )

    print(
        f"Total keypoints: {total_keypoints}"
    )

    print(
        f"Total ORB extraction time: "
        f"{total_runtime:.2f} seconds"
    )

    if total_images:

        print(
            f"Average keypoints/image: "
            f"{total_keypoints / total_images:.2f}"
        )

        print(
            f"Average extraction time/image: "
            f"{total_runtime / total_images:.4f} seconds"
        )

    print()
    print(
        f"Lighting samples: {lighting_output}"
    )

    print(
        f"Viewpoint samples: {viewpoint_output}"
    )

    print("=" * 72)

    return (
        lighting_samples,
        viewpoint_samples
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run ORB on the full HPatches sequence "
            "dataset and save sample visualizations "
            "for illumination and viewpoint sequences."
        )
    )

    parser.add_argument(
        "--lighting-sequence",
        default=DEFAULT_LIGHTING_SEQUENCE,
        help="HPatches i_ sequence for illumination samples."
    )

    parser.add_argument(
        "--viewpoint-sequence",
        default=DEFAULT_VIEWPOINT_SEQUENCE,
        help="HPatches v_ sequence for viewpoint samples."
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum keypoints drawn per sample."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory."
    )

    args = parser.parse_args()

    process_dataset(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        max_points=args.max_points,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()