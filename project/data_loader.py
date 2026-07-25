import os
import cv2
import numpy as np

# =============================================================================
# Dataset Configuration
# =============================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.abspath(
    os.path.join(
        THIS_DIR,
        "..",
        "hpatches-benchmark",
        "data",
        "hpatches-release",
    )
)

PATCH_SIZE = 65

VALID_IMAGE_TYPES = [
    "ref",
    "e1", "e2", "e3", "e4", "e5",
    "h1", "h2", "h3", "h4", "h5",
    "t1", "t2", "t3", "t4", "t5",
]


# =============================================================================
# Dataset Utilities
# =============================================================================

def list_sequences():
    """Return all HPatches sequence names."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(DATASET_PATH)

    sequences = [
        seq for seq in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, seq))
    ]

    sequences.sort()

    return sequences


def load_image(sequence, image_type):
    """
    Load the stacked HPatches image.

    Example:
        ref.png
        e1.png
        h3.png
    """

    if image_type not in VALID_IMAGE_TYPES:
        raise ValueError(f"Invalid image type: {image_type}")

    path = os.path.join(
        DATASET_PATH,
        sequence,
        image_type + ".png"
    )

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    return image


def split_into_patches(stacked_image):
    """
    Split a stacked HPatches image into individual 65x65 patches.
    """

    height = stacked_image.shape[0]

    if height % PATCH_SIZE != 0:
        raise ValueError(
            "Image height is not divisible by patch size."
        )

    num_patches = height // PATCH_SIZE

    patches = []

    for i in range(num_patches):

        start = i * PATCH_SIZE
        end = start + PATCH_SIZE

        patch = stacked_image[start:end, :, :]

        patches.append(patch)

    return patches


def load_patches(sequence, image_type):
    """
    Load a sequence and return a list of patches.

    Returns:
        List[np.ndarray]
    """

    stacked = load_image(sequence, image_type)

    return split_into_patches(stacked)


def load_homography(sequence, target):
    """
    Load the ground-truth homography.
    """

    if target == "ref":
        return np.eye(3)

    number = target[1:]

    path = os.path.join(
        DATASET_PATH,
        sequence,
        f"H_ref_{number}"
    )

    H = np.loadtxt(
        path,
        delimiter=","
    )

    return H


def load_pair(sequence, target):
    """
    Returns stacked images.

    Returns:
        reference_stack
        target_stack
        homography
    """

    ref = load_image(sequence, "ref")
    target_img = load_image(sequence, target)
    H = load_homography(sequence, target)

    return ref, target_img, H


def load_patch_pair(sequence, target):
    """
    Returns lists of corresponding patches.

    Returns:
        ref_patches
        target_patches
        homography
    """

    ref = load_patches(sequence, "ref")
    target_img = load_patches(sequence, target)
    H = load_homography(sequence, target)

    return ref, target_img, H
