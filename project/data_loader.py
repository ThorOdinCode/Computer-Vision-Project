import os
import cv2
import numpy as np


# =============================================================================
# Dataset Paths
# =============================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# Patch dataset (65x65 patches)
PATCH_DATASET_PATH = os.path.abspath(
    os.path.join(
        THIS_DIR,
        "..",
        "hpatches-benchmark",
        "data",
        "hpatches-release"
    )
)


# Full image sequence dataset
SEQUENCE_DATASET_PATH = os.path.abspath(
    os.path.join(
        THIS_DIR,
        "..",
         "hpatches-benchmark",
         "data",
        "hpatches-sequences-release"
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
# PATCH DATASET FUNCTIONS
# =============================================================================


def list_patch_sequences():

    if not os.path.exists(PATCH_DATASET_PATH):
        raise FileNotFoundError(PATCH_DATASET_PATH)

    sequences = [
        seq for seq in os.listdir(PATCH_DATASET_PATH)
        if os.path.isdir(
            os.path.join(PATCH_DATASET_PATH, seq)
        )
    ]

    return sorted(sequences)



def load_patch_image(sequence, image_type):

    if image_type not in VALID_IMAGE_TYPES:
        raise ValueError(
            f"Invalid image type: {image_type}"
        )


    path = os.path.join(
        PATCH_DATASET_PATH,
        sequence,
        image_type + ".png"
    )


    image = cv2.imread(path)


    if image is None:
        raise FileNotFoundError(path)


    # BGR -> RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    return image



def split_into_patches(stacked_image):

    height = stacked_image.shape[0]


    if height % PATCH_SIZE != 0:
        raise ValueError(
            "Invalid stacked image size"
        )


    patches = []


    for i in range(height // PATCH_SIZE):

        start = i * PATCH_SIZE
        end = start + PATCH_SIZE

        patches.append(
            stacked_image[start:end,:,:]
        )


    return patches



def load_patches(sequence, image_type):

    image = load_patch_image(
        sequence,
        image_type
    )

    return split_into_patches(image)



def load_patch_homography(sequence, target):

    if target == "ref":
        return np.eye(3)


    number = target[1:]


    path = os.path.join(
        PATCH_DATASET_PATH,
        sequence,
        f"H_ref_{number}"
    )


    return np.loadtxt(
        path,
        delimiter=","
    )



def load_patch_pair(sequence, target):

    ref = load_patch_image(
        sequence,
        "ref"
    )


    target_image = load_patch_image(
        sequence,
        target
    )


    H = load_patch_homography(
        sequence,
        target
    )


    return ref, target_image, H



# =============================================================================
# FULL IMAGE SEQUENCE DATASET FUNCTIONS
# =============================================================================


def list_image_sequences():

    if not os.path.exists(SEQUENCE_DATASET_PATH):
        raise FileNotFoundError(
            SEQUENCE_DATASET_PATH
        )


    sequences = [
        seq for seq in os.listdir(SEQUENCE_DATASET_PATH)
        if os.path.isdir(
            os.path.join(
                SEQUENCE_DATASET_PATH,
                seq
            )
        )
    ]


    return sorted(sequences)



def load_sequence_image(sequence, image_number):

    path = os.path.join(
        SEQUENCE_DATASET_PATH,
        sequence,
        f"{image_number}.ppm"
    )


    image = cv2.imread(path)


    if image is None:
        raise FileNotFoundError(path)


    # BGR -> RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    return image



def load_sequence_homography(sequence, target):

    if target == 1:
        return np.eye(3)


    path = os.path.join(
        SEQUENCE_DATASET_PATH,
        sequence,
        f"H_1_{target}"
    )


    return np.loadtxt(path)



def load_sequence_pair(sequence, target):

    reference = load_sequence_image(
        sequence,
        1
    )


    target_image = load_sequence_image(
        sequence,
        target
    )


    H = load_sequence_homography(
        sequence,
        target
    )


    return reference, target_image, H