from data_loader import *

import cv2


def main():

    print("=" * 60)
    print("HPatches Data Loader Test")
    print("=" * 60)

    sequences = list_sequences()

    print(f"\nSequences: {len(sequences)}")

    print("\nFirst five:")

    for seq in sequences[:5]:
        print(seq)

    print("\nLoading stacked images...")

    ref_stack, target_stack, H = load_pair(
        "v_bark",
        "e1"
    )

    print("\nStacked image shape:")
    print(ref_stack.shape)

    print("\nGround Truth Homography:")
    print(H)

    print("\nSplitting into patches...")

    ref_patches, target_patches, _ = load_patch_pair(
        "v_bark",
        "e1"
    )

    print(f"\nNumber of patches: {len(ref_patches)}")

    print("Shape of first patch:")
    print(ref_patches[0].shape)

    # Save a few patches for visual inspection

    cv2.imwrite("ref_patch_0.png", ref_patches[0])
    cv2.imwrite("target_patch_0.png", target_patches[0])

    cv2.imwrite("ref_patch_100.png", ref_patches[100])
    cv2.imwrite("target_patch_100.png", target_patches[100])

    print("\nSaved test patches:")
    print("  ref_patch_0.png")
    print("  target_patch_0.png")
    print("  ref_patch_100.png")
    print("  target_patch_100.png")
    print("\nFirst pixel RGB values:")
    print(ref_stack[0,0])


if __name__ == "__main__":
    main()
