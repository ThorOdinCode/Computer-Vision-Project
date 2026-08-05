import sys
import os


# Add project directory to path

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(PROJECT_DIR)


from data_loader import (
    list_image_sequences,
    load_sequence_image,
    load_sequence_pair
)



def main():

    print("=" * 60)
    print("HPatches Sequence Dataset Loader Test")
    print("=" * 60)


    # --------------------------------------------------
    # List sequences
    # --------------------------------------------------

    sequences = list_image_sequences()


    print("\nNumber of sequences:")
    print(len(sequences))


    print("\nFirst five sequences:")

    for seq in sequences[:5]:
        print(seq)



    # --------------------------------------------------
    # Load sample image
    # --------------------------------------------------

    sequence = sequences[0]


    print("\nLoading sample:")
    print(sequence)


    image = load_sequence_image(
        sequence,
        1
    )


    print("\nReference image:")
    print(image.shape)



    target = load_sequence_image(
        sequence,
        2
    )


    print("\nTarget image:")
    print(target.shape)



    # --------------------------------------------------
    # Load homography
    # --------------------------------------------------

    ref, target, H = load_sequence_pair(
        sequence,
        2
    )


    print("\nHomography:")
    print(H)



    # --------------------------------------------------
    # Display images
    # --------------------------------------------------

    import cv2


    ref_bgr = cv2.cvtColor(
        ref,
        cv2.COLOR_RGB2BGR
    )


    target_bgr = cv2.cvtColor(
        target,
        cv2.COLOR_RGB2BGR
    )


    cv2.imshow(
        "Reference Image",
        ref_bgr
    )


    cv2.imshow(
        "Target Image",
        target_bgr
    )


    print("\nPress any key to exit")

    cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()