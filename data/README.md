# HPatches dataset

This project uses the **HPatches full image sequences** for wide-baseline
SIFT and ORB feature detection, description, and matching.

## Local layout

The downloaded dataset is intentionally excluded from Git:

```text
data/
|-- hpatches-sequences-release/
|   |-- i_*/                 # illumination-change sequences
|   `-- v_*/                 # viewpoint-change sequences
|-- dataset_manifest.csv     # generated list of 580 evaluated image pairs
`-- README.md
```

Each sequence contains six full images (`1.ppm` through `6.ppm`). Image
`1.ppm` is the reference. Files `H_1_2` through `H_1_6` contain 3x3
ground-truth homographies mapping points from the reference image to each
target image.

The target image number is recorded as `difficulty` 1 through 5 in the
manifest. This is the conventional HPatches progression within a sequence;
it should not be presented as a separately annotated scale category.

## Validate and regenerate the manifest

From the repository root:

```powershell
python code/prepare_hpatches.py
```

The script checks all 116 sequences, 696 images, and 580 homographies. It
also confirms that each image has a PPM/PGM header and every homography is a
finite, nonsingular 3x3 matrix.

## Dataset source

- Full sequences:
  https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
- Dataset documentation:
  https://github.com/hpatches/hpatches-dataset

## Citation

V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk,
"HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local
Descriptors," CVPR, 2017.

