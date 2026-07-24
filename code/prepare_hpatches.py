"""Validate HPatches full sequences and generate the project pair manifest.

Usage:
    python code/prepare_hpatches.py
    python code/prepare_hpatches.py --dataset-root path/to/hpatches-sequences-release
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EXPECTED_SEQUENCES = 116
EXPECTED_ILLUMINATION = 57
EXPECTED_VIEWPOINT = 59
IMAGES_PER_SEQUENCE = 6


def read_homography(path: Path) -> list[float]:
    try:
        values = [float(value) for value in path.read_text().split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read numeric homography: {path}") from exc
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expected 9 finite homography values: {path}")
    determinant = (
        values[0] * (values[4] * values[8] - values[5] * values[7])
        - values[1] * (values[3] * values[8] - values[5] * values[6])
        + values[2] * (values[3] * values[7] - values[4] * values[6])
    )
    if abs(determinant) < 1e-12:
        raise ValueError(f"singular homography: {path}")
    return values


def validate_ppm(path: Path) -> None:
    try:
        with path.open("rb") as image:
            magic = image.read(2)
    except OSError as exc:
        raise ValueError(f"cannot read image: {path}") from exc
    if magic not in {b"P5", b"P6"}:
        raise ValueError(f"invalid PPM/PGM header in {path}: {magic!r}")


def build_manifest(dataset_root: Path, manifest_path: Path) -> None:
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_root}")

    sequences = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith(("i_", "v_"))
    )
    illumination = [path for path in sequences if path.name.startswith("i_")]
    viewpoint = [path for path in sequences if path.name.startswith("v_")]

    errors: list[str] = []
    rows: list[dict[str, str | int]] = []

    for sequence in sequences:
        change_type = (
            "illumination" if sequence.name.startswith("i_") else "viewpoint"
        )
        for image_number in range(1, IMAGES_PER_SEQUENCE + 1):
            image_path = sequence / f"{image_number}.ppm"
            if not image_path.is_file():
                errors.append(f"missing image: {image_path}")
            else:
                try:
                    validate_ppm(image_path)
                except ValueError as exc:
                    errors.append(str(exc))

        for target_number in range(2, IMAGES_PER_SEQUENCE + 1):
            homography_path = sequence / f"H_1_{target_number}"
            if not homography_path.is_file():
                errors.append(f"missing homography: {homography_path}")
            else:
                try:
                    read_homography(homography_path)
                except ValueError as exc:
                    errors.append(str(exc))

            rows.append(
                {
                    "sequence": sequence.name,
                    "change_type": change_type,
                    "difficulty": target_number - 1,
                    "reference_image": f"{sequence.name}/1.ppm",
                    "target_image": f"{sequence.name}/{target_number}.ppm",
                    "homography": f"{sequence.name}/H_1_{target_number}",
                }
            )

    expected_counts = {
        "total sequences": (len(sequences), EXPECTED_SEQUENCES),
        "illumination sequences": (len(illumination), EXPECTED_ILLUMINATION),
        "viewpoint sequences": (len(viewpoint), EXPECTED_VIEWPOINT),
        "image pairs": (len(rows), EXPECTED_SEQUENCES * 5),
    }
    for label, (actual, expected) in expected_counts.items():
        if actual != expected:
            errors.append(f"{label}: expected {expected}, found {actual}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:25])
        remainder = len(errors) - 25
        if remainder > 0:
            preview += f"\n- ...and {remainder} more"
        raise SystemExit(f"HPatches validation failed:\n{preview}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=[
                "sequence",
                "change_type",
                "difficulty",
                "reference_image",
                "target_image",
                "homography",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Validated dataset: {dataset_root}")
    print(
        f"{len(sequences)} sequences "
        f"({len(illumination)} illumination, {len(viewpoint)} viewpoint)"
    )
    print(f"{len(rows)} reference-target pairs")
    print(f"Wrote manifest: {manifest_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "hpatches-sequences-release",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "dataset_manifest.csv",
    )
    args = parser.parse_args()
    build_manifest(args.dataset_root.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()

