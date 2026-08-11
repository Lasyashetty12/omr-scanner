import argparse
import json

import cv2

from omr_preprocess import canonicalize_omr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--reference",
        default="references/neet_reference.png",
    )
    parser.add_argument(
        "--output",
        default="canonical_output.jpg",
    )
    args = parser.parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:
        raise SystemExit(
            f"Could not read image: {args.image}"
        )

    result, debug = canonicalize_omr(
        image=image,
        reference_path=args.reference,
        output_size=(
            1600,
            2200,
        ),
        use_orb=True,
        use_ecc=True,
        ecc_minimum_score=0.75,
        debug_dir="alignment_debug",
    )

    cv2.imwrite(
        args.output,
        result,
    )

    print(
        "Saved:",
        args.output,
    )

    print(
        "Shape:",
        result.shape,
    )

    print(
        "Alignment:"
    )

    print(
        json.dumps(
            debug,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
