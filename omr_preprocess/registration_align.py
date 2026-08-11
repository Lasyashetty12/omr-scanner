from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 2200

# Canonical registration-mark centres in the user's clean NEET reference.
CANONICAL_MARKERS_1600_2200 = np.array(
    [
        [81.2, 78.3],       # TL
        [1522.0, 78.3],     # TR
        [1523.3, 2124.2],   # BR
        [79.9, 2120.4],     # BL
    ],
    dtype=np.float32,
)


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]      # TL
    ordered[2] = points[np.argmax(sums)]      # BR
    ordered[1] = points[np.argmin(diffs)]     # TR
    ordered[3] = points[np.argmax(diffs)]     # BL

    return ordered


def _resize_for_detection(
    image: np.ndarray,
    max_side: int = 1500,
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))

    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    return image.copy(), 1.0


def _binary_dark(gray: np.ndarray) -> np.ndarray:
    """
    Produce a dark-object mask robust to uneven mobile lighting.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Global Otsu + adaptive threshold, then combine.
    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9,
    )

    mask = cv2.bitwise_and(otsu, adaptive)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
        iterations=1,
    )

    return mask


def _candidate_black_blocks(
    image: np.ndarray,
) -> list[Dict[str, Any]]:
    """
    Find compact dark filled rectangles/squares that could be registration marks.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = _binary_dark(gray)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = gray.shape[:2]
    image_area = float(h * w)

    candidates: list[Dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        # Registration blocks are visually large but still small relative to frame.
        if area < image_area * 0.00015:
            continue

        if area > image_area * 0.035:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < 10 or bh < 10:
            continue

        aspect = bw / float(bh)

        # Bottom marks can merge slightly with page rules, so allow some elongation.
        if not 0.45 <= aspect <= 2.2:
            continue

        rect_area = float(bw * bh)
        fill = area / max(rect_area, 1.0)

        if fill < 0.52:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True,
        )

        compactness = (
            4.0 * np.pi * area /
            max(perimeter * perimeter, 1.0)
        )

        cx = x + bw / 2.0
        cy = y + bh / 2.0

        candidates.append(
            {
                "center": np.array(
                    [cx, cy],
                    dtype=np.float32,
                ),
                "bbox": (x, y, bw, bh),
                "area": area,
                "fill": fill,
                "aspect": aspect,
                "vertices": len(approx),
                "compactness": float(compactness),
            }
        )

    return candidates


def _corner_region_score(
    candidate: Dict[str, Any],
    corner: str,
    width: int,
    height: int,
) -> float:
    cx, cy = candidate["center"]

    nx = cx / max(float(width), 1.0)
    ny = cy / max(float(height), 1.0)

    target = {
        "TL": (0.12, 0.10),
        "TR": (0.88, 0.10),
        "BR": (0.88, 0.90),
        "BL": (0.12, 0.90),
    }[corner]

    distance = np.hypot(
        nx - target[0],
        ny - target[1],
    )

    area_score = min(
        candidate["area"] / max(width * height * 0.004, 1.0),
        2.0,
    )

    square_score = max(
        0.0,
        1.0 - abs(
            np.log(
                max(candidate["aspect"], 1e-6)
            )
        ),
    )

    fill_score = candidate["fill"]

    vertex_score = (
        1.0
        if 4 <= candidate["vertices"] <= 8
        else 0.5
    )

    return (
        -distance * 8.0
        + area_score * 1.3
        + square_score * 1.5
        + fill_score * 1.6
        + vertex_score
    )


def _pick_corner_candidate(
    candidates: list[Dict[str, Any]],
    corner: str,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    """
    Search a generous corner quadrant but reject centre-page content.
    """
    chosen = []

    for candidate in candidates:
        cx, cy = candidate["center"]
        nx = cx / float(width)
        ny = cy / float(height)

        if corner == "TL":
            inside = nx < 0.48 and ny < 0.35
        elif corner == "TR":
            inside = nx > 0.52 and ny < 0.35
        elif corner == "BR":
            inside = nx > 0.52 and ny > 0.70
        else:
            inside = nx < 0.48 and ny > 0.70

        if not inside:
            continue

        score = _corner_region_score(
            candidate,
            corner,
            width,
            height,
        )

        chosen.append(
            (
                score,
                candidate,
            )
        )

    if not chosen:
        return None

    chosen.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return chosen[0][1]


def _validate_marker_geometry(
    markers: np.ndarray,
    width: int,
    height: int,
) -> None:
    tl, tr, br, bl = order_points(markers)

    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    if min(top, bottom, left, right) < min(width, height) * 0.32:
        raise ValueError(
            "Registration markers are too close together. "
            "A wrong black object was probably selected."
        )

    polygon = np.array(
        [tl, tr, br, bl],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    area = abs(float(cv2.contourArea(polygon)))
    coverage = area / float(width * height)

    if coverage < 0.35:
        raise ValueError(
            "Registration-marker quadrilateral is too small."
        )

    # Opposite sides should not differ absurdly.
    if max(top, bottom) / max(min(top, bottom), 1.0) > 1.8:
        raise ValueError(
            "Top/bottom registration geometry is inconsistent."
        )

    if max(left, right) / max(min(left, right), 1.0) > 1.8:
        raise ValueError(
            "Left/right registration geometry is inconsistent."
        )


def detect_registration_blocks(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect the four large black registration blocks visible on the OMR sheet.

    Output order: TL, TR, BR, BL.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    small, scale = _resize_for_detection(
        image,
        max_side=1500,
    )

    h, w = small.shape[:2]

    candidates = _candidate_black_blocks(
        small
    )

    picked = []

    for corner in (
        "TL",
        "TR",
        "BR",
        "BL",
    ):
        candidate = _pick_corner_candidate(
            candidates,
            corner,
            w,
            h,
        )

        if candidate is None:
            raise ValueError(
                f"Could not detect {corner} registration block. "
                "Keep the whole OMR sheet visible, reduce glare, "
                "and avoid covering the corner marks."
            )

        picked.append(
            candidate["center"] / scale
        )

    markers = order_points(
        np.array(
            picked,
            dtype=np.float32,
        )
    )

    full_h, full_w = image.shape[:2]

    _validate_marker_geometry(
        markers,
        full_w,
        full_h,
    )

    debug = {
        "candidate_count":
            len(candidates),

        "scale":
            float(scale),

        "markers": [
            [
                round(
                    float(point[0]),
                    2,
                ),
                round(
                    float(point[1]),
                    2,
                ),
            ]
            for point
            in markers
        ],
    }

    return markers, debug


def _canonical_marker_positions(
    width: int,
    height: int,
) -> np.ndarray:
    markers = (
        CANONICAL_MARKERS_1600_2200
        .copy()
    )

    markers[:, 0] *= (
        width / 1600.0
    )

    markers[:, 1] *= (
        height / 2200.0
    )

    return markers.astype(
        np.float32
    )


def warp_from_registration_blocks(
    image: np.ndarray,
    source_markers: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source = order_points(
        source_markers
    ).astype(np.float32)

    destination = (
        _canonical_marker_positions(
            width,
            height,
        )
    )

    matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    return corrected, matrix


def _prepare_feature_image(
    image: np.ndarray,
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    return cv2.GaussianBlur(
        clahe.apply(gray),
        (
            3,
            3,
        ),
        0,
    )


def _alignment_feature_mask(
    width: int,
    height: int,
) -> np.ndarray:
    """
    Use stable printed structure and de-emphasize changing filled bubbles.
    """
    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    # Header / identity / paper-code region.
    mask[
        :
        int(
            height * 0.36
        ),
        :
    ] = 255

    # Side timing/registration bars.
    mask[
        :,
        :
        int(
            width * 0.12
        )
    ] = 255

    mask[
        :,
        int(
            width * 0.88
        )
        :
    ] = 255

    # Vertical response-column separators.
    for fraction in (
        0.25,
        0.50,
        0.75,
    ):
        x = int(
            width * fraction
        )

        half = int(
            width * 0.018
        )

        mask[
            int(
                height * 0.32
            )
            :,
            max(
                0,
                x - half,
            )
            :
            min(
                width,
                x + half,
            ),
        ] = 255

    # Bottom rules / signature strip.
    mask[
        int(
            height * 0.90
        )
        :,
        :
    ] = 255

    return mask


def _orb_refine(
    moving: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = reference.shape[:2]

    moving_gray = _prepare_feature_image(
        moving
    )

    reference_gray = _prepare_feature_image(
        reference
    )

    mask = _alignment_feature_mask(
        w,
        h,
    )

    orb = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=20,
        patchSize=31,
        fastThreshold=10,
    )

    kp_m, des_m = orb.detectAndCompute(
        moving_gray,
        mask,
    )

    kp_r, des_r = orb.detectAndCompute(
        reference_gray,
        mask,
    )

    debug = {
        "orb_keypoints_moving":
            len(
                kp_m or []
            ),

        "orb_keypoints_reference":
            len(
                kp_r or []
            ),

        "orb_good_matches":
            0,

        "orb_inliers":
            0,

        "orb_applied":
            False,
    }

    if (
        des_m is None
        or des_r is None
    ):
        return moving, debug

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        des_m,
        des_r,
        k=2,
    )

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if (
            first.distance
            < 0.72
            * second.distance
        ):
            good.append(
                first
            )

    debug[
        "orb_good_matches"
    ] = len(good)

    if len(good) < 25:
        return moving, debug

    source_points = np.float32(
        [
            kp_m[
                match.queryIdx
            ].pt
            for match
            in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    destination_points = np.float32(
        [
            kp_r[
                match.trainIdx
            ].pt
            for match
            in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    homography, inlier_mask = (
        cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            3.0,
        )
    )

    if homography is None:
        return moving, debug

    inliers = (
        int(
            inlier_mask.sum()
        )
        if inlier_mask
        is not None
        else 0
    )

    debug[
        "orb_inliers"
    ] = inliers

    if inliers < 18:
        return moving, debug

    # Homography must leave reference corners close to the output canvas.
    corners = np.float32(
        [
            [
                0,
                0,
            ],
            [
                w - 1,
                0,
            ],
            [
                w - 1,
                h - 1,
            ],
            [
                0,
                h - 1,
            ],
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    transformed = (
        cv2.perspectiveTransform(
            corners,
            homography,
        )
        .reshape(
            4,
            2,
        )
    )

    expected = corners.reshape(
        4,
        2,
    )

    corner_error = float(
        np.mean(
            np.linalg.norm(
                transformed
                - expected,
                axis=1,
            )
        )
    )

    debug[
        "orb_corner_error"
    ] = corner_error

    # This is only a fine registration. Reject aggressive warps.
    if corner_error > 90.0:
        return moving, debug

    refined = cv2.warpPerspective(
        moving,
        homography,
        (
            w,
            h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    debug[
        "orb_applied"
    ] = True

    return refined, debug


def _ecc_refine(
    moving: np.ndarray,
    reference: np.ndarray,
    minimum_score: float = 0.75,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Small affine ECC refinement.

    IMPORTANT:
    A low-confidence ECC result is rejected. The previous alignment is kept.
    """
    full_h, full_w = reference.shape[:2]

    small_w = 800
    small_h = int(
        round(
            full_h
            * (
                small_w
                / full_w
            )
        )
    )

    moving_small = cv2.resize(
        _prepare_feature_image(
            moving
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_AREA,
    ).astype(
        np.float32
    ) / 255.0

    reference_small = cv2.resize(
        _prepare_feature_image(
            reference
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_AREA,
    ).astype(
        np.float32
    ) / 255.0

    mask_small = cv2.resize(
        _alignment_feature_mask(
            full_w,
            full_h,
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    warp_small = np.eye(
        2,
        3,
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )

    debug = {
        "ecc_attempted":
            True,

        "ecc_applied":
            False,

        "ecc_score":
            None,

        "ecc_minimum_score":
            float(
                minimum_score
            ),
    }

    try:
        score, warp_small = (
            cv2.findTransformECC(
                reference_small,
                moving_small,
                warp_small,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=mask_small,
                gaussFiltSize=5,
            )
        )
    except cv2.error:
        return moving, debug

    debug[
        "ecc_score"
    ] = float(score)

    # Critical safety gate.
    if score < minimum_score:
        return moving, debug

    sx = (
        full_w
        / float(
            small_w
        )
    )

    sy = (
        full_h
        / float(
            small_h
        )
    )

    scale_to_full = np.array(
        [
            [
                sx,
                0,
                0,
            ],
            [
                0,
                sy,
                0,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float32,
    )

    scale_to_small = np.array(
        [
            [
                1.0 / sx,
                0,
                0,
            ],
            [
                0,
                1.0 / sy,
                0,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float32,
    )

    affine3 = np.vstack(
        [
            warp_small,
            [
                0,
                0,
                1,
            ],
        ]
    ).astype(
        np.float32
    )

    full_affine3 = (
        scale_to_full
        @ affine3
        @ scale_to_small
    )

    full_affine = (
        full_affine3[
            :
            2,
            :
        ]
    )

    refined = cv2.warpAffine(
        moving,
        full_affine,
        (
            full_w,
            full_h,
        ),
        flags=(
            cv2.INTER_LINEAR
            |
            cv2.WARP_INVERSE_MAP
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    debug[
        "ecc_applied"
    ] = True

    return refined, debug


def _draw_marker_debug(
    image: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    names = (
        "TL",
        "TR",
        "BR",
        "BL",
    )

    for name, point in zip(
        names,
        markers,
    ):
        x = int(
            round(
                float(
                    point[0]
                )
            )
        )

        y = int(
            round(
                float(
                    point[1]
                )
            )
        )

        cv2.circle(
            output,
            (
                x,
                y,
            ),
            18,
            (
                0,
                0,
                255,
            ),
            4,
        )

        cv2.putText(
            output,
            name,
            (
                x + 20,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    polygon = (
        markers
        .astype(
            np.int32
        )
        .reshape(
            -1,
            1,
            2,
        )
    )

    cv2.polylines(
        output,
        [
            polygon
        ],
        True,
        (
            255,
            0,
            0,
        ),
        3,
    )

    return output


def canonicalize_omr(
    image: np.ndarray,
    reference_path: str | Path,
    output_size: Tuple[int, int] = (
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
    ),
    use_orb: bool = True,
    use_ecc: bool = True,
    ecc_minimum_score: float = 0.75,
    debug_dir: Optional[
        str | Path
    ] = None,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    """
    Convert the mobile photo into canonical reference geometry using
    the four printed black registration blocks as the primary anchors.

    Pipeline:
      1. detect four registration blocks in original photo
      2. validate their geometry
      3. homography from detected block centres to canonical centres
      4. optional conservative ORB refinement
      5. optional ECC refinement ONLY when score >= threshold
      6. guarantee exact 1600x2200 output
    """
    width, height = map(
        int,
        output_size,
    )

    reference = cv2.imread(
        str(
            reference_path
        )
    )

    if reference is None:
        raise ValueError(
            "Could not load canonical reference: "
            f"{reference_path}"
        )

    reference = cv2.resize(
        reference,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    markers, marker_debug = (
        detect_registration_blocks(
            image
        )
    )

    coarse, homography = (
        warp_from_registration_blocks(
            image,
            markers,
            width,
            height,
        )
    )

    result = coarse

    debug: Dict[
        str,
        Any,
    ] = {
        "alignment_method":
            "registration_blocks",

        "output_size": {
            "width":
                width,

            "height":
                height,
        },

        "registration":
            marker_debug,

        "coarse_homography":
            homography.tolist(),
    }

    if use_orb:
        result, orb_debug = (
            _orb_refine(
                result,
                reference,
            )
        )

        debug.update(
            orb_debug
        )

    if use_ecc:
        result, ecc_debug = (
            _ecc_refine(
                result,
                reference,
                minimum_score=
                    ecc_minimum_score,
            )
        )

        debug.update(
            ecc_debug
        )

    if (
        result.shape[1]
        != width
        or result.shape[0]
        != height
    ):
        result = cv2.resize(
            result,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    if debug_dir is not None:
        debug_dir = Path(
            debug_dir
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "00_registration_detection.jpg"
            ),
            _draw_marker_debug(
                image,
                markers,
            ),
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "01_registration_warp.jpg"
            ),
            coarse,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "02_canonical_aligned.jpg"
            ),
            result,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "03_reference.jpg"
            ),
            reference,
        )

    return result, debug
