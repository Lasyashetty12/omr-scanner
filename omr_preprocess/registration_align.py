"""
REFERENCE USAGE POLICY

The clean NEET reference image is used ONLY for geometric preprocessing:
- canonical orientation selection
- ORB fine registration
- optional high-confidence ECC fine registration

It is NEVER used to:
- decide whether a bubble is filled
- read answers
- read the paper code
- score marks

All answer/paper-code reading happens later from the corrected scan using
the JSON coordinates + runtime column calibration + ML/classical reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .canonical import detect_document_quad, warp_document_quad



DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 2200

# Registration-block centres are the template's geometry authority.  Feature
# refinement may only correct very small residual error after that mapping;
# a larger warp would move valid bubble coordinates away from their bubbles.
MAX_FINE_ALIGNMENT_CORNER_ERROR = 24.0

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
        # A dark desk/background can surround the bright page and otherwise
        # hide the inset black registration blocks from EXTERNAL retrieval.
        cv2.RETR_LIST,
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


def _candidate_corner_blocks(
    image: np.ndarray,
) -> list[Dict[str, Any]]:
    """Find filled corner squares without being confused by page borders.

    The generated NEET/KCET sheet joins each lower registration square to a
    horizontal rule, while the JEE sheet places its squares almost on the
    outer border.  Removing long horizontal and vertical strokes before
    contour extraction separates the compact filled blocks from those rules.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    height, width = mask.shape[:2]

    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(25, width // 25), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(25, height // 25)),
        ),
    )
    compact_mask = cv2.subtract(mask, cv2.bitwise_or(horizontal, vertical))
    compact_mask = cv2.morphologyEx(
        compact_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    contours, _ = cv2.findContours(
        compact_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(width * height)
    candidates: list[Dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not image_area * 0.00006 <= area <= image_area * 0.006:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 8 or bh < 8:
            continue

        aspect = bw / float(bh)
        if not 0.45 <= aspect <= 2.2:
            continue

        fill = area / max(float(bw * bh), 1.0)
        if fill < 0.42:
            continue

        candidates.append(
            {
                "center": np.array(
                    [x + bw / 2.0, y + bh / 2.0],
                    dtype=np.float32,
                ),
                "bbox": (x, y, bw, bh),
                "area": area,
                "fill": fill,
                "aspect": aspect,
            }
        )

    return candidates


def _pick_extreme_corner_candidate(
    candidates: list[Dict[str, Any]],
    corner: str,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    """Pick the compact dark block nearest an image corner."""
    target = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([float(width), 0.0], dtype=np.float32),
        "BR": np.array([float(width), float(height)], dtype=np.float32),
        "BL": np.array([0.0, float(height)], dtype=np.float32),
    }[corner]

    selected: list[Tuple[float, Dict[str, Any]]] = []
    for candidate in candidates:
        cx, cy = candidate["center"]
        nx = cx / max(float(width), 1.0)
        ny = cy / max(float(height), 1.0)
        if corner == "TL":
            inside = nx < 0.45 and ny < 0.45
        elif corner == "TR":
            inside = nx > 0.55 and ny < 0.45
        elif corner == "BR":
            inside = nx > 0.55 and ny > 0.55
        else:
            inside = nx < 0.45 and ny > 0.55
        if not inside:
            continue

        normalized = np.array(
            [cx / max(float(width), 1.0), cy / max(float(height), 1.0)],
            dtype=np.float32,
        )
        normalized_target = np.array(
            [target[0] / max(float(width), 1.0), target[1] / max(float(height), 1.0)],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(normalized - normalized_target))
        square_penalty = abs(float(np.log(max(candidate["aspect"], 1e-6))))
        score = distance + square_penalty * 0.025 - candidate["fill"] * 0.01
        selected.append((score, candidate))

    if not selected:
        return None
    return min(selected, key=lambda item: item[0])[1]


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
            inside = nx < 0.48 and ny < 0.50
        elif corner == "TR":
            inside = nx > 0.52 and ny < 0.50
        elif corner == "BR":
            inside = nx > 0.52 and ny > 0.50
        else:
            inside = nx < 0.48 and ny > 0.50

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


def _validate_marker_corner_regions(
    markers: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Require four detected marks to occupy the outer page-corner regions.

    Printed margins vary between OMR versions and camera page detection may
    choose either the paper edge or the inset border.  Broad corner regions
    accept both, while preventing response bubbles near the central grid from
    serving as registration points.
    """
    tl, tr, br, bl = order_points(markers)
    normalized = np.array(
        [[x / max(float(width), 1.0), y / max(float(height), 1.0)] for x, y in (tl, tr, br, bl)],
        dtype=np.float32,
    )
    valid = bool(
        normalized[0, 0] < 0.30 and normalized[0, 1] < 0.30
        and normalized[1, 0] > 0.70 and normalized[1, 1] < 0.30
        and normalized[2, 0] > 0.70 and normalized[2, 1] > 0.70
        and normalized[3, 0] < 0.30 and normalized[3, 1] > 0.70
    )
    debug = {
        "valid": valid,
        "normalized_markers": [
            [round(float(x), 4), round(float(y), 4)] for x, y in normalized
        ],
    }
    if not valid:
        raise ValueError(
            "Unable to align the complete OMR sheet. Four genuine corner "
            "registration boxes were not found in the page-corner regions."
        )
    return debug


def _detect_solid_corner_boxes_on_canonical_page(
    image: np.ndarray,
    expected_markers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect solid registration squares even when they touch page borders.

    Long-line removal is useful on raw images, but a lower registration box
    connected to the border can be removed with that border. On an already
    flattened page we can safely search only the four outer corner regions and
    select compact windows that are much darker than their surrounding ring.
    Filled answer bubbles are outside these regions and do not qualify.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    height, width = gray.shape[:2]
    min_side = max(8, int(round(width * 0.006)))
    max_side = max(14, int(round(width * 0.040)))
    step = max(2, int(round(width * 0.0025)))
    expected = (
        order_points(expected_markers).astype(np.float32)
        if expected_markers is not None
        else _canonical_marker_positions(width, height)
    )
    # A mobile page contour may follow either the physical paper edge or the
    # inset printed border. That can move a real marker by more than the old
    # seven-percent window after the first perspective warp, especially at BR.
    # The wider window is still confined to a known template corner and every
    # selected set must pass the four-marker geometry checks below.
    search_half_width = max(36, int(round(width * 0.13)))
    # Mobile document contours sometimes include a large blank paper margin
    # below the printed OMR frame. In that case the lower blocks can sit about
    # 16% above their canonical y-coordinate even though the complete sheet is
    # visible. Keep the x search tighter, but allow that vertical displacement.
    search_half_height = max(48, int(round(height * 0.20)))
    regions = {}
    for name, (expected_x, expected_y) in zip(
        ("TL", "TR", "BR", "BL"),
        expected,
    ):
        regions[name] = (
            max(0, int(round(expected_x)) - search_half_width),
            min(width, int(round(expected_x)) + search_half_width + 1),
            max(0, int(round(expected_y)) - search_half_height),
            min(height, int(round(expected_y)) + search_half_height + 1),
            float(expected_x),
            float(expected_y),
        )
    points: list[list[float]] = []
    details: Dict[str, Any] = {}

    for name in ("TL", "TR", "BR", "BL"):
        x0, x1, y0, y1, target_x, target_y = regions[name]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(
            (xx - target_x) / max(float(search_half_width), 1.0),
            (yy - target_y) / max(float(search_half_height), 1.0),
        )
        best: Optional[Tuple[float, int, int, int, float, float]] = None

        for side in range(min_side, max_side + 1, step):
            inner = cv2.boxFilter(blurred, -1, (side, side), normalize=True)
            outer_side = side * 2 + 1
            outer = cv2.boxFilter(
                blurred,
                -1,
                (outer_side, outer_side),
                normalize=True,
            )
            inner_roi = inner[y0:y1, x0:x1]
            contrast_roi = (outer - inner)[y0:y1, x0:x1]
            scores = (
                contrast_roi / 80.0
                + (180.0 - np.minimum(inner_roi, 180.0)) / 180.0 * 0.25
                - distance * 0.35
            )
            # boxFilter extrapolates image borders. Without this guard, the
            # physical page edge can look like a high-contrast square at x=0
            # or y=height-1 and impersonate BL/BR. Require the complete outer
            # comparison ring to be present inside the image.
            edge_margin = side + 1
            valid = (
                (contrast_roi >= 8.0)
                & (inner_roi < 180.0)
                & (xx >= edge_margin)
                & (xx < width - edge_margin)
                & (yy >= edge_margin)
                & (yy < height - edge_margin)
            )
            scores = np.where(valid, scores, -9.0)
            local_y, local_x = np.unravel_index(np.argmax(scores), scores.shape)
            score = float(scores[local_y, local_x])
            center_x = x0 + int(local_x)
            center_y = y0 + int(local_y)
            candidate = (
                score,
                center_x,
                center_y,
                side,
                float(inner[center_y, center_x]),
                float(contrast_roi[local_y, local_x]),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None or best[0] <= 0.20:
            raise ValueError(
                f"Could not detect {name} solid registration box on the complete page."
            )

        score, center_x, center_y, side, mean_gray, contrast = best
        points.append([float(center_x), float(center_y)])
        details[name] = {
            "center": [center_x, center_y],
            "side": side,
            "mean_gray": round(mean_gray, 2),
            "surrounding_contrast": round(contrast, 2),
            "score": round(score, 4),
        }

    markers = order_points(np.array(points, dtype=np.float32))
    _validate_marker_geometry(markers, width, height)
    _validate_marker_corner_regions(markers, width, height)
    return markers, {
        "method": "solid_square_corner_contrast",
        "candidate_count": 4,
        "markers": [[round(float(x), 2), round(float(y), 2)] for x, y in markers],
        "details": details,
    }


def _detect_registration_blocks_internal(
    small: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], list[Dict[str, Any]]]:
    h, w = small.shape[:2]
    candidates = _candidate_corner_blocks(small)

    picked_dict = {}
    for corner in ("TL", "TR", "BR", "BL"):
        cand = _pick_extreme_corner_candidate(candidates, corner, w, h)
        if cand is not None:
            picked_dict[corner] = cand["center"]

    # Older sheets without isolated corner blocks retain the original broad
    # filled-rectangle search as a compatibility fallback.
    if len(picked_dict) < 4:
        legacy_candidates = _candidate_black_blocks(small)
        for corner in ("TL", "TR", "BR", "BL"):
            if corner in picked_dict:
                continue
            cand = _pick_corner_candidate(legacy_candidates, corner, w, h)
            if cand is not None:
                picked_dict[corner] = cand["center"]
        candidates.extend(legacy_candidates)

    return picked_dict, candidates


def detect_registration_blocks(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect the four large black registration blocks visible on the OMR sheet.
    Handles both portrait and landscape input photos automatically.

    Output order: TL, TR, BR, BL.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    small, scale = _resize_for_detection(
        image,
        max_side=1500,
    )

    h, w = small.shape[:2]

    # Primary attempt in natural orientation (0°)
    picked_dict, candidates = _detect_registration_blocks_internal(small)

    # If not all 4 blocks found in natural view, try remaining cardinal rotations
    if len(picked_dict) < 4:
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, "90_cw"),
            (cv2.ROTATE_180, "180"),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, "270_ccw"),
        ]

        for rot_code, rot_name in rotations:
            small_rot = cv2.rotate(small, rot_code)
            rot_picked, rot_cands = _detect_registration_blocks_internal(small_rot)

            if len(rot_picked) > len(picked_dict):
                mapped_dict = {}
                for k, pt in rot_picked.items():
                    x_rot, y_rot = pt
                    if rot_name == "90_cw":
                        mapped_dict[k] = np.array([y_rot, h - x_rot], dtype=np.float32)
                    elif rot_name == "180":
                        mapped_dict[k] = np.array([w - x_rot, h - y_rot], dtype=np.float32)
                    elif rot_name == "270_ccw":
                        mapped_dict[k] = np.array([w - y_rot, x_rot], dtype=np.float32)
                picked_dict = mapped_dict
                if len(picked_dict) >= 4:
                    break

    picked = []
    for corner in ("TL", "TR", "BR", "BL"):
        if corner not in picked_dict:
            raise ValueError(
                f"Could not detect {corner} registration block. "
                "Keep the whole OMR sheet visible, reduce glare, "
                "and avoid covering the corner marks."
            )
        picked.append(picked_dict[corner] / scale)

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


def _validate_canonical_marker_positions(
    markers: np.ndarray,
    width: int,
    height: int,
    expected_markers: Optional[np.ndarray] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Reject an A4 warp whose internal registration blocks are implausible.

    Page corners establish the coordinate system.  These blocks are a
    validation signal only; they are deliberately not used for another crop
    or page-boundary warp.
    """
    expected = (
        order_points(expected_markers).astype(np.float32)
        if expected_markers is not None
        else _canonical_marker_positions(width, height)
    )
    distances = np.linalg.norm(markers - expected, axis=1)
    mean_error = float(np.mean(distances))
    max_error = float(np.max(distances))
    valid = mean_error <= 85.0 and max_error <= 130.0
    debug = {
        "detected": True,
        "expected_markers": [[round(float(x), 2), round(float(y), 2)] for x, y in expected],
        "mean_position_error": round(mean_error, 2),
        "max_position_error": round(max_error, 2),
        "valid": valid,
    }
    if not valid and strict:
        raise ValueError(
            "Unable to align the complete OMR sheet. Please place the entire "
            "A4 OMR inside the camera frame with all four corners visible and capture again."
        )
    return debug


def warp_from_registration_blocks(
    image: np.ndarray,
    source_markers: np.ndarray,
    width: int,
    height: int,
    destination_markers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    source = order_points(
        source_markers
    ).astype(np.float32)

    destination = (
        order_points(destination_markers).astype(np.float32)
        if destination_markers is not None
        else _canonical_marker_positions(width, height)
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


def _warp_best_marker_correspondence(
    image: np.ndarray,
    source_markers: np.ndarray,
    destination_markers: np.ndarray,
    reference: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Resolve 0/90/180/270 capture orientation during registration.

    Geometric point ordering names the corners of the camera frame, not the
    printed sheet.  Trying the four cyclic correspondences maps the actual
    printed top-left marker to the reference top-left marker without rotating
    and then stretching an already-canonical image.
    """
    source = order_points(source_markers).astype(np.float32)
    destination = order_points(destination_markers).astype(np.float32)
    candidates: list[Tuple[float, int, np.ndarray, np.ndarray]] = []

    for shift in range(4):
        shifted_source = np.roll(source, -shift, axis=0)
        matrix = cv2.getPerspectiveTransform(shifted_source, destination)
        corrected = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        score = _header_structure_score(corrected, reference)
        candidates.append((score, shift, corrected, matrix))

    score, shift, corrected, matrix = max(candidates, key=lambda item: item[0])
    rotation = (shift * 90) % 360
    debug = {
        "selected_rotation": int(rotation),
        "orientation_method": "registration_marker_correspondence",
        "orientation_correction_applied": bool(shift),
        "orientation_scores": {
            str((candidate_shift * 90) % 360): round(float(candidate_score), 3)
            for candidate_score, candidate_shift, _, _ in candidates
        },
    }
    return corrected, matrix, debug


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

    # This is only a fine registration. Reject aggressive warps: the four
    # registration blocks have already established the complete sheet frame.
    if corner_error > MAX_FINE_ALIGNMENT_CORNER_ERROR:
        debug["orb_rejected_reason"] = "fine_alignment_exceeds_geometry_limit"
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

    corners = np.float32(
        [[0, 0], [full_w - 1, 0], [full_w - 1, full_h - 1], [0, full_h - 1]]
    ).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, full_affine).reshape(4, 2)
    affine_corner_error = float(
        np.mean(
            np.linalg.norm(transformed_corners - corners.reshape(4, 2), axis=1)
        )
    )
    debug["ecc_corner_error"] = affine_corner_error
    if affine_corner_error > MAX_FINE_ALIGNMENT_CORNER_ERROR:
        debug["ecc_rejected_reason"] = "fine_alignment_exceeds_geometry_limit"
        return moving, debug

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



# ============================================================
# CANONICAL ORIENTATION
# ============================================================

def _rotate_to_candidate(
    image: np.ndarray,
    rotation: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Rotate a canonical-size image by 0/90/180/270 degrees and resize
    back to the exact canonical canvas.

    This is only used for orientation selection after perspective
    correction, never for JSON coordinate modification.
    """

    rotation = int(rotation) % 360

    if rotation == 0:
        candidate = image.copy()

    elif rotation == 90:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    elif rotation == 180:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    elif rotation == 270:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    else:
        raise ValueError(
            f"Unsupported rotation: {rotation}"
        )

    if (
        candidate.shape[1] != width
        or candidate.shape[0] != height
    ):
        candidate = cv2.resize(
            candidate,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    return candidate


def _header_structure_score(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compare ONLY the canonical top/header region.

    This avoids the response grid dominating orientation because the
    response section is visually repetitive and can look similar at 180°.
    """
    if candidate.ndim == 3:
        cand_gray = cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        cand_gray = candidate.copy()

    if reference.ndim == 3:
        ref_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        ref_gray = reference.copy()

    h, w = ref_gray.shape[:2]

    # Use only top 32%: Manchester header + instructions + student fields.
    top_h = int(
        round(
            h * 0.32
        )
    )

    cand_top = cand_gray[
        :top_h,
        :
    ]

    ref_top = ref_gray[
        :top_h,
        :
    ]

    target_w = 900

    target_h = int(
        round(
            top_h
            *
            (
                target_w
                /
                float(w)
            )
        )
    )

    cand_top = cv2.resize(
        cand_top,
        (
            target_w,
            target_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    ref_top = cv2.resize(
        ref_top,
        (
            target_w,
            target_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Mild normalization so lighting does not dominate.
    cand_top = cv2.equalizeHist(
        cand_top
    )

    ref_top = cv2.equalizeHist(
        ref_top
    )

    cand_edges = cv2.Canny(
        cand_top,
        45,
        140,
    )

    ref_edges = cv2.Canny(
        ref_top,
        45,
        140,
    )

    # Direct normalized correlation of header structure.
    cand_f = cand_edges.astype(
        np.float32
    )

    ref_f = ref_edges.astype(
        np.float32
    )

    cand_f -= float(
        cand_f.mean()
    )

    ref_f -= float(
        ref_f.mean()
    )

    denominator = float(
        np.linalg.norm(
            cand_f
        )
        *
        np.linalg.norm(
            ref_f
        )
    )

    correlation = (
        float(
            np.sum(
                cand_f
                *
                ref_f
            )
        )
        /
        denominator
        if denominator > 1e-6
        else 0.0
    )

    # ORB header matches provide a second independent orientation signal.
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10,
    )

    kp_c, des_c = orb.detectAndCompute(
        cand_top,
        None,
    )

    kp_r, des_r = orb.detectAndCompute(
        ref_top,
        None,
    )

    good_count = 0
    inlier_count = 0

    if (
        des_c is not None
        and des_r is not None
        and len(kp_c) >= 8
        and len(kp_r) >= 8
    ):
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        pairs = matcher.knnMatch(
            des_c,
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
                <
                0.72
                *
                second.distance
            ):
                good.append(
                    first
                )

        good_count = len(
            good
        )

        if good_count >= 8:
            src_pts = np.float32(
                [
                    kp_c[
                        m.queryIdx
                    ].pt
                    for m
                    in good
                ]
            ).reshape(
                -1,
                1,
                2,
            )

            dst_pts = np.float32(
                [
                    kp_r[
                        m.trainIdx
                    ].pt
                    for m
                    in good
                ]
            ).reshape(
                -1,
                1,
                2,
            )

            _, inlier_mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                4.0,
            )

            if inlier_mask is not None:
                inlier_count = int(
                    inlier_mask.sum()
                )

    # Header correlation dominates; ORB/inliers refine the choice.
    score = (
        correlation
        *
        1000.0
        +
        good_count
        *
        1.5
        +
        inlier_count
        *
        4.0
    )

    return float(
        score
    )


def ensure_canonical_orientation(
    image: np.ndarray,
    reference: np.ndarray,
    width: int,
    height: int,
    allowed_rotations: Tuple[int, ...] = (0, 90, 180, 270),
) -> tuple[np.ndarray, dict]:
    """
    Force the Manchester header to the TOP.

    Evaluates the permitted cardinal rotations against the canonical reference
    header. Production registration narrows this to 0°/180° after the sensor
    frame has already been normalized to portrait.
    """

    allowed = set(allowed_rotations)
    rotations = tuple(
        rotation
        for rotation in (0, 90, 180, 270)
        if rotation in allowed
    )
    if not rotations:
        raise ValueError("At least one canonical orientation must be allowed.")

    candidates = {
        rotation: _rotate_to_candidate(
            image,
            rotation,
            width,
            height,
        )
        for rotation in rotations
    }

    scores = {
        rotation:
            _header_structure_score(
                candidate,
                reference,
            )
        for rotation, candidate
        in candidates.items()
    }

    best_rotation = max(
        scores,
        key=scores.get,
    )

    oriented = candidates[
        best_rotation
    ]

    return oriented, {
        "selected_rotation":
            int(
                best_rotation
            ),

        "orientation_scores": {
            str(rotation):
                round(
                    float(score),
                    3,
                )
            for rotation, score
            in scores.items()
        },

        "orientation_method":
            "header_structural_matching_" + "_".join(map(str, rotations)),
    }


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

    reference_markers, reference_marker_debug = detect_registration_blocks(reference)

    # Establish the complete A4 boundary before using any internal marks. This
    # guarantees that a nearby filled answer bubble can never become a guessed
    # page corner and crop/elongate only part of the uploaded sheet.
    page_input = image
    capture_pre_rotation = 0
    if image.shape[1] > image.shape[0]:
        # Put a landscape sensor frame into portrait geometry before looking
        # for an A4 boundary. Header matching below resolves which end is top.
        page_input = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        capture_pre_rotation = 90

    document_quad, document_quad_debug = detect_document_quad(
        page_input,
        expected_ratio=width / float(height),
        return_debug=True,
    )
    full_page = warp_document_quad(page_input, document_quad, width, height)
    oriented, orientation_debug = ensure_canonical_orientation(
        full_page,
        reference,
        width,
        height,
        # The sensor frame has already been normalized to portrait. From this
        # point the sheet can only be upright or upside-down. Excluding 90/270
        # prevents a correctly uploaded portrait OMR from turning sideways due
        # to repetitive bubble-grid structure.
        allowed_rotations=(0, 180),
    )
    orientation_debug["selected_rotation"] = (
        capture_pre_rotation + int(orientation_debug["selected_rotation"])
    ) % 360
    orientation_debug["capture_pre_rotation"] = capture_pre_rotation

    # Registration boxes now perform a small geometry correction inside the
    # already-complete page. All four must be genuinely present near their
    # template locations; missing boxes are not inferred from response ink.
    markers, marker_debug = _detect_solid_corner_boxes_on_canonical_page(
        oriented,
        expected_markers=reference_markers,
    )
    pre_registration_validation = _validate_canonical_marker_positions(
        markers,
        width,
        height,
        expected_markers=reference_markers,
        # The full-page contour can follow either the physical paper edge or
        # the inset printed border, so exact pre-warp marker margins vary.
        strict=False,
    )
    pre_registration_validation["corner_regions"] = _validate_marker_corner_regions(
        markers,
        width,
        height,
    )
    coarse, homography = warp_from_registration_blocks(
        oriented,
        markers,
        width,
        height,
        destination_markers=reference_markers,
    )

    result = coarse

    debug: Dict[
        str,
        Any,
    ] = {
        "alignment_method":
            "registration_blocks",

        "document_detection": {
            "document_detected": True,
            "bounds": {
                name: [
                    round(float(point[0]), 2),
                    round(float(point[1]), 2),
                ]
                for name, point in zip(
                    ("top_left", "top_right", "bottom_right", "bottom_left"),
                    document_quad,
                )
            },
            "perspective_correction_applied": True,
        },

        "page_detection": {
            "method": "complete_a4_then_four_registration_blocks",
            "candidate_count": document_quad_debug["candidate_count"],
            "valid_candidate_count": document_quad_debug["valid_candidate_count"],
            "page_area_ratio": document_quad_debug["page_area_ratio"],
            "selected_candidate": 1,
        },

        "document_quad": document_quad.tolist(),

        "output_size": {
            "width":
                width,

            "height":
                height,
        },

        "registration":
            marker_debug,

        "reference_registration":
            reference_marker_debug,

        "orientation":
            orientation_debug,

        "coarse_homography":
            homography.tolist(),
        "homography":
            homography,
    }
    debug["registration"]["pre_registration_validation"] = pre_registration_validation

    debug[
        "document_detection"
    ][
        "rotation_angle"
    ] = orientation_debug[
        "selected_rotation"
    ]

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

    # Fine alignment is optional, but it must not invalidate the complete
    # page geometry established above.  Validate the final recognition image,
    # not merely the pre-refinement warp.
    final_markers, final_marker_debug = detect_registration_blocks(result)
    final_validation = _validate_canonical_marker_positions(
        final_markers,
        width,
        height,
        expected_markers=reference_markers,
        # The coarse homography maps the four validated source markers exactly
        # to these reference points, and both optional refinements are already
        # limited to a 24px corner displacement. A second contour pass can
        # choose nearby border/text ink on dense JEE sheets; record that as
        # diagnostic information instead of rejecting an otherwise valid page.
        strict=False,
    )
    debug["registration"]["final_markers"] = final_marker_debug["markers"]
    debug["registration"]["final_canonical_position_validation"] = final_validation

    if debug_dir is not None:
        page_debug_image = page_input.copy()
        cv2.polylines(
            page_debug_image,
            [document_quad.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 0, 255),
            4,
        )
        # Full-page geometry trace.  The selected quadrilateral is repeated
        # in candidates/selected views because detection metadata records the
        # candidate count and score without altering the source image.
        cv2.imwrite(str(debug_dir / "01_raw_camera.jpg"), image)
        cv2.imwrite(str(debug_dir / "02_page_candidates.jpg"), page_debug_image)
        cv2.imwrite(str(debug_dir / "03_selected_a4_page.jpg"), page_debug_image)
        cv2.imwrite(str(debug_dir / "04_a4_perspective_corrected.jpg"), coarse)
        cv2.imwrite(str(debug_dir / "05_canonical_omr.jpg"), result)
        cv2.imwrite(
            str(debug_dir / "06_corner_block_validation.jpg"),
            _draw_marker_debug(result, final_markers),
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "00_registration_detection.jpg"
            ),
            _draw_marker_debug(
                oriented,
                markers,
            ),
        )

        cv2.imwrite(str(debug_dir / "00_a4_page_quad.jpg"), page_debug_image)

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
                "02_oriented_canonical.jpg"
            ),
            oriented,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "03_canonical_aligned.jpg"
            ),
            result,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "04_reference.jpg"
            ),
            reference,
        )

    return result, debug
