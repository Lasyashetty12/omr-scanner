# scanner.py
from ml_omr.hybrid_reader import scan_answers_ml
from omr_preprocess import canonicalize_omr
import json
import os

import cv2
import numpy as np

from config import (
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_CONTRAST,
)
from ml_omr.column_calibration import (
    auto_calibrate_neet_columns,
    generate_calibrated_bubble_coordinates,
    draw_calibration_debug,
    validate_column_alignment,
)
from ml_omr.grid_detector import (
    fit_response_grid,
    draw_grid_detection_debug,
)


# ============================================================
# CANONICAL NEET/KCET REGISTRATION MARKERS
# ============================================================
#
# Measured from the user's blank Manchester NEET reference sheet
# and scaled to the 1600 x 2200 coordinate system used by neet.json.
#
# Order:
#   TL, TR, BR, BL
#
CANONICAL_REGISTRATION_MARKERS = np.array(
    [
        [81.2, 78.3],
        [1522.0, 78.3],
        [1523.3, 2124.2],
        [79.9, 2120.4],
    ],
    dtype="float32",
)


# ============================================================
# ML MODEL CHECK
# ============================================================

def ensure_ml_model_available():
    from pathlib import Path

    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "bubble_classifier.onnx"
    )

    if not model_path.exists():
        raise ValueError(
            "ML bubble model is missing. Expected: "
            f"{model_path}"
        )


# ============================================================
# TEMPLATE
# ============================================================

def load_template(template_path):

    try:
        with open(
            template_path,
            "r",
            encoding="utf-8",
        ) as file:
            template = json.load(file)

    except FileNotFoundError:
        raise ValueError(
            f"Template not found: {template_path}"
        )

    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON template: {template_path}"
        )

    common_required = [
        "template_name",
        "exam_name",
        "sheet_width",
        "sheet_height",
    ]

    for field in common_required:
        if field not in template:
            raise ValueError(
                f"Template missing required field: {field}"
            )

    exam_name = (
        str(
            template.get(
                "exam_name",
                ""
            )
        )
        .strip()
        .upper()
    )

    if exam_name in [
        "NEET",
        "KCET",
    ]:

        required = [
            "total_questions",
            "questions_per_column",
            "options",
            "columns",
            "paper_code",
        ]

        for field in required:
            if field not in template:
                raise ValueError(
                    f"{exam_name} template missing field: {field}"
                )

    elif exam_name == "JEE":

        if "series" not in template:
            raise ValueError(
                "JEE template requires 'series'."
            )

    else:
        raise ValueError(
            f"Unsupported exam in template: {exam_name}"
        )

    return template


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            "Unable to read uploaded image."
        )

    return image



# ============================================================
# CANONICAL PREPROCESSING
# ============================================================

def preprocess_to_canonical(
    image,
    template,
):
    """
    Convert a mobile photo into the same geometry as the clean
    canonical NEET/KCET reference before any paper-code or answer
    reading occurs.

    Pipeline:
      1. generous registration-marker crop
      2. detect registration marks inside crop
      3. marker-to-marker homography
      4. force exact template size
      5. mild grayscale normalization for downstream reading

    The template JSON coordinates are never modified.
    """

    cropped_image, crop_debug = (
        crop_omr_by_corner_boxes(
            image
        )
    )

    registration_markers = (
        detect_registration_markers_in_crop(
            cropped_image
        )
    )

    corrected = (
        perspective_transform_from_registration_markers(
            cropped_image,
            registration_markers,
            template,
        )
    )

    expected_width = int(
        template["sheet_width"]
    )

    expected_height = int(
        template["sheet_height"]
    )

    actual_height, actual_width = (
        corrected.shape[:2]
    )

    if (
        actual_width != expected_width
        or actual_height != expected_height
    ):
        corrected = cv2.resize(
            corrected,
            (
                expected_width,
                expected_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    gray = cv2.cvtColor(
        corrected,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    normalized_gray = clahe.apply(
        gray
    )

    normalized_gray = cv2.GaussianBlur(
        normalized_gray,
        (3, 3),
        0,
    )

    debug = {
        "crop":
            crop_debug,

        "registration_markers": [
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
            in registration_markers
        ],

        "output_size": {
            "width":
                int(expected_width),

            "height":
                int(expected_height),
        },
    }

    return (
        corrected,
        normalized_gray,
        cropped_image,
        debug,
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_blur_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def calculate_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        np.mean(gray)
    )


def calculate_contrast(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        np.std(gray)
    )


def validate_image_quality(image):

    blur = calculate_blur_score(
        image
    )

    brightness = calculate_brightness(
        image
    )

    contrast = calculate_contrast(
        image
    )

    if blur < MIN_BLUR_SCORE:
        raise ValueError(
            "Image is too blurry. Please scan again."
        )

    if brightness < MIN_BRIGHTNESS:
        raise ValueError(
            "Image is too dark."
        )

    if brightness > MAX_BRIGHTNESS:
        raise ValueError(
            "Image is overexposed."
        )

    if contrast < MIN_CONTRAST:
        raise ValueError(
            "Image contrast is too low."
        )

    return {
        "blur": round(
            blur,
            2
        ),
        "brightness": round(
            brightness,
            2
        ),
        "contrast": round(
            contrast,
            2
        ),
    }



# ============================================================
# CORNER-BOX FIRST CROP
# ============================================================

def find_omr_corner_boxes(image):
    """
    Detect large black registration marks near the OMR corners.

    BL is allowed to be missing because on real sheets it may merge
    with the printed border. The initial crop can still be estimated
    from TL/TR/BR.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]
    image_area = float(height * width)

    candidates = []

    for contour in contours:
        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < image_area * 0.00015:
            continue

        if area > image_area * 0.03:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 8 or h < 8:
            continue

        aspect = w / float(h)

        if not 0.35 <= aspect <= 2.5:
            continue

        rect_area = float(w * h)

        fill_ratio = (
            area / rect_area
            if rect_area > 0
            else 0.0
        )

        if fill_ratio < 0.35:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        candidates.append(
            {
                "area": area,
                "center": (
                    float(cx),
                    float(cy),
                ),
                "box": (
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                ),
                "fill_ratio": float(
                    fill_ratio
                ),
            }
        )

    quadrants = {
        "tl": [],
        "tr": [],
        "br": [],
        "bl": [],
    }

    for candidate in candidates:
        cx, cy = candidate["center"]

        if (
            cx < width * 0.45
            and cy < height * 0.35
        ):
            quadrants["tl"].append(
                candidate
            )

        if (
            cx > width * 0.55
            and cy < height * 0.35
        ):
            quadrants["tr"].append(
                candidate
            )

        if (
            cx > width * 0.55
            and cy > height * 0.65
        ):
            quadrants["br"].append(
                candidate
            )

        if (
            cx < width * 0.45
            and cy > height * 0.65
        ):
            quadrants["bl"].append(
                candidate
            )

    # Score by closeness to the corresponding outer image corner,
    # with a small preference for large/dark marks.
    targets = {
        "tl": np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        "tr": np.array(
            [float(width), 0.0],
            dtype=np.float32,
        ),
        "br": np.array(
            [float(width), float(height)],
            dtype=np.float32,
        ),
        "bl": np.array(
            [0.0, float(height)],
            dtype=np.float32,
        ),
    }

    selected = {}

    for name, items in quadrants.items():

        if not items:
            continue

        ranked = []

        for item in items:
            point = np.array(
                item["center"],
                dtype=np.float32,
            )

            distance = float(
                np.linalg.norm(
                    point
                    - targets[name]
                )
            )

            score = (
                distance
                - item["area"] * 0.004
                - item["fill_ratio"] * 8.0
            )

            ranked.append(
                (
                    score,
                    item,
                )
            )

        ranked.sort(
            key=lambda pair:
            pair[0]
        )

        selected[name] = (
            ranked[0][1]
        )

    if (
        "tl" not in selected
        or "tr" not in selected
        or "br" not in selected
    ):
        raise ValueError(
            "Could not detect enough OMR registration marks. "
            "Keep the full page inside the A4 guide."
        )

    return selected


def crop_omr_by_corner_boxes(
    image,
    padding_ratio=0.10,
):
    """
    Crop the camera image around the OMR before perspective correction.

    IMPORTANT:
    This crop is deliberately generous. It is only used to remove
    unrelated background. It must NOT crop away the paper edges.
    """

    boxes = find_omr_corner_boxes(
        image
    )

    height, width = image.shape[:2]

    tl = np.array(
        boxes["tl"]["center"],
        dtype=np.float32,
    )

    tr = np.array(
        boxes["tr"]["center"],
        dtype=np.float32,
    )

    br = np.array(
        boxes["br"]["center"],
        dtype=np.float32,
    )

    if "bl" in boxes:
        bl = np.array(
            boxes["bl"]["center"],
            dtype=np.float32,
        )
    else:
        # Parallelogram estimate is good enough for the INITIAL CROP.
        # The final perspective transform does NOT use this estimate.
        bl = (
            tl
            + (
                br - tr
            )
        )

    points = np.array(
        [
            tl,
            tr,
            br,
            bl,
        ],
        dtype=np.float32,
    )

    min_x = float(
        np.min(
            points[:, 0]
        )
    )

    max_x = float(
        np.max(
            points[:, 0]
        )
    )

    min_y = float(
        np.min(
            points[:, 1]
        )
    )

    max_y = float(
        np.max(
            points[:, 1]
        )
    )

    span_w = max(
        1.0,
        max_x - min_x,
    )

    span_h = max(
        1.0,
        max_y - min_y,
    )

    # Registration marks sit inside the paper boundary.
    # Use generous margins so the true paper corners remain visible.
    pad_x = int(
        span_w * padding_ratio
    )

    pad_y = int(
        span_h * padding_ratio
    )

    x1 = max(
        0,
        int(min_x) - pad_x,
    )

    y1 = max(
        0,
        int(min_y) - pad_y,
    )

    x2 = min(
        width,
        int(max_x) + pad_x,
    )

    y2 = min(
        height,
        int(max_y) + pad_y,
    )

    if (
        x2 - x1 < width * 0.35
        or y2 - y1 < height * 0.35
    ):
        raise ValueError(
            "Initial OMR crop is too small."
        )

    cropped = image[
        y1:y2,
        x1:x2
    ].copy()

    return cropped, {
        "method":
            "registration_marker_crop",

        "x1":
            int(x1),

        "y1":
            int(y1),

        "x2":
            int(x2),

        "y2":
            int(y2),
    }


def detect_page_corners_from_crop(
    cropped_image,
    template,
):
    """
    Detect the OUTER paper corners after the background has already
    been removed.

    The four black registration marks are NOT used as final warp
    corners. This avoids the 'halved/clipped' result that happens
    when marker centres are incorrectly mapped to the canvas edges.
    """

    gray = cv2.cvtColor(
        cropped_image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    image_area = float(
        width * height
    )

    expected_ratio = (
        float(
            template["sheet_width"]
        )
        /
        float(
            template["sheet_height"]
        )
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # A white-paper mask is now reliable because most unrelated
    # background was removed by the first crop.
    _, white = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15),
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        white,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    best_points = None
    best_score = -1.0

    for contour in contours[:30]:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < image_area * 0.35:
            continue

        # First try an actual 4-point approximation.
        perimeter = cv2.arcLength(
            contour,
            True,
        )

        candidate_sets = []

        for epsilon in (
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
        ):
            approx = cv2.approxPolyDP(
                contour,
                epsilon * perimeter,
                True,
            )

            if len(approx) == 4:
                candidate_sets.append(
                    approx
                    .reshape(4, 2)
                    .astype(
                        "float32"
                    )
                )

        # Also use minAreaRect as a robust fallback when one page edge
        # is curved/occluded or merged with another sheet underneath.
        rect = cv2.minAreaRect(
            contour
        )

        box = cv2.boxPoints(
            rect
        ).astype(
            "float32"
        )

        candidate_sets.append(
            box
        )

        for points in candidate_sets:

            ordered = order_points(
                points
            )

            tl, tr, br, bl = ordered

            top_w = float(
                np.linalg.norm(
                    tr - tl
                )
            )

            bottom_w = float(
                np.linalg.norm(
                    br - bl
                )
            )

            left_h = float(
                np.linalg.norm(
                    bl - tl
                )
            )

            right_h = float(
                np.linalg.norm(
                    br - tr
                )
            )

            avg_w = (
                top_w
                + bottom_w
            ) / 2.0

            avg_h = (
                left_h
                + right_h
            ) / 2.0

            if (
                avg_w <= 0
                or avg_h <= 0
            ):
                continue

            ratio = (
                avg_w
                / avg_h
            )

            ratio_error = abs(
                ratio
                - expected_ratio
            ) / expected_ratio

            # Perspective can alter the apparent ratio, but an OMR
            # should still be broadly portrait-shaped.
            if ratio_error > 0.45:
                continue

            quad_area = abs(
                float(
                    cv2.contourArea(
                        ordered.reshape(
                            (-1, 1, 2)
                        )
                    )
                )
            )

            coverage = (
                quad_area
                / image_area
            )

            if coverage < 0.45:
                continue

            score = (
                coverage * 3.0
                + max(
                    0.0,
                    1.0 - ratio_error
                ) * 2.0
            )

            if score > best_score:
                best_score = score
                best_points = ordered

    if best_points is None:
        # Final safe fallback: use the full generous crop.
        # Because the crop already follows the OMR region closely,
        # this is safer than mapping marker centres to canvas edges.
        return np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )

    return best_points


# ============================================================
# REGISTRATION MARKERS
# ============================================================

def find_marker_candidates(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, binary = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = (
        image.shape[:2]
    )

    image_area = (
        height * width
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if (
            area
            < image_area * 0.00015
        ):
            continue

        if (
            area
            > image_area * 0.03
        ):
            continue

        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )

        if h == 0:
            continue

        aspect_ratio = (
            w / float(h)
        )

        if not (
            0.70
            <= aspect_ratio
            <= 1.30
        ):
            continue

        rectangle_area = (
            w * h
        )

        if rectangle_area == 0:
            continue

        fill_ratio = (
            area
            / float(
                rectangle_area
            )
        )

        if fill_ratio < 0.60:
            continue

        candidates.append(
            {
                "area": float(area),

                "center": (
                    x + w / 2.0,
                    y + h / 2.0,
                ),

                "box": (
                    x,
                    y,
                    w,
                    h,
                ),
            }
        )

    return candidates


def select_four_markers(
    image,
    candidates,
):

    if len(candidates) < 4:
        raise ValueError(
            "Could not find all four registration markers."
        )

    height, width = (
        image.shape[:2]
    )

    centers = np.array(
        [
            candidate["center"]
            for candidate
            in candidates
        ],
        dtype="float32",
    )

    top_left = min(
        centers,
        key=lambda p:
        p[0] + p[1],
    )

    top_right = min(
        centers,
        key=lambda p:
        (width - p[0])
        + p[1],
    )

    bottom_right = min(
        centers,
        key=lambda p:
        (width - p[0])
        + (height - p[1]),
    )

    bottom_left = min(
        centers,
        key=lambda p:
        p[0]
        + (height - p[1]),
    )

    corners = np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype="float32",
    )

    unique = np.unique(
        corners.astype(int),
        axis=0,
    )

    if len(unique) != 4:
        raise ValueError(
            "Registration marker detection is ambiguous."
        )

    return corners

def detect_corner_markers(
    image,
    template,
):
    """
    Detect the actual white OMR sheet inside a mobile-camera image.

    Candidate quadrilaterals are ranked using:
    - expected template aspect ratio
    - brightness / white-paper fraction
    - candidate size

    This prevents the laptop screen, browser frame, table, or the
    complete camera frame from being mistaken for the OMR sheet.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        40,
        150,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "Could not detect OMR sheet."
        )

    image_height, image_width = image.shape[:2]

    image_area = float(
        image_height
        * image_width
    )

    expected_width = float(
        template["sheet_width"]
    )

    expected_height = float(
        template["sheet_height"]
    )

    expected_ratio = (
        expected_width
        / expected_height
    )

    best_candidate = None
    best_score = -1.0

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    for contour in contours[:150]:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        area_ratio = (
            area
            / image_area
        )

        # The paper may occupy only part of the mobile frame.
        if area_ratio < 0.06:
            continue

        # Never accept almost the entire camera frame.
        if area_ratio > 0.92:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        for epsilon_factor in (
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
        ):

            approx = cv2.approxPolyDP(
                contour,
                epsilon_factor
                * perimeter,
                True,
            )

            if len(approx) != 4:
                continue

            points = (
                approx
                .reshape(4, 2)
                .astype("float32")
            )

            ordered = order_points(
                points
            )

            tl, tr, br, bl = ordered

            top_width = float(
                np.linalg.norm(
                    tr - tl
                )
            )

            bottom_width = float(
                np.linalg.norm(
                    br - bl
                )
            )

            left_height = float(
                np.linalg.norm(
                    bl - tl
                )
            )

            right_height = float(
                np.linalg.norm(
                    br - tr
                )
            )

            average_width = (
                top_width
                + bottom_width
            ) / 2.0

            average_height = (
                left_height
                + right_height
            ) / 2.0

            if (
                average_width <= 0
                or average_height <= 0
            ):
                continue

            ratio = (
                average_width
                / average_height
            )

            # Allow perspective variation around the template ratio.
            if not (
                expected_ratio * 0.68
                <= ratio
                <= expected_ratio * 1.35
            ):
                continue

            mask = np.zeros(
                gray.shape,
                dtype=np.uint8,
            )

            polygon = (
                ordered
                .astype(np.int32)
                .reshape((-1, 1, 2))
            )

            cv2.fillConvexPoly(
                mask,
                polygon,
                255,
            )

            pixels = gray[
                mask > 0
            ]

            if pixels.size == 0:
                continue

            mean_brightness = float(
                np.mean(
                    pixels
                )
            )

            white_fraction = float(
                np.mean(
                    pixels > 150
                )
            )

            ratio_error = abs(
                ratio
                - expected_ratio
            )

            ratio_score = max(
                0.0,
                1.0
                - (
                    ratio_error
                    / expected_ratio
                ),
            )

            brightness_score = (
                mean_brightness
                / 255.0
            )

            size_score = min(
                area_ratio / 0.40,
                1.0,
            )

            score = (
                ratio_score * 4.0
                + white_fraction * 4.0
                + brightness_score * 2.0
                + size_score
            )

            if score > best_score:
                best_score = score
                best_candidate = ordered

            break

    if best_candidate is None:
        raise ValueError(
            "Could not locate the white OMR sheet. "
            "Keep all four paper corners visible, "
            "hold the phone parallel to the page, "
            "and let the paper fill most of the yellow A4 guide."
        )

    return best_candidate



# ============================================================
# REGISTRATION-MARKER ALIGNMENT TO CANONICAL TEMPLATE
# ============================================================

def _marker_candidate_score(
    gray,
    contour,
    roi_origin,
    target_point,
    full_width,
    full_height,
):
    """
    Score one dark contour as a possible large registration mark.
    Lower score is better.
    """

    x0, y0 = roi_origin

    area = float(
        cv2.contourArea(
            contour
        )
    )

    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(
        contour
    )

    if (
        w < 8
        or h < 8
    ):
        return None

    aspect = (
        w / float(h)
    )

    # Square marks and the round BR mark are both compact.
    # BL can merge slightly with printed borders, so stay tolerant.
    if not (
        0.35
        <= aspect
        <= 2.8
    ):
        return None

    rect_area = float(
        w * h
    )

    fill_ratio = (
        area / rect_area
        if rect_area > 0
        else 0.0
    )

    if fill_ratio < 0.28:
        return None

    gx = int(
        x0 + x
    )

    gy = int(
        y0 + y
    )

    roi = gray[
        gy:
        min(
            gray.shape[0],
            gy + h,
        ),
        gx:
        min(
            gray.shape[1],
            gx + w,
        )
    ]

    if roi.size == 0:
        return None

    mean_darkness = (
        255.0
        - float(
            np.mean(
                roi
            )
        )
    ) / 255.0

    cx = (
        gx + w / 2.0
    )

    cy = (
        gy + h / 2.0
    )

    target_x, target_y = (
        target_point
    )

    diagonal = max(
        1.0,
        float(
            np.hypot(
                full_width,
                full_height,
            )
        ),
    )

    distance = float(
        np.hypot(
            cx - target_x,
            cy - target_y,
        )
    ) / diagonal

    normalized_size = min(
        area
        / max(
            1.0,
            (
                full_width
                * full_height
                * 0.003
            ),
        ),
        1.0,
    )

    # Distance dominates.
    # Large, dark, compact regions receive a small bonus.
    score = (
        distance
        - 0.10 * normalized_size
        - 0.06 * fill_ratio
        - 0.05 * mean_darkness
    )

    return {
        "score":
            float(score),

        "center":
            (
                float(cx),
                float(cy),
            ),

        "box":
            (
                int(gx),
                int(gy),
                int(w),
                int(h),
            ),

        "area":
            area,

        "fill_ratio":
            float(fill_ratio),

        "mean_darkness":
            float(mean_darkness),
    }


def detect_registration_markers_in_crop(
    image,
):
    """
    Detect the four large Manchester registration marks inside the
    already-cropped OMR image.

    The initial crop is intentionally generous. At this stage the
    four marks should be near the outer corners of the crop.

    If BL cannot be isolated because it touches the border, it is
    estimated from TL/TR/BR rather than rejecting the scan.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape[:2]

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    # Remove/thin printed rules while retaining the large solid marks.
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1,
    )

    # Broad ROIs. They are deliberately larger than the physical
    # marker locations to tolerate camera perspective.
    regions = {
        "tl": (
            0,
            0,
            int(
                width * 0.40
            ),
            int(
                height * 0.28
            ),
        ),

        "tr": (
            int(
                width * 0.60
            ),
            0,
            width,
            int(
                height * 0.28
            ),
        ),

        "br": (
            int(
                width * 0.60
            ),
            int(
                height * 0.72
            ),
            width,
            height,
        ),

        "bl": (
            0,
            int(
                height * 0.72
            ),
            int(
                width * 0.40
            ),
            height,
        ),
    }

    targets = {
        "tl": (
            width * 0.08,
            height * 0.06,
        ),

        "tr": (
            width * 0.92,
            height * 0.06,
        ),

        "br": (
            width * 0.92,
            height * 0.94,
        ),

        "bl": (
            width * 0.08,
            height * 0.94,
        ),
    }

    selected = {}

    for name, (
        x1,
        y1,
        x2,
        y2,
    ) in regions.items():

        roi = cleaned[
            y1:y2,
            x1:x2
        ]

        contours, _ = cv2.findContours(
            roi,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        ranked = []

        for contour in contours:

            result = _marker_candidate_score(
                gray=gray,
                contour=contour,
                roi_origin=(
                    x1,
                    y1,
                ),
                target_point=targets[
                    name
                ],
                full_width=width,
                full_height=height,
            )

            if result is None:
                continue

            ranked.append(
                result
            )

        if ranked:

            ranked.sort(
                key=lambda item:
                item["score"]
            )

            selected[name] = (
                ranked[0]
            )

    # TL/TR/BR are required. BL gets a geometric fallback because
    # the user's real sheet can merge the BL block with the border.
    for name in (
        "tl",
        "tr",
        "br",
    ):

        if name not in selected:

            raise ValueError(
                "Could not identify the OMR "
                f"{name.upper()} registration mark."
            )

    tl = np.array(
        selected[
            "tl"
        ][
            "center"
        ],
        dtype=np.float32,
    )

    tr = np.array(
        selected[
            "tr"
        ][
            "center"
        ],
        dtype=np.float32,
    )

    br = np.array(
        selected[
            "br"
        ][
            "center"
        ],
        dtype=np.float32,
    )

    predicted_bl = (
        tl
        + (
            br - tr
        )
    )

    if "bl" in selected:

        detected_bl = np.array(
            selected[
                "bl"
            ][
                "center"
            ],
            dtype=np.float32,
        )

        page_diagonal = max(
            1.0,
            float(
                np.hypot(
                    width,
                    height,
                )
            ),
        )

        difference = float(
            np.linalg.norm(
                detected_bl
                - predicted_bl
            )
        )

        # Trust the detected BL only when it agrees reasonably well
        # with the other three markers.
        if (
            difference
            <= page_diagonal * 0.10
        ):
            bl = detected_bl
        else:
            bl = predicted_bl

    else:
        bl = predicted_bl

    bl[0] = np.clip(
        bl[0],
        0,
        width - 1,
    )

    bl[1] = np.clip(
        bl[1],
        0,
        height - 1,
    )

    source = np.array(
        [
            tl,
            tr,
            br,
            bl,
        ],
        dtype="float32",
    )

    quad_area = abs(
        float(
            cv2.contourArea(
                source.reshape(
                    (-1, 1, 2)
                )
            )
        )
    )

    image_area = float(
        width * height
    )

    if (
        quad_area
        < image_area * 0.35
    ):
        raise ValueError(
            "Detected registration marks do not form "
            "a valid OMR page."
        )

    return source


def perspective_transform_from_registration_markers(
    image,
    source_markers,
    template,
):
    """
    Warp detected registration-marker centres to the exact marker
    centres measured from the blank reference sheet.

    This is different from mapping marker centres to the OUTER image
    corners. Mapping them to the canvas edges caused the previously
    observed clipped / 'halved' corrected image.
    """

    width = int(
        template[
            "sheet_width"
        ]
    )

    height = int(
        template[
            "sheet_height"
        ]
    )

    source = order_points(
        source_markers
    )

    # Scale the canonical NEET reference positions if another
    # Manchester template uses a different canonical canvas size.
    scale_x = (
        width
        / 1600.0
    )

    scale_y = (
        height
        / 2200.0
    )

    destination = (
        CANONICAL_REGISTRATION_MARKERS.copy()
    )

    destination[:, 0] *= (
        scale_x
    )

    destination[:, 1] *= (
        scale_y
    )

    matrix = cv2.getPerspectiveTransform(
        source.astype(
            "float32"
        ),
        destination.astype(
            "float32"
        ),
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

    return corrected


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def order_points(points):

    points = np.array(
        points,
        dtype="float32",
    )

    ordered = np.zeros(
        (4, 2),
        dtype="float32",
    )

    sums = points.sum(
        axis=1
    )

    differences = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    ordered[0] = points[
        np.argmin(sums)
    ]

    ordered[2] = points[
        np.argmax(sums)
    ]

    ordered[1] = points[
        np.argmin(
            differences
        )
    ]

    ordered[3] = points[
        np.argmax(
            differences
        )
    ]

    return ordered


def perspective_transform(
    image,
    corners,
    template,
):

    width = int(
        template[
            "sheet_width"
        ]
    )

    height = int(
        template[
            "sheet_height"
        ]
    )

    source = order_points(
        corners
    )

    destination = np.array(
        [
            [0, 0],

            [
                width - 1,
                0,
            ],

            [
                width - 1,
                height - 1,
            ],

            [
                0,
                height - 1,
            ],
        ],
        dtype="float32",
    )

    matrix = (
        cv2.getPerspectiveTransform(
            source,
            destination,
        )
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
    )

    return corrected


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    gray = clahe.apply(
        gray
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return gray


def create_threshold_image(image):

    gray = normalize_grayscale(
        image
    )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )


# ============================================================
# BUBBLE FILL
# ============================================================

def get_fill_ratio(
    gray_image,
    x,
    y,
    template,
):

    radius = int(
        template.get(
            "bubble_radius",
            10,
        )
    )

    dark_threshold = int(
        template.get(
            "dark_pixel_threshold",
            100,
        )
    )

    height, width = (
        gray_image.shape[:2]
    )

    x = int(x)
    y = int(y)

    x1 = max(
        0,
        x - radius,
    )

    y1 = max(
        0,
        y - radius,
    )

    x2 = min(
        width,
        x + radius + 1,
    )

    y2 = min(
        height,
        y + radius + 1,
    )

    roi = gray_image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return 0.0

    mask = np.zeros_like(
        roi,
        dtype=np.uint8,
    )

    local_x = (
        x - x1
    )

    local_y = (
        y - y1
    )

    cv2.circle(
        mask,
        (
            local_x,
            local_y,
        ),
        radius,
        255,
        -1,
    )

    dark_pixels = np.logical_and(
        roi < dark_threshold,
        mask > 0,
    )

    dark_count = (
        np.count_nonzero(
            dark_pixels
        )
    )

    total_count = (
        np.count_nonzero(
            mask
        )
    )

    if total_count == 0:
        return 0.0

    return (
        dark_count
        / float(total_count)
    )


# ============================================================
# GENERATE NEET / KCET QUESTION COORDINATES
# ============================================================

def generate_bubble_coordinates(template):

    total_questions = int(
        template["total_questions"]
    )

    questions_per_column = int(
        template["questions_per_column"]
    )

    options = template["options"]
    columns = template["columns"]

    question_y_positions = template.get(
        "question_y_positions"
    )

    if question_y_positions:

        if (
            len(question_y_positions)
            != questions_per_column
        ):
            raise ValueError(
                "question_y_positions must contain "
                f"{questions_per_column} values."
            )

    else:

        start_y = int(
            template.get(
                "question_start_y",
                0
            )
        )

        row_gap = int(
            template.get(
                "question_row_gap",
                0
            )
        )

        if row_gap <= 0:
            raise ValueError(
                "question_row_gap must be greater than zero."
            )

    coordinates = {}

    for question in range(
        1,
        total_questions + 1
    ):

        column_index = (
            question - 1
        ) // questions_per_column

        row_index = (
            question - 1
        ) % questions_per_column

        if column_index >= len(columns):

            raise ValueError(
                "Template does not contain "
                "enough answer columns."
            )

        if question_y_positions:

            y = int(
                question_y_positions[
                    row_index
                ]
            )

        else:

            y = (
                start_y
                +
                row_index
                *
                row_gap
            )

        coordinates[
            question
        ] = {}

        for option in options:

            if (
                option
                not in
                columns[column_index]
            ):

                raise ValueError(
                    f"Missing coordinate "
                    f"for option {option}."
                )

            x = int(
                columns[
                    column_index
                ][option]
            )

            coordinates[
                question
            ][option] = (
                x,
                y
            )

    return coordinates


# ============================================================
# SINGLE MCQ
# ============================================================

def detect_question_answer(
    gray_image,
    coordinates,
    template,
):

    options = template[
        "options"
    ]

    blank_threshold = float(
        template.get(
            "blank_threshold",
            0.18,
        )
    )

    filled_threshold = float(
        template.get(
            "filled_threshold",
            0.50,
        )
    )

    multiple_threshold = float(
        template.get(
            "multiple_threshold",
            filled_threshold,
        )
    )

    scores = {}

    for option in options:

        x, y = coordinates[
            option
        ]

        scores[
            option
        ] = get_fill_ratio(
            gray_image,
            x,
            y,
            template,
        )

    ranked = sorted(
        scores.items(),
        key=lambda item:
        item[1],
        reverse=True,
    )

    if not ranked:
        return {
            "answer": "BLANK",
            "scores": {},
            "highest_score": 0.0,
            "confidence_gap": 0.0,
        }

    highest_option = (
        ranked[0][0]
    )

    highest_score = float(
        ranked[0][1]
    )

    second_score = float(
        ranked[1][1]
        if len(ranked) > 1
        else 0
    )

    filled_options = [
        option

        for option, score
        in scores.items()

        if (
            score
            >= multiple_threshold
        )
    ]

    if len(
        filled_options
    ) >= 2:

        answer = "MULTIPLE"

    elif (
        highest_score
        >= filled_threshold
    ):

        answer = (
            highest_option
        )

    elif (
        highest_score
        < blank_threshold
    ):

        answer = "BLANK"

    else:

        answer = "UNCERTAIN"

    return {
        "answer":
            answer,

        "scores": {
            key: round(
                float(value),
                4,
            )

            for key, value
            in scores.items()
        },

        "highest_score":
            round(
                highest_score,
                4,
            ),

        "confidence_gap":
            round(
                highest_score
                - second_score,
                4,
            ),
    }


# ============================================================
# SCAN NEET / KCET ANSWERS WITH ML
# ============================================================

def scan_answers(
    corrected_image,
    template,
):
    """
    ML-based answer reader for NEET / KCET.

    Runtime flow:
      1. convert corrected image to grayscale
      2. auto-calibrate the four response columns
      3. build calibrated bubble coordinates
      4. run the relative hybrid ML reader
      5. convert its output to the existing scanner/scorer format

    The template JSON itself is never modified.
    """

    if corrected_image.ndim == 3:
        gray = cv2.cvtColor(
            corrected_image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = corrected_image.copy()

    column_offsets = (
        auto_calibrate_neet_columns(
            gray,
            template,
        )
    )

    alignment_quality = (
        validate_column_alignment(
            column_offsets
        )
    )

    coordinates = (
        generate_calibrated_bubble_coordinates(
            template,
            column_offsets,
        )
    )

    # --------------------------------------------------------
    # ACTUAL PRINTED BUBBLE-GRID DETECTION
    # --------------------------------------------------------
    fitted_coordinates, grid_debug_info = (
        fit_response_grid(
            gray,
            coordinates,
            template,
        )
    )

    if not os.environ.get(
        "VERCEL"
    ):
        calibration_debug_base = (
            cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )
        )

        calibration_debug = (
            draw_calibration_debug(
                calibration_debug_base,
                template,
                column_offsets,
            )
        )

        cv2.imwrite(
            "column_calibration_debug.jpg",
            calibration_debug,
        )
        grid_detection_debug = (
            draw_grid_detection_debug(
                calibration_debug_base,
                template,
                coordinates,
                fitted_coordinates,
                grid_debug_info,
            )
        )

        cv2.imwrite(
            "grid_detection_debug.jpg",
            grid_detection_debug,
        )

    raw_answers, ml_debug = (
        scan_answers_ml(
            gray=gray,
            coordinates=fitted_coordinates,
            crop_radius=int(
                template.get(
                    "ml_crop_radius",
                    16,
                )
            ),
            filled_confidence=float(
                template.get(
                    "ml_filled_confidence",
                    0.70,
                )
            ),
            ambiguous_confidence=float(
                template.get(
                    "ml_ambiguous_confidence",
                    0.60,
                )
            ),
        )
    )

    answers = {}

    for question in coordinates:

        detected = raw_answers.get(
            question
        )

        details = ml_debug.get(
            question,
            {},
        )

        status = details.get(
            "status",
            "blank",
        )

        if detected == "MULTIPLE":
            final_answer = "MULTIPLE"

        elif detected in template[
            "options"
        ]:
            final_answer = detected

        elif status == "ambiguous":
            final_answer = "UNCERTAIN"

        else:
            final_answer = "BLANK"

        answers[
            question
        ] = {
            "answer":
                final_answer,

            "ml_status":
                status,

            "ml":
                details,
        }

    return answers


def draw_answer_analysis(
    corrected_image,
    template,
    answers,
):
    """
    Draw the final OMR decisions on the corrected image.

    IMPORTANT:
    This recalculates the same per-column calibration used by scan_answers(),
    so the circles shown here correspond to the actual runtime sampling
    positions rather than the raw JSON coordinates.

    Green  = selected single answer
    Red    = multiple
    Yellow = uncertain
    Gray   = blank/unselected
    """

    debug_image = (
        corrected_image.copy()
    )

    if corrected_image.ndim == 3:
        gray = cv2.cvtColor(
            corrected_image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = corrected_image.copy()

    column_offsets = (
        auto_calibrate_neet_columns(
            gray,
            template,
        )
    )

    coordinates = (
        generate_calibrated_bubble_coordinates(
            template,
            column_offsets,
        )
    )
    coordinates, _grid_debug_info = (
        fit_response_grid(
            gray,
            coordinates,
            template,
        )
    )

    option_colors = {
        "selected":
            (0, 200, 0),

        "multiple":
            (0, 0, 255),

        "uncertain":
            (0, 215, 255),

        "blank":
            (160, 160, 160),
    }

    radius = (
        int(
            template.get(
                "bubble_radius",
                11,
            )
        )
        + 5
    )

    for question, option_map in (
        coordinates.items()
    ):

        result = answers.get(
            question,
            {},
        )

        final_answer = result.get(
            "answer",
            "BLANK",
        )

        ml_details = result.get(
            "ml",
            {},
        )

        multiple_options = (
            ml_details.get(
                "multiple_options",
                [],
            )
        )

        best_option = (
            ml_details.get(
                "best_option"
            )
        )

        for option, (
            x,
            y,
        ) in option_map.items():

            color = option_colors[
                "blank"
            ]

            thickness = 1

            if (
                final_answer
                in template["options"]
                and option == final_answer
            ):
                color = option_colors[
                    "selected"
                ]

                thickness = 4

            elif (
                final_answer
                == "MULTIPLE"
                and option
                in multiple_options
            ):
                color = option_colors[
                    "multiple"
                ]

                thickness = 4

            elif (
                final_answer
                == "UNCERTAIN"
                and option
                == best_option
            ):
                color = option_colors[
                    "uncertain"
                ]

                thickness = 3

            cv2.circle(
                debug_image,
                (
                    int(x),
                    int(y),
                ),
                radius,
                color,
                thickness,
            )

        first_option = (
            template["options"][0]
        )

        label_x = int(
            option_map[
                first_option
            ][0]
            - 80
        )

        label_y = int(
            option_map[
                first_option
            ][1]
            + 5
        )

        if (
            final_answer
            == "MULTIPLE"
        ):
            text_color = (
                0,
                0,
                255,
            )

            label = (
                f"{question}:MULTI"
            )

        elif (
            final_answer
            == "UNCERTAIN"
        ):
            text_color = (
                0,
                165,
                255,
            )

            label = (
                f"{question}:?"
            )

        elif (
            final_answer
            == "BLANK"
        ):
            text_color = (
                100,
                100,
                100,
            )

            label = (
                f"{question}:-"
            )

        else:
            text_color = (
                0,
                180,
                0,
            )

            label = (
                f"{question}:{final_answer}"
            )

        cv2.putText(
            debug_image,
            label,
            (
                label_x,
                label_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return debug_image


# ============================================================
# QUESTION PAPER CODE
# NEET / KCET
# ============================================================

def detect_paper_code(
    gray_image,
    template,
):

    paper_code_config = (
        template.get(
            "paper_code"
        )
    )

    if not paper_code_config:
        raise ValueError(
            "Paper code configuration is missing."
        )

    if not paper_code_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "Paper code detection is disabled."
        )

    characters = (
        paper_code_config.get(
            "characters",
            [],
        )
    )

    if not characters:
        raise ValueError(
            "Paper code character coordinates are missing."
        )

    detected_characters = []

    character_details = []

    for position_index, character_config in enumerate(
        characters,
        start=1,
    ):

        x = int(
            character_config[
                "x"
            ]
        )

        start_y = int(
            character_config[
                "start_y"
            ]
        )

        gap = int(
            character_config[
                "gap"
            ]
        )

        values = (
            character_config[
                "values"
            ]
        )

        scores = {}

        for index, value in enumerate(
            values
        ):

            y = (
                start_y
                +
                index
                *
                gap
            )

            score = get_fill_ratio(
                gray_image,
                x,
                y,
                template,
            )

            scores[
                str(value)
            ] = score

        ranked = sorted(
            scores.items(),
            key=lambda item:
            item[1],
            reverse=True,
        )

        if not ranked:
            raise ValueError(
                f"Could not detect paper code position {position_index}."
            )

        best_value = (
            ranked[0][0]
        )

        best_score = float(
            ranked[0][1]
        )

        second_score = float(
            ranked[1][1]
            if len(ranked) > 1
            else 0
        )

        threshold = float(
            paper_code_config.get(
                "filled_threshold",
                template.get(
                    "filled_threshold",
                    0.50,
                ),
            )
        )

        confidence_gap_required = float(
            paper_code_config.get(
                "minimum_confidence_gap",
                0.05,
            )
        )

        if (
            best_score
            < threshold
        ):
            raise ValueError(
                "Question paper code could not be detected "
                f"at position {position_index}."
            )

        confidence_gap = (
            best_score
            - second_score
        )

        if (
            confidence_gap
            < confidence_gap_required
        ):
            readable_scores = ", ".join(
                f"{key}={float(value):.4f}"
                for key, value
                in scores.items()
            )

            raise ValueError(
                "Question paper code is ambiguous "
                f"at position {position_index}. "
                f"Best={best_value} "
                f"score={best_score:.4f}, "
                f"second={second_score:.4f}, "
                f"gap={confidence_gap:.4f}. "
                f"Scores: {readable_scores}"
            )

        detected_characters.append(
            best_value
        )

        character_details.append(
            {
                "position":
                    position_index,

                "value":
                    best_value,

                "score":
                    round(
                        best_score,
                        4,
                    ),

                "confidence_gap":
                    round(
                        confidence_gap,
                        4,
                    ),

                "scores": {
                    key: round(
                        float(value),
                        4,
                    )

                    for key, value
                    in scores.items()
                },
            }
        )

    return {
        "value":
            "".join(
                detected_characters
            ),

        "characters":
            character_details,
    }


# ============================================================
# JEE SERIES
# ============================================================

def detect_jee_series(
    gray_image,
    template,
):

    series_config = (
        template.get(
            "series"
        )
    )

    if not series_config:
        raise ValueError(
            "JEE series configuration is missing."
        )

    if not series_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "JEE series detection is disabled."
        )

    coordinates = (
        series_config.get(
            "coordinates",
            {},
        )
    )

    if not coordinates:
        raise ValueError(
            "JEE series coordinates are missing."
        )

    scores = {}

    for (
        series,
        position
    ) in coordinates.items():

        if (
            not isinstance(
                position,
                list,
            )
            or len(position) != 2
        ):
            raise ValueError(
                f"Invalid coordinate for JEE series {series}."
            )

        x, y = position

        scores[
            str(series)
        ] = get_fill_ratio(
            gray_image,
            int(x),
            int(y),
            template,
        )

    ranked = sorted(
        scores.items(),
        key=lambda item:
        item[1],
        reverse=True,
    )

    if not ranked:
        raise ValueError(
            "Unable to detect JEE series."
        )

    selected_series = (
        ranked[0][0]
    )

    selected_score = float(
        ranked[0][1]
    )

    second_score = float(
        ranked[1][1]
        if len(ranked) > 1
        else 0
    )

    threshold = float(
        series_config.get(
            "filled_threshold",
            template.get(
                "filled_threshold",
                0.50,
            ),
        )
    )

    minimum_confidence_gap = float(
        series_config.get(
            "minimum_confidence_gap",
            0.05,
        )
    )

    if (
        selected_score
        < threshold
    ):
        raise ValueError(
            "Unable to detect JEE series."
        )

    confidence_gap = (
        selected_score
        - second_score
    )

    if (
        confidence_gap
        < minimum_confidence_gap
    ):
        raise ValueError(
            "JEE series detection is ambiguous."
        )

    return {
        "value":
            selected_series,

        "score":
            round(
                selected_score,
                4,
            ),

        "confidence_gap":
            round(
                confidence_gap,
                4,
            ),

        "scores": {
            key: round(
                float(value),
                4,
            )

            for key, value
            in scores.items()
        },
    }


# ============================================================
# JEE MCQ SCANNER
# ============================================================

def scan_jee_mcq_sections(
    corrected_image,
    template,
):

    gray = normalize_grayscale(
        corrected_image
    )

    sections = (
        template.get(
            "mcq_sections",
            [],
        )
    )

    detected = {}

    for section in sections:

        start_question = int(
            section[
                "start_question"
            ]
        )

        total_questions = int(
            section[
                "total_questions"
            ]
        )

        start_y = int(
            section[
                "start_y"
            ]
        )

        row_gap = int(
            section[
                "row_gap"
            ]
        )

        options = (
            section.get(
                "options",
                [
                    "A",
                    "B",
                    "C",
                    "D",
                ],
            )
        )

        option_x = (
            section[
                "option_x"
            ]
        )

        for row_index in range(
            total_questions
        ):

            question_number = (
                start_question
                +
                row_index
            )

            y = (
                start_y
                +
                row_index
                *
                row_gap
            )

            coordinates = {}

            for option in options:

                coordinates[
                    option
                ] = (
                    int(
                        option_x[
                            option
                        ]
                    ),
                    int(y),
                )

            temporary_template = (
                template.copy()
            )

            temporary_template[
                "options"
            ] = options

            detected[
                question_number
            ] = detect_question_answer(
                gray,
                coordinates,
                temporary_template,
            )

    return detected


# ============================================================
# JEE NUMERICAL SCANNER
# ============================================================

def detect_numerical_value(
    gray_image,
    question_config,
    template,
):

    columns = (
        question_config.get(
            "columns",
            []
        )
    )

    if not columns:
        return {
            "answer": "BLANK",
            "columns": [],
        }

    detected_digits = []

    column_details = []

    for column_index, column in enumerate(
        columns,
        start=1,
    ):

        x = int(
            column[
                "x"
            ]
        )

        start_y = int(
            column[
                "start_y"
            ]
        )

        gap = int(
            column[
                "gap"
            ]
        )

        values = (
            column.get(
                "values",
                [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ],
            )
        )

        scores = {}

        for row_index, value in enumerate(
            values
        ):

            y = (
                start_y
                +
                row_index
                *
                gap
            )

            scores[
                str(value)
            ] = get_fill_ratio(
                gray_image,
                x,
                y,
                template,
            )

        ranked = sorted(
            scores.items(),
            key=lambda item:
            item[1],
            reverse=True,
        )

        if not ranked:
            detected_digits.append(
                ""
            )
            continue

        best_value = (
            ranked[0][0]
        )

        best_score = float(
            ranked[0][1]
        )

        second_score = float(
            ranked[1][1]
            if len(ranked) > 1
            else 0
        )

        blank_threshold = float(
            question_config.get(
                "blank_threshold",
                template.get(
                    "blank_threshold",
                    0.18,
                ),
            )
        )

        filled_threshold = float(
            question_config.get(
                "filled_threshold",
                template.get(
                    "filled_threshold",
                    0.50,
                ),
            )
        )

        if (
            best_score
            < blank_threshold
        ):

            detected_value = ""

        elif (
            best_score
            >= filled_threshold
        ):

            detected_value = (
                best_value
            )

        else:

            detected_value = (
                "?"
            )

        detected_digits.append(
            detected_value
        )

        column_details.append(
            {
                "column":
                    column_index,

                "value":
                    detected_value,

                "best_score":
                    round(
                        best_score,
                        4,
                    ),

                "confidence_gap":
                    round(
                        best_score
                        - second_score,
                        4,
                    ),

                "scores": {
                    key: round(
                        float(value),
                        4,
                    )

                    for key, value
                    in scores.items()
                },
            }
        )

    if all(
        value == ""
        for value in detected_digits
    ):

        answer = "BLANK"

    elif any(
        value == "?"
        for value in detected_digits
    ):

        answer = "UNCERTAIN"

    else:

        answer = "".join(
            detected_digits
        )

    return {
        "answer":
            answer,

        "columns":
            column_details,
    }


def scan_jee_numerical_sections(
    corrected_image,
    template,
):

    gray = normalize_grayscale(
        corrected_image
    )

    sections = (
        template.get(
            "numerical_sections",
            [],
        )
    )

    detected = {}

    for section in sections:

        questions = (
            section.get(
                "questions",
                []
            )
        )

        for question in questions:

            question_number = int(
                question[
                    "question"
                ]
            )

            detected[
                question_number
            ] = detect_numerical_value(
                gray,
                question,
                template,
            )

    return detected


def scan_jee_answers(
    corrected_image,
    template,
):

    mcq_answers = (
        scan_jee_mcq_sections(
            corrected_image,
            template,
        )
    )

    numerical_answers = (
        scan_jee_numerical_sections(
            corrected_image,
            template,
        )
    )

    return {
        "mcq":
            mcq_answers,

        "numerical":
            numerical_answers,
    }


# ============================================================
# PAPER CODE DEBUG IMAGE
# ============================================================

def draw_paper_code_debug(
    corrected_image,
    template,
):
    """
    Draw the exact paper-code sampling circles on the corrected
    1600 x 2200 image. Useful for coordinate calibration.
    """

    debug = corrected_image.copy()

    config = template.get(
        "paper_code",
        {},
    )

    characters = config.get(
        "characters",
        [],
    )

    for position_index, character in enumerate(
        characters,
        start=1,
    ):

        x = int(
            character["x"]
        )

        start_y = int(
            character["start_y"]
        )

        gap = int(
            character["gap"]
        )

        values = character[
            "values"
        ]

        for index, value in enumerate(
            values
        ):

            y = (
                start_y
                + index * gap
            )

            cv2.circle(
                debug,
                (
                    int(x),
                    int(y),
                ),
                14,
                (
                    0,
                    0,
                    255,
                ),
                2,
            )

            cv2.putText(
                debug,
                str(value),
                (
                    int(x) + 18,
                    int(y) + 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    255,
                    0,
                    0,
                ),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            debug,
            f"CODE POS {position_index}",
            (
                max(
                    0,
                    int(x) - 80,
                ),
                max(
                    25,
                    int(start_y) - 20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    return debug


# ============================================================
# DEBUG IMAGE
# ============================================================

def create_debug_image(
    corrected_image,
    answers,
    template,
):

    debug = (
        corrected_image.copy()
    )

    exam_name = (
        str(
            template.get(
                "exam_name",
                ""
            )
        )
        .strip()
        .upper()
    )

    if exam_name in [
        "NEET",
        "KCET",
    ]:

        if corrected_image.ndim == 3:
            calibration_gray = cv2.cvtColor(
                corrected_image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            calibration_gray = corrected_image.copy()

        column_offsets = (
            auto_calibrate_neet_columns(
                calibration_gray,
                template,
            )
        )

        coordinates = (
            generate_calibrated_bubble_coordinates(
                template,
                column_offsets,
            )
        )

        radius = int(
            template.get(
                "bubble_radius",
                10,
            )
        )

        for (
            question,
            options
        ) in coordinates.items():

            for (
                option,
                position
            ) in options.items():

                x, y = position

                cv2.circle(
                    debug,
                    (
                        int(x),
                        int(y),
                    ),
                    radius,
                    (
                        0,
                        255,
                        0,
                    ),
                    1,
                )

            first_option = (
                template[
                    "options"
                ][0]
            )

            x, y = options[
                first_option
            ]

            answer = (
                answers.get(
                    question,
                    {}
                )
                .get(
                    "answer",
                    "-"
                )
            )

            cv2.putText(
                debug,
                f"Q{question}:{answer}",
                (
                    max(
                        0,
                        int(x) - 60,
                    ),
                    int(y) + 30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (
                    0,
                    0,
                    255,
                ),
                1,
            )

    elif exam_name == "JEE":

        # JEE debug drawing can be expanded
        # once exact MCQ and numerical coordinates
        # are calibrated.

        cv2.putText(
            debug,
            "JEE TEMPLATE",
            (
                20,
                40,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (
                0,
                0,
                255,
            ),
            2,
        )

    return debug


# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_omr(
    image_path,
    template_path,
):

    # --------------------------------
    # Load template
    # --------------------------------

    template = load_template(
        template_path
    )

    # --------------------------------
    # Load original image
    # --------------------------------

    image = load_image(
        image_path
    )

    # --------------------------------
    # Quality validation
    # --------------------------------

    quality = (
        validate_image_quality(
            image
        )
    )

    # --------------------------------
    # Determine exam before alignment
    # --------------------------------

    template_exam_name = (
        str(
            template.get(
                "exam_name",
                ""
            )
        )
        .strip()
        .upper()
    )

    # ========================================================
    # NEET / KCET:
    # PREPROCESS TO THE SAME CANONICAL GEOMETRY AS THE CLEAN
    # REFERENCE SHEET BEFORE USING ANY JSON COORDINATES.
    # ========================================================

    if template_exam_name in [
        "NEET",
        "KCET",
    ]:

        project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        reference_path = os.path.join(
            project_dir,
            "references",
            "neet_reference.png",
        )

        local_debug_dir = (
            os.path.join(
                project_dir,
                "alignment_debug",
            )
            if not os.environ.get("VERCEL")
            else None
        )

        corrected, alignment_debug = (
            canonicalize_omr(
                image=image,
                reference_path=reference_path,
                output_size=(
                    int(template["sheet_width"]),
                    int(template["sheet_height"]),
                ),
                use_orb=True,
                use_ecc=True,
                ecc_minimum_score=0.75,
                debug_dir=local_debug_dir,
            )
        )

        expected_width = int(
            template["sheet_width"]
        )

        expected_height = int(
            template["sheet_height"]
        )

        actual_height, actual_width = (
            corrected.shape[:2]
        )

        if (
            actual_width != expected_width
            or actual_height != expected_height
        ):
            corrected = cv2.resize(
                corrected,
                (
                    expected_width,
                    expected_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        if corrected.shape[:2] != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "Canonical OMR normalization failed. "
                f"Expected {expected_width}x{expected_height}, "
                f"got {corrected.shape[1]}x{corrected.shape[0]}."
            )

        gray = normalize_grayscale(
            corrected
        )

        crop_debug = {
            "method":
                "canonical_reference_alignment",

            "document_quad":
                alignment_debug.get(
                    "document_quad"
                ),

            "orb_applied":
                alignment_debug.get(
                    "orb_applied",
                    False,
                ),

            "orb_inliers":
                alignment_debug.get(
                    "orb_inliers",
                    0,
                ),

            "ecc_applied":
                alignment_debug.get(
                    "ecc_applied",
                    False,
                ),

            "ecc_score":
                alignment_debug.get(
                    "ecc_score"
                ),
        }

        if not os.environ.get(
            "VERCEL"
        ):
            cv2.imwrite(
                "corrected_omr.jpg",
                corrected,
            )

    # ========================================================
    # JEE:
    # RETAIN EXISTING OUTER-PAGE CORRECTION
    # ========================================================

    else:

        corners = (
            detect_corner_markers(
                image,
                template,
            )
        )

        corrected = (
            perspective_transform(
                image,
                corners,
                template,
            )
        )

        gray = normalize_grayscale(
            corrected
        )

        crop_debug = {
            "method":
                "white_page_contour",

            "corners": [
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
                in corners
            ],
        }

        alignment_debug = {
            "crop":
                crop_debug,

            "output_size": {
                "width":
                    int(
                        template[
                            "sheet_width"
                        ]
                    ),

                "height":
                    int(
                        template[
                            "sheet_height"
                        ]
                    ),
            },
        }


    # --------------------------------
    # Grayscale is already prepared by canonical preprocessing.
    # --------------------------------

    # --------------------------------
    # Paper-code coordinate debug
    # Local only: never write into the Vercel deployment root.
    # --------------------------------

    paper_code_debug = (
        draw_paper_code_debug(
            corrected,
            template,
        )
    )

    if not os.environ.get(
        "VERCEL"
    ):
        cv2.imwrite(
            "paper_code_debug.jpg",
            paper_code_debug,
        )

    exam_name = (
        str(
            template.get(
                "exam_name",
                ""
            )
        )
        .strip()
        .upper()
    )

    paper_code = None
    jee_series = None

    answers = {}

    # ========================================================
    # NEET
    # ========================================================

    if exam_name == "NEET":

        ensure_ml_model_available()

        # Scan paper code first
        paper_code = (
            detect_paper_code(
                gray,
                template,
            )
        )

        # Then scan 180 answers
        answers = (
            scan_answers(
                corrected,
                template,
            )
        )

        if not os.environ.get(
            "VERCEL"
        ):
            answer_debug = (
                draw_answer_analysis(
                    corrected,
                    template,
                    answers,
                )
            )

            cv2.imwrite(
                "bubble_analysis_debug.jpg",
                answer_debug,
            )

    # ========================================================
    # KCET
    # ========================================================

    elif exam_name == "KCET":

        ensure_ml_model_available()

        # Scan paper code first
        paper_code = (
            detect_paper_code(
                gray,
                template,
            )
        )

        # Then scan 240 answers
        answers = (
            scan_answers(
                corrected,
                template,
            )
        )

    # ========================================================
    # JEE
    # ========================================================

    elif exam_name == "JEE":

        # Scan series first
        jee_series = (
            detect_jee_series(
                gray,
                template,
            )
        )

        # Then scan JEE MCQ + numerical sections
        answers = (
            scan_jee_answers(
                corrected,
                template,
            )
        )

    else:

        raise ValueError(
            f"Unsupported exam type: {exam_name}"
        )

    # --------------------------------
    # Threshold image
    # --------------------------------

    threshold = (
        create_threshold_image(
            corrected
        )
    )

    # --------------------------------
    # Debug image
    # --------------------------------

    debug = (
        create_debug_image(
            corrected,
            answers,
            template,
        )
    )

    # --------------------------------------------------------
    # FINAL OMR ANALYSIS DEBUG
    # --------------------------------------------------------
    # Local development only.
    #
    # debug_omr.jpg shows:
    # - the final corrected/canonical OMR
    # - calibrated runtime bubble locations
    # - detected answer beside each question
    #
    # The clean reference image is NOT blended into this image and is
    # NOT used for answer decisions. It is only used earlier for geometry.
    if not os.environ.get(
        "VERCEL"
    ):
        cv2.imwrite(
            "debug_omr.jpg",
            debug,
        )

        # Keep the more detailed colored analysis image as well.
        if exam_name in [
            "NEET",
            "KCET",
        ]:
            detailed_debug = (
                draw_answer_analysis(
                    corrected,
                    template,
                    answers,
                )
            )

            cv2.imwrite(
                "bubble_analysis_debug.jpg",
                detailed_debug,
            )

    # --------------------------------
    # Final output
    # --------------------------------

    return {
        "template":
            template,

        "exam_name":
            exam_name,

        "quality":
            quality,

        "crop_debug":
            crop_debug,

        "alignment_debug":
            alignment_debug,

        "paper_code":
            paper_code,

        "jee_series":
            jee_series,

        "answers":
            answers,

        "corrected":
            corrected,

        "threshold":
            threshold,

        "debug":
            debug,
    }