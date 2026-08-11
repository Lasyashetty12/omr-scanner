# scanner.py
from ml_omr.hybrid_reader import scan_answers_ml
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
    Find the four large black registration marks on the OMR:
    top-left, top-right, bottom-right, bottom-left.
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
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]
    image_area = float(height * width)

    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < image_area * 0.00025:
            continue

        if area > image_area * 0.02:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)

        if not 0.55 <= aspect <= 1.55:
            continue

        rect_area = float(w * h)
        fill = area / rect_area if rect_area else 0.0

        if fill < 0.45:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        candidates.append(
            {
                "area": area,
                "box": (x, y, w, h),
                "center": (cx, cy),
            }
        )

    if not candidates:
        raise ValueError(
            "Could not find OMR corner boxes."
        )

    quadrants = {
        "tl": [],
        "tr": [],
        "br": [],
        "bl": [],
    }

    for candidate in candidates:
        cx, cy = candidate["center"]

        if cx < width * 0.45 and cy < height * 0.45:
            quadrants["tl"].append(candidate)

        if cx > width * 0.55 and cy < height * 0.45:
            quadrants["tr"].append(candidate)

        if cx > width * 0.55 and cy > height * 0.55:
            quadrants["br"].append(candidate)

        if cx < width * 0.45 and cy > height * 0.55:
            quadrants["bl"].append(candidate)

    selected = {}

    for name, items in quadrants.items():
        if not items:
            raise ValueError(
                f"Could not find OMR {name.upper()} corner box. "
                "Fit the full sheet inside the yellow A4 guide."
            )

        selected[name] = max(
            items,
            key=lambda item: item["area"],
        )

    return selected


def crop_omr_by_corner_boxes(
    image,
    padding_ratio=0.07,
):
    """
    Crop around the four large OMR registration marks first.
    This removes the table/laptop/background before perspective
    correction.
    """

    boxes = find_omr_corner_boxes(
        image
    )

    height, width = image.shape[:2]

    tl = boxes["tl"]["box"]
    tr = boxes["tr"]["box"]
    br = boxes["br"]["box"]
    bl = boxes["bl"]["box"]

    left = min(
        tl[0],
        bl[0],
    )

    right = max(
        tr[0] + tr[2],
        br[0] + br[2],
    )

    top = min(
        tl[1],
        tr[1],
    )

    bottom = max(
        bl[1] + bl[3],
        br[1] + br[3],
    )

    span_w = max(
        1,
        right - left,
    )

    span_h = max(
        1,
        bottom - top,
    )

    pad_x = int(
        span_w * padding_ratio
    )

    pad_y = int(
        span_h * padding_ratio
    )

    x1 = max(
        0,
        left - pad_x,
    )

    y1 = max(
        0,
        top - pad_y,
    )

    x2 = min(
        width,
        right + pad_x,
    )

    y2 = min(
        height,
        bottom + pad_y,
    )

    if (
        x2 - x1 < width * 0.35
        or y2 - y1 < height * 0.35
    ):
        raise ValueError(
            "Detected OMR corner boxes produced an invalid crop."
        )

    cropped = image[
        y1:y2,
        x1:x2
    ].copy()

    return cropped, {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "corner_boxes": boxes,
    }


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

def detect_omr_registration_corners(image):
    """
    Detect NEET/KCET corner registration marks.

    If one corner — especially BL — cannot be isolated because it
    touches a printed border, estimate it from the other 3 corners.
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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = float(
        width * height
    )

    candidates = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        # Ignore normal bubbles/text.
        if area < image_area * 0.00015:
            continue

        # Ignore huge regions/background.
        if area > image_area * 0.03:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if (
            w < 8
            or h < 8
        ):
            continue

        aspect = (
            w / float(h)
        )

        # Fairly tolerant because a corner may merge
        # slightly with sheet lines.
        if not (
            0.35
            <= aspect
            <= 2.5
        ):
            continue

        rect_area = float(
            w * h
        )

        fill_ratio = (
            area / rect_area
            if rect_area > 0
            else 0
        )

        if fill_ratio < 0.35:
            continue

        cx = (
            x + w / 2.0
        )

        cy = (
            y + h / 2.0
        )

        candidates.append(
            {
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

                "area":
                    area,

                "fill":
                    fill_ratio,
            }
        )

    if len(candidates) < 3:

        raise ValueError(
            "Could not detect enough OMR registration marks. "
            "Keep the full sheet visible."
        )


    # ========================================================
    # CORNER TARGET LOCATIONS
    # ========================================================

    targets = {

        "tl": np.array(
            [
                width * 0.10,
                height * 0.08,
            ],
            dtype=np.float32,
        ),

        "tr": np.array(
            [
                width * 0.90,
                height * 0.08,
            ],
            dtype=np.float32,
        ),

        "br": np.array(
            [
                width * 0.92,
                height * 0.88,
            ],
            dtype=np.float32,
        ),

        "bl": np.array(
            [
                width * 0.08,
                height * 0.88,
            ],
            dtype=np.float32,
        ),
    }


    selected = {}


    # ========================================================
    # FIND BEST CANDIDATE FOR EACH CORNER
    # ========================================================

    for name, target in targets.items():

        ranked = []

        for candidate in candidates:

            cx, cy = (
                candidate[
                    "center"
                ]
            )

            # Broad region limits prevent selecting
            # interior filled answer bubbles.
            if name == "tl":

                if not (
                    cx < width * 0.45
                    and
                    cy < height * 0.35
                ):
                    continue


            elif name == "tr":

                if not (
                    cx > width * 0.55
                    and
                    cy < height * 0.35
                ):
                    continue


            elif name == "br":

                if not (
                    cx > width * 0.55
                    and
                    cy > height * 0.65
                ):
                    continue


            elif name == "bl":

                if not (
                    cx < width * 0.45
                    and
                    cy > height * 0.65
                ):
                    continue


            point = np.array(
                [
                    cx,
                    cy,
                ],
                dtype=np.float32,
            )

            distance = float(
                np.linalg.norm(
                    point - target
                )
            )

            # Prefer bigger, darker marks slightly.
            score = (
                distance
                - candidate["area"] * 0.005
                - candidate["fill"] * 10
            )

            ranked.append(
                (
                    score,
                    candidate,
                )
            )


        if ranked:

            ranked.sort(
                key=lambda item:
                item[0]
            )

            selected[name] = (
                ranked[0][1]
            )


    # ========================================================
    # MUST HAVE TL + TR + BR
    #
    # These are normally clean on your sheet.
    # ========================================================

    required = [
        "tl",
        "tr",
        "br",
    ]


    for name in required:

        if name not in selected:

            raise ValueError(
                f"Could not identify OMR "
                f"{name.upper()} registration mark."
            )


    tl = np.array(
        selected["tl"]["center"],
        dtype=np.float32,
    )

    tr = np.array(
        selected["tr"]["center"],
        dtype=np.float32,
    )

    br = np.array(
        selected["br"]["center"],
        dtype=np.float32,
    )


    # ========================================================
    # BOTTOM LEFT
    # ========================================================

    if "bl" in selected:

        bl_detected = np.array(
            selected["bl"]["center"],
            dtype=np.float32,
        )


        # Also calculate geometric prediction.
        predicted_bl = (
            tl
            + (
                br - tr
            )
        )


        # If detected BL is wildly different from where
        # geometry says it should be, don't trust the contour.
        difference = float(
            np.linalg.norm(
                bl_detected
                - predicted_bl
            )
        )


        page_diagonal = float(
            np.hypot(
                width,
                height,
            )
        )


        if (
            difference
            > page_diagonal * 0.08
        ):

            bl = predicted_bl

        else:

            bl = bl_detected


    else:

        # ----------------------------------------------------
        # BL wasn't detected.
        #
        # Infer it from the other three:
        #
        # BL ≈ TL + (BR - TR)
        # ----------------------------------------------------

        bl = (
            tl
            + (
                br - tr
            )
        )


    # Keep prediction inside image.
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


    points = np.array(
        [
            tl,
            tr,
            br,
            bl,
        ],
        dtype=np.float32,
    )


    # ========================================================
    # SANITY CHECK
    # ========================================================

    quad_area = abs(
        float(
            cv2.contourArea(
                points.reshape(
                    (-1, 1, 2)
                )
            )
        )
    )


    if (
        quad_area
        < image_area * 0.30
    ):

        raise ValueError(
            "Registration marks do not form a valid OMR area."
        )


    return points


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

    The return shape remains compatible with the existing
    scorer/debug code.
    """

    if corrected_image.ndim == 3:
        gray = cv2.cvtColor(
            corrected_image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = corrected_image.copy()

    coordinates = generate_bubble_coordinates(
        template
    )

    raw_answers, ml_debug = scan_answers_ml(
        gray=gray,
        coordinates=coordinates,
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

    answers = {}

    for question in coordinates:

        detected = raw_answers.get(
            question
        )

        details = ml_debug.get(
            question,
            {}
        )

        status = details.get(
            "status",
            "blank",
        )

        if detected == "MULTIPLE":
            final_answer = "MULTIPLE"

        elif detected in template["options"]:
            final_answer = detected

        elif status == "ambiguous":
            final_answer = "UNCERTAIN"

        else:
            final_answer = "BLANK"

        answers[question] = {
            "answer": final_answer,
            "ml_status": status,
            "ml": details,
        }

    return answers


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

        coordinates = (
            generate_bubble_coordinates(
                template
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
    # Determine exam from the loaded template before alignment.
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
    # DIRECTLY USE THE FOUR LARGE BLACK REGISTRATION MARKS
    # ========================================================

    if template_exam_name in [
        "NEET",
        "KCET",
    ]:

        corners = (
            detect_omr_registration_corners(
                image
            )
        )

        corrected = (
            perspective_transform(
                image,
                corners,
                template,
            )
        )

        crop_debug = {
            "method":
                "direct_registration_markers",

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

    # ========================================================
    # JEE:
    # KEEP EXISTING WHITE-PAPER DETECTION
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

    # Local alignment debug.
    if not os.environ.get(
        "VERCEL"
    ):
        cv2.imwrite(
            "corrected_omr.jpg",
            corrected,
        )

    # --------------------------------
    # Prepare grayscale
    # --------------------------------

    gray = normalize_grayscale(
        corrected
    )

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