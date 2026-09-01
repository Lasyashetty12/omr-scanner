# scanner.py
from ml_omr.hybrid_reader import scan_answers_ml
from omr_preprocess import canonicalize_omr
from omr_preprocess.document_mode import prepare_omr_document_mode
from omr_preprocess.quality import assess_document_quality
from identity_reader import detect_identity_fields
import json
import logging
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("omr_scanner")

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

def safe_reference_name(name):
    """Limit template-selected reference assets to the references folder."""
    value = str(name or "").strip()
    basename = os.path.basename(value)
    if not basename or basename != value:
        raise ValueError("Invalid template reference_image.")
    return basename


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
            "series",
        ]

        for field in required:
            if field not in template:
                raise ValueError(
                    f"{exam_name} template missing field: {field}"
                )

    elif exam_name == "JEE":

        for field in (
            "series",
            "mcq_sections",
            "numerical_sections",
            "numerical_layout",
        ):
            if field not in template:
                raise ValueError(
                    f"JEE template requires '{field}'."
                )

    else:
        raise ValueError(
            f"Unsupported exam in template: {exam_name}"
        )

    return template


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_source, *, filename=None, mime_type=None, return_debug=False):
    """Decode an upload once into the BGR working image used by OMR.

    ``image_source`` may be the raw request bytes or a path, retaining path
    support for existing callers and command-line workflows.  Pillow owns
    EXIF handling here so neither browser preview behaviour nor OpenCV build
    options can decide the orientation reaching registration.
    """
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(image_source)
        source_name = filename or "uploaded_image"
    else:
        source_path = Path(image_source)
        raw_bytes = source_path.read_bytes()
        source_name = filename or source_path.name

    if not raw_bytes:
        raise ValueError("Unable to read uploaded image.")

    raw_decoded = None
    orientation = None
    orientation_applied = False
    decoder = "Pillow"
    try:
        from PIL import Image, ImageOps

        with Image.open(BytesIO(raw_bytes)) as pil_source:
            # Load before leaving the file object and record the physical
            # matrix before EXIF transpose for diagnostics.
            raw_decoded = np.asarray(pil_source.convert("RGB")).copy()
            exif = pil_source.getexif()
            orientation_value = exif.get(274)
            orientation = int(orientation_value) if orientation_value else None
            oriented = ImageOps.exif_transpose(pil_source).convert("RGB")
            oriented_rgb = np.asarray(oriented).copy()
        orientation_applied = orientation not in (None, 1)
        decoded_bgr = cv2.cvtColor(oriented_rgb, cv2.COLOR_RGB2BGR)
    except Exception as error:
        # A few valid images are unsupported by Pillow.  OpenCV is used only
        # as a byte decoder and told to ignore EXIF, avoiding version-specific
        # automatic orientation behaviour.  There is no EXIF to apply in this
        # fallback path.
        decoder = "OpenCV (Pillow fallback)"
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        decoded_bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), flags)
        if decoded_bgr is None:
            raise ValueError("Unable to decode uploaded image.") from error
        raw_decoded = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    original_height, original_width = raw_decoded.shape[:2]
    oriented_height, oriented_width = decoded_bgr.shape[:2]
    normalized = normalize_image_resolution(decoded_bgr)
    normalized_height, normalized_width = normalized.shape[:2]
    debug = {
        "source": "uploaded_bytes",
        "original_filename": os.path.basename(source_name),
        "mime_type": mime_type or "",
        "original_width": int(original_width),
        "original_height": int(original_height),
        "original_aspect_ratio": round(original_width / float(original_height), 6),
        "image_width": int(original_width),
        "image_height": int(original_height),
        "aspect_ratio": round(original_width / float(original_height), 6),
        "orientation": orientation,
        "orientation_normalized_width": int(oriented_width),
        "orientation_normalized_height": int(oriented_height),
        "orientation_normalized_aspect_ratio": round(
            oriented_width / float(oriented_height), 6
        ),
        "normalized_width": int(normalized_width),
        "normalized_height": int(normalized_height),
        "normalized_aspect_ratio": round(
            normalized_width / float(normalized_height), 6
        ),
        "preprocessed_width": int(normalized_width),
        "preprocessed_height": int(normalized_height),
        "preprocessed_aspect_ratio": round(
            normalized_width / float(normalized_height), 6
        ),
        "exif_orientation": orientation,
        "orientation_applied": orientation_applied,
        "decoder": decoder,
        "resolution_normalized": (oriented_width, oriented_height) != (normalized_width, normalized_height),
        "color_space": "BGR uint8",
    }
    return (normalized, debug, raw_decoded, decoded_bgr) if return_debug else normalized


# ============================================================
# RESOLUTION NORMALIZATION
# ============================================================
#
# Laptop webcams / older reference images are typically well under
# 2MP (e.g. ~1280x720). Mobile phone cameras routinely capture
# 8-12MP+ photos (e.g. 4000x3000) for the exact same physical sheet.
#
# Every blur/morphology step used for registration-marker and
# page-corner detection (crop_omr_by_corner_boxes,
# detect_page_corners_from_crop, detect_registration_markers_in_crop,
# find_marker_candidates, etc.) uses FIXED, absolute-pixel kernel
# sizes such as GaussianBlur(..., (5, 5)) or a (15, 15) morphological
# close. Those kernels behave very differently depending on the raw
# resolution of the photo: on a high-resolution mobile photo the same
# kernel is proportionally tiny, so noise isn't suppressed and gaps
# in the paper/marker mask aren't bridged the way they are on a
# lower-resolution laptop image. That is what makes the corner-box /
# page-corner detection pick different points for a mobile photo than
# for a laptop image of the same sheet, which shows up downstream as
# a wrong or inconsistent crop.
#
# Downscaling (never upscaling) every image to a common working
# resolution before detection removes resolution as a variable, so
# detection behaves the same regardless of source device. The final
# canonical sheet is still forced to template["sheet_width"] /
# ["sheet_height"] later in preprocess_to_canonical, so this does not
# reduce final output quality.
#
NORMALIZED_MAX_DIMENSION = 1600


def normalize_image_resolution(
    image,
    max_dimension=NORMALIZED_MAX_DIMENSION,
):
    height, width = image.shape[:2]
    longer_side = max(height, width)

    if longer_side <= max_dimension:
        return image

    scale = max_dimension / float(longer_side)

    resized = cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized


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

    # Pre-scan quality metrics are informational only.
    # Actual OMR validity is decided by page/registration-marker detection.
    # This avoids rejecting a valid white OMR sheet before corner detection.

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
            questions_per_column=int(
                template[
                    "questions_per_column"
                ]
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



def _analysis_center(
    analysis_debug,
    question_no,
    option_label,
    fallback_coordinates=None,
):
    """
    Return the exact center used by the reader.

    Priority:
      1. draw_center
      2. crop_center
      3. fitted runtime coordinates passed as fallback

    This guarantees debug circles use the same center as classification.
    """

    q_debug = {}

    if isinstance(
        analysis_debug,
        dict,
    ):
        q_debug = analysis_debug.get(
            question_no,
            analysis_debug.get(
                str(
                    question_no
                ),
                {},
            ),
        )

    option_debug = (
        q_debug.get(
            "options",
            {},
        ).get(
            option_label,
            {},
        )
        if isinstance(
            q_debug,
            dict,
        )
        else {}
    )

    center = (
        option_debug.get(
            "draw_center"
        )
        or option_debug.get(
            "crop_center"
        )
    )

    if (
        isinstance(
            center,
            (list, tuple),
        )
        and
        len(
            center
        )
        == 2
    ):
        return (
            int(
                round(
                    float(
                        center[
                            0
                        ]
                    )
                )
            ),
            int(
                round(
                    float(
                        center[
                            1
                        ]
                    )
                )
            ),
        )

    if (
        fallback_coordinates
        and
        question_no
        in fallback_coordinates
        and
        option_label
        in fallback_coordinates[
            question_no
        ]
    ):
        x, y = fallback_coordinates[
            question_no
        ][
            option_label
        ]

        return (
            int(
                round(
                    float(
                        x
                    )
                )
            ),
            int(
                round(
                    float(
                        y
                    )
                )
            ),
        )

    return (
        0,
        0,
    )


def draw_answer_analysis(
    corrected_image,
    template,
    answers,
    original_image=None,
    homography=None,
):
    """
    Clean debug overlay using the exact centers used by the reader.
    Projects canonical coordinates back onto original_image when provided.
    """

    inv_homography = None
    if original_image is not None and homography is not None:
        try:
            inv_homography = np.linalg.inv(homography)
            debug_image = original_image.copy()
            scale_ratio = max(original_image.shape[:2]) / 1600.0
        except Exception:
            inv_homography = None
            debug_image = corrected_image.copy()
            scale_ratio = 1.0
    else:
        debug_image = corrected_image.copy()
        scale_ratio = 1.0

    # Reduced bubble and pin point radius for subtle, compact indicators
    base_radius = max(3.0, float(template.get("bubble_radius", 11)) - 6.0)
    bubble_radius = max(
        3,
        int(round(base_radius * scale_ratio)),
    )
    dot_radius = max(1, int(round(1.0 * scale_ratio)))
    line_thick = 1
    ring_thick = max(1, int(round(1.2 * scale_ratio)))

    option_order = list(
        template.get(
            "options",
            ["A", "B", "C", "D"],
        )
    )

    total_questions = int(
        template.get(
            "total_questions",
            len(
                answers
            ),
        )
    )

    # Hard response-grid vertical limits from the exact JSON.
    y_positions = [
        float(
            value
        )
        for value
        in template.get(
            "question_y_positions",
            [],
        )
    ]

    if y_positions:
        response_y_min = min(
            y_positions
        ) - 10.0

        response_y_max = max(
            y_positions
        ) + 10.0
    else:
        response_y_min = 0.0
        response_y_max = float(
            corrected_image.shape[
                0
            ]
        )

    def center_for(
        answer_record,
        option_label,
    ):
        ml_details = (
            answer_record.get(
                "ml",
                {},
            )
            if isinstance(
                answer_record,
                dict,
            )
            else {}
        )

        option_details = (
            ml_details.get(
                "options",
                {},
            ).get(
                option_label,
                {},
            )
        )

        center = (
            option_details.get(
                "draw_center"
            )
            or
            option_details.get(
                "crop_center"
            )
            or
            option_details.get(
                "calibrated_center"
            )
        )

        if (
            not isinstance(
                center,
                (list, tuple),
            )
            or
            len(
                center
            ) != 2
        ):
            return None

        cx = int(
            round(
                float(
                    center[
                        0
                    ]
                )
            )
        )

        cy = int(
            round(
                float(
                    center[
                        1
                    ]
                )
            )
        )

        # Do not draw any accidental candidate below/above response grid.
        if (
            cy < response_y_min
            or cy > response_y_max
        ):
            return None

        def transform_pt(x_c, y_c):
            if inv_homography is None:
                return (x_c, y_c)
            pt = np.array([x_c, y_c, 1.0], dtype=np.float64)
            opt = inv_homography @ pt
            if opt[2] != 0:
                return (
                    int(round(opt[0] / opt[2])),
                    int(round(opt[1] / opt[2])),
                )
            return (int(round(opt[0])), int(round(opt[1])))

        return transform_pt(cx, cy)

    for question_no, answer_record in answers.items():

        try:
            qno = int(
                question_no
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        # Prevent any accidental Q181 / extra debug row.
        if (
            qno < 1
            or qno > total_questions
        ):
            continue

        if not isinstance(
            answer_record,
            dict,
        ):
            continue

        detected_answer = answer_record.get(
            "answer",
            "BLANK",
        )

        ml_details = answer_record.get(
            "ml",
            {},
        )

        status = str(
            answer_record.get(
                "ml_status",
                ml_details.get(
                    "status",
                    "",
                ),
            )
        ).lower()

        # Draw a small neutral ring and exact pin dot for all A/B/C/D.
        for option_label in option_order:
            center = center_for(
                answer_record,
                option_label,
            )

            if center is None:
                continue

            cx, cy = center

            # Exact detected center.
            cv2.circle(
                debug_image,
                center,
                dot_radius,
                (
                    0,
                    165,
                    255,
                ),
                -1,
                lineType=cv2.LINE_AA,
            )

            # Smaller neutral circle; no overlap.
            cv2.circle(
                debug_image,
                center,
                bubble_radius,
                (
                    180,
                    180,
                    180,
                ),
                line_thick,
                lineType=cv2.LINE_AA,
            )

        # MULTIPLE
        if (
            detected_answer == "MULTIPLE"
            or status == "multiple"
        ):
            multiple_options = ml_details.get(
                "multiple_options",
                [],
            )

            for option_label in multiple_options:
                center = center_for(
                    answer_record,
                    option_label,
                )

                if center is None:
                    continue

                cv2.circle(
                    debug_image,
                    center,
                    bubble_radius,
                    (
                        0,
                        0,
                        255,
                    ),
                    ring_thick,
                    lineType=cv2.LINE_AA,
                )

            continue

        # SINGLE ANSWER
        if detected_answer in option_order:
            center = center_for(
                answer_record,
                detected_answer,
            )

            if center is not None:
                cv2.circle(
                    debug_image,
                    center,
                    bubble_radius,
                    (
                        0,
                        255,
                        0,
                    ),
                    ring_thick,
                    lineType=cv2.LINE_AA,
                )

            continue

        # AMBIGUOUS
        if (
            detected_answer == "UNCERTAIN"
            or status == "ambiguous"
        ):
            best_option = ml_details.get(
                "best_option",
                option_order[
                    0
                ],
            )

            center = center_for(
                answer_record,
                best_option,
            )

            if center is not None:
                cv2.circle(
                    debug_image,
                    center,
                    bubble_radius,
                    (
                        0,
                        255,
                        255,
                    ),
                    ring_thick,
                    lineType=cv2.LINE_AA,
                )

    return debug_image



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
# EXAM SERIES (P/Q/R/S)
# ============================================================

def detect_exam_series(
    gray_image,
    template,
    exam_name=None,
):

    exam_label = str(
        exam_name or template.get("exam_name") or "OMR"
    ).strip().upper()

    series_config = (
        template.get(
            "series"
        )
    )

    if not series_config:
        raise ValueError(
            f"{exam_label} series configuration is missing."
        )

    if not series_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            f"{exam_label} series detection is disabled."
        )

    coordinates = (
        series_config.get(
            "coordinates",
            {},
        )
    )

    if not coordinates:
        raise ValueError(
            f"{exam_label} series coordinates are missing."
        )

    scores = {}
    sampling_centres = {}
    search_radius = int(
        series_config.get(
            "search_radius",
            max(3, min(8, round(float(template.get("bubble_radius", 11)) * 0.55))),
        )
    )
    search_step = max(1, int(series_config.get("search_step", 2)))
    offsets = sorted(set(range(-search_radius, search_radius + 1, search_step)) | {0})

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
                f"Invalid coordinate for {exam_label} series {series}."
            )

        x, y = position
        best_score = -1.0
        best_centre = (int(x), int(y))
        for dx in offsets:
            for dy in offsets:
                score = float(
                    get_fill_ratio(
                        gray_image,
                        int(x) + dx,
                        int(y) + dy,
                        template,
                    )
                )
                if score > best_score:
                    best_score = score
                    best_centre = (int(x) + dx, int(y) + dy)

        scores[str(series)] = best_score
        sampling_centres[str(series)] = list(best_centre)

    ranked = sorted(
        scores.items(),
        key=lambda item:
        item[1],
        reverse=True,
    )

    if not ranked:
        raise ValueError(
            f"Unable to detect {exam_label} series."
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

    confidence_gap = (
        selected_score
        - second_score
    )

    # A shaded/mobile capture can lower the absolute fill ratio while still
    # leaving one clearly dominant P/Q/R/S bubble. Accept that strong relative
    # signal, but never accept a weak or ambiguous row.
    relaxed_threshold = threshold * 0.78
    strong_gap = max(minimum_confidence_gap * 2.0, 0.08)
    if selected_score < threshold and not (
        selected_score >= relaxed_threshold
        and confidence_gap >= strong_gap
    ):
        raise ValueError(
            f"Unable to detect {exam_label} series."
        )

    if (
        confidence_gap
        < minimum_confidence_gap
    ):
        raise ValueError(
            f"{exam_label} series detection is ambiguous."
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

        "sampling_centres": sampling_centres,
    }


def detect_jee_series(gray_image, template):
    """Backward-compatible JEE-specific entry point used by existing callers."""
    return detect_exam_series(gray_image, template, exam_name="JEE")


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

        y_positions = section.get(
            "question_y_positions"
        )

        if y_positions is not None:
            if len(y_positions) != total_questions:
                raise ValueError(
                    "JEE MCQ question_y_positions must match "
                    "the section question count."
                )
        else:
            start_y = int(section["start_y"])
            row_gap = int(section["row_gap"])

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

            if y_positions is not None:
                y = int(y_positions[row_index])
            else:
                y = start_y + row_index * row_gap

            coordinates = {}

            # JEE MCQ bubbles can move a few pixels after camera/page
            # registration. Refine each configured bubble centre locally
            # before applying the normal fill thresholds.
            #
            # This is the production path that previously produced the
            # validated 75/75 JEE result. The search is deliberately small
            # enough that it cannot jump to the neighbouring option.
            search_radius = int(
                section.get(
                    "search_radius",
                    4,
                )
            )

            search_step = max(
                1,
                int(
                    section.get(
                        "search_step",
                        1,
                    )
                ),
            )

            for option in options:

                base_x = int(
                    option_x[
                        option
                    ]
                )

                base_y = int(y)

                best_x = base_x
                best_y = base_y
                best_score = -1.0

                for dx in range(
                    -search_radius,
                    search_radius + 1,
                    search_step,
                ):
                    for dy in range(
                        -search_radius,
                        search_radius + 1,
                        search_step,
                    ):

                        candidate_x = (
                            base_x + dx
                        )

                        candidate_y = (
                            base_y + dy
                        )

                        candidate_score = (
                            get_fill_ratio(
                                gray,
                                candidate_x,
                                candidate_y,
                                template,
                            )
                        )

                        if (
                            candidate_score
                            > best_score
                        ):
                            best_score = (
                                candidate_score
                            )

                            best_x = (
                                candidate_x
                            )

                            best_y = (
                                candidate_y
                            )

                coordinates[
                    option
                ] = (
                    best_x,
                    best_y,
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

        y_positions = column.get(
            "y_positions"
        )

        if y_positions is not None:
            if len(y_positions) != len(values):
                raise ValueError(
                    "JEE numerical digit y-position count does not "
                    "match the configured values."
                )
        else:
            start_y = int(column["start_y"])
            gap = int(column["gap"])

        scores = {}

        for row_index, value in enumerate(
            values
        ):

            if y_positions is not None:
                y = int(y_positions[row_index])
            else:
                y = start_y + row_index * gap

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

    decimal_details = []
    selected_decimal = None
    decimal_candidates = []

    for decimal in question_config.get("decimal_points", []):
        score = get_fill_ratio(
            gray_image,
            int(decimal["x"]),
            int(decimal["y"]),
            template,
        )
        detail = {
            "after_column": int(decimal["after_column"]),
            "score": round(float(score), 4),
        }
        decimal_details.append(detail)
        if score >= float(
            question_config.get(
                "filled_threshold",
                template.get("filled_threshold", 0.42),
            )
        ):
            decimal_candidates.append(detail)

    if len(decimal_candidates) == 1:
        selected_decimal = decimal_candidates[0]["after_column"]

    sign_detail = None
    negative = False
    sign_config = question_config.get("sign")
    if sign_config:
        sign_score = get_fill_ratio(
            gray_image,
            int(sign_config["x"]),
            int(sign_config["y"]),
            template,
        )
        negative = sign_score >= float(
            question_config.get(
                "filled_threshold",
                template.get("filled_threshold", 0.42),
            )
        )
        sign_detail = {
            "value": "-" if negative else "",
            "score": round(float(sign_score), 4),
        }

    if all(
        value == ""
        for value in detected_digits
    ):

        answer = "BLANK"

    elif any(
        value == "?"
        for value in detected_digits
    ) or len(decimal_candidates) > 1:

        answer = "UNCERTAIN"

    else:

        first = next(
            index for index, value in enumerate(detected_digits) if value
        )
        last = max(
            index for index, value in enumerate(detected_digits) if value
        )
        used_digits = detected_digits[first:last + 1]

        if any(value == "" for value in used_digits):
            answer = "UNCERTAIN"
        else:
            answer = "".join(used_digits)
            if selected_decimal is not None:
                insertion = selected_decimal - first
                if insertion <= 0 or insertion >= len(answer):
                    answer = "UNCERTAIN"
                else:
                    answer = answer[:insertion] + "." + answer[insertion:]
            if negative and answer != "UNCERTAIN":
                answer = "-" + answer

    return {
        "answer":
            answer,

        "columns":
            column_details,

        "decimal_points":
            decimal_details,

        "sign":
            sign_detail,
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

    layout = template.get(
        "numerical_layout",
        {},
    )

    for section in sections:

        questions = section.get("questions")

        if questions is None:
            start_question = int(section["start_question"])
            digit_offsets = [int(value) for value in layout["digit_offsets"]]
            decimal_offsets = [int(value) for value in layout["decimal_offsets"]]
            decimal_after = [int(value) for value in layout["decimal_after_columns"]]
            digit_values = [str(value) for value in layout["digit_values"]]
            digit_y_positions = [int(value) for value in section["digit_y_positions"]]

            questions = []
            for index, base_x in enumerate(section["question_x_positions"]):
                base_x = int(base_x)
                questions.append(
                    {
                        "question": start_question + index,
                        "columns": [
                            {
                                "x": base_x + offset,
                                "y_positions": digit_y_positions,
                                "values": digit_values,
                            }
                            for offset in digit_offsets
                        ],
                        "decimal_points": [
                            {
                                "x": base_x + offset,
                                "y": int(section["decimal_y"]),
                                "after_column": after_column,
                            }
                            for offset, after_column in zip(
                                decimal_offsets,
                                decimal_after,
                            )
                        ],
                        "sign": {
                            "x": base_x + int(layout.get("sign_offset", 0)),
                            "y": int(section["sign_y"]),
                        },
                    }
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


def draw_jee_answer_analysis(corrected_image, template, answers):
    # Draw JEE MCQ and numerical bubbles actually detected by the reader.
    debug = corrected_image.copy()
    gray = normalize_grayscale(corrected_image)

    answers = answers or {}
    mcq_answers = answers.get("mcq", {})
    numerical_answers = answers.get("numerical", {})

    radius = max(
        10,
        int(template.get("bubble_radius", 10)) + 4,
    )
    ring_thickness = 2
    options_default = ["A", "B", "C", "D"]

    def refine_center(
        base_x,
        base_y,
        search_radius=4,
        search_step=1,
    ):
        best = (
            int(base_x),
            int(base_y),
        )
        best_score = -1.0

        for dx in range(
            -search_radius,
            search_radius + 1,
            search_step,
        ):
            for dy in range(
                -search_radius,
                search_radius + 1,
                search_step,
            ):
                x = int(base_x) + dx
                y = int(base_y) + dy

                score = float(
                    get_fill_ratio(
                        gray,
                        x,
                        y,
                        template,
                    )
                )

                if score > best_score:
                    best_score = score
                    best = (x, y)

        return best

    for section in template.get(
        "mcq_sections",
        [],
    ):
        start_question = int(
            section["start_question"]
        )
        total_questions = int(
            section["total_questions"]
        )
        y_positions = section.get(
            "question_y_positions"
        )
        options = section.get(
            "options",
            options_default,
        )
        option_x = section["option_x"]

        search_radius = int(
            section.get(
                "search_radius",
                4,
            )
        )
        search_step = max(
            1,
            int(
                section.get(
                    "search_step",
                    1,
                )
            ),
        )

        multiple_threshold = float(
            section.get(
                "multiple_threshold",
                template.get(
                    "multiple_threshold",
                    template.get(
                        "filled_threshold",
                        0.50,
                    ),
                ),
            )
        )

        for row_index in range(
            total_questions
        ):
            question_number = (
                start_question
                + row_index
            )

            record = mcq_answers.get(
                question_number,
                mcq_answers.get(
                    str(question_number),
                    {},
                ),
            )

            if not isinstance(
                record,
                dict,
            ):
                continue

            if y_positions is not None:
                y = int(
                    y_positions[
                        row_index
                    ]
                )
            else:
                y = (
                    int(
                        section[
                            "start_y"
                        ]
                    )
                    + row_index
                    * int(
                        section[
                            "row_gap"
                        ]
                    )
                )

            detected = str(
                record.get(
                    "answer",
                    "BLANK",
                )
            ).upper()

            scores = (
                record.get(
                    "scores",
                    {},
                )
                or {}
            )

            if detected in options:
                selected_options = [
                    detected
                ]
                ring_color = (
                    0,
                    200,
                    0,
                )

            elif detected == "MULTIPLE":
                selected_options = [
                    option
                    for option
                    in options
                    if float(
                        scores.get(
                            option,
                            0.0,
                        )
                    )
                    >= multiple_threshold
                ]
                ring_color = (
                    0,
                    0,
                    255,
                )

            elif detected == "UNCERTAIN":
                selected_options = (
                    [
                        max(
                            options,
                            key=lambda option:
                            float(
                                scores.get(
                                    option,
                                    0.0,
                                )
                            ),
                        )
                    ]
                    if scores
                    else []
                )
                ring_color = (
                    0,
                    215,
                    255,
                )

            else:
                selected_options = []

            for option in selected_options:
                calibrated_centres = (
                    record.get(
                        "option_centres",
                        {},
                    )
                    if isinstance(
                        record,
                        dict,
                    )
                    else {}
                )

                calibrated_point = calibrated_centres.get(
                    option
                )

                if (
                    isinstance(
                        calibrated_point,
                        (list, tuple),
                    )
                    and len(calibrated_point) == 2
                ):
                    center = (
                        int(
                            round(
                                float(
                                    calibrated_point[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    calibrated_point[1]
                                )
                            )
                        ),
                    )
                else:
                    center = (
                        int(
                            option_x[
                                option
                            ]
                        ),
                        int(y),
                    )

                cv2.circle(
                    debug,
                    center,
                    radius,
                    ring_color,
                    ring_thickness,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    debug,
                    center,
                    2,
                    ring_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

    layout = template.get(
        "numerical_layout",
        {},
    )

    digit_offsets = [
        int(value)
        for value
        in layout.get(
            "digit_offsets",
            [],
        )
    ]

    decimal_offsets = [
        int(value)
        for value
        in layout.get(
            "decimal_offsets",
            [],
        )
    ]

    decimal_after_columns = [
        int(value)
        for value
        in layout.get(
            "decimal_after_columns",
            [],
        )
    ]

    for section in template.get(
        "numerical_sections",
        [],
    ):
        start_question = int(
            section[
                "start_question"
            ]
        )

        digit_y_positions = [
            int(value)
            for value
            in section[
                "digit_y_positions"
            ]
        ]

        decimal_y = int(
            section[
                "decimal_y"
            ]
        )

        sign_y = int(
            section[
                "sign_y"
            ]
        )

        threshold = float(
            template.get(
                "jee_numeric_special_threshold",
                template.get(
                    "filled_threshold",
                    0.50,
                ),
            )
        )

        for index, base_x_value in enumerate(
            section[
                "question_x_positions"
            ]
        ):
            question_number = (
                start_question
                + index
            )

            record = numerical_answers.get(
                question_number,
                numerical_answers.get(
                    str(question_number),
                    {},
                ),
            )

            if not isinstance(
                record,
                dict,
            ):
                continue

            base_x = int(
                base_x_value
            )

            detected_answer = str(
                record.get(
                    "answer",
                    "BLANK",
                )
            ).upper()

            ring_color = (
                (
                    0,
                    215,
                    255,
                )
                if detected_answer
                == "UNCERTAIN"
                else (
                    0,
                    200,
                    0,
                )
            )

            for detail in record.get(
                "columns",
                [],
            ):
                value = str(
                    detail.get(
                        "value",
                        "",
                    )
                )

                column_index = (
                    int(
                        detail.get(
                            "column",
                            0,
                        )
                    )
                    - 1
                )

                if (
                    not value.isdigit()
                    or column_index < 0
                    or column_index
                    >= len(
                        digit_offsets
                    )
                ):
                    continue

                digit = int(
                    value
                )

                if (
                    digit < 0
                    or digit
                    >= len(
                        digit_y_positions
                    )
                ):
                    continue

                detected_center = detail.get(
                    "center"
                )

                if (
                    isinstance(
                        detected_center,
                        (list, tuple),
                    )
                    and len(detected_center) == 2
                ):
                    center = (
                        int(
                            round(
                                float(
                                    detected_center[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    detected_center[1]
                                )
                            )
                        ),
                    )
                else:
                    center = (
                        base_x
                        + digit_offsets[
                            column_index
                        ],
                        digit_y_positions[
                            digit
                        ],
                    )

                cv2.circle(
                    debug,
                    center,
                    radius,
                    ring_color,
                    ring_thickness,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    debug,
                    center,
                    2,
                    ring_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

            for detail in record.get(
                "decimal_points",
                [],
            ):
                if float(
                    detail.get(
                        "score",
                        0.0,
                    )
                ) < threshold:
                    continue

                after_column = int(
                    detail.get(
                        "after_column",
                        -1,
                    )
                )

                if (
                    after_column
                    not in
                    decimal_after_columns
                ):
                    continue

                decimal_index = (
                    decimal_after_columns
                    .index(
                        after_column
                    )
                )

                if (
                    decimal_index
                    >= len(
                        decimal_offsets
                    )
                ):
                    continue

                detected_center = detail.get(
                    "center"
                )

                if (
                    isinstance(
                        detected_center,
                        (list, tuple),
                    )
                    and len(detected_center) == 2
                ):
                    center = (
                        int(
                            round(
                                float(
                                    detected_center[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    detected_center[1]
                                )
                            )
                        ),
                    )
                else:
                    center = (
                        base_x
                        + decimal_offsets[
                            decimal_index
                        ],
                        decimal_y,
                    )

                cv2.circle(
                    debug,
                    center,
                    radius,
                    ring_color,
                    ring_thickness,
                    lineType=cv2.LINE_AA,
                )

            sign_detail = (
                record.get(
                    "sign"
                )
                or {}
            )

            if (
                sign_detail.get(
                    "value"
                )
                == "-"
            ):
                detected_center = sign_detail.get(
                    "center"
                )

                if (
                    isinstance(
                        detected_center,
                        (list, tuple),
                    )
                    and len(detected_center) == 2
                ):
                    center = (
                        int(
                            round(
                                float(
                                    detected_center[0]
                                )
                            )
                        ),
                        int(
                            round(
                                float(
                                    detected_center[1]
                                )
                            )
                        ),
                    )
                else:
                    center = (
                        base_x
                        + int(
                            layout.get(
                                "sign_offset",
                                0,
                            )
                        ),
                        sign_y,
                    )

                cv2.circle(
                    debug,
                    center,
                    radius,
                    ring_color,
                    ring_thickness,
                    lineType=cv2.LINE_AA,
                )

    cv2.putText(
        debug,
        "JEE DETECTED BUBBLES",
        (
            20,
            40,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (
            0,
            0,
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
    original_image=None,
    homography=None,
):
    # The debug overlay is intentionally tied to the exact canonical image
    # used for recognition.  A browser/original-photo overlay is visually
    # useful but cannot be compared reliably across devices.
    debug = corrected_image.copy()

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
        return draw_answer_analysis(
            corrected_image,
            template,
            answers,
        )

    elif exam_name == "JEE":
        return draw_jee_answer_analysis(
            corrected_image,
            template,
            answers,
        )

    return debug


# ============================================================
# MAIN PROCESSOR
# ============================================================

def _last_row_geometry_debug(corrected_image, template):
    """Capture the final-row-to-bottom-edge margin without changing any bubble geometry."""
    question_y_positions = template.get("question_y_positions") or []
    if not question_y_positions:
        return {
            "last_row_y": None,
            "last_row_bubble_center_plus_radius": None,
            "full_image_height": int(corrected_image.shape[0]),
            "distance_from_last_row_to_bottom": None,
            "bottom_margin_pixels": None,
            "bottom_row_vertical_spacing_px": None,
        }

    last_row_y = float(question_y_positions[-1])
    bubble_radius = float(template.get("bubble_radius", 11.0))
    last_row_bottom = last_row_y + bubble_radius
    distance_to_bottom = float(corrected_image.shape[0] - last_row_bottom)
    row_spacing = 0.0
    if len(question_y_positions) > 1:
        row_spacing = float(question_y_positions[-1] - question_y_positions[-2])

    return {
        "last_row_y": int(round(last_row_y)),
        "last_row_bubble_center_plus_radius": int(round(last_row_bottom)),
        "full_image_height": int(corrected_image.shape[0]),
        "distance_from_last_row_to_bottom": round(distance_to_bottom, 2),
        "bottom_margin_pixels": round(distance_to_bottom, 2),
        "bottom_row_vertical_spacing_px": round(row_spacing, 2),
    }


def process_omr(
    image_path,
    template_path,
    *,
    input_filename=None,
    input_mime_type=None,
    diagnostic_dir=None,
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

    image, input_debug, raw_decoded_rgb, orientation_normalized = load_image(
        image_path,
        filename=input_filename,
        mime_type=input_mime_type,
        return_debug=True,
    )

    debug_input_enabled = bool(os.environ.get("OMR_DEBUG_INPUT"))
    if debug_input_enabled and diagnostic_dir is None and not os.environ.get("VERCEL"):
        diagnostic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alignment_debug", "input")
    if debug_input_enabled and diagnostic_dir:
        os.makedirs(diagnostic_dir, exist_ok=True)
        cv2.imwrite(os.path.join(diagnostic_dir, "01_raw_decoded.jpg"), cv2.cvtColor(raw_decoded_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(diagnostic_dir, "02_orientation_normalized.jpg"), orientation_normalized)
        cv2.imwrite(os.path.join(diagnostic_dir, "03_resolution_normalized.jpg"), image)
        # Registration uses the normalized full frame directly; it has no
        # independent crop operation before registration.
        cv2.imwrite(os.path.join(diagnostic_dir, "04_document_crop.jpg"), image)

    logger.info(
        "OMR input geometry | source=%s | decoded=%dx%d (%.6f) | "
        "orientation_normalized=%dx%d (%.6f) | registration_input=%dx%d (%.6f) | "
        "EXIF orientation=%s | orientation_applied=%s",
        input_debug["original_filename"],
        input_debug["original_width"],
        input_debug["original_height"],
        input_debug["original_aspect_ratio"],
        input_debug["orientation_normalized_width"],
        input_debug["orientation_normalized_height"],
        input_debug["orientation_normalized_aspect_ratio"],
        input_debug["normalized_width"],
        input_debug["normalized_height"],
        input_debug["normalized_aspect_ratio"],
        input_debug["exif_orientation"],
        input_debug["orientation_applied"],
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

    # Every generated sheet is registered against its own clean PDF render.
    # This keeps fixed bubble coordinates stable across mobile and laptop
    # captures and prevents one exam from inheriting another sheet's geometry.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    reference_name = template.get("reference_image")
    if not reference_name:
        raise ValueError(
            f"{template_exam_name} template is missing reference_image."
        )

    reference_path = os.path.join(
        project_dir,
        "references",
        safe_reference_name(reference_name),
    )
    local_debug_dir = (
        os.path.join(project_dir, "alignment_debug", template["template_name"])
        if not os.environ.get("VERCEL")
        else None
    )

    corrected, alignment_debug = canonicalize_omr(
        image=image,
        reference_path=reference_path,
        output_size=(
            int(template["sheet_width"]),
            int(template["sheet_height"]),
        ),
        # Geometry is established only from the complete page and its four
        # registration boxes. Content-feature refinements can introduce a
        # small mobile-dependent tilt/scale and are deliberately disabled.
        # Four registration boxes establish the canonical page geometry.
        # JEE answer grids are calibrated locally by the robust reader below,
        # so feature/ECC warps are deliberately disabled for every exam.
        use_orb=False,
        use_ecc=False,
        debug_dir=local_debug_dir,
    )
    alignment_debug["input"] = input_debug

    document_preview, recognition_image, document_mode_debug = (
        prepare_omr_document_mode(
            corrected,
            debug_dir=local_debug_dir,
        )
    )
    alignment_debug["document_mode"] = document_mode_debug

    expected_width = int(template["sheet_width"])
    expected_height = int(template["sheet_height"])
    if corrected.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "Canonical OMR normalization failed. "
            f"Expected {expected_width}x{expected_height}, "
            f"got {corrected.shape[1]}x{corrected.shape[0]}."
        )

    # Bubble/series decisions use the colour-neutral, shadow-normalized image.
    # It has exactly the same dimensions and coordinates as ``corrected``.
    gray = normalize_grayscale(recognition_image)
    crop_debug = {
        "method": "canonical_reference_alignment",
        "reference_image": reference_name,
        "document_quad": alignment_debug.get("document_quad"),
        "orb_applied": alignment_debug.get("orb_applied", False),
        "orb_inliers": alignment_debug.get("orb_inliers", 0),
        "ecc_applied": alignment_debug.get("ecc_applied", False),
        "ecc_score": alignment_debug.get("ecc_score"),
    }

    if not os.environ.get("VERCEL"):
        cv2.imwrite("corrected_omr.jpg", corrected)
        cv2.imwrite("document_mode_preview.jpg", document_preview)


    if debug_input_enabled and diagnostic_dir:
        cv2.imwrite(os.path.join(diagnostic_dir, "05_canonical_registered.jpg"), corrected)
        cv2.imwrite(os.path.join(diagnostic_dir, "06_existing_recognition_input.jpg"), corrected)

    if not os.environ.get("VERCEL"):
        cv2.imwrite(os.path.join(os.getcwd(), "preprocessed_full_omr.png"), corrected)

    last_row_debug = _last_row_geometry_debug(corrected, template)
    if debug_input_enabled and diagnostic_dir:
        with open(os.path.join(diagnostic_dir, "bottom_row_geometry.json"), "w", encoding="utf-8") as bottom_file:
            json.dump(last_row_debug, bottom_file, indent=2)

    # --------------------------------
    # Grayscale is already prepared by canonical preprocessing.
    # --------------------------------

    # --------------------------------
    # Informational document-quality assessment. This never changes the
    # canonical image, recognition thresholds, or recognition result.
    # --------------------------------
    quality[
        "document_quality"
    ] = assess_document_quality(
        original_bgr=image,
        canonical_bgr=corrected,
        alignment_debug=alignment_debug,
    )

    document_quality = quality[
        "document_quality"
    ]

    if not document_quality[
        "can_scan"
    ]:
        details = " ".join(
            document_quality[
                "warnings"
            ]
        )

        raise ValueError(
            "Image quality rejected before OMR recognition. "
            f"Sharpness: {document_quality['sharpness']:.2f}. "
            f"Brightness: {document_quality['brightness']:.2f}. "
            f"{details} Please retake the complete sheet again."
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
    exam_series = None
    jee_series = None

    # Read printed identity bubbles from the same canonical image used for
    # answer recognition. Identity is informative and must never block scoring.
    try:
        identity = detect_identity_fields(
            recognition_image,
            template,
        )
    except Exception as identity_error:
        identity = {
            "warning": str(identity_error),
        }

    answers = {}

    # ========================================================
    # NEET
    # ========================================================

    if exam_name == "NEET":

        ensure_ml_model_available()

        # The generated combined sheet selects P/Q/R/S horizontally.
        exam_series = (
            detect_exam_series(
                gray,
                template,
                exam_name="NEET",
            )
        )
        paper_code = exam_series

        # Scan every physical response row in the generated sheet.
        answers = (
            scan_answers(
                recognition_image,
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

        exam_series = (
            detect_exam_series(
                gray,
                template,
                exam_name="KCET",
            )
        )
        paper_code = exam_series

        # Scan every physical response row in the generated sheet.
        answers = (
            scan_answers(
                recognition_image,
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
        exam_series = jee_series

        # Then scan JEE MCQ + numerical sections.
        #
        # IMPORTANT:
        # Use the previously validated production reader. Its MCQ path
        # performs only a small local centre correction and its numerical
        # path uses the established JEE template coordinates.
        answers = (
            scan_jee_answers(
                recognition_image,
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
            recognition_image
        )
    )

    # --------------------------------
    # Debug image
    # --------------------------------

    homography = alignment_debug.get("homography")

    debug = (
        create_debug_image(
            corrected,
            answers,
            template,
            original_image=image,
            homography=homography,
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
                    original_image=image,
                    homography=homography,
                )
            )

            cv2.imwrite(
                "bubble_analysis_debug.jpg",
                detailed_debug,
            )

    if debug_input_enabled and diagnostic_dir:
        cv2.imwrite(os.path.join(diagnostic_dir, "07_bubble_debug.jpg"), debug)
        diagnostic_json = {
            "input": input_debug,
            "orientation": {
                "exif_orientation": input_debug["exif_orientation"],
                "orientation_applied": input_debug["orientation_applied"],
            },
            "resolution": {
                "width": input_debug["normalized_width"],
                "height": input_debug["normalized_height"],
                "aspect_ratio": input_debug["normalized_aspect_ratio"],
                "normalized": input_debug["resolution_normalized"],
            },
            "registration": {
                "document_bounds": alignment_debug.get("document_detection", {}).get("bounds"),
                "markers": alignment_debug.get("registration", {}).get("markers"),
                "orb_applied": alignment_debug.get("orb_applied", False),
                "orb_inliers": alignment_debug.get("orb_inliers", 0),
                "ecc_applied": alignment_debug.get("ecc_applied", False),
                "ecc_score": alignment_debug.get("ecc_score"),
            },
            "final_geometry": {
                "width": int(corrected.shape[1]),
                "height": int(corrected.shape[0]),
                "aspect_ratio": round(corrected.shape[1] / float(corrected.shape[0]), 6),
                "stage": "immediately_before_bubble_recognition",
            },
            "bottom_row_geometry": last_row_debug,
            "preprocessed_full_omr_path": os.path.join(os.getcwd(), "preprocessed_full_omr.png"),
        }
        with open(os.path.join(diagnostic_dir, "diagnostics.json"), "w", encoding="utf-8") as diagnostic_file:
            json.dump(diagnostic_json, diagnostic_file, indent=2, default=str)

    # --------------------------------
    # Diagnostic Logging
    # --------------------------------
    try:
        orig_h, orig_w = image.shape[:2]
        corr_h, corr_w = corrected.shape[:2]
        doc_quad = alignment_debug.get("document_quad", [])
        stages = alignment_debug.get("document_mode", {}).get("stages", [])

        if exam_name == "JEE":
            answer_rows = [
                *answers.get("mcq", {}).values(),
                *answers.get("numerical", {}).values(),
            ]
        else:
            answer_rows = list(answers.values())

        detected_values = [
            row.get("answer", "BLANK") if isinstance(row, dict) else row
            for row in answer_rows
        ]
        num_blank = sum(1 for value in detected_values if value in (None, "BLANK"))
        num_multiple = sum(1 for value in detected_values if value == "MULTIPLE")
        num_uncertain = sum(1 for value in detected_values if value == "UNCERTAIN")
        num_evaluated = len(answer_rows)

        source = input_debug.get("source", "uploaded_bytes")
        image_width = int(input_debug.get("image_width", orig_w))
        image_height = int(input_debug.get("image_height", orig_h))
        aspect_ratio = float(input_debug.get("aspect_ratio", image_width / float(image_height) if image_height else 0.0))
        orientation = input_debug.get("orientation")
        pre_w = int(input_debug.get("preprocessed_width", corr_w))
        pre_h = int(input_debug.get("preprocessed_height", corr_h))
        pre_ratio = float(input_debug.get("preprocessed_aspect_ratio", pre_w / float(pre_h) if pre_h else 0.0))

        logger.info(
            "Scan Diagnostic Log | source=%s | image=%dx%d (%.6f) | orientation=%s | "
            "preprocessed=%dx%d (%.6f) | registration_input=%dx%d (%.6f) | "
            "Before bubble recognition: %dx%d (%.6f) | Corners: %s | "
            "Preprocessing: %s | Exam: %s | Template: %s | Evaluated Bubbles: %d | "
            "Blank: %d | Multiple: %d | Uncertain: %d",
            source, image_width, image_height, aspect_ratio, orientation,
            pre_w, pre_h, pre_ratio, orig_w, orig_h, orig_w / float(orig_h),
            corr_w, corr_h, corr_w / float(corr_h), doc_quad, stages, exam_name,
            template.get("template_name"), num_evaluated, num_blank,
            num_multiple, num_uncertain
        )
    except Exception as log_err:
        logger.warning("Diagnostic logging error: %s", log_err)

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

        "input_debug": input_debug,

        "crop_debug":
            crop_debug,

        "alignment_debug":
            alignment_debug,

        "paper_code":
            paper_code,

        "series":
            exam_series,

        "jee_series":
            jee_series,

        "identity":
            identity,

        "answers":
            answers,

        "corrected":
            corrected,

        "threshold":
            threshold,

        "debug":
            debug,
    }


def compare_omr_inputs(laptop_image, mobile_image, template_path, *, diagnostic_dir=None):
    """Run two files through the frozen OMR pipeline and report first-order differences.

    This is deliberately a helper rather than a second recognition path.  It
    is useful in a shell or regression test with a known laptop/mobile pair.
    """
    laptop = process_omr(laptop_image, template_path)
    mobile = process_omr(mobile_image, template_path)

    def registration(result):
        debug = result.get("alignment_debug", {})
        return {
            "canonical_dimensions": debug.get("output_size"),
            "document_bounds": debug.get("document_detection", {}).get("bounds"),
            "orb_inliers": debug.get("orb_inliers", 0),
            "ecc_score": debug.get("ecc_score"),
        }

    laptop_answers = laptop.get("answers", {})
    mobile_answers = mobile.get("answers", {})
    answer_differences = {
        str(question): {"laptop": laptop_answers.get(question), "mobile": mobile_answers.get(question)}
        for question in sorted(set(laptop_answers) | set(mobile_answers))
        if laptop_answers.get(question) != mobile_answers.get(question)
    }
    comparison = {
        "laptop": {"input": laptop["input_debug"], "registration": registration(laptop)},
        "mobile": {"input": mobile["input_debug"], "registration": registration(mobile)},
        "answer_differences": answer_differences,
        "recognition_same": not answer_differences,
    }

    if diagnostic_dir is not None:
        output_dir = Path(diagnostic_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "laptop_preprocessed.png"), laptop["corrected"])
        cv2.imwrite(str(output_dir / "laptop_bubble_debug.png"), laptop["debug"])
        cv2.imwrite(str(output_dir / "mobile_preprocessed.png"), mobile["corrected"])
        cv2.imwrite(str(output_dir / "mobile_bubble_debug.png"), mobile["debug"])
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str),
            encoding="utf-8",
        )

    return comparison
