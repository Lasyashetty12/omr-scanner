# scanner.py

import cv2
import numpy as np

from config import (
    SHEET_WIDTH,
    SHEET_HEIGHT,
    TOTAL_QUESTIONS,
    OPTIONS,

    BUBBLE_RADIUS,
    DARK_PIXEL_THRESHOLD,
    BLANK_THRESHOLD,
    FILLED_THRESHOLD,
    MULTIPLE_THRESHOLD,

    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_CONTRAST,

    QUESTIONS_PER_COLUMN,
    QUESTION_START_Y,
    QUESTION_ROW_GAP,
    COLUMN_OPTION_X,
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


# ============================================================
# IMAGE QUALITY CHECKS
# ============================================================

def calculate_blur_score(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def calculate_brightness(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.mean(gray)
    )


def calculate_contrast(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
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

    print(
        f"Blur score: {blur:.2f}"
    )

    print(
        f"Brightness: {brightness:.2f}"
    )

    print(
        f"Contrast: {contrast:.2f}"
    )

    if blur < MIN_BLUR_SCORE:
        raise ValueError(
            "Image is too blurry. "
            "Please scan again."
        )

    if brightness < MIN_BRIGHTNESS:
        raise ValueError(
            "Image is too dark. "
            "Please improve lighting and scan again."
        )

    if brightness > MAX_BRIGHTNESS:
        raise ValueError(
            "Image is overexposed. "
            "Please reduce lighting and scan again."
        )

    if contrast < MIN_CONTRAST:
        raise ValueError(
            "Image contrast is too low. "
            "Please scan again."
        )

    return {
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
    }


# ============================================================
# CORNER / REGISTRATION MARKERS
# ============================================================

def order_points(points):
    points = np.array(
        points,
        dtype="float32"
    )

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    point_sum = points.sum(
        axis=1
    )

    point_diff = np.diff(
        points,
        axis=1
    ).reshape(-1)

    rect[0] = points[
        np.argmin(point_sum)
    ]

    rect[2] = points[
        np.argmax(point_sum)
    ]

    rect[1] = points[
        np.argmin(point_diff)
    ]

    rect[3] = points[
        np.argmax(point_diff)
    ]

    return rect


def find_marker_candidates(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_height, image_width = (
        image.shape[:2]
    )

    image_area = (
        image_height *
        image_width
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        # Ignore very small objects.
        if area < image_area * 0.00015:
            continue

        # Ignore huge objects.
        if area > image_area * 0.03:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h == 0:
            continue

        aspect_ratio = (
            w / float(h)
        )

        # Registration marker should be
        # approximately square.
        if not (
            0.70 <=
            aspect_ratio <=
            1.30
        ):
            continue

        rectangle_area = (
            w * h
        )

        if rectangle_area == 0:
            continue

        fill_ratio = (
            area /
            float(rectangle_area)
        )

        # Marker should be mostly solid.
        if fill_ratio < 0.60:
            continue

        center_x = (
            x + w / 2.0
        )

        center_y = (
            y + h / 2.0
        )

        candidates.append({
            "area": area,

            "center": (
                center_x,
                center_y
            ),

            "rect": (
                x,
                y,
                w,
                h
            )
        })

    return candidates


def select_four_corner_markers(
    image,
    candidates
):
    if len(candidates) < 4:
        raise ValueError(
            "All four OMR corner markers "
            "are not visible. Please scan again."
        )

    height, width = (
        image.shape[:2]
    )

    centers = np.array(
        [
            item["center"]
            for item in candidates
        ],
        dtype="float32"
    )

    top_left = min(
        centers,
        key=lambda p:
            p[0] + p[1]
    )

    top_right = min(
        centers,
        key=lambda p:
            (width - p[0]) +
            p[1]
    )

    bottom_right = min(
        centers,
        key=lambda p:
            (width - p[0]) +
            (height - p[1])
    )

    bottom_left = min(
        centers,
        key=lambda p:
            p[0] +
            (height - p[1])
    )

    corners = np.array(
        [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ],
        dtype="float32"
    )

    unique_points = np.unique(
        corners.astype(int),
        axis=0
    )

    if len(unique_points) != 4:
        raise ValueError(
            "Unable to determine four "
            "distinct OMR markers."
        )

    return corners


def detect_corner_markers(image):
    candidates = find_marker_candidates(
        image
    )

    return select_four_corner_markers(
        image,
        candidates
    )


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def perspective_transform(
    image,
    corners
):
    source = order_points(
        corners
    )

    destination = np.array(
        [
            [
                0,
                0
            ],

            [
                SHEET_WIDTH - 1,
                0
            ],

            [
                SHEET_WIDTH - 1,
                SHEET_HEIGHT - 1
            ],

            [
                0,
                SHEET_HEIGHT - 1
            ]
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(
        source,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            SHEET_WIDTH,
            SHEET_HEIGHT
        )
    )

    return corrected


# ============================================================
# LIGHTING NORMALIZATION
# ============================================================

def normalize_grayscale(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    normalized = clahe.apply(
        gray
    )

    normalized = cv2.GaussianBlur(
        normalized,
        (3, 3),
        0
    )

    return normalized


def preprocess_sheet(image):
    normalized = normalize_grayscale(
        image
    )

    threshold = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    return threshold


# ============================================================
# OMR TEMPLATE COORDINATES
# ============================================================

def generate_bubble_coordinates():
    coordinates = {}

    for question in range(
        1,
        TOTAL_QUESTIONS + 1
    ):

        column_index = (
            question - 1
        ) // QUESTIONS_PER_COLUMN

        row_index = (
            question - 1
        ) % QUESTIONS_PER_COLUMN

        if column_index >= len(
            COLUMN_OPTION_X
        ):
            raise ValueError(
                "OMR template configuration "
                "does not contain enough columns."
            )

        y = (
            QUESTION_START_Y +
            row_index *
            QUESTION_ROW_GAP
        )

        coordinates[
            question
        ] = {}

        for option in OPTIONS:
            x = COLUMN_OPTION_X[
                column_index
            ][option]

            coordinates[
                question
            ][option] = (
                int(x),
                int(y)
            )

    return coordinates


# ============================================================
# BUBBLE DARKNESS MEASUREMENT
# ============================================================

def get_circular_fill_ratio(
    gray_image,
    center_x,
    center_y,
    radius=BUBBLE_RADIUS
):
    height, width = (
        gray_image.shape[:2]
    )

    x1 = max(
        0,
        center_x - radius
    )

    y1 = max(
        0,
        center_y - radius
    )

    x2 = min(
        width,
        center_x + radius + 1
    )

    y2 = min(
        height,
        center_y + radius + 1
    )

    roi = gray_image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return 0.0

    mask = np.zeros(
        roi.shape,
        dtype=np.uint8
    )

    local_x = (
        center_x - x1
    )

    local_y = (
        center_y - y1
    )

    cv2.circle(
        mask,
        (
            local_x,
            local_y
        ),
        radius,
        255,
        -1
    )

    dark_pixels = np.logical_and(
        roi <
        DARK_PIXEL_THRESHOLD,

        mask >
        0
    )

    dark_count = np.count_nonzero(
        dark_pixels
    )

    total_count = np.count_nonzero(
        mask
    )

    if total_count == 0:
        return 0.0

    return (
        dark_count /
        float(total_count)
    )


# ============================================================
# ANSWER DETECTION
# ============================================================

def detect_question_answer(
    gray_image,
    option_coordinates
):
    scores = {}

    for option, (
        x,
        y
    ) in option_coordinates.items():

        scores[option] = (
            get_circular_fill_ratio(
                gray_image,
                x,
                y
            )
        )

    ranked = sorted(
        scores.items(),
        key=lambda item:
            item[1],
        reverse=True
    )

    highest_option = (
        ranked[0][0]
    )

    highest_score = (
        ranked[0][1]
    )

    second_highest_score = (
        ranked[1][1]
    )

    filled_options = [
        option

        for option, score
        in scores.items()

        if score >=
        MULTIPLE_THRESHOLD
    ]

    if len(filled_options) >= 2:
        answer = "MULTIPLE"

    elif len(filled_options) == 1:
        answer = (
            filled_options[0]
        )

    elif (
        highest_score <
        BLANK_THRESHOLD
    ):
        answer = "BLANK"

    else:
        # Borderline response.
        # Safer than automatically selecting.
        answer = "BLANK"

    # Difference between top two choices.
    confidence_gap = (
        highest_score -
        second_highest_score
    )

    return {
        "answer": answer,

        "scores": scores,

        "highest_score":
            highest_score,

        "confidence_gap":
            confidence_gap,
    }


# ============================================================
# SCAN ALL QUESTIONS
# ============================================================

def scan_answers(
    corrected_image
):
    gray = normalize_grayscale(
        corrected_image
    )

    threshold_image = (
        preprocess_sheet(
            corrected_image
        )
    )

    coordinates = (
        generate_bubble_coordinates()
    )

    answers = {}

    for (
        question_number,
        option_coordinates
    ) in coordinates.items():

        answers[
            question_number
        ] = detect_question_answer(
            gray,
            option_coordinates
        )

    return (
        answers,
        threshold_image
    )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def draw_marker_debug(
    image,
    corners
):
    debug = image.copy()

    labels = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for index, point in enumerate(
        corners
    ):
        x = int(
            point[0]
        )

        y = int(
            point[1]
        )

        cv2.circle(
            debug,
            (x, y),
            15,
            (0, 0, 255),
            3
        )

        cv2.putText(
            debug,
            labels[index],
            (
                x + 20,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return debug


def draw_debug_image(
    corrected_image,
    answers
):
    debug_image = (
        corrected_image.copy()
    )

    coordinates = (
        generate_bubble_coordinates()
    )

    for (
        question_number,
        option_coordinates
    ) in coordinates.items():

        result = answers[
            question_number
        ]

        detected_answer = (
            result["answer"]
        )

        for option, (
            x,
            y
        ) in option_coordinates.items():

            score = (
                result[
                    "scores"
                ][option]
            )

            # Green circle for each
            # expected bubble center.
            cv2.circle(
                debug_image,
                (x, y),
                BUBBLE_RADIUS,
                (0, 255, 0),
                1
            )

            cv2.putText(
                debug_image,
                f"{score:.2f}",
                (
                    x - 12,
                    y - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.23,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Print answer near first bubble.
        first_x, first_y = (
            option_coordinates[
                OPTIONS[0]
            ]
        )

        cv2.putText(
            debug_image,
            (
                f"Q{question_number}: "
                f"{detected_answer}"
            ),
            (
                max(
                    0,
                    first_x - 70
                ),
                first_y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return debug_image


# ============================================================
# MAIN OMR PROCESSING
# ============================================================

def process_omr(
    image_path
):
    # 1. Load image
    image = load_image(
        image_path
    )

    # 2. Validate original scan
    quality = (
        validate_image_quality(
            image
        )
    )

    # 3. Find registration markers
    corners = (
        detect_corner_markers(
            image
        )
    )

    marker_debug = (
        draw_marker_debug(
            image,
            corners
        )
    )

    # 4. Correct rotation,
    # perspective and scale.
    corrected_image = (
        perspective_transform(
            image,
            corners
        )
    )

    # 5. Read bubbles.
    (
        answers,
        threshold_image
    ) = scan_answers(
        corrected_image
    )

    # 6. Debug visualization.
    debug_image = (
        draw_debug_image(
            corrected_image,
            answers
        )
    )

    return {
        "original":
            image,

        "marker_debug":
            marker_debug,

        "corrected":
            corrected_image,

        "threshold":
            threshold_image,

        "debug":
            debug_image,

        "answers":
            answers,

        "quality":
            quality,
    }