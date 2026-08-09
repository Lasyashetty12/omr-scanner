# scanner.py

import json

import cv2
import numpy as np

from config import (
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_CONTRAST,
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

    # Common fields required for all exams
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
        template.get(
            "exam_name",
            ""
        )
        .strip()
        .upper()
    )

    # NEET / KCET use normal MCQ layout
    if exam_name in [
        "NEET",
        "KCET",
    ]:

        required = [
            "total_questions",
            "questions_per_column",
            "options",
            "columns",
        ]

        for field in required:

            if field not in template:

                raise ValueError(
                    f"Template missing required field: {field}"
                )

    # JEE may use different sections
    elif exam_name == "JEE":

        if "series" not in template:

            raise ValueError(
                "JEE template requires series configuration."
            )

    return template


# ============================================================
# IMAGE
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
# QUALITY
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

    height, width = image.shape[:2]

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

        x, y, w, h = cv2.boundingRect(
            contour
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
                "area": area,

                "center": (
                    x + w / 2.0,
                    y + h / 2.0,
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


def detect_corner_markers(image):

    candidates = (
        find_marker_candidates(
            image
        )
    )

    return select_four_markers(
        image,
        candidates,
    )


# ============================================================
# PERSPECTIVE
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

    return cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
    )


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
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
# MCQ TEMPLATE COORDINATES
# ============================================================

def generate_bubble_coordinates(
    template
):

    total_questions = int(
        template[
            "total_questions"
        ]
    )

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    start_y = int(
        template.get(
            "question_start_y",
            0,
        )
    )

    row_gap = int(
        template.get(
            "question_row_gap",
            0,
        )
    )

    options = template[
        "options"
    ]

    columns = template[
        "columns"
    ]

    coordinates = {}

    for question in range(
        1,
        total_questions + 1,
    ):

        column_index = (
            question - 1
        ) // questions_per_column

        row_index = (
            question - 1
        ) % questions_per_column

        if (
            column_index
            >= len(columns)
        ):

            raise ValueError(
                "Template does not contain enough columns."
            )

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
                columns[
                    column_index
                ]
            ):

                raise ValueError(
                    f"Missing coordinate for option {option}"
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
                int(y),
            )

    return coordinates


# ============================================================
# BUBBLE DETECTION
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

    dark = np.logical_and(
        roi < dark_threshold,
        mask > 0,
    )

    dark_count = (
        np.count_nonzero(
            dark
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

    highest_option = (
        ranked[0][0]
    )

    highest_score = float(
        ranked[0][1]
    )

    second_score = float(
        ranked[1][1]
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

        answer = highest_option

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


def scan_answers(
    corrected_image,
    template,
):

    gray = normalize_grayscale(
        corrected_image
    )

    coordinates = (
        generate_bubble_coordinates(
            template
        )
    )

    answers = {}

    for (
        question,
        option_coordinates
    ) in coordinates.items():

        answers[
            question
        ] = detect_question_answer(
            gray,
            option_coordinates,
            template,
        )

    return answers


# ============================================================
# QUESTION PAPER CODE
# NEET + KCET
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

        return None

    if not paper_code_config.get(
        "enabled",
        False,
    ):

        return None

    detected_characters = []

    character_details = []

    for character_config in (
        paper_code_config[
            "characters"
        ]
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
                value
            ] = score

        ranked = sorted(
            scores.items(),
            key=lambda item:
            item[1],
            reverse=True,
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

        minimum_score = float(
            template.get(
                "filled_threshold",
                0.50,
            )
        )

        if (
            best_score
            < minimum_score
        ):

            raise ValueError(
                "Question paper code could not be detected."
            )

        detected_characters.append(
            best_value
        )

        character_details.append(
            {
                "value":
                    best_value,

                "score":
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

        return None

    if not series_config.get(
        "enabled",
        False,
    ):

        return None

    scores = {}

    for series, position in (
        series_config[
            "coordinates"
        ].items()
    ):

        x, y = position

        scores[
            series
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

    filled_threshold = float(
        template.get(
            "filled_threshold",
            0.50,
        )
    )

    if (
        selected_score
        < filled_threshold
    ):

        raise ValueError(
            "Unable to detect JEE series."
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
                selected_score
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


# ============================================================
# JEE MCQ
# Placeholder until exact JEE calibration is completed
# ============================================================

def scan_jee_answers(
    corrected_image,
    template,
):

    """
    JEE has MCQ + numerical answer sections.

    Exact question coordinates still need
    calibration from the original JEE OMR.

    For now we return empty sections safely
    instead of crashing.
    """

    result = {
        "mcq": {},
        "numerical": {},
    }

    return result


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
        template.get(
            "exam_name",
            ""
        )
        .strip()
        .upper()
    )

    # JEE uses a different layout.
    # Don't call normal MCQ coordinate generator.
    if exam_name == "JEE":

        return debug

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

        for option, (
            x,
            y
        ) in options.items():

            cv2.circle(
                debug,
                (x, y),
                radius,
                (0, 255, 0),
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

        answer = answers[
            question
        ]["answer"]

        cv2.putText(
            debug,
            f"Q{question}:{answer}",
            (
                max(
                    0,
                    x - 60,
                ),
                y + 30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 0, 255),
            1,
        )

    return debug


# ============================================================
# MAIN SCANNER
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
    # Load image
    # --------------------------------

    image = load_image(
        image_path
    )

    # --------------------------------
    # Validate image quality
    # --------------------------------

    quality = (
        validate_image_quality(
            image
        )
    )

    # --------------------------------
    # Detect four registration markers
    # --------------------------------

    corners = (
        detect_corner_markers(
            image
        )
    )

    # --------------------------------
    # Perspective correction
    # --------------------------------

    corrected = (
        perspective_transform(
            image,
            corners,
            template,
        )
    )

    # --------------------------------
    # Normalized grayscale
    # --------------------------------

    gray = normalize_grayscale(
        corrected
    )

    # --------------------------------
    # Exam
    # --------------------------------

    exam_name = (
        template.get(
            "exam_name",
            ""
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

        # Paper code MUST be scanned first
        paper_code = (
            detect_paper_code(
                gray,
                template,
            )
        )

        answers = scan_answers(
            corrected,
            template,
        )

    # ========================================================
    # KCET
    # ========================================================

    elif exam_name == "KCET":

        # Paper code MUST be scanned first
        paper_code = (
            detect_paper_code(
                gray,
                template,
            )
        )

        answers = scan_answers(
            corrected,
            template,
        )

    # ========================================================
    # JEE
    # ========================================================

    elif exam_name == "JEE":

        # Detect series first
        jee_series = (
            detect_jee_series(
                gray,
                template,
            )
        )

        # JEE MCQ + numerical scanning
        answers = scan_jee_answers(
            corrected,
            template,
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

    if exam_name in [
        "NEET",
        "KCET",
    ]:

        debug = create_debug_image(
            corrected,
            answers,
            template,
        )

    else:

        debug = (
            corrected.copy()
        )

    # --------------------------------
    # Return EVERYTHING
    # --------------------------------

    return {
        "template":
            template,

        "exam_name":
            exam_name,

        "quality":
            quality,

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