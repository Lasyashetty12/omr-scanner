# debug_mapping.py

import sys
import cv2

from scanner import (
    load_template,
    load_image,
    detect_corner_markers,
    perspective_transform,
    generate_bubble_coordinates,
)


# ============================================================
# MAIN DEBUG FUNCTION
# ============================================================

def draw_mapping(
    image_path,
    template_path,
    output_path="bubble_mapping_debug.jpg",
):

    # ========================================================
    # LOAD TEMPLATE
    # ========================================================

    template = load_template(
        template_path
    )

    exam_name = (
        template.get(
            "exam_name",
            ""
        )
        .strip()
        .upper()
    )

    print()
    print("=======================================")
    print("LOADING OMR TEMPLATE")
    print("=======================================")
    print(f"Template: {template_path}")
    print(f"Exam: {exam_name}")

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = load_image(
        image_path
    )

    print(
        f"Input image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # ========================================================
    # ALIGN / PERSPECTIVE CORRECT
    # ========================================================

    corners = detect_corner_markers(
        image
    )

    corrected = perspective_transform(
        image,
        corners,
        template,
    )

    debug = corrected.copy()

    corrected_height, corrected_width = (
        debug.shape[:2]
    )

    print(
        f"Corrected image size: "
        f"{corrected_width} x {corrected_height}"
    )

    # ========================================================
    # TEMPLATE SIZE
    # ========================================================

    template_width = int(
        template.get(
            "sheet_width",
            corrected_width
        )
    )

    template_height = int(
        template.get(
            "sheet_height",
            corrected_height
        )
    )

    print(
        f"Template coordinate size: "
        f"{template_width} x {template_height}"
    )

    # ========================================================
    # SCALE
    # ========================================================

    scale_x = (
        corrected_width
        / float(template_width)
    )

    scale_y = (
        corrected_height
        / float(template_height)
    )

    print(
        f"Coordinate scale X: "
        f"{scale_x:.6f}"
    )

    print(
        f"Coordinate scale Y: "
        f"{scale_y:.6f}"
    )

    # ========================================================
    # HELPERS
    # ========================================================

    def scale_point(
        x,
        y
    ):

        return (
            int(
                round(
                    float(x)
                    * scale_x
                )
            ),

            int(
                round(
                    float(y)
                    * scale_y
                )
            ),
        )


    def scale_radius(
        radius
    ):

        radius_x = (
            float(radius)
            * scale_x
        )

        radius_y = (
            float(radius)
            * scale_y
        )

        return max(
            2,
            int(
                round(
                    (
                        radius_x
                        +
                        radius_y
                    )
                    / 2
                )
            )
        )


    # ========================================================
    # NEET / KCET
    # ========================================================

    if exam_name in [
        "NEET",
        "KCET",
    ]:

        print()
        print("=======================================")
        print("DRAWING MCQ BUBBLES")
        print("=======================================")

        coordinates = (
            generate_bubble_coordinates(
                template
            )
        )

        bubble_radius = (
            scale_radius(
                int(
                    template.get(
                        "bubble_radius",
                        10
                    )
                )
            )
        )

        options_list = (
            template[
                "options"
            ]
        )

        for (
            question,
            options
        ) in coordinates.items():

            for (
                option,
                position
            ) in options.items():

                template_x, template_y = (
                    position
                )

                x, y = scale_point(
                    template_x,
                    template_y
                )

                # Bubble circle
                cv2.circle(
                    debug,
                    (
                        x,
                        y
                    ),
                    bubble_radius,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )

                # Center dot
                cv2.circle(
                    debug,
                    (
                        x,
                        y
                    ),
                    2,
                    (
                        255,
                        0,
                        0
                    ),
                    -1
                )

                # Option label
                cv2.putText(
                    debug,
                    str(option),
                    (
                        x - 5,
                        y - 12
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (
                        255,
                        0,
                        0
                    ),
                    1
                )

            # Question number
            first_option = (
                options_list[0]
            )

            first_x, first_y = (
                options[
                    first_option
                ]
            )

            first_x, first_y = (
                scale_point(
                    first_x,
                    first_y
                )
            )

            cv2.putText(
                debug,
                str(question),
                (
                    max(
                        0,
                        first_x - 45
                    ),
                    first_y + 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (
                    0,
                    0,
                    255
                ),
                1
            )

        # ====================================================
        # PAPER CODE
        # ====================================================

        paper_code = (
            template.get(
                "paper_code"
            )
        )

        if (
            paper_code
            and paper_code.get(
                "enabled",
                False
            )
        ):

            print()
            print("=======================================")
            print("DRAWING PAPER CODE")
            print("=======================================")

            paper_radius = (
                scale_radius(
                    9
                )
            )

            for character in (
                paper_code.get(
                    "characters",
                    []
                )
            ):

                template_x = int(
                    character[
                        "x"
                    ]
                )

                template_start_y = int(
                    character[
                        "start_y"
                    ]
                )

                template_gap = int(
                    character[
                        "gap"
                    ]
                )

                values = (
                    character[
                        "values"
                    ]
                )

                for (
                    row_index,
                    value
                ) in enumerate(
                    values
                ):

                    template_y = (
                        template_start_y
                        +
                        row_index
                        *
                        template_gap
                    )

                    x, y = scale_point(
                        template_x,
                        template_y
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        paper_radius,
                        (
                            255,
                            0,
                            255
                        ),
                        2
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        2,
                        (
                            0,
                            0,
                            255
                        ),
                        -1
                    )

                    cv2.putText(
                        debug,
                        str(value),
                        (
                            x + 12,
                            y + 4
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (
                            255,
                            0,
                            255
                        ),
                        1
                    )

    # ========================================================
    # JEE
    # ========================================================

    elif exam_name == "JEE":

        print()
        print("=======================================")
        print("DRAWING JEE MCQ SECTIONS")
        print("=======================================")

        # ====================================================
        # JEE MCQ SETTINGS
        # ====================================================

        mcq_settings = (
            template.get(
                "mcq_settings",
                {}
            )
        )

        mcq_radius = (
            scale_radius(
                int(
                    mcq_settings.get(
                        "bubble_radius",
                        9
                    )
                )
            )
        )

        # ====================================================
        # JEE MCQ SECTIONS
        # ====================================================

        for section in (
            template.get(
                "mcq_sections",
                []
            )
        ):

            section_name = (
                section.get(
                    "name",
                    "MCQ"
                )
            )

            question_start = int(
                section[
                    "question_start"
                ]
            )

            question_end = int(
                section[
                    "question_end"
                ]
            )

            columns = (
                section[
                    "columns"
                ]
            )

            y_positions = (
                section[
                    "question_y_positions"
                ]
            )

            questions_per_column = (
                len(
                    y_positions
                )
            )

            total_questions = (
                question_end
                -
                question_start
                +
                1
            )

            print(
                f"{section_name}: "
                f"{question_start}-{question_end}"
            )

            for local_index in range(
                total_questions
            ):

                question_number = (
                    question_start
                    +
                    local_index
                )

                column_index = (
                    local_index
                    //
                    questions_per_column
                )

                row_index = (
                    local_index
                    %
                    questions_per_column
                )

                if (
                    column_index
                    >= len(columns)
                ):
                    print(
                        f"WARNING: "
                        f"Q{question_number} "
                        f"column missing"
                    )
                    continue

                if (
                    row_index
                    >= len(y_positions)
                ):
                    continue

                template_y = int(
                    y_positions[
                        row_index
                    ]
                )

                # --------------------------------------------
                # Draw options
                # --------------------------------------------

                for (
                    option,
                    template_x
                ) in (
                    columns[
                        column_index
                    ].items()
                ):

                    x, y = scale_point(
                        template_x,
                        template_y
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        mcq_radius,
                        (
                            0,
                            255,
                            0
                        ),
                        2
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        2,
                        (
                            255,
                            0,
                            0
                        ),
                        -1
                    )

                    cv2.putText(
                        debug,
                        str(option),
                        (
                            x - 5,
                            y - 12
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (
                            255,
                            0,
                            0
                        ),
                        1
                    )

                # --------------------------------------------
                # Question label
                # --------------------------------------------

                first_template_x = (
                    columns[
                        column_index
                    ]["A"]
                )

                first_x, first_y = (
                    scale_point(
                        first_template_x,
                        template_y
                    )
                )

                cv2.putText(
                    debug,
                    str(
                        question_number
                    ),
                    (
                        max(
                            0,
                            first_x - 45
                        ),
                        first_y + 4
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (
                        0,
                        0,
                        255
                    ),
                    1
                )

        # ====================================================
        # JEE NUMERICAL
        # ====================================================

        print()
        print("=======================================")
        print("DRAWING JEE NUMERICAL SECTIONS")
        print("=======================================")

        numerical_settings = (
            template.get(
                "numerical_settings",
                {}
            )
        )

        numerical_radius = (
            scale_radius(
                int(
                    numerical_settings.get(
                        "bubble_radius",
                        8
                    )
                )
            )
        )

        digit_values = (
            numerical_settings.get(
                "digit_values",
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
                ]
            )
        )

        for section in (
            template.get(
                "numerical_sections",
                []
            )
        ):

            section_name = (
                section.get(
                    "name",
                    "NUMERICAL"
                )
            )

            question_start = int(
                section[
                    "question_start"
                ]
            )

            question_end = int(
                section[
                    "question_end"
                ]
            )

            question_x = int(
                section[
                    "question_x"
                ]
            )

            question_y_positions = (
                section[
                    "question_y_positions"
                ]
            )

            digit_x_positions = (
                section[
                    "digit_x_positions"
                ]
            )

            digit_y_offset = int(
                section[
                    "digit_y_offset"
                ]
            )

            digit_row_gap = int(
                section[
                    "digit_row_gap"
                ]
            )

            total_questions = (
                question_end
                -
                question_start
                +
                1
            )

            print(
                f"{section_name}: "
                f"{question_start}-{question_end}"
            )

            for q_index in range(
                total_questions
            ):

                if (
                    q_index
                    >= len(
                        question_y_positions
                    )
                ):
                    continue

                question_number = (
                    question_start
                    +
                    q_index
                )

                template_base_y = int(
                    question_y_positions[
                        q_index
                    ]
                )

                question_label_x, question_label_y = (
                    scale_point(
                        question_x,
                        template_base_y
                    )
                )

                cv2.putText(
                    debug,
                    str(
                        question_number
                    ),
                    (
                        question_label_x,
                        question_label_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (
                        0,
                        0,
                        255
                    ),
                    1
                )

                # --------------------------------------------
                # Digit columns
                # --------------------------------------------

                for template_x in (
                    digit_x_positions
                ):

                    for (
                        digit_index,
                        digit
                    ) in enumerate(
                        digit_values
                    ):

                        template_y = (
                            template_base_y
                            +
                            digit_y_offset
                            +
                            digit_index
                            *
                            digit_row_gap
                        )

                        x, y = (
                            scale_point(
                                template_x,
                                template_y
                            )
                        )

                        cv2.circle(
                            debug,
                            (
                                x,
                                y
                            ),
                            numerical_radius,
                            (
                                255,
                                0,
                                255
                            ),
                            1
                        )

                        cv2.circle(
                            debug,
                            (
                                x,
                                y
                            ),
                            1,
                            (
                                0,
                                0,
                                255
                            ),
                            -1
                        )

                        cv2.putText(
                            debug,
                            str(
                                digit
                            ),
                            (
                                x + 9,
                                y + 3
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.22,
                            (
                                255,
                                0,
                                255
                            ),
                            1
                        )

        # ====================================================
        # JEE SERIES / 7-DIGIT CODE
        # ====================================================

        print()
        print("=======================================")
        print("DRAWING JEE SERIES CODE")
        print("=======================================")

        series = (
            template.get(
                "series",
                {}
            )
        )

        if series.get(
            "enabled",
            False
        ):

            x_positions = (
                series.get(
                    "x_positions",
                    []
                )
            )

            start_y = int(
                series.get(
                    "start_y",
                    0
                )
            )

            row_gap = int(
                series.get(
                    "row_gap",
                    0
                )
            )

            values = (
                series.get(
                    "values",
                    []
                )
            )

            series_radius = (
                scale_radius(
                    8
                )
            )

            for (
                column_index,
                template_x
            ) in enumerate(
                x_positions,
                start=1
            ):

                for (
                    value_index,
                    value
                ) in enumerate(
                    values
                ):

                    template_y = (
                        start_y
                        +
                        value_index
                        *
                        row_gap
                    )

                    x, y = scale_point(
                        template_x,
                        template_y
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        series_radius,
                        (
                            0,
                            255,
                            255
                        ),
                        2
                    )

                    cv2.circle(
                        debug,
                        (
                            x,
                            y
                        ),
                        1,
                        (
                            0,
                            0,
                            255
                        ),
                        -1
                    )

                    cv2.putText(
                        debug,
                        str(
                            value
                        ),
                        (
                            x + 9,
                            y + 3
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.22,
                        (
                            0,
                            255,
                            255
                        ),
                        1
                    )

                # Column number
                column_x, column_y = (
                    scale_point(
                        template_x,
                        start_y
                    )
                )

                cv2.putText(
                    debug,
                    f"C{column_index}",
                    (
                        column_x - 8,
                        max(
                            15,
                            column_y - 15
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (
                        0,
                        255,
                        255
                    ),
                    1
                )

    else:

        raise ValueError(
            f"Unsupported exam type: "
            f"{exam_name}"
        )

    # ========================================================
    # CORNER MARKERS
    # ========================================================

    height, width = (
        debug.shape[:2]
    )

    corner_points = [
        (
            0,
            0
        ),
        (
            width - 1,
            0
        ),
        (
            width - 1,
            height - 1
        ),
        (
            0,
            height - 1
        ),
    ]

    corner_labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for (
        point,
        label
    ) in zip(
        corner_points,
        corner_labels
    ):

        x, y = point

        cv2.circle(
            debug,
            (
                x,
                y
            ),
            15,
            (
                0,
                255,
                255
            ),
            3
        )

        text_x = min(
            max(
                5,
                x
            ),
            width - 50
        )

        text_y = min(
            max(
                25,
                y
            ),
            height - 10
        )

        cv2.putText(
            debug,
            label,
            (
                text_x,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                255,
                255
            ),
            2
        )

    # ========================================================
    # SAVE DEBUG IMAGE
    # ========================================================

    success = cv2.imwrite(
        output_path,
        debug
    )

    if not success:

        raise RuntimeError(
            "Could not save debug image."
        )

    print()
    print("=======================================")
    print("BUBBLE MAPPING DEBUG CREATED")
    print("=======================================")

    print(
        f"Exam: {exam_name}"
    )

    print(
        f"Template: {template_path}"
    )

    print(
        f"Input: {image_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Corrected size: "
        f"{width} x {height}"
    )

    print(
        f"Template size: "
        f"{template_width} x {template_height}"
    )

    print(
        f"Scale: "
        f"X={scale_x:.6f}, "
        f"Y={scale_y:.6f}"
    )

    print("=======================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(
        sys.argv
    ) < 3:

        print()
        print("Usage:")

        print(
            "python debug_mapping.py "
            "<image> <template>"
        )

        print()
        print("NEET example:")

        print(
            "python debug_mapping.py "
            "neet.png "
            "templates\\neet.json"
        )

        print()
        print("JEE example:")

        print(
            "python debug_mapping.py "
            "jee.png "
            "templates\\jee.json"
        )

        sys.exit(
            1
        )

    input_image = (
        sys.argv[1]
    )

    input_template = (
        sys.argv[2]
    )

    draw_mapping(
        input_image,
        input_template,
        "bubble_mapping_debug.jpg",
    )