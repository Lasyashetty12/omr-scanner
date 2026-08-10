import os
import sys
import cv2

from scanner import (
    load_template,
    load_image,
    detect_corner_markers,
    perspective_transform,
    generate_bubble_coordinates,
)


def draw_mapping(
    image_path,
    template_path,
    output_path="bubble_mapping_debug.jpg",
):

    # ========================================================
    # LOAD TEMPLATE
    # ========================================================

    template = load_template(template_path)

    print()
    print("=======================================")
    print("LOADING OMR TEMPLATE")
    print("=======================================")
    print(f"Template: {template_path}")

    exam_name = (
        template.get("exam_name", "")
        .strip()
        .upper()
    )

    print(f"Exam: {exam_name}")

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = load_image(image_path)

    print(f"Input image size: {image.shape[1]} x {image.shape[0]}")

    # ========================================================
    # ALIGN SHEET
    # ========================================================

    corners = detect_corner_markers(image)

    corrected = perspective_transform(
        image,
        corners,
        template,
    )

    debug = corrected.copy()

    # ========================================================
    # GET ACTUAL CORRECTED IMAGE SIZE
    # ========================================================

    corrected_height, corrected_width = debug.shape[:2]

    print(
        f"Corrected image size: "
        f"{corrected_width} x {corrected_height}"
    )

    # ========================================================
    # TEMPLATE COORDINATE SYSTEM
    #
    # These values come ONLY from neet.json
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
    # SCALE TEMPLATE COORDINATES TO CORRECTED IMAGE
    #
    # If perspective_transform() already produces exactly
    # the template size, these values will both be 1.0.
    # ========================================================

    scale_x = corrected_width / template_width
    scale_y = corrected_height / template_height

    print(f"Coordinate scale X: {scale_x:.6f}")
    print(f"Coordinate scale Y: {scale_y:.6f}")

    # ========================================================
    # HELPER FUNCTION
    # ========================================================

    def scale_point(x, y):

        return (
            int(round(x * scale_x)),
            int(round(y * scale_y))
        )

    # ========================================================
    # ANSWER BUBBLES
    #
    # ALL POSITIONS COME FROM neet.json
    # ========================================================

    

    radius = int(
        template.get(
            "bubble_radius",
            10
        )
    )

    # Scale radius too
    radius_x = radius * scale_x
    radius_y = radius * scale_y

    scaled_radius = int(
        round(
            (radius_x + radius_y) / 2
        )
    )

    print()
    print("=======================================")
    print("DRAWING ANSWER BUBBLES")
    print("=======================================")

    for question, options in coordinates.items():

        for option, position in options.items():

            # Position from neet.json
            x, y = position

            # Convert template coordinates to
            # corrected-image coordinates
            x, y = scale_point(x, y)

            # ------------------------------------------------
            # Bubble circle
            # ------------------------------------------------

            cv2.circle(
                debug,
                (x, y),
                scaled_radius,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Center dot
            # ------------------------------------------------

            cv2.circle(
                debug,
                (x, y),
                2,
                (255, 0, 0),
                -1
            )

            # ------------------------------------------------
            # Option label
            # ------------------------------------------------

            cv2.putText(
                debug,
                option,
                (
                    x - 5,
                    y - 12
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 0, 0),
                1
            )

        # ====================================================
        # QUESTION NUMBER
        # ====================================================

        first_option = template["options"][0]

        first_x, first_y = options[first_option]

        first_x, first_y = scale_point(
            first_x,
            first_y
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
            (0, 0, 255),
            1
        )

    # ========================================================
    # PAPER CODE
    #
    # POSITIONS ALSO COME FROM neet.json
    # ========================================================

    paper_code = template.get(
        "paper_code"
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

        for character in paper_code.get(
            "characters",
            []
        ):

            # Position comes from JSON
            template_x = int(
                character["x"]
            )

            template_start_y = int(
                character["start_y"]
            )

            template_gap = int(
                character["gap"]
            )

            values = character["values"]

            for row_index, value in enumerate(values):

                template_y = (
                    template_start_y
                    +
                    row_index * template_gap
                )

                # Scale JSON coordinate
                x, y = scale_point(
                    template_x,
                    template_y
                )

                # Scale paper-code radius
                paper_radius = max(
                    5,
                    int(round(9 * ((scale_x + scale_y) / 2)))
                )

                # ------------------------------------------------
                # Paper code bubble
                # ------------------------------------------------

                cv2.circle(
                    debug,
                    (x, y),
                    paper_radius,
                    (255, 0, 255),
                    2
                )

                # ------------------------------------------------
                # Center
                # ------------------------------------------------

                cv2.circle(
                    debug,
                    (x, y),
                    2,
                    (0, 0, 255),
                    -1
                )

                # ------------------------------------------------
                # Value
                # ------------------------------------------------

                cv2.putText(
                    debug,
                    str(value),
                    (
                        x + 12,
                        y + 4
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (255, 0, 255),
                    1
                )

    # ========================================================
    # DRAW ALIGNMENT CORNERS
    # ========================================================

    height, width = debug.shape[:2]

    corner_points = [
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1),
    ]

    corner_labels = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for point, label in zip(
        corner_points,
        corner_labels
    ):

        x, y = point

        cv2.circle(
            debug,
            (x, y),
            15,
            (0, 255, 255),
            3
        )

        text_x = min(
            max(5, x),
            width - 50
        )

        text_y = min(
            max(25, y),
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
            (0, 255, 255),
            2
        )

    # ========================================================
    # SAVE
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
    print(f"Exam: {exam_name}")
    print(f"Template: {template_path}")
    print(f"Input: {image_path}")
    print(f"Output: {output_path}")
    print(
        f"Corrected size: "
        f"{width} x {height}"
    )
    print(
        f"Template size: "
        f"{template_width} x {template_height}"
    )
    print(
        f"Scale: X={scale_x:.6f}, "
        f"Y={scale_y:.6f}"
    )
    print("=======================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 3:

        print("Usage:")
        print(
            "python debug_mapping.py "
            "<image> <template>"
        )

        print()

        print("Example:")
        print(
            "python debug_mapping.py "
            "neet_test.png "
            "templates/neet.json"
        )

        sys.exit(1)

    input_image = sys.argv[1]

    input_template = sys.argv[2]

    draw_mapping(
        input_image,
        input_template,
        "bubble_mapping_debug.jpg",
    )