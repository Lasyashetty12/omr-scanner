import cv2

for filename in [
    "grid_detection_debug.jpg",
    "bubble_analysis_debug.jpg",
    "corrected_omr.jpg",
]:
    image = cv2.imread(
        filename
    )

    if image is None:
        print(
            f"{filename}: not generated"
        )
    else:
        print(
            f"{filename}: "
            f"{image.shape[1]}x{image.shape[0]}"
        )
