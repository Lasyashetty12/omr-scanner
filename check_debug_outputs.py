import cv2

files = [
    "corrected_omr.jpg",
    "debug_omr.jpg",
    "bubble_analysis_debug.jpg",
    "paper_code_debug.jpg",
    "column_calibration_debug.jpg",
]

for filename in files:
    image = cv2.imread(filename)

    if image is None:
        print(f"{filename}: not generated")
        continue

    print(
        f"{filename}: "
        f"{image.shape[1]}x{image.shape[0]}"
    )
