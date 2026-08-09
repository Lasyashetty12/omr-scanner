# config.py

import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates"
)


# ------------------------------------------------
# IMAGE QUALITY
# ------------------------------------------------

MIN_BLUR_SCORE = 80

MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 245

MIN_CONTRAST = 20


# ------------------------------------------------
# BUBBLE DETECTION DEFAULTS
# ------------------------------------------------

DEFAULT_DARK_PIXEL_THRESHOLD = 100

DEFAULT_BLANK_THRESHOLD = 0.18

DEFAULT_FILLED_THRESHOLD = 0.50

DEFAULT_MULTIPLE_THRESHOLD = 0.50


# Automatically create directories
os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    TEMPLATE_DIR,
    exist_ok=True
)