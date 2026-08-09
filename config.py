import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

ANSWER_KEY_DIR = os.path.join(
    BASE_DIR,
    "answer_keys"
)

# Vercel writable temporary directories
UPLOAD_DIR = "/tmp/uploads"
RESULT_DIR = "/tmp/results"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# IMAGE QUALITY SETTINGS
# ============================================================

MIN_BLUR_SCORE = 80

MIN_BRIGHTNESS = 60

MAX_BRIGHTNESS = 245

MIN_CONTRAST = 20