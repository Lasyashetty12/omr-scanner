# config.py

SHEET_WIDTH = 1200
SHEET_HEIGHT = 1700

TOTAL_QUESTIONS = 50

OPTIONS = ["A", "B", "C", "D"]

# -----------------------------
# MARKING SCHEME
# -----------------------------

CORRECT_MARKS = 4
WRONG_MARKS = -1
BLANK_MARKS = 0
MULTIPLE_MARKS = -1


# -----------------------------
# IMAGE QUALITY
# -----------------------------

MIN_BLUR_SCORE = 80

MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 245

MIN_CONTRAST = 20


# -----------------------------
# BUBBLE DETECTION
# -----------------------------

BUBBLE_RADIUS = 10

# Pixels darker than this grayscale value
# are considered ink/filled.
DARK_PIXEL_THRESHOLD = 100

# A completely empty bubble should normally
# be much lower than this.
BLANK_THRESHOLD = 0.18

# If a bubble reaches this value,
# consider it filled.
FILLED_THRESHOLD = 0.50

# If two bubbles exceed this,
# classify as MULTIPLE.
MULTIPLE_THRESHOLD = 0.50


# -----------------------------
# CURRENT SAMPLE OMR TEMPLATE
# -----------------------------

QUESTIONS_PER_COLUMN = 10

QUESTION_START_Y = 680
QUESTION_ROW_GAP = 90

COLUMN_OPTION_X = [
    {
        "A": 52,
        "B": 99,
        "C": 145,
        "D": 189,
    },

    {
        "A": 310,
        "B": 354,
        "C": 399,
        "D": 445,
    },

    {
        "A": 561,
        "B": 607,
        "C": 652,
        "D": 696,
    },

    {
        "A": 810,
        "B": 855,
        "C": 899,
        "D": 944,
    },

    {
        "A": 1060,
        "B": 1106,
        "C": 1151,
        "D": 1192,
    },
]