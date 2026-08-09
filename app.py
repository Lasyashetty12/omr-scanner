# app.py

import json
import os
import uuid

import cv2
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from config import (
    UPLOAD_DIR,
    RESULT_DIR,
    TEMPLATE_DIR,
)

from scanner import process_omr

from scorer import calculate_score


app = FastAPI(
    title="OMR Scanner API",
    version="1.0.0",
)
from config import STATIC_DIR

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        os.path.join(
            STATIC_DIR,
            "index.html"
        )
    )

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# SCAN
# ============================================================

@app.post("/scan")
async def scan_omr(
    image: UploadFile = File(...),

    template_name: str = Form(
        "sample_template.json"
    ),

    answer_key_json: str = Form(...),

    correct_marks: float = Form(
        4
    ),

    wrong_marks: float = Form(
        -1
    ),

    blank_marks: float = Form(
        0
    ),

    multiple_marks: float = Form(
        -1
    ),
):
    scan_id = str(
        uuid.uuid4()
    )

    # --------------------------------
    # Validate file extension
    # --------------------------------

    filename = (
        image.filename
        or "omr.jpg"
    )

    extension = (
        os.path.splitext(
            filename
        )[1]
        .lower()
    )

    allowed = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            ),
        )

    # --------------------------------
    # Answer key
    # --------------------------------

    try:
        answer_key = json.loads(
            answer_key_json
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=(
                "answer_key_json "
                "contains invalid JSON."
            ),
        )

    # --------------------------------
    # Template
    # --------------------------------

    # Prevent path traversal
    template_name = (
        os.path.basename(
            template_name
        )
    )

    template_path = (
        os.path.join(
            TEMPLATE_DIR,
            template_name,
        )
    )

    if not os.path.exists(
        template_path
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "OMR template "
                "not found."
            ),
        )

    # --------------------------------
    # Save upload
    # --------------------------------

    upload_filename = (
        f"{scan_id}{extension}"
    )

    upload_path = (
        os.path.join(
            UPLOAD_DIR,
            upload_filename,
        )
    )

    file_data = await image.read()

    with open(
        upload_path,
        "wb",
    ) as file:
        file.write(
            file_data
        )

    try:
        # --------------------------------
        # OMR processing
        # --------------------------------

        processing = (
            process_omr(
                upload_path,
                template_path,
            )
        )

        detected_answers = (
            processing[
                "answers"
            ]
        )

        # --------------------------------
        # Score
        # --------------------------------

        scoring = calculate_score(
            detected_answers,
            answer_key,

            correct_marks=
                correct_marks,

            wrong_marks=
                wrong_marks,

            blank_marks=
                blank_marks,

            multiple_marks=
                multiple_marks,
        )

        # --------------------------------
        # Save debug images
        # --------------------------------

        corrected_path = (
            os.path.join(
                RESULT_DIR,
                f"{scan_id}_corrected.jpg",
            )
        )

        threshold_path = (
            os.path.join(
                RESULT_DIR,
                f"{scan_id}_threshold.jpg",
            )
        )

        debug_path = (
            os.path.join(
                RESULT_DIR,
                f"{scan_id}_debug.jpg",
            )
        )

        cv2.imwrite(
            corrected_path,
            processing[
                "corrected"
            ],
        )

        cv2.imwrite(
            threshold_path,
            processing[
                "threshold"
            ],
        )

        cv2.imwrite(
            debug_path,
            processing[
                "debug"
            ],
        )

        # --------------------------------
        # API-safe answers
        # --------------------------------

        answer_output = {}

        for question, data in (
            detected_answers.items()
        ):
            answer_output[
                str(question)
            ] = data

        # --------------------------------
        # Result
        # --------------------------------

        result = {
            "scan_id":
                scan_id,

            "status":
                "success",

            "template":
                processing[
                    "template"
                ].get(
                    "template_name"
                ),

            "quality":
                processing[
                    "quality"
                ],

            "score":
                scoring[
                    "score"
                ],

            "correct":
                scoring[
                    "correct"
                ],

            "wrong":
                scoring[
                    "wrong"
                ],

            "blank":
                scoring[
                    "blank"
                ],

            "multiple":
                scoring[
                    "multiple"
                ],

            "answers":
                answer_output,

            "question_results":
                scoring[
                    "questions"
                ],
        }

        # --------------------------------
        # Save JSON
        # --------------------------------

        result_path = (
            os.path.join(
                RESULT_DIR,
                f"{scan_id}.json",
            )
        )

        with open(
            result_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
            )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )

    except Exception as error:
        print(
            "Internal error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "OMR processing failed."
            ),
        )


# ============================================================
# GET RESULT
# ============================================================

@app.get(
    "/result/{scan_id}"
)
def get_result(
    scan_id: str
):
    # UUID-like safety
    scan_id = (
        os.path.basename(
            scan_id
        )
    )

    result_path = (
        os.path.join(
            RESULT_DIR,
            f"{scan_id}.json",
        )
    )

    if not os.path.exists(
        result_path
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Result not found."
            ),
        )

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )