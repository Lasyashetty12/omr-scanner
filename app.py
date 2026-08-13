# app.py

import json
import os
import uuid

import cv2

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    BASE_DIR,
    UPLOAD_DIR,
    RESULT_DIR,
    TEMPLATE_DIR,
    STATIC_DIR,
)

from scanner import process_omr

from scorer import (
    calculate_score,
    calculate_jee_score,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="OMR Scanner API",
    version="2.1.0",
)


# ============================================================
# PATHS
# ============================================================

ANSWER_KEY_DIR = os.path.join(
    BASE_DIR,
    "answer_keys",
)


# ============================================================
# STATIC FILES
# ============================================================

if os.path.exists(STATIC_DIR):

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static",
    )


# ============================================================
# HELPERS
# ============================================================

def safe_filename(name):

    return os.path.basename(
        str(name)
    )


def save_json(
    path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def load_answer_key_for_exam(
    exam_name,
    identifier,
):

    exam_name = (
        str(exam_name)
        .strip()
        .lower()
    )

    identifier = (
        safe_filename(
            str(identifier)
            .strip()
            .upper()
        )
    )

    if not identifier:

        raise ValueError(
            "Answer key identifier is empty."
        )

    answer_key_path = os.path.join(
        ANSWER_KEY_DIR,
        exam_name,
        f"{identifier}.json",
    )

    if not os.path.exists(
        answer_key_path
    ):

        raise ValueError(
            f"No answer key found for "
            f"{exam_name.upper()} "
            f"paper/series {identifier}."
        )

    try:

        with open(
            answer_key_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except json.JSONDecodeError:

        raise ValueError(
            f"Invalid answer key JSON: "
            f"{answer_key_path}"
        )

    return data


def save_debug_images(
    scan_id,
    processing,
):

    corrected = processing.get(
        "corrected"
    )

    threshold = processing.get(
        "threshold"
    )

    debug = processing.get(
        "debug"
    )

    if corrected is not None:

        corrected_path = os.path.join(
            RESULT_DIR,
            f"{scan_id}_corrected.jpg",
        )

        cv2.imwrite(
            corrected_path,
            corrected,
        )

    if threshold is not None:

        threshold_path = os.path.join(
            RESULT_DIR,
            f"{scan_id}_threshold.jpg",
        )

        cv2.imwrite(
            threshold_path,
            threshold,
        )

    if debug is not None:

        debug_path = os.path.join(
            RESULT_DIR,
            f"{scan_id}_debug.jpg",
        )

        cv2.imwrite(
            debug_path,
            debug,
        )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    index_path = os.path.join(
        STATIC_DIR,
        "index.html",
    )

    if os.path.exists(
        index_path
    ):

        return FileResponse(
            index_path
        )

    return {
        "status": "ok",
        "message": "OMR Scanner API is running",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "OMR Scanner",
        "version": "2.1.0",
    }


# ============================================================
# SCAN OMR
# ============================================================

@app.post("/scan")
async def scan_omr(

    image: UploadFile = File(...),

    exam: str = Form(...),

):

    # ========================================================
    # VALIDATE EXAM
    # ========================================================

    exam = (
        exam
        .strip()
        .lower()
    )

    allowed_exams = [
        "neet",
        "kcet",
        "jee",
    ]

    if exam not in allowed_exams:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exam must be "
                "NEET, KCET or JEE."
            ),
        )


    # ========================================================
    # TEMPLATE
    # ========================================================

    template_path = os.path.join(
        TEMPLATE_DIR,
        f"{exam}.json",
    )

    if not os.path.exists(
        template_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                f"Template not found "
                f"for {exam.upper()}."
            ),
        )


    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    original_filename = (
        image.filename
        or "camera_omr.jpg"
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            ),
        )


    # ========================================================
    # CREATE SCAN ID
    # ========================================================

    scan_id = str(
        uuid.uuid4()
    )

    upload_filename = (
        f"{scan_id}{extension}"
    )

    upload_path = os.path.join(
        UPLOAD_DIR,
        upload_filename,
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        contents = await image.read()

        if not contents:

            raise ValueError(
                "Captured image is empty."
            )

        with open(
            upload_path,
            "wb",
        ) as file:

            file.write(
                contents
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save "
                "captured OMR image: "
                f"{str(error)}"
            ),
        )


    # ========================================================
    # PROCESS OMR
    # ========================================================

    try:

        processing = process_omr(
            upload_path,
            template_path,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "OMR processing failed: "
                f"{str(error)}"
            ),
        )


    # ========================================================
    # BASIC RESULT
    # ========================================================

    result = {

        "scan_id":
            scan_id,

        "status":
            "processed",

        "exam":
            exam.upper(),

        "quality":
            processing.get(
                "quality"
            ),

    }


    # ========================================================
    # NEET
    # ========================================================

    if exam == "neet":

        paper_code_data = (
            processing.get(
                "paper_code"
            )
        )

        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "NEET question paper "
                    "code could not be detected."
                ),
            )


        paper_code = (
            paper_code_data.get(
                "value"
            )
        )


        if not paper_code:

            raise HTTPException(
                status_code=400,
                detail=(
                    "NEET question paper "
                    "code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "neet",
                    paper_code,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        if "answers" not in answer_key_data:

            raise HTTPException(
                status_code=500,
                detail=(
                    "NEET answer key does not "
                    "contain 'answers'."
                ),
            )


        detected_answers = (
            processing.get(
                "answers",
                {},
            )
        )


        marking = (
            answer_key_data.get(
                "marking",
                {},
            )
        )


        score_data = calculate_score(

            detected_answers=
                detected_answers,

            answer_key=
                answer_key_data[
                    "answers"
                ],

            correct_marks=
                marking.get(
                    "correct",
                    4,
                ),

            wrong_marks=
                marking.get(
                    "wrong",
                    -1,
                ),

            blank_marks=
                marking.get(
                    "blank",
                    0,
                ),

            multiple_marks=
                marking.get(
                    "multiple",
                    -1,
                ),

        )


        result.update(
            {

                "paper_code":
                    paper_code,

                "paper_code_details":
                    paper_code_data,

                "score":
                    score_data[
                        "score"
                    ],

                "correct":
                    score_data[
                        "correct"
                    ],

                "wrong":
                    score_data[
                        "wrong"
                    ],

                "blank":
                    score_data[
                        "blank"
                    ],

                "multiple":
                    score_data[
                        "multiple"
                    ],

                "uncertain":
                    score_data[
                        "uncertain"
                    ],

                "answers":
                    detected_answers,

                "question_results":
                    score_data[
                        "questions"
                    ],

            }
        )


    # ========================================================
    # KCET
    # ========================================================

    elif exam == "kcet":

        paper_code_data = (
            processing.get(
                "paper_code"
            )
        )


        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET question paper "
                    "code could not be detected."
                ),
            )


        paper_code = (
            paper_code_data.get(
                "value"
            )
        )


        if not paper_code:

            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET question paper "
                    "code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "kcet",
                    paper_code,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        if "answers" not in answer_key_data:

            raise HTTPException(
                status_code=500,
                detail=(
                    "KCET answer key does not "
                    "contain 'answers'."
                ),
            )


        detected_answers = (
            processing.get(
                "answers",
                {},
            )
        )


        marking = (
            answer_key_data.get(
                "marking",
                {},
            )
        )


        score_data = calculate_score(

            detected_answers=
                detected_answers,

            answer_key=
                answer_key_data[
                    "answers"
                ],

            correct_marks=
                marking.get(
                    "correct",
                    1,
                ),

            wrong_marks=
                marking.get(
                    "wrong",
                    0,
                ),

            blank_marks=
                marking.get(
                    "blank",
                    0,
                ),

            multiple_marks=
                marking.get(
                    "multiple",
                    0,
                ),

        )


        result.update(
            {

                "paper_code":
                    paper_code,

                "paper_code_details":
                    paper_code_data,

                "score":
                    score_data[
                        "score"
                    ],

                "correct":
                    score_data[
                        "correct"
                    ],

                "wrong":
                    score_data[
                        "wrong"
                    ],

                "blank":
                    score_data[
                        "blank"
                    ],

                "multiple":
                    score_data[
                        "multiple"
                    ],

                "uncertain":
                    score_data[
                        "uncertain"
                    ],

                "answers":
                    detected_answers,

                "question_results":
                    score_data[
                        "questions"
                    ],

            }
        )


    # ========================================================
    # JEE
    # ========================================================

    elif exam == "jee":

        series_data = (
            processing.get(
                "jee_series"
            )
        )


        if not series_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "JEE series/code could "
                    "not be detected."
                ),
            )


        series = (
            series_data.get(
                "value"
            )
        )


        if not series:

            raise HTTPException(
                status_code=400,
                detail=(
                    "JEE series/code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "jee",
                    series,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        detected = (
            processing.get(
                "answers",
                {},
            )
        )


        mcq_detected = (
            detected.get(
                "mcq",
                {},
            )
        )


        numerical_detected = (
            detected.get(
                "numerical",
                {},
            )
        )


        # ----------------------------------------------------
        # JEE SCORING
        # ----------------------------------------------------

        try:

            score_data = calculate_jee_score(

                detected_answers=
                    detected,

                answer_key=
                    answer_key_data,

            )

        except Exception as error:

            # Allows calibration/testing even if
            # JEE answer key format is incomplete.

            result.update(
                {

                    "series":
                        series,

                    "series_details":
                        series_data,

                    "mcq_answers":
                        mcq_detected,

                    "numerical_answers":
                        numerical_detected,

                    "score":
                        None,

                    "correct":
                        None,

                    "wrong":
                        None,

                    "blank":
                        None,

                    "multiple":
                        None,

                    "message":
                        (
                            "JEE sheet detected, "
                            "but scoring could not "
                            "be completed: "
                            f"{str(error)}"
                        ),

                }
            )

        else:

            result.update(
                {

                    "series":
                        series,

                    "series_details":
                        series_data,

                    "score":
                        score_data.get(
                            "score"
                        ),

                    "correct":
                        score_data.get(
                            "correct"
                        ),

                    "wrong":
                        score_data.get(
                            "wrong"
                        ),

                    "blank":
                        score_data.get(
                            "blank"
                        ),

                    "multiple":
                        score_data.get(
                            "multiple",
                            0,
                        ),

                    "mcq_answers":
                        mcq_detected,

                    "numerical_answers":
                        numerical_detected,

                    "score_details":
                        score_data,

                }
            )


    # ========================================================
    # SAVE DEBUG IMAGES
    # ========================================================

    try:

        save_debug_images(
            scan_id,
            processing,
        )

    except Exception as error:

        print(
            "Debug image save warning:",
            error,
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_path = os.path.join(
        RESULT_DIR,
        f"{scan_id}.json",
    )


    try:

        save_json(
            result_path,
            result,
        )

    except Exception as error:

        print(
            "Result save warning:",
            error,
        )


    return result


# ============================================================
# GET RESULT
# ============================================================

@app.get(
    "/result/{scan_id}"
)
def get_result(
    scan_id: str,
):

    scan_id = safe_filename(
        scan_id
    )


    result_path = os.path.join(
        RESULT_DIR,
        f"{scan_id}.json",
    )


    if not os.path.exists(
        result_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Result not found.",
        )


    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# API INFO
# ============================================================

@app.get("/api")
def api_info():

    return {

        "service":
            "OMR Scanner",

        "version":
            "2.1.0",

        "supported_exams": [
            "NEET",
            "JEE",
            "KCET",
        ],

        "workflow":
            (
                "Select exam -> "
                "capture OMR using camera -> "
                "detect paper code/series -> "
                "load answer key automatically -> "
                "calculate score"
            ),

        "endpoints": {

            "scan":
                "POST /scan",

            "health":
                "GET /health",

            "result":
                "GET /result/{scan_id}",

        },

    }
