# app.py

import json
import os
import uuid
from datetime import datetime
from typing import List

import cv2

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Query,
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

from database import (
    save_omr_result_to_db,
    get_omr_results_from_db,
    get_omr_result_by_id_from_db,
    get_omr_result_by_scan_id_from_db,
    get_database_diagnostics,
    is_db_configured,
)

from cloudinary_storage import (
    cloudinary_enabled,
    upload_evaluation_json,
    upload_scan_images,
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

NEET_QUESTION_LIMIT = 180
KCET_PCM_QUESTION_LIMIT = 180
KCET_PCMB_QUESTION_LIMIT = 240


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


def normalize_detected_class(value):
    """Convert the printed I/II/LT class bubbles to dashboard values."""
    normalized = str(value or "").strip().upper()
    return {
        "I": "11",
        "II": "12",
        "11": "11",
        "12": "12",
        "LT": "LT",
    }.get(normalized, "")


def limit_question_mapping(question_mapping, question_limit):
    """Return only valid question entries up to the evaluation limit."""
    limited = {}

    for question_number, value in (question_mapping or {}).items():
        try:
            numeric_question = int(question_number)
        except (TypeError, ValueError):
            continue

        if 1 <= numeric_question <= question_limit:
            limited[numeric_question] = value

    return limited


def sanitize_optional_result_assets(result):
    """Keep optional result-image URLs in sync with files on local storage.

    Older local JSON records may not contain ``bubble_debug_image_url`` because
    previous versions saved the JSON before adding image URLs.  Reconstruct the
    URL when the scan-specific debug image is actually present, while still
    suppressing stale URLs whose files no longer exist.
    """
    if not isinstance(result, dict):
        return result

    scan_id = safe_filename(result.get("scan_id") or result.get("id") or "")
    debug_url = result.get("bubble_debug_image_url")

    if scan_id:
        debug_filename = f"{scan_id}_bubble_debug.jpg"
        debug_path = os.path.join(RESULT_DIR, debug_filename)
        if os.path.exists(debug_path):
            result["bubble_debug_image_url"] = f"/results/{debug_filename}"
        elif isinstance(debug_url, str) and debug_url.startswith("/results/"):
            result["bubble_debug_image_url"] = None
    elif isinstance(debug_url, str) and debug_url.startswith("/results/"):
        debug_filename = safe_filename(debug_url.split("?", 1)[0])
        debug_path = os.path.join(RESULT_DIR, debug_filename)
        if not os.path.exists(debug_path):
            result["bubble_debug_image_url"] = None

    return result


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

    def _fast_write(path, img):
        if img is None:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not written or not os.path.exists(path):
            raise OSError(f"Could not write debug image: {path}")
        return True

    corrected = processing.get("corrected")
    if corrected is not None:
        _fast_write(os.path.join(RESULT_DIR, f"{scan_id}_corrected.jpg"), corrected)

    debug_img = processing.get("debug")
    if debug_img is not None:
        _fast_write(os.path.join(RESULT_DIR, f"{scan_id}_bubble_debug.jpg"), debug_img)


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
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    return {
        "status": "ok",
        "message": "OMR Scanner API is running",
    }


# ============================================================
# TEACHER DASHBOARD PAGE
# ============================================================

@app.get("/dashboard")
@app.get("/dashboard.html")
def dashboard_page():

    dashboard_path = os.path.join(
        STATIC_DIR,
        "dashboard.html",
    )

    if os.path.exists(dashboard_path):
        return FileResponse(
            dashboard_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    raise HTTPException(
        status_code=404,
        detail="Teacher dashboard page not found.",
    )


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
# STORAGE STATUS - SAFE PRODUCTION DIAGNOSTIC
# ============================================================

@app.get("/api/storage-status")
def storage_status():
    db_status = get_database_diagnostics()

    return {
        "supabase": db_status,
        "cloudinary": {
            "configured": bool(
                cloudinary_enabled()
            ),
        },
    }


# ============================================================
# SCAN OMR
# ============================================================

@app.post("/scan")
async def scan_omr(

    image: UploadFile = File(...),

    exam: str = Form(...),

    stream: str = Form("pcmb"),

    class_name: str = Form(""),

    section: str = Form(""),

    exam_date: str = Form(""),

    session: str = Form(""),

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
        "kcet_neet",
        "jee",
    ]

    if exam not in allowed_exams:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exam must be "
                "KCET/NEET auto-detect or JEE."
            ),
        )

    class_name = str(
        class_name
        or ""
    ).strip()

    section = str(
        section
        or ""
    ).strip().upper()

    exam_date = str(
        exam_date
        or ""
    ).strip()

    session = str(
        session
        or ""
    ).strip()

    if exam == "jee":
        if (
            class_name not in {"11", "12", "LT"}
            or section not in {"A", "B", "C"}
            or session not in {"Morning", "Afternoon"}
            or not exam_date
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "JEE scan requires Class, Section, "
                    "Exam Date and Session."
                ),
            )

        try:
            datetime.strptime(
                exam_date,
                "%Y-%m-%d",
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Exam Date must be a valid date "
                    "in YYYY-MM-DD format."
                ),
            )

    elif exam in {"neet", "kcet", "kcet_neet"}:
        if (
            section not in {"A", "B", "C"}
            or session not in {"Morning", "Afternoon"}
            or not exam_date
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET/NEET scan requires Section, "
                    "Exam Date and Session. Roll number, Class, "
                    "Series and Exam are detected from the OMR."
                ),
            )

        try:
            datetime.strptime(
                exam_date,
                "%Y-%m-%d",
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Exam Date must be a valid date "
                    "in YYYY-MM-DD format."
                ),
            )


    # ========================================================
    # TEMPLATE
    # ========================================================

    template_exam = (
        "kcet"
        if exam == "kcet_neet"
        else exam
    )

    template_path = os.path.join(
        TEMPLATE_DIR,
        f"{template_exam}.json",
    )

    if not os.path.exists(
        template_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                f"Template not found "
                f"for {template_exam.upper()}."
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
            contents,
            template_path,
            input_filename=original_filename,
            input_mime_type=image.content_type,
            diagnostic_dir=(
                os.path.join(RESULT_DIR, f"{scan_id}_input_diagnostics")
                if os.environ.get("OMR_DEBUG_INPUT")
                else None
            ),
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

        "input": processing.get("input_debug"),

        "identity":
            processing.get(
                "identity",
                {},
            ),

    }

    identity_data = (
        result.get(
            "identity"
        )
        or {}
    )

    if identity_data.get(
        "roll_number"
    ):
        result["roll_number"] = (
            identity_data[
                "roll_number"
            ]
        )

    detected_class = normalize_detected_class(
        identity_data.get("class")
    )

    if detected_class:
        result["class"] = detected_class

    detected_exam = str(
        identity_data.get("exam")
        or ""
    ).strip().upper()

    if exam in {"neet", "kcet", "kcet_neet"}:
        if detected_exam not in {"NEET", "KCET"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not detect the KCET/NEET exam bubble "
                    "from the OMR sheet."
                ),
            )

        if not result.get("roll_number"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not detect a complete roll number "
                    "from the OMR sheet."
                ),
            )

        if not detected_class:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not detect the Class bubble "
                    "from the OMR sheet."
                ),
            )

        exam = detected_exam.lower()
        result["exam"] = detected_exam
        result["detected_exam"] = detected_exam
        result["section"] = section
        result["exam_date"] = exam_date
        result["session"] = session

    if exam == "jee":
        result["class"] = class_name
        result["section"] = section
        result["exam_date"] = exam_date
        result["session"] = session
        result["stream"] = "PCM"


    # ========================================================
    # NEET
    # ========================================================

    if exam == "neet":

        paper_code_data = (
            processing.get(
                "series"
            )
        )

        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "NEET series (P/Q/R/S) "
                    "could not be detected."
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
                    "NEET series is empty."
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


        detected_answers = limit_question_mapping(
            processing.get("answers", {}),
            NEET_QUESTION_LIMIT,
        )

        neet_answer_key = {
            key: value
            for key, value in answer_key_data["answers"].items()
            if int(key) <= NEET_QUESTION_LIMIT
        }


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
                neet_answer_key,

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

                "series":
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

                "total_questions":
                    NEET_QUESTION_LIMIT,

            }
        )


    # ========================================================
    # KCET
    # ========================================================

    elif exam == "kcet":

        paper_code_data = (
            processing.get(
                "series"
            )
        )


        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET series (P/Q/R/S) "
                    "could not be detected."
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
                    "KCET series is empty."
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


        question_limit = (
            KCET_PCM_QUESTION_LIMIT
            if stream and stream.lower().strip() == "pcm"
            else KCET_PCMB_QUESTION_LIMIT
        )

        detected_answers = limit_question_mapping(
            processing.get("answers", {}),
            question_limit,
        )


        marking = (
            answer_key_data.get(
                "marking",
                {},
            )
        )


        kcet_answer_key = {
            key: value
            for key, value in answer_key_data["answers"].items()
            if int(key) <= question_limit
        }

        score_data = calculate_score(

            detected_answers=
                detected_answers,

            answer_key=
                kcet_answer_key,

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

                "stream":
                    (stream or "PCMB").upper(),

                "paper_code":
                    paper_code,

                "series":
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

                "total_questions":
                    question_limit,

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

            configured_answers = answer_key_data.get(
                "answers",
                {},
            )

            if (
                isinstance(configured_answers, dict)
                and "mcq" in configured_answers
                and "numerical" in configured_answers
            ):
                mcq_answer_key = configured_answers["mcq"]
                numerical_answer_key = configured_answers["numerical"]
            else:
                numerical_numbers = {
                    *range(21, 26),
                    *range(46, 51),
                    *range(71, 76),
                }
                mcq_answer_key = {}
                numerical_answer_key = {}
                for question_number, answer in configured_answers.items():
                    target = (
                        numerical_answer_key
                        if int(question_number) in numerical_numbers
                        else mcq_answer_key
                    )
                    target[str(question_number)] = answer

            score_data = calculate_jee_score(

                detected_mcq=
                    mcq_detected,

                detected_numerical=
                    numerical_detected,

                mcq_answer_key=
                    mcq_answer_key,

                numerical_answer_key=
                    numerical_answer_key,

                marking=
                    answer_key_data.get("marking", {}),

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

            # Merge MCQ + numerical scoring so the result page and database
            # receive the complete 75-question JEE breakdown.
            jee_question_results = {}
            jee_question_results.update(
                score_data.get("mcq", {}).get("questions", {})
            )
            jee_question_results.update(
                score_data.get("numerical", {}).get("questions", {})
            )

            result.update(
                {
                    "series": series,
                    "paper_code": series,
                    "series_details": series_data,
                    "stream": "PCM",
                    "max_score": 300,
                    "marking_scheme": {
                        "name": "JEE Main 2026 Paper 1",
                        "mcq_correct": 4,
                        "mcq_wrong": -1,
                        "mcq_blank": 0,
                        "numerical_correct": 4,
                        "numerical_wrong": -1,
                        "numerical_blank": 0,
                    },
                    "score": score_data.get("score"),
                    "correct": score_data.get("correct"),
                    "wrong": score_data.get("wrong"),
                    "blank": score_data.get("blank"),
                    "multiple": score_data.get("multiple", 0),
                    "uncertain": score_data.get("uncertain", 0),
                    "mcq_answers": mcq_detected,
                    "numerical_answers": numerical_detected,
                    "question_results": jee_question_results,
                    "score_details": score_data,
                }
            )

    result["student"] = {
        "name": "Student Candidate",
        "roll_number": result.get("roll_number"),
        "class": result.get("class"),
        "section": section,
    }

    result["exam_info"] = {
        "exam_type": result.get("exam"),
        "paper_code": result.get("paper_code") or result.get("series"),
        "paper_series": result.get("paper_code") or result.get("series"),
        "exam_date": exam_date,
        "session": session,
    }


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
    # RESULT IMAGE URLS
    # ========================================================

    # Add image URLs BEFORE persisting the result JSON.  The individual
    # result page reloads this JSON when no external database is configured;
    # saving first used to make the bubble debug image disappear on View Result.
    result["original_image_url"] = f"/uploads/{upload_filename}"
    result["corrected_image_url"] = f"/results/{scan_id}_corrected.jpg"

    bubble_debug_path = os.path.join(
        RESULT_DIR,
        f"{scan_id}_bubble_debug.jpg",
    )
    result["bubble_debug_image_url"] = (
        f"/results/{scan_id}_bubble_debug.jpg"
        if os.path.exists(bubble_debug_path)
        else None
    )

    # Cloudinary is optional. When configured, persist every successful scan
    # remotely and return durable URLs; local files remain a development and
    # outage fallback.
    if cloudinary_enabled():
        try:
            cloudinary_images = upload_scan_images(
                scan_id=scan_id,
                original_bytes=contents,
                corrected_image=processing.get("corrected"),
                evaluated_image=processing.get("debug"),
            )
            result["cloudinary"] = cloudinary_images
            result["original_image_url"] = cloudinary_images["original"]["url"]
            result["corrected_image_url"] = cloudinary_images["corrected"]["url"]
            result["bubble_debug_image_url"] = cloudinary_images["evaluated"]["url"]

            evaluation_asset = upload_evaluation_json(
                scan_id=scan_id,
                evaluation=result,
            )
            result["cloudinary"]["evaluation"] = evaluation_asset
            result["evaluation_json_url"] = evaluation_asset["url"]
        except Exception as error:
            result["cloudinary_warning"] = (
                "OMR evaluation completed, but Cloudinary upload failed: "
                f"{str(error)}"
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

    # ========================================================
    # SAVE TO DATABASE
    # ========================================================

    db_student_info = {
        "name": "Student Candidate",
        "roll_number": result.get("roll_number"),
        "class_name": result.get("class"),
        "section": section,
        "batch": exam_date[:4],
    }

    db_id = save_omr_result_to_db(
        result,
        student_info=db_student_info,
    )
    if db_id:
        result["id"] = db_id
        result["database_saved"] = True
        result["database_result_id"] = db_id
    else:
        result["id"] = scan_id
        result["database_saved"] = False
        result["database_warning"] = (
            "Evaluation completed, but the result was not persisted "
            "to Supabase. Check /api/storage-status."
        )

    return result


# ============================================================
# GET UPLOADED IMAGE (AS CLICKED)
# ============================================================

@app.get(
    "/uploads/{filename}"
)
def get_upload_image(
    filename: str,
):

    filename = safe_filename(
        filename
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        filename,
    )

    if not os.path.exists(
        image_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Uploaded image not found.",
        )

    return FileResponse(
        image_path,
    )


# ============================================================
# GET RESULT IMAGE
# ============================================================

@app.get(
    "/results/{filename}"
)
def get_result_image(
    filename: str,
):

    filename = safe_filename(
        filename
    )

    image_path = os.path.join(
        RESULT_DIR,
        filename,
    )

    if not os.path.exists(
        image_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Result image not found.",
        )

    return FileResponse(
        image_path,
        media_type="image/jpeg",
    )


# ============================================================
# INDIVIDUAL RESULT HTML PAGE
# ============================================================

@app.get("/result.html")
def get_result_html():

    result_path = os.path.join(
        STATIC_DIR,
        "result.html",
    )

    if os.path.exists(
        result_path
    ):

        return FileResponse(
            result_path,
        )

    raise HTTPException(
        status_code=404,
        detail="Result page not found.",
    )


# ============================================================
# TEACHER DASHBOARD API â€” LIST EVALUATED RESULTS
# ============================================================


@app.get("/api/omr-results")
def list_omr_results(
    class_name: str = Query(None, alias="class"),
    section: str = Query(None),
    exam: str = Query(None),
):

    return get_omr_results_from_db(
        class_filter=class_name,
        section_filter=section,
        exam_filter=exam,
    )


# ============================================================
# GET INDIVIDUAL OMR RESULT BY DB ID / SCAN ID
# ============================================================

@app.get("/api/omr-results/{result_id}")
@app.get("/result/{scan_id}")
def get_result(
    scan_id: str = None,
    result_id: str = None,
):

    target_id = result_id or scan_id
    if not target_id:
        raise HTTPException(
            status_code=404,
            detail="Result not found.",
        )

    target_id = safe_filename(target_id)

    # 1. Database lookup.  Accept both the database result ID and the
    # stable scan UUID used by result links.  The scan UUID is resolved
    # through scans.image_reference -> omr_result_id, which works across
    # serverless/Vercel instances where local result files are ephemeral.
    if is_db_configured():
        db_res = get_omr_result_by_id_from_db(target_id)
        if not db_res:
            db_res = get_omr_result_by_scan_id_from_db(target_id)
        if db_res:
            return sanitize_optional_result_assets(db_res)

    # 2. Local Fallback JSON Lookup
    result_path = os.path.join(
        RESULT_DIR,
        f"{target_id}.json",
    )

    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as file:
                return sanitize_optional_result_assets(json.load(file))
        except Exception:
            pass

    raise HTTPException(
        status_code=404,
        detail="Result not found.",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# ============================================================
# BATCH SCAN OMR — MAX 500 FILES
# ============================================================

@app.post("/scan-batch")
async def scan_omr_batch(
    images: List[UploadFile] = File(...),
    exam: str = Form(...),
    stream: str = Form("pcmb"),
    class_name: str = Form(""),
    section: str = Form(""),
    exam_date: str = Form(""),
    session: str = Form(""),
):
    if not images:
        raise HTTPException(
            status_code=400,
            detail="No OMR images were uploaded.",
        )

    if len(images) > 500:
        raise HTTPException(
            status_code=400,
            detail=(
                "A maximum of 500 OMR images "
                "can be uploaded at one time."
            ),
        )

    results = []
    failures = []

    for index, image in enumerate(images, start=1):
        try:
            result = await scan_omr(
                image=image,
                exam=exam,
                stream=stream,
                class_name=class_name,
                section=section,
                exam_date=exam_date,
                session=session,
            )

            results.append(
                {
                    "index": index,
                    "filename": image.filename,
                    "result": result,
                }
            )

        except HTTPException as error:
            failures.append(
                {
                    "index": index,
                    "filename": image.filename,
                    "status_code": error.status_code,
                    "error": str(error.detail),
                }
            )

        except Exception as error:
            failures.append(
                {
                    "index": index,
                    "filename": image.filename,
                    "status_code": 500,
                    "error": str(error),
                }
            )

    return {
        "requested": len(images),
        "processed": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
