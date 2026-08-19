# scorer.py


# ============================================================
# NORMAL MCQ SCORER
# USED FOR NEET + KCET
# ============================================================

def calculate_score(
    detected_answers,
    answer_key,
    correct_marks=4,
    wrong_marks=-1,
    blank_marks=0,
    multiple_marks=-1,
):
    """
    Score normal A/B/C/D OMR answers.

    detected_answers format:

    {
        1: {
            "answer": "A"
        },
        2: {
            "answer": "BLANK"
        }
    }

    answer_key format:

    {
        "1": "A",
        "2": "C"
    }
    """

    correct_count = 0
    wrong_count = 0
    blank_count = 0
    multiple_count = 0
    uncertain_count = 0

    total_score = 0

    question_results = {}


    for (
        question_number,
        correct_answer
    ) in answer_key.items():

        question_number = int(
            question_number
        )

        correct_answer = (
            str(correct_answer)
            .strip()
            .upper()
        )

        detected_data = (
            detected_answers.get(
                question_number
            )
        )

        if detected_data is None:

            detected_answer = (
                "BLANK"
            )

        else:

            detected_answer = (
                str(
                    detected_data.get(
                        "answer",
                        "BLANK"
                    )
                )
                .strip()
                .upper()
            )


        # ----------------------------------------------------
        # CORRECT
        # ----------------------------------------------------

        if (
            detected_answer
            == correct_answer
        ):

            status = "CORRECT"

            marks = (
                correct_marks
            )

            correct_count += 1


        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        elif (
            detected_answer
            == "BLANK"
        ):

            status = "BLANK"

            marks = (
                blank_marks
            )

            blank_count += 1


        # ----------------------------------------------------
        # MULTIPLE
        # ----------------------------------------------------

        elif (
            detected_answer
            == "MULTIPLE"
        ):

            status = "MULTIPLE"

            marks = (
                multiple_marks
            )

            multiple_count += 1


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        elif (
            detected_answer
            == "UNCERTAIN"
        ):

            status = "UNCERTAIN"

            marks = (
                blank_marks
            )

            uncertain_count += 1


        # ----------------------------------------------------
        # WRONG
        # ----------------------------------------------------

        else:

            status = "WRONG"

            marks = (
                wrong_marks
            )

            wrong_count += 1


        total_score += marks


        question_results[
            question_number
        ] = {

            "detected":
                detected_answer,

            "correct_answer":
                correct_answer,

            "status":
                status,

            "marks":
                marks,
        }


    return {

        "correct":
            correct_count,

        "wrong":
            wrong_count,

        "blank":
            blank_count,

        "multiple":
            multiple_count,

        "uncertain":
            uncertain_count,

        "score":
            total_score,

        "questions":
            question_results,
    }


# ============================================================
# JEE MCQ SCORER
# ============================================================

def calculate_jee_mcq_score(
    detected_answers,
    answer_key,
    correct_marks=4,
    wrong_marks=-1,
    blank_marks=0,
    multiple_marks=-1,
):
    """
    Score only JEE MCQ questions.

    This reuses the normal MCQ scoring logic.
    """

    return calculate_score(

        detected_answers=
            detected_answers,

        answer_key=
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


# ============================================================
# JEE NUMERICAL SCORER
# ============================================================

def calculate_jee_numerical_score(
    detected_answers,
    answer_key,
    correct_marks=4,
    wrong_marks=0,
    blank_marks=0,
):
    """
    Score JEE numerical-response questions.

    detected_answers format:

    {
        21: {
            "answer": "12"
        },

        22: {
            "answer": "BLANK"
        }
    }

    answer_key format:

    {
        "21": "12",
        "22": "7"
    }
    """

    correct_count = 0
    wrong_count = 0
    blank_count = 0
    uncertain_count = 0

    total_score = 0

    question_results = {}


    for (
        question_number,
        correct_answer
    ) in answer_key.items():

        question_number = int(
            question_number
        )

        correct_answer = (
            str(correct_answer)
            .strip()
        )

        detected_data = (
            detected_answers.get(
                question_number
            )
        )

        if detected_data is None:

            detected_answer = (
                "BLANK"
            )

        else:

            detected_answer = (
                str(
                    detected_data.get(
                        "answer",
                        "BLANK"
                    )
                )
                .strip()
            )


        # ----------------------------------------------------
        # CORRECT
        # ----------------------------------------------------

        if (
            detected_answer
            == correct_answer
        ):

            status = "CORRECT"

            marks = (
                correct_marks
            )

            correct_count += 1


        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        elif (
            detected_answer
            == "BLANK"
        ):

            status = "BLANK"

            marks = (
                blank_marks
            )

            blank_count += 1


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        elif (
            detected_answer
            == "UNCERTAIN"
        ):

            status = "UNCERTAIN"

            marks = (
                blank_marks
            )

            uncertain_count += 1


        # ----------------------------------------------------
        # WRONG
        # ----------------------------------------------------

        else:

            status = "WRONG"

            marks = (
                wrong_marks
            )

            wrong_count += 1


        total_score += marks


        question_results[
            question_number
        ] = {

            "detected":
                detected_answer,

            "correct_answer":
                correct_answer,

            "status":
                status,

            "marks":
                marks,
        }


    return {

        "correct":
            correct_count,

        "wrong":
            wrong_count,

        "blank":
            blank_count,

        "uncertain":
            uncertain_count,

        "score":
            total_score,

        "questions":
            question_results,
    }


# ============================================================
# COMPLETE JEE SCORER
# ============================================================

def calculate_jee_score(
    detected_mcq_or_full,
    detected_num_or_key,
    mcq_answer_key=None,
    numerical_answer_key=None,
    marking=None,
):
    """
    Combine JEE MCQ and numerical scoring. Supports both 2-arg and 4-arg signatures.
    """
    if mcq_answer_key is not None and numerical_answer_key is not None:
        detected_mcq = detected_mcq_or_full
        detected_numerical = detected_num_or_key
    else:
        detected_full = detected_mcq_or_full or {}
        answer_key_full = detected_num_or_key or {}

        if marking is None:
            marking = answer_key_full.get("marking", {})

        detected_mcq = detected_full.get("mcq", {})
        detected_numerical = detected_full.get("numerical", {})

        answers_dict = answer_key_full.get("answers", {})
        mcq_answer_key = {}
        numerical_answer_key = {}

        if "mcq" in answers_dict and "numerical" in answers_dict:
            mcq_answer_key = answers_dict["mcq"]
            numerical_answer_key = answers_dict["numerical"]
        else:
            for q_num, ans in answers_dict.items():
                try:
                    q_int = int(q_num)
                    if (21 <= q_int <= 25) or (46 <= q_int <= 50) or (71 <= q_int <= 75):
                        numerical_answer_key[str(q_int)] = str(ans)
                    else:
                        mcq_answer_key[str(q_int)] = str(ans)
                except ValueError:
                    mcq_answer_key[str(q_num)] = str(ans)

    if marking is None:
        marking = {}

    mcq_result = calculate_jee_mcq_score(
        detected_answers=detected_mcq,
        answer_key=mcq_answer_key,
        correct_marks=marking.get("mcq_correct", marking.get("correct", 4)),
        wrong_marks=marking.get("mcq_wrong", marking.get("wrong", -1)),
        blank_marks=marking.get("mcq_blank", marking.get("blank", 0)),
        multiple_marks=marking.get("mcq_multiple", marking.get("multiple", -1)),
    )

    numerical_result = calculate_jee_numerical_score(
        detected_answers=detected_numerical,
        answer_key=numerical_answer_key,
        correct_marks=marking.get("numerical_correct", marking.get("correct", 4)),
        wrong_marks=marking.get("numerical_wrong", 0),
        blank_marks=marking.get("numerical_blank", marking.get("blank", 0)),
    )

    total_score = mcq_result["score"] + numerical_result["score"]


    return {

        "score":
            total_score,

        "correct":
            (
                mcq_result[
                    "correct"
                ]
                +
                numerical_result[
                    "correct"
                ]
            ),

        "wrong":
            (
                mcq_result[
                    "wrong"
                ]
                +
                numerical_result[
                    "wrong"
                ]
            ),

        "blank":
            (
                mcq_result[
                    "blank"
                ]
                +
                numerical_result[
                    "blank"
                ]
            ),

        "multiple":
            mcq_result.get(
                "multiple",
                0,
            ),

        "uncertain":
            (
                mcq_result.get(
                    "uncertain",
                    0,
                )
                +
                numerical_result.get(
                    "uncertain",
                    0,
                )
            ),

        "mcq":
            mcq_result,

        "numerical":
            numerical_result,
    }