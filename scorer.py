# scorer.py


def calculate_score(
    detected_answers,
    answer_key,
    correct_marks=4,
    wrong_marks=-1,
    blank_marks=0,
    multiple_marks=-1,
):
    correct_count = 0
    wrong_count = 0
    blank_count = 0
    multiple_count = 0

    total_score = 0

    question_results = {}

    for question_number, correct_answer in (
        answer_key.items()
    ):
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
            detected_answer = "BLANK"
        else:
            detected_answer = (
                detected_data["answer"]
            )

        if (
            detected_answer
            == correct_answer
        ):
            status = "CORRECT"
            marks = correct_marks

            correct_count += 1

        elif detected_answer == "BLANK":
            status = "BLANK"
            marks = blank_marks

            blank_count += 1

        elif detected_answer == "MULTIPLE":
            status = "MULTIPLE"
            marks = multiple_marks

            multiple_count += 1

        else:
            status = "WRONG"
            marks = wrong_marks

            wrong_count += 1

        total_score += marks

        question_results[
            question_number
        ] = {
            "detected": detected_answer,
            "correct_answer": correct_answer,
            "status": status,
            "marks": marks,
        }

    return {
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "multiple": multiple_count,
        "score": total_score,
        "questions": question_results,
    }