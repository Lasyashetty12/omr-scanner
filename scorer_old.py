# scorer.py

from config import (
    CORRECT_MARKS,
    WRONG_MARKS,
    BLANK_MARKS,
    MULTIPLE_MARKS,
)


def calculate_score(detected_answers, answer_key):
    correct_count = 0
    wrong_count = 0
    blank_count = 0
    multiple_count = 0

    total_score = 0

    question_results = {}

    for question_number, correct_answer in answer_key.items():

        detected_data = detected_answers.get(question_number)

        if detected_data is None:
            detected_answer = "BLANK"
        else:
            detected_answer = detected_data["answer"]

        if detected_answer == correct_answer:
            status = "CORRECT"
            marks = CORRECT_MARKS
            correct_count += 1

        elif detected_answer == "BLANK":
            status = "BLANK"
            marks = BLANK_MARKS
            blank_count += 1

        elif detected_answer == "MULTIPLE":
            status = "MULTIPLE"
            marks = MULTIPLE_MARKS
            multiple_count += 1

        else:
            status = "WRONG"
            marks = WRONG_MARKS
            wrong_count += 1

        total_score += marks

        question_results[question_number] = {
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