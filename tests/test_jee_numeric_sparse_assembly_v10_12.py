
from pathlib import Path

import scorer
from jee_reader import _assemble_sparse_jee_numerical

ROOT = Path(__file__).resolve().parents[1]


def test_sparse_gap_inside_number_is_not_uncertain():
    result = _assemble_sparse_jee_numerical(
        ["2", "2", "1", "4", "", "2", "4"],
        selected_decimal=4,
        negative=False,
    )
    assert result["answer"] == "2214.24"
    assert result["used_columns"] == [1, 2, 3, 4, 6, 7]


def test_decimal_uses_physical_column_boundary():
    result = _assemble_sparse_jee_numerical(
        ["0", "1", "3", "", "4", "", "9"],
        selected_decimal=3,
        negative=False,
    )
    assert result["raw_answer"] == "013.49"
    assert result["answer"] == "13.49"
    assert result["integer_columns"] == [1, 2, 3]
    assert result["fractional_columns"] == [5, 7]


def test_blank_slot_after_decimal_is_skipped():
    result = _assemble_sparse_jee_numerical(
        ["2", "", "3", "5", "3", "1", "7"],
        selected_decimal=1,
        negative=False,
    )
    assert result["answer"] == "2.35317"


def test_minus_sign_prefixes_number():
    result = _assemble_sparse_jee_numerical(
        ["4", "2", "2", "1", "", "1", "9"],
        selected_decimal=4,
        negative=True,
    )
    assert result["answer"] == "-4221.19"


def test_decimal_empty_left_gets_zero():
    result = _assemble_sparse_jee_numerical(
        ["", "", "5", "", "", "", ""],
        selected_decimal=2,
        negative=False,
    )
    assert result["answer"] == "0.5"


def test_decimal_empty_right_gets_zero():
    result = _assemble_sparse_jee_numerical(
        ["1", "2", "", "", "", "", ""],
        selected_decimal=4,
        negative=False,
    )
    assert result["answer"] == "12.0"


def test_no_decimal_concatenates_marked_columns_only():
    result = _assemble_sparse_jee_numerical(
        ["", "1", "", "2", "", "", "3"],
        selected_decimal=None,
        negative=False,
    )
    assert result["answer"] == "123"


def test_all_blank_is_blank():
    result = _assemble_sparse_jee_numerical(
        ["", "", "", "", "", "", ""],
        selected_decimal=None,
        negative=False,
    )
    assert result["answer"] == "BLANK"


def test_score_9_equals_9_point_0():
    result = scorer.calculate_jee_numerical_score(
        detected_answers={25: {"answer": "9"}},
        answer_key={"25": "9.0"},
    )
    assert result["correct"] == 1
    assert result["questions"][25]["status"] == "CORRECT"


def test_score_negative_equivalent_decimals():
    result = scorer.calculate_jee_numerical_score(
        detected_answers={21: {"answer": "-1.20"}},
        answer_key={"21": "-1.2"},
    )
    assert result["correct"] == 1
    assert result["questions"][21]["detected_normalized"] == "-1.2"


def test_uncertain_stays_uncertain():
    result = scorer.calculate_jee_numerical_score(
        detected_answers={21: {"answer": "UNCERTAIN"}},
        answer_key={"21": "-1.2"},
    )
    assert result["uncertain"] == 1


def test_numeric_algorithms_unchanged():
    source = (ROOT / "jee_reader.py").read_text(encoding="utf-8")
    assert '"local_grid_affine_v10_2"' in source
    assert '"uniform_core_v6"' in source
    assert '"decimal_local_contrast_v6_4"' in source


def test_mcq_v10_11_untouched():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")
    assert "jee_ml_hybrid_gate_v10_11" in source


def test_batch_limit_500_untouched():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "MAX_BATCH_OMR_FILES = 500" in source
