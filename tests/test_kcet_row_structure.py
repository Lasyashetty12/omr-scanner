import ast
from pathlib import Path


def test_hybrid_reader_uses_the_template_row_count_for_final_row_handling():
    """
    v10.19 architecture:
      - hybrid_reader owns final-row logic.
      - json_anchor_reader owns KCET/NEET calls into scan_answers_ml.
      - scanner no longer has to call scan_answers_ml directly for KCET/NEET.
    """
    module = ast.parse(
        Path("ml_omr/hybrid_reader.py").read_text(
            encoding="utf-8"
        )
    )

    scan_answers_ml = next(
        node
        for node in module.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "scan_answers_ml"
        )
    )

    assert "questions_per_column" in [
        arg.arg
        for arg in scan_answers_ml.args.args
    ]

    modulo_checks = [
        node
        for node in ast.walk(scan_answers_ml)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.right, ast.Name)
            and node.right.id == "questions_per_column"
        )
    ]

    assert modulo_checks

    anchor_module = ast.parse(
        Path(
            "ml_omr/json_anchor_reader.py"
        ).read_text(
            encoding="utf-8"
        )
    )

    hybrid_calls = [
        node
        for node in ast.walk(anchor_module)
        if (
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "scan_answers_ml"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "scan_answers_ml"
                )
            )
        )
    ]

    assert hybrid_calls

    questions_per_column_keywords = [
        keyword
        for call in hybrid_calls
        for keyword in call.keywords
        if keyword.arg == "questions_per_column"
    ]

    assert questions_per_column_keywords

    assert any(
        isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "int"
        for keyword in questions_per_column_keywords
    )

    def uses_template_row_count(node):
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "template"
            ):
                return True

            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "template"
                and child.func.attr == "get"
            ):
                return True

        return False

    assert any(
        uses_template_row_count(keyword.value)
        for keyword in questions_per_column_keywords
    )


