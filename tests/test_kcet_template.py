import json
from pathlib import Path


TEMPLATE_PATH = Path("templates/kcet.json")


def test_kcet_template_uses_its_own_identity_and_crop_geometry():
    """KCET follows the 52-row generated combined NEET/KCET sheet."""
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

    assert template["template_name"] == "kcet"
    assert template["exam_name"] == "KCET"
    assert template["total_questions"] == 208
    assert template["questions_per_column"] == 52
    assert len(template["question_y_positions"]) == 52
    assert template["reference_image"] == "neet_kcet_generated.png"
    assert set(template["series"]["coordinates"]) == {"P", "Q", "R", "S"}
    assert template["ml_crop_radius"] == 12
    assert template["ml_crop_radius"] < 16
    assert template["grid_max_initial_match_distance"] == 12
    assert template["grid_max_direct_pin_dy"] == 10
    assert template["grid_max_final_dy_from_input"] == 16
