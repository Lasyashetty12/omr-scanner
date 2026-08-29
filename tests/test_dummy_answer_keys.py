import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = ("P", "Q", "R", "S")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_dummy_neet_and_kcet_keys_cover_the_physical_208_rows():
    for exam in ("neet", "kcet"):
        for series in SERIES:
            key = load(ROOT / "answer_keys" / exam / f"{series}.json")
            assert key["dummy"] is True
            assert key["series"] == series
            assert len(key["answers"]) == 208
            assert set(key["answers"].values()) == {"A", "B", "C", "D"}


def test_dummy_jee_keys_cover_all_60_mcq_and_15_numerical_questions():
    for series in SERIES:
        key = load(ROOT / "answer_keys" / "jee" / f"{series}.json")
        assert key["dummy"] is True
        assert key["series"] == series
        assert len(key["answers"]["mcq"]) == 60
        assert len(key["answers"]["numerical"]) == 15
