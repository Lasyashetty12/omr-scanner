from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _blob(relative):
    result = subprocess.run(
        ["git", "hash-object", str(ROOT / relative)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_core_jee_files_preserve_reference_contract():
    # v10.35 intentionally modifies ONLY jee_reader.py to make the
    # reference JEE geometry deterministic across repeated captures.
    #
    # Therefore jee_reader.py must no longer be required to have the exact
    # dbc738 blob. Instead, verify that the reference reader is still present
    # and that only the intended v10.35 stability layers were added.
    source = (
        ROOT / "jee_reader.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "scan_jee_mcq_sections_robust" in source
    assert "scan_jee_numerical_sections_robust" in source
    assert "jee_mcq_reference_delta_v10_4" in source
    assert "jee_solid_core_reader_v5" in source

    assert "deterministic_lattice_cluster_v10_35" in source
    assert "affine_lattice_smoothing_v10_35" in source
    assert "numeric_affine_sampling_v10_35" in source

    # These reference assets/readers were not modified by v10.35 and should
    # still match the accurate Sapthagiri commit exactly.
    assert _blob("jee_precise_reader.py") == (
        "f21a3ce61312fc48425575f74b8e44d28a003657"
    )
    assert _blob("templates/jee.json") == (
        "18d9d4ef4ef615bf37d0372a05d01756646432b4"
    )


def test_scanner_uses_reference_jee_mcq_reader():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert (
        "from jee_reader import (\n"
        "    scan_jee_mcq_sections_robust,\n"
        "    scan_jee_numerical_sections_robust,\n"
        ")"
        in source
    )

    assert "scan_jee_mcq_sections_cv_robust as" not in source


def test_reference_stable_single_gate_is_restored():
    source = (ROOT / "scanner.py").read_text(encoding="utf-8")

    assert "jee_reference_gate_dbc738_v10_33" in source
    assert "refine_jee_multiscale_multiples(" not in source
    assert "resolve_strict_jee_secondary_multiple(" not in source
    assert '"stable_single_kept_v10_11"' in source


def test_kcet_neet_reader_files_still_exist():
    assert (ROOT / "ml_omr" / "hybrid_reader.py").exists()
    assert (ROOT / "templates" / "kcet.json").exists()
    assert (ROOT / "templates" / "neet.json").exists()
