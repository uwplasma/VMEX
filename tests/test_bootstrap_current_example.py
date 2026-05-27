from __future__ import annotations

from pathlib import Path


def test_bootstrap_current_example_is_explicit_user_workflow() -> None:
    text = (Path(__file__).resolve().parents[1] / "examples" / "bootstrap_current_fixed_point.py").read_text()

    assert "FIXED_POINT_OPTIONS = vj.BootstrapCurrentOptions" in text
    assert "VMEC_RUN_KWARGS = {" in text
    assert "result = vj.bootstrap_current_fixed_point" in text
    assert "vj.write_indata(FINAL_INPUT, result.indata)" in text
    assert "HISTORY_JSON.write_text" in text
    assert "solver_device" in text
