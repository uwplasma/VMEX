from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from vmec_jax.mirror import load_mirror_output

pytestmark = pytest.mark.mirror


def _load_run_case(script_name: str):
    script = Path("examples/mirror") / script_name
    return runpy.run_path(str(script))["run_case"]


def test_mirror_examples_write_readable_outputs_without_plots(tmp_path):
    cases = [
        ("fixed_cylinder.py", {"maxiter": 1, "write_plots": False}),
        ("fixed_flared_tube.py", {"maxiter": 1, "write_plots": False}),
        ("wham_vacuum_boundary.py", {"midplane_radius": 0.25, "maxiter": 1, "write_plots": False}),
    ]
    for script_name, kwargs in cases:
        run_case = _load_run_case(script_name)
        mout = run_case(tmp_path / script_name.removesuffix(".py"), **kwargs)
        output = load_mirror_output(mout)
        assert output.attributes["geometry_type"] == "mirror"
        assert output.diagnostics.min_sqrtg > 0.0
        assert output.diagnostics.mirror_ratio >= 1.0
