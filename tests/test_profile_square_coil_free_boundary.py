from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools.diagnostics import profile_square_coil_free_boundary as profile


def test_square_coil_profile_residual_payload_keeps_solver_mode_and_history_tails():
    diagnostics = {
        "solver_mode": "parity",
        "use_scan": True,
        "performance_mode": False,
        "converged": False,
        "converged_strict": False,
        "requested_ftol": 1.0e-12,
        "final_fsqr": 1.0e-5,
        "final_fsqz": 2.0e-5,
        "final_fsql": 3.0e-6,
        "bad_resets": 0,
        "ijacob": 1,
        "free_boundary": {
            "nestor_model": "vmec2000_like_dense_integral",
            "couple_edge": True,
            "activate_fsq": 1.0e-3,
            "ivac": 3,
            "ivacskip": 0,
            "nvacskip": 2,
            "last_nestor_diagnostics": {
                "bnormal_rms": 4.0e-4,
                "bsqvac_rms": 1.5e-2,
            },
        },
        "freeb_ivac_history": np.array([1, 2, 3]),
        "freeb_ivacskip_history": np.array([0, 1, 0]),
        "freeb_full_update_history": np.array([1, 0, 1]),
        "freeb_nestor_reused_history": np.array([0, 1, 0]),
        "freeb_nestor_bnormal_rms_history": np.array([1.0e-3, 7.0e-4, 4.0e-4]),
        "include_edge_history": np.array([0, 1, 1]),
        "bad_jacobian_history": np.array([0, 0, 0]),
    }
    result = SimpleNamespace(
        n_iter=3,
        diagnostics=diagnostics,
        w_history=np.array([1.0, 0.5, 0.25, 0.125]),
        fsqr2_history=np.array([1.0e-3, 1.0e-4, 1.0e-5]),
        fsqz2_history=np.array([2.0e-3, 2.0e-4, 2.0e-5]),
        fsql2_history=np.array([3.0e-4, 3.0e-5, 3.0e-6]),
    )
    run = SimpleNamespace(result=result)

    payload = profile._final_residuals(run)

    assert payload["solver_mode"] == "parity"
    assert payload["use_scan"] is True
    assert payload["free_boundary_active"] is True
    assert payload["final_fsq_component_sum"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail"] == pytest.approx([0.0033, 0.00033, 3.3e-5])
    assert payload["history"]["freeb_ivac_tail"] == [1, 2, 3]
    assert payload["history"]["include_edge_tail"] == [0, 1, 1]


def test_square_coil_profile_partial_vmec2000_payload_reads_timeout_rows(tmp_path: Path):
    workdir = tmp_path / "vmec2000_mgrid"
    workdir.mkdir()
    (workdir / "threed1.case").write_text(
        "\n".join(
            [
                " NS =    9 NO. FOURIER MODES =  113 FTOLV =  1.000E-08 NITER =   5000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT    RAX(v=0)      WMHD      BETA      <M>   DEL-BSQ   FEDGE",
                "    1   4.00E-03  2.00E-03  1.00E-03  1.00E-04  2.00E-04  3.00E-04  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-03",
                " 'Plasma Boundary exceeded Vacuum Grid Size         '",
                "  200   4.00E-06  2.00E-06  1.00E-06  1.00E-08  2.00E-08  3.00E-08  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-06",
            ]
        )
        + "\n"
    )

    payload = profile._partial_vmec2000_payload(workdir)

    assert payload["iteration_row_count"] == 2
    assert payload["last_row"]["it"] == 200
    assert payload["last_row"]["total"] == pytest.approx(7.0e-6)
    assert payload["min_total"] == pytest.approx(7.0e-6)
    assert payload["vacuum_grid_exceeded_count"] == 1
