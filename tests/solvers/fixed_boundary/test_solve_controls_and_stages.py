import numpy as np

from vmec_jax import solve
from vmec_jax.config import config_from_indata
from vmec_jax.namelist import InData
from vmec_jax.optimization_workflow import build_fixed_boundary_objective_stage
from vmec_jax.solve import (
    _resolve_cg_tol,
    _resolve_grad_tol,
    _resolve_lbfgs_curvature_tol,
    _resolve_lm_damping,
)


def test_axis_reset_dump_returns_false_when_filesystem_write_fails(monkeypatch, tmp_path):
    def fail_mkdir(self, *args, **kwargs):
        raise OSError("synthetic mkdir failure")

    monkeypatch.setattr(solve.Path, "mkdir", fail_mkdir)

    assert not solve._write_axis_reset_dump(
        axis_dump_dir=tmp_path / "axis",
        ns=3,
        ntor=1,
        used_state_guess=True,
        raxis_cc=np.asarray([1.0, 0.1]),
        raxis_cs=np.asarray([0.0, 0.0]),
        zaxis_cc=np.asarray([0.0, 0.0]),
        zaxis_cs=np.asarray([0.0, 0.2]),
    )


def test_scan_chunk_settings_wrapper_uses_runtime_backend_and_env(monkeypatch):
    monkeypatch.setattr(solve, "_scan_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "7")

    assert solve._scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
    ) == (7, True)


def test_axis_m0_mask_falls_back_to_modes_when_precomputed_mask_missing():
    static = type("Static", (), {"modes": type("Modes", (), {"m": np.asarray([0, 1, 0])})()})()

    np.testing.assert_allclose(np.asarray(solve._axis_m0_mask(static, dtype=np.float64)), [1.0, 0.0, 1.0])


def test_resolve_grad_tol_scales_with_initial_gradient():
    tol_small = _resolve_grad_tol(None, grad_rms0=1.0e-6, dtype=np.float64)
    tol_large = _resolve_grad_tol(None, grad_rms0=1.0, dtype=np.float64)

    assert tol_small > 0.0
    assert tol_large > tol_small
    assert np.isclose(tol_large / tol_small, 1.0e6, rtol=1.0e-12)


def test_resolve_lbfgs_curvature_tol_tracks_vector_scale():
    s1 = np.array([1.0, 2.0, 3.0])
    y1 = np.array([2.0, 4.0, 6.0])
    s2 = 10.0 * s1
    y2 = 10.0 * y1

    tol1 = _resolve_lbfgs_curvature_tol(s1, y1)
    tol2 = _resolve_lbfgs_curvature_tol(s2, y2)

    assert tol1 > 0.0
    assert np.isclose(tol2 / tol1, 100.0, rtol=1.0e-12)


def test_resolve_cg_tol_tightens_with_progress():
    tol0 = _resolve_cg_tol(None, current_obj=1.0, initial_obj=1.0, target_obj=1.0e-12, dtype=np.float64)
    tol1 = _resolve_cg_tol(None, current_obj=1.0e-6, initial_obj=1.0, target_obj=1.0e-12, dtype=np.float64)

    assert 0.0 < tol1 < tol0 < 1.0


def test_resolve_lm_damping_uses_curvature_scale():
    d1 = _resolve_lm_damping(None, curvature_scale=1.0, dtype=np.float64)
    d2 = _resolve_lm_damping(None, curvature_scale=1.0e6, dtype=np.float64)

    assert d1 > 0.0
    assert d2 > d1


def _with_extra_high_mode(indata):
    indexed = {name: dict(values) for name, values in indata.indexed.items()}
    indexed.setdefault("RBC", {})[(2, 0)] = 0.123
    indexed.setdefault("ZBS", {})[(2, 0)] = 0.456
    return InData(
        scalars=dict(indata.scalars),
        indexed=indexed,
        source_path=indata.source_path,
    )


def test_projected_continuation_stage_keeps_high_modes_zero(load_case_circular_tokamak):
    _cfg, base_indata, _static, _boundary, _state0 = load_case_circular_tokamak
    indata = _with_extra_high_mode(base_indata)

    stage1 = build_fixed_boundary_objective_stage(
        config_from_indata(indata),
        indata,
        stage_mode=1,
        objectives=[],
        project_input_boundary_to_max_mode=True,
        inner_max_iter=1,
        trial_max_iter=1,
    )

    assert (2, 0) not in stage1.ctx.indata.indexed.get("RBC", {})
    assert (2, 0) not in stage1.ctx.indata.indexed.get("ZBS", {})

    next_indata = stage1.optimizer._indata_from_params(np.zeros(len(stage1.specs)))
    assert next_indata.indexed["RBC"][(2, 0)] == 0.0
    assert next_indata.indexed["ZBS"][(2, 0)] == 0.0

    stage2 = build_fixed_boundary_objective_stage(
        config_from_indata(next_indata),
        next_indata,
        stage_mode=2,
        objectives=[],
        project_input_boundary_to_max_mode=True,
        inner_max_iter=1,
        trial_max_iter=1,
    )

    assert stage2.ctx.indata.indexed["RBC"][(2, 0)] == 0.0
    assert stage2.ctx.indata.indexed["ZBS"][(2, 0)] == 0.0
