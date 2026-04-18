#!/usr/bin/env python

"""Optimize a VMEC-JAX equilibrium for quasi-helical symmetry.

This standalone example mirrors the SIMSOPT fixed-resolution QH workflow, but
stays entirely inside vmec_jax. It uses the recovered exact discrete-adjoint
Jacobian path, not finite differences.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import vmec_jax as vj


max_nfev = 10  # Maximum number of function evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary
ftol = 1e-4  # Function tolerance for least-squares termination
gtol = 1e-4  # Gradient tolerance for least-squares termination
xtol = 1e-4  # Step tolerance for least-squares termination


def main() -> None:
    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)
    os.environ.setdefault("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1024")

    print("Running examples/optimization/qh_fixed_resolution_exact.py")
    print("==========================================================")

    filename = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"
    cfg, indata = vj.load_config(str(filename))
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes)

    def _last_array_value(key: str, scalar_key: str, default, cast):
        value = indata.get(key, None)
        if isinstance(value, list) and value:
            return cast(value[-1])
        return cast(indata.get(scalar_key, default))

    inner_max_iter = int(_last_array_value("NITER_ARRAY", "NITER", 1500, int))
    inner_ftol = float(_last_array_value("FTOL_ARRAY", "FTOL", 1e-13, float))
    step_size = float(indata.get_float("DELT", 1.0))

    # Define parameter space:
    specs = vj.boundary_param_specs(
        boundary,
        static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    params0 = jnp.zeros((len(specs),), dtype=jnp.float64)
    print("Parameter space:", vj.boundary_param_names(specs))

    state_guess0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess0, static).sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    layout = state_guess0.layout

    def residuals_from_state(state):
        aspect = vj.equilibrium_aspect_ratio_from_state(state=state, static=static)
        qs = vj.quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=np.arange(0, 1.01, 0.1),
            helicity_m=1,
            helicity_n=-1,
        )
        aspect_residual = jnp.asarray([aspect - 7.0], dtype=jnp.float64)
        return jnp.concatenate([aspect_residual, jnp.asarray(qs["residuals1d"], dtype=jnp.float64)])

    def total_from_residual(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        return float(np.dot(residual, residual))

    def _boundary_from_params(params):
        return vj.apply_boundary_params(boundary, specs, jnp.asarray(params, dtype=jnp.float64))

    def solve_forward_state(params):
        boundary_now = _boundary_from_params(params)
        state0 = initial_guess_from_boundary(static, boundary_now, indata, vmec_project=True)
        result = vj.solve_fixed_boundary_residual_iter(
            state0,
            static,
            indata=indata,
            signgs=signgs,
            ftol=inner_ftol,
            max_iter=inner_max_iter,
            step_size=step_size,
            vmec2000_control=True,
            reference_mode=False,
            backtracking=True,
            limit_dt_from_force=True,
            limit_update_rms=True,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces=True,
            use_scan=False,
            light_history=True,
            resume_state_mode="full",
        )
        return result.state

    def solve_exact_state(params, *, return_payload: bool = False):
        boundary_now = _boundary_from_params(params)
        state0 = initial_guess_from_boundary(static, boundary_now, indata, vmec_project=True)
        axis_override = extract_axis_override_from_state(state0, static)
        tape = vj.build_residual_checkpoint_tape_direct(
            state0,
            static,
            max_iter=inner_max_iter,
            solver_kwargs=dict(
                indata=indata,
                signgs=signgs,
                ftol=inner_ftol,
                step_size=step_size,
                vmec2000_control=True,
                reference_mode=False,
                backtracking=True,
                limit_dt_from_force=True,
                limit_update_rms=True,
                verbose=False,
                verbose_vmec2000_table=False,
                jit_forces=True,
                use_scan=False,
                light_history=True,
                resume_state_mode="full",
            ),
            indata=indata,
            signgs=signgs,
            ftol=inner_ftol,
            step_size=step_size,
            light_history=True,
            store_trace=False,
            store_full_step_traces=False,
        )
        state = unpack_state(jnp.asarray(tape.final_packed_state, dtype=jnp.float64), layout)
        if return_payload:
            return state, {"tape": tape, "axis_override": axis_override}
        return state

    def residual_fun(params):
        state = solve_exact_state(params)
        return np.asarray(residuals_from_state(state), dtype=float)

    def forward_residual_fun(params):
        state = solve_forward_state(params)
        return np.asarray(residuals_from_state(state), dtype=float)

    def jacobian_fun(params):
        params = jnp.asarray(params, dtype=jnp.float64)
        state, payload = solve_exact_state(params, return_payload=True)
        packed_final = jnp.asarray(pack_state(state), dtype=jnp.float64)

        def _initial_state_packed(p):
            boundary_now = _boundary_from_params(p)
            state0 = initial_guess_from_boundary(
                static,
                boundary_now,
                indata,
                vmec_project=True,
                axis_override=payload["axis_override"],
            )
            return jnp.asarray(pack_state(state0), dtype=jnp.float64)

        def _residuals_from_packed(packed):
            return residuals_from_state(unpack_state(packed, layout))

        directions = jnp.eye(int(params.size), dtype=params.dtype)
        _, initial_linear = jax.linearize(_initial_state_packed, params)
        initial_tangents = jax.vmap(initial_linear)(directions)
        final_tangents = vj.checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
        )
        _, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
        columns = jax.vmap(residual_linear)(final_tangents)
        return np.asarray(columns, dtype=float).T

    residual0 = residual_fun(params0)
    state0 = solve_forward_state(params0)
    qs0 = vj.quasisymmetry_ratio_residual_from_state(
        state=state0,
        static=static,
        indata=indata,
        signgs=signgs,
        flux_local=flux,
        prof_local={"pressure": pressure},
        pressure_local=pressure,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )

    print("Quasisymmetry objective before optimization:", float(np.asarray(qs0["total"])))
    print("Total objective before optimization:", total_from_residual(residual0))

    result = vj.gauss_newton_least_squares(
        residual_fun,
        jacobian_fun,
        np.asarray(params0, dtype=float),
        forward_residual_fun=forward_residual_fun,
        max_nfev=max_nfev,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        verbose=1,
    )

    state = solve_forward_state(result["x"])
    residual = np.asarray(residuals_from_state(state), dtype=float)
    qs = vj.quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        flux_local=flux,
        prof_local={"pressure": pressure},
        pressure_local=pressure,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )

    print("Final aspect ratio:", float(np.asarray(vj.equilibrium_aspect_ratio_from_state(state=state, static=static))))
    print("Quasisymmetry objective after optimization:", float(np.asarray(qs["total"])))
    print("Total objective after optimization:", total_from_residual(residual))
    print("End of examples/optimization/qh_fixed_resolution_exact.py")
    print("=======================================================")


if __name__ == "__main__":
    main()
