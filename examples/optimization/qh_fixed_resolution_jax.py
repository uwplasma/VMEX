#!/usr/bin/env python
"""Quasi-helical symmetry optimization with vmec_jax.

This is intentionally written like the SIMSOPT fixed-resolution examples:
edit top-level parameters, build the objective list, build the VMEC problem,
run the optimizer, then save and plot outputs.  There is no argparse and no
``main()`` wrapper.
"""

from pathlib import Path

import numpy as np

try:
    from fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        build_qs_stage,
        combine_stage_histories,
        load_qs_input,
        print_final_summary,
        print_problem_summary,
        quasisymmetry_objective,
        run_qs_stage,
        save_final_outputs,
        save_stage_artifacts,
        stage_budget,
        stage_mode_sequence,
        stage_params_from_previous,
    )
except ModuleNotFoundError:
    from examples.optimization.fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        build_qs_stage,
        combine_stage_histories,
        load_qs_input,
        print_final_summary,
        print_problem_summary,
        quasisymmetry_objective,
        run_qs_stage,
        save_final_outputs,
        save_stage_artifacts,
        stage_budget,
        stage_mode_sequence,
        stage_params_from_previous,
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ── User parameters ──────────────────────────────────────────────────────────
INPUT_FILE = DATA_DIR / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path("results/qh_opt")

VMEC_MPOL = 5
VMEC_NTOR = 5
MAX_MODE = 1

MAX_NFEV = 15
CONTINUATION_NFEV = 10
USE_MODE_CONTINUATION = True

METHOD = "scipy"  # "scipy", "gauss_newton", "lbfgs_adjoint", or "scipy_matrix_free"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-3
GTOL = 1.0e-3
XTOL = 1.0e-3

# 0 means use NITER/FTOL from the VMEC input deck for accepted exact points.
INNER_MAX_ITER = 0
INNER_FTOL = 0.0
TRIAL_MAX_ITER = 300
TRIAL_FTOL = 1.0e-10
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend

HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 7.0
ASPECT_WEIGHT = 1.0
QS_WEIGHT = 1.0

USE_ESS = False
ALPHA = 2.5


# ── Objective function ───────────────────────────────────────────────────────
# Add an objective by appending another ObjectiveTerm.  The callback receives
# (ctx, state) and returns a scalar or vector; vmec_jax minimizes
# weight * (value - target) in least-squares form.
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # ObjectiveTerm("major_radius", lambda ctx, state: state.rmncc[0, 0], target=1.0, weight=0.1),
]


# ── Problem setup ────────────────────────────────────────────────────────────
RUN = FixedBoundaryQSConfig(
    input_file=INPUT_FILE,
    output_dir=OUTPUT_DIR,
    vmec_mpol=VMEC_MPOL,
    vmec_ntor=VMEC_NTOR,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    use_mode_continuation=USE_MODE_CONTINUATION,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    method=METHOD,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    target_aspect=TARGET_ASPECT,
    label=f"QH opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
)

cfg, indata = load_qs_input(INPUT_FILE, vmec_mpol=VMEC_MPOL, vmec_ntor=VMEC_NTOR)
stage_modes = stage_mode_sequence(RUN)


# ── Optimization ─────────────────────────────────────────────────────────────
stage_records = []
params_stage = None
prev_specs = None

for stage_mode in stage_modes:
    stage = build_qs_stage(RUN, cfg, indata, stage_mode, OBJECTIVES)
    params0 = stage_params_from_previous(stage, params_stage=params_stage, prev_specs=prev_specs)
    nfev = stage_budget(RUN, stage_mode)

    if stage_mode == MAX_MODE:
        print_problem_summary(RUN, OBJECTIVES, stage, params0)
    else:
        print(f"Stage {stage_mode} -> {stage_mode + 1} continuation seed (budget={nfev}) ...")

    result = run_qs_stage(
        RUN,
        stage,
        params0,
        nfev=nfev,
        verbose=1 if stage_mode == MAX_MODE else 0,
    )
    save_stage_artifacts(
        RUN,
        OUTPUT_DIR / f"stage_{stage_mode:02d}",
        stage.optimizer,
        params0,
        result["x"],
        result,
    )
    stage_records.append((stage_mode, stage, params0, result))
    prev_specs = stage.ctx.specs
    params_stage = result["x"]


# ── Output ───────────────────────────────────────────────────────────────────
final_stage = stage_records[-1][1]
final_result = stage_records[-1][3]
combined_history = combine_stage_histories(RUN, stage_modes, stage_records)
if combined_history is not None:
    final_result["_history_dump"] = combined_history

print_final_summary(RUN, final_result)
save_final_outputs(RUN, stage_records, final_stage, final_result)
