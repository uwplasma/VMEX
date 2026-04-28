#!/usr/bin/env python
"""Quasi-poloidal symmetry optimization with vmec_jax.

QP is still a quasisymmetry problem: ``HELICITY_M = 0`` targets
``|B| ~ B(N zeta)``.  This standalone script follows the same linear
SIMSOPT-style workflow as the QA/QH examples.
"""

from pathlib import Path

import numpy as np

try:
    from fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        abs_mean_iota_floor_objective,
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
        abs_mean_iota_floor_objective,
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

VMEC_MPOL = 5
VMEC_NTOR = 5
MAX_MODE = 3

MAX_NFEV = 20
CONTINUATION_NFEV = 0
USE_MODE_CONTINUATION = False

METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-4

# QP remains exploratory; bounded VMEC budgets avoid spending minutes on poor
# rejected trial points from the QH seed.
INNER_MAX_ITER = 80
INNER_FTOL = 1.0e-8
TRIAL_MAX_ITER = 80
TRIAL_FTOL = 1.0e-8
SOLVER_DEVICE = None

HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 7.0
TARGET_ABS_IOTA_MIN = 0.31
TARGET_IOTA_DISPLAY = -TARGET_ABS_IOTA_MIN

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 20.0
QS_WEIGHT = 1.0

USE_ESS = True
ALPHA = 2.5

OUTPUT_DIR = Path(f"results/qp_opt/n{HELICITY_N:+d}/mode{MAX_MODE}/{'ess' if USE_ESS else 'no_ess'}")


# ── Objective function ───────────────────────────────────────────────────────
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # ObjectiveTerm("custom_vector", lambda ctx, state: your_vector(ctx, state), target=0.0, weight=0.1),
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
    target_iota=TARGET_IOTA_DISPLAY,
    label=f"QP opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
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
