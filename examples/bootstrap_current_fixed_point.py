#!/usr/bin/env python
"""Compute a self-consistent Redl bootstrap-current VMEC input profile.

This example is intentionally explicit: the user selects the finite-beta
profiles, VMEC input, fixed-point controls, solver controls, output files, and
then calls ``vj.bootstrap_current_fixed_point``.  The plasma boundary is not an
optimization variable here; the loop only updates VMEC's current profile from
the Redl bootstrap-current formula.

Run from the repository root:

    PYTHONPATH=. python examples/bootstrap_current_fixed_point.py
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import vmec_jax as vj
from vmec_jax.driver import write_wout_from_fixed_boundary_run


REPO_ROOT = Path(__file__).resolve().parents[1]

# Physics/profile controls.  For stellarator studies set HELICITY_N to the
# target quasisymmetry helicity used by the Redl formula.
INPUT_PATH = REPO_ROOT / "examples" / "data" / "input.shaped_tokamak_pressure"
BETA_PERCENT = 1.0
HELICITY_N = 0
REDL_SURFACES = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
N_CURRENT = 32

# Fixed-point controls.  `integrating_factor` is the most faithful deterministic
# update; `low_beta` and `lagged_pressure` are useful diagnostics.
FIXED_POINT_OPTIONS = vj.BootstrapCurrentOptions(
    helicity_n=HELICITY_N,
    surfaces=REDL_SURFACES,
    n_current=N_CURRENT,
    policy="integrating_factor",  # alternatives: "low_beta", "lagged_pressure"
    damping=0.5,  # set closer to 1.0 for aggressive Picard updates
    max_fixed_point_iter=3,
    mismatch_tol=1.0e-2,
    current_tol=1.0e-2,
)

# VMEC solve controls for each fixed-point stage.
VMEC_RUN_KWARGS = {
    "max_iter": 250,
    "multigrid": False,
    "verbose": False,
    "jit_forces": "auto",
    "solver_device": None,  # use "cpu" or "gpu" to force a backend
}

RESULTS_DIR = REPO_ROOT / "results" / "bootstrap_current_fixed_point"
FINAL_INPUT = RESULTS_DIR / "input.bootstrap_current_final"
FINAL_WOUT = RESULTS_DIR / "wout_bootstrap_current_final.nc"
HISTORY_JSON = RESULTS_DIR / "history.json"


RESULTS_DIR.mkdir(parents=True, exist_ok=True)

base_indata = vj.read_indata(INPUT_PATH)
profiles = vj.standard_finite_beta_profiles(BETA_PERCENT)
indata = vj.with_pressure_profile(base_indata, profiles.pressure_pa, pres_scale=1.0)

result = vj.bootstrap_current_fixed_point(
    indata,
    options=FIXED_POINT_OPTIONS,
    ne_coeffs=profiles.ne_coeffs,
    Te_coeffs=profiles.Te_coeffs,
    Ti_coeffs=profiles.Ti_coeffs,
    Zeff_coeffs=profiles.Zeff_coeffs,
    run_kwargs=VMEC_RUN_KWARGS,
)

vj.write_indata(FINAL_INPUT, result.indata)
if result.last_run is not None:
    write_wout_from_fixed_boundary_run(FINAL_WOUT, result.last_run, include_fsq=True)

history = [asdict(item) for item in result.history]
HISTORY_JSON.write_text(
    json.dumps(
        {
            "input": str(INPUT_PATH),
            "final_input": str(FINAL_INPUT),
            "final_wout": str(FINAL_WOUT),
            "beta_percent": BETA_PERCENT,
            "helicity_n": HELICITY_N,
            "converged": result.converged,
            "reason": result.reason,
            "history": history,
        },
        indent=2,
    )
    + "\n"
)

last = result.history[-1] if result.history else None
print(f"Wrote final input: {FINAL_INPUT}")
if result.last_run is not None:
    print(f"Wrote final WOUT:  {FINAL_WOUT}")
print(f"Wrote history:     {HISTORY_JSON}")
print(f"Converged:         {result.converged} ({result.reason})")
if last is not None:
    print(f"Iterations:        {last.iteration}")
    print(f"CURTOR:            {last.curtor:.6e}")
    print(f"Mismatch norm:     {last.mismatch_norm:.6e}")
    print(f"Current update:    {last.current_update_norm:.6e}")
