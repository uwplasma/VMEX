"""Direct-coil free-boundary workflow helpers for examples and tests.

This module owns the lightweight pieces needed to build a direct-coil
free-boundary optimization loop: simple coil providers, optimizer-coordinate
packing, complete-solve execution, JSON-safe summaries, and scalar objective
terms.  It deliberately does not contain same-branch derivative reports or
custom-VJP promotion gates; those remain in ``coil_optimization``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.external_fields import CoilFieldParams
from vmec_jax.external_fields.coils_jax import coil_current_norm, coil_lengths
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

__all__ = [
    "apply_coil_variables",
    "array_history",
    "coil_diagnostics",
    "direct_coil_optimization_workflow_metadata",
    "direct_coil_qs_summary_configs",
    "float_or_none",
    "json_safe_payload",
    "make_circle_provider",
    "make_free_boundary_indata",
    "objective_from_summary",
    "objective_terms_from_summary",
    "parse_float_list",
    "run_direct_free_boundary",
    "select_coil_variables",
    "summarize_run",
    "variable_records",
    "write_json",
]


def _json_default(value: Any) -> Any:
    """Encode common NumPy/path values for diagnostic JSON artifacts."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def json_safe_payload(value: Any) -> Any:
    """Return a JSON-native copy using the same encoding as report files."""

    return json.loads(json.dumps(value, default=_json_default))


def write_json(path: Path, data: Any) -> None:
    """Write an indented JSON diagnostic artifact with VMEC-friendly encoders."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def make_circle_provider(
    *,
    current_scale: float,
    chunk_size: int | None = None,
    current: float = 2.0,
    radius: float = 1.4,
    n_segments: int = 96,
    nfp: int = 1,
    stellsym: bool = False,
) -> tuple[CoilFieldParams, dict[str, Any]]:
    """Return a synthetic circular direct-coil provider for smoke examples.

    The curve has two nonzero Fourier coefficients, so examples can exercise
    both current and coil-shape degrees of freedom without shipping external
    coil assets.
    """

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(float(radius))
    dofs = dofs.at[0, 1, 1].set(float(radius))
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([float(current)]),
        n_segments=int(n_segments),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        current_scale=float(current_scale),
        chunk_size=None if chunk_size is None else int(chunk_size),
    )
    return params, {
        "provider": "circle",
        "current": float(current),
        "radius": float(radius),
        "n_segments": int(n_segments),
        "nfp": int(nfp),
        "stellsym": bool(stellsym),
        "current_scale_multiplier": float(current_scale),
        "chunk_size": None if chunk_size is None else int(chunk_size),
    }


def make_free_boundary_indata(
    input_path: Path,
    output_path: Path,
    *,
    vmec_max_iter: int,
    ftol: float,
    ns: int,
    mpol: int,
    ntor: int,
    nzeta: int,
    beta_percent: float,
    pressure_profile: str,
    pressure_scale: float,
    phiedge: float,
) -> Path:
    """Write a small direct-coil free-boundary VMEC input deck for examples."""

    from vmec_jax.namelist import read_indata, write_indata
    from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles

    indata = deepcopy(read_indata(input_path))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": [int(ns)],
            "NITER_ARRAY": [int(vmec_max_iter)],
            "FTOL_ARRAY": [float(ftol)],
            "NITER": int(vmec_max_iter),
            "FTOL": float(ftol),
            "PHIEDGE": float(phiedge),
            "MPOL": int(mpol),
            "NTOR": int(ntor),
            "NZETA": int(nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(nzeta)),
        }
    )
    pressure_profile = str(pressure_profile).strip().lower()
    if pressure_profile == "standard":
        profiles = standard_finite_beta_profiles(float(beta_percent))
        am, pres_scale = pressure_profile_to_vmec_am(profiles.pressure_pa, pres_scale=1.0)
        indata.scalars["PMASS_TYPE"] = "power_series"
        indata.scalars["PRES_SCALE"] = pres_scale
        indata.scalars["AM"] = am
    elif pressure_profile in {"linear", "linear-scale", "legacy"}:
        indata.scalars["PMASS_TYPE"] = "power_series"
        indata.scalars["PRES_SCALE"] = float(pressure_scale)
        indata.scalars["AM"] = [1.0, -1.0]
    else:
        raise ValueError("pressure_profile must be 'standard' or 'linear-scale'")
    write_indata(output_path, indata)
    return output_path


def select_coil_variables(
    params: CoilFieldParams,
    *,
    max_current_vars: int,
    max_fourier_vars: int,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...]]]]:
    """Select simple current and nonzero Fourier DOFs for coil-only examples."""

    base_currents = np.asarray(params.base_currents, dtype=float)
    base_dofs = np.asarray(params.base_curve_dofs, dtype=float)
    variables: list[tuple[str, tuple[int, ...]]] = []

    for i in range(min(int(max_current_vars), base_currents.size)):
        variables.append(("current", (i,)))

    if max_fourier_vars > 0:
        nonzero_dofs = np.argwhere(np.abs(base_dofs) > 0.0)
        dof_indices = nonzero_dofs[: int(max_fourier_vars)]
        for index in dof_indices:
            variables.append(("fourier_dof", tuple(int(i) for i in index)))

    return np.zeros(len(variables), dtype=float), variables


def apply_coil_variables(
    base_params: CoilFieldParams,
    x: np.ndarray,
    variables: list[tuple[str, tuple[int, ...]]],
    *,
    current_step: float,
    dof_step: float,
) -> CoilFieldParams:
    """Apply optimizer coordinates to coil currents and Fourier coefficients."""

    currents = np.asarray(base_params.base_currents, dtype=float).copy()
    dofs = np.asarray(base_params.base_curve_dofs, dtype=float).copy()

    for value, (kind, index) in zip(np.asarray(x, dtype=float), variables, strict=True):
        if kind == "current":
            i = index[0]
            currents[i] *= 1.0 + float(current_step) * float(value)
        elif kind == "fourier_dof":
            dofs[index] += float(dof_step) * float(value)
        else:  # pragma: no cover - defensive programming for future variable kinds.
            raise ValueError(f"unknown coil variable kind {kind!r}")

    return base_params.with_arrays(base_curve_dofs=jnp.asarray(dofs), base_currents=jnp.asarray(currents))


def coil_diagnostics(params: CoilFieldParams) -> dict[str, Any]:
    """Return compact current/geometry diagnostics for a direct-coil provider."""

    lengths = np.asarray(coil_lengths(params), dtype=float).reshape(-1)
    currents = np.asarray(params.base_currents, dtype=float).reshape(-1)
    dofs = np.asarray(params.base_curve_dofs, dtype=float)
    return {
        "n_base_coils": int(currents.size),
        "n_segments": int(params.n_segments),
        "nfp": int(params.nfp),
        "stellsym": bool(params.stellsym),
        "current_scale": float(params.current_scale),
        "current_min": float(np.min(currents)) if currents.size else None,
        "current_max": float(np.max(currents)) if currents.size else None,
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "mean_coil_length": float(np.mean(lengths)) if lengths.size else None,
        "min_coil_length": float(np.min(lengths)) if lengths.size else None,
        "max_coil_length": float(np.max(lengths)) if lengths.size else None,
        "base_curve_dofs_shape": [int(v) for v in dofs.shape],
        "nonzero_base_curve_dofs": int(np.count_nonzero(np.abs(dofs) > 0.0)),
    }


def variable_records(
    variables: list[tuple[str, tuple[int, ...]]],
    base_params: CoilFieldParams,
    *,
    current_step: float,
    dof_step: float,
) -> list[dict[str, Any]]:
    """Describe optimizer coordinates in physical coil units."""

    currents = np.asarray(base_params.base_currents, dtype=float)
    dofs = np.asarray(base_params.base_curve_dofs, dtype=float)
    records: list[dict[str, Any]] = []
    for kind, index in variables:
        record: dict[str, Any] = {"kind": kind, "index": index}
        if kind == "current":
            i = index[0]
            record.update(
                {
                    "base_value": float(currents[i]),
                    "parameterization": "multiplicative",
                    "unit_x_delta": float(currents[i]) * float(current_step),
                    "current_step_fraction": float(current_step),
                }
            )
        elif kind == "fourier_dof":
            record.update(
                {
                    "base_value": float(dofs[index]),
                    "parameterization": "additive",
                    "unit_x_delta": float(dof_step),
                    "dof_step": float(dof_step),
                }
            )
        else:  # pragma: no cover - defensive programming for future variable kinds.
            record["parameterization"] = "unknown"
        records.append(record)
    return records


def float_or_none(value: Any) -> float | None:
    """Convert a scalar-like value to a finite float, or return ``None``."""

    if value is None:
        return None
    try:
        result = float(np.asarray(value))
    except Exception:
        return None
    return result if np.isfinite(result) else None


def array_history(value: Any) -> list[float]:
    """Return a one-dimensional float history from a result payload."""

    if value is None:
        return []
    try:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]
    except Exception:
        return []


def run_direct_free_boundary(
    input_path: Path,
    params: CoilFieldParams,
    *,
    vmec_max_iter: int,
    activate_fsq: float,
    jit_forces: bool = True,
) -> tuple[Any, float]:
    """Run one complete direct-coil free-boundary solve and return wall time."""

    from vmec_jax.driver import run_free_boundary

    start = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(vmec_max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(jit_forces),
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=float(activate_fsq),
    )
    return run, time.perf_counter() - start


def summarize_run(
    run: Any,
    params: CoilFieldParams,
    *,
    objective: float,
    wall_s: float,
    target_aspect: float,
    target_iota: float,
    helicity_m: int = 1,
    helicity_n: int = 0,
    qs_surfaces: list[float] | None = None,
    qs_ntheta: int = 31,
    qs_nphi: int = 32,
) -> dict[str, Any]:
    """Summarize a complete direct-coil run into objective-ready scalars."""

    qs_surfaces = [0.25, 0.5, 0.75] if qs_surfaces is None else qs_surfaces
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
    fsqr = float_or_none(diag.get("final_fsqr"))
    fsqz = float_or_none(diag.get("final_fsqz"))
    fsql = float_or_none(diag.get("final_fsql"))
    residual_proxy = sum(value for value in (fsqr, fsqz, fsql) if value is not None)
    result = run.result

    aspect = None
    mean_iota = None
    try:
        aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static)))
    except Exception:
        pass
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        iota_arr = np.asarray(iotas, dtype=float)
        mean_iota = float(np.nanmean(iota_arr[1:] if iota_arr.size > 1 else iota_arr))
    except Exception:
        pass
    qs_total = None
    try:
        qs = quasisymmetry_ratio_residual_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            surfaces=qs_surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            ntheta=int(qs_ntheta),
            nphi=int(qs_nphi),
        )
        qs_total = float(np.asarray(qs["total"]))
    except Exception:
        pass

    return {
        "objective": float(objective),
        "wall_s": float(wall_s),
        "vmec_n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "residual_proxy": float(residual_proxy),
        "aspect": aspect,
        "target_aspect": float(target_aspect),
        "mean_iota": mean_iota,
        "target_iota": float(target_iota),
        "qs_total": qs_total,
        "qs_helicity_m": int(helicity_m),
        "qs_helicity_n": int(helicity_n),
        "qs_surfaces": [float(value) for value in qs_surfaces],
        "qs_ntheta": int(qs_ntheta),
        "qs_nphi": int(qs_nphi),
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "mean_coil_length": float(np.mean(np.asarray(coil_lengths(params), dtype=float))),
        "free_boundary_vacuum_stub": freeb.get("vacuum_stub") if isinstance(freeb, dict) else None,
        "free_boundary_nestor_model": freeb.get("nestor_model") if isinstance(freeb, dict) else None,
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
        "vmec_history": {
            "w": array_history(getattr(result, "w_history", None)),
            "fsqr2": array_history(getattr(result, "fsqr2_history", None)),
            "fsqz2": array_history(getattr(result, "fsqz2_history", None)),
            "fsql2": array_history(getattr(result, "fsql2_history", None)),
        },
    }


def objective_from_summary(
    summary: dict[str, Any],
    *,
    residual_weight: float,
    aspect_weight: float,
    iota_weight: float,
    qs_weight: float = 0.0,
) -> float:
    """Return the weighted scalar objective used by direct-coil examples."""

    return float(
        objective_terms_from_summary(
            summary,
            residual_weight=residual_weight,
            qs_weight=qs_weight,
            aspect_weight=aspect_weight,
            iota_weight=iota_weight,
        )["total"]
    )


def objective_terms_from_summary(
    summary: dict[str, Any],
    *,
    residual_weight: float,
    aspect_weight: float,
    iota_weight: float,
    qs_weight: float = 0.0,
) -> dict[str, Any]:
    """Return weighted residual/QS/aspect/iota objective contributions."""

    residual = float(summary.get("residual_proxy") or 0.0)
    qs_total = summary.get("qs_total")
    aspect = summary.get("aspect")
    mean_iota = summary.get("mean_iota")
    aspect_error = None if aspect is None else float(aspect) - float(summary["target_aspect"])
    iota_error = None if mean_iota is None else float(mean_iota) - float(summary["target_iota"])
    qs_penalty = 0.0 if qs_total is None else float(qs_total)
    aspect_penalty = 0.0 if aspect_error is None else aspect_error**2
    iota_penalty = 0.0 if iota_error is None else iota_error**2
    residual_term = float(residual_weight) * residual
    qs_term = float(qs_weight) * qs_penalty
    aspect_term = float(aspect_weight) * aspect_penalty
    iota_term = float(iota_weight) * iota_penalty
    missing_terms = []
    if qs_total is None and float(qs_weight) != 0.0:
        missing_terms.append("qs_total")
    if aspect is None and float(aspect_weight) != 0.0:
        missing_terms.append("aspect")
    if mean_iota is None and float(iota_weight) != 0.0:
        missing_terms.append("mean_iota")
    return {
        "total": float(residual_term + qs_term + aspect_term + iota_term),
        "residual": {
            "value": residual,
            "weight": float(residual_weight),
            "contribution": float(residual_term),
        },
        "quasisymmetry": {
            "value": None if qs_total is None else float(qs_total),
            "target": 0.0,
            "weight": float(qs_weight),
            "contribution": float(qs_term),
            "helicity_m": int(summary.get("qs_helicity_m", 1)),
            "helicity_n": int(summary.get("qs_helicity_n", 0)),
            "surfaces": [float(value) for value in summary.get("qs_surfaces", [])],
        },
        "aspect": {
            "value": None if aspect is None else float(aspect),
            "target": float(summary["target_aspect"]),
            "error": aspect_error,
            "squared_error": float(aspect_penalty),
            "weight": float(aspect_weight),
            "contribution": float(aspect_term),
        },
        "mean_iota": {
            "value": None if mean_iota is None else float(mean_iota),
            "target": float(summary["target_iota"]),
            "error": iota_error,
            "squared_error": float(iota_penalty),
            "weight": float(iota_weight),
            "contribution": float(iota_term),
        },
        "missing_unweighted_terms": missing_terms,
    }


def direct_coil_optimization_workflow_metadata(repo_root: Any) -> dict[str, Any]:
    """Return the pedagogic workflow contract recorded in summary artifacts."""

    return {
        "flow": "single_stage_direct_coil_no_mgrid",
        "field_backend": "direct_coils",
        "workflow_steps": [
            "load or synthesize direct coils",
            "select coil-current and coil-Fourier optimization variables",
            "write VMEC input with MGRID_FILE='DIRECT_COILS'",
            "run complete free-boundary solves with direct JAX Biot-Savart sampling",
            "score VMEC residual, VMEC-state QS residual, aspect, and mean-iota terms",
        ],
        "optimized_dofs": "coil currents and selected coil Fourier coefficients only",
        "plasma_boundary_optimized": False,
        "python_provider_required": True,
        "uses_mgrid_file": False,
        "mgrid_compatibility_example": str(repo_root / "examples" / "free_boundary_essos_mgrid_forward.py"),
        "vmec_input_replay": (
            "MGRID_FILE='DIRECT_COILS' is a vmec_jax Python-provider tag. "
            "Run this optimization script, or call run_free_boundary with CoilFieldParams, "
            "so the solver receives the direct-coil provider."
        ),
    }


def direct_coil_qs_summary_configs(
    args: Any,
    *,
    input_path: Any,
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return objective, VMEC, and optimizer summary configs for direct-coil QS examples."""

    objective_model = {
        "description": "Deterministic direct-coil free-boundary objective with VMEC residual, QS, aspect, and iota terms.",
        "qs_note": (
            "The QS term is evaluated from the accepted VMEC state. Full coil-to-Boozer/QS exact "
            "gradients through adaptive free-boundary branch selection remain a separate promotion gate."
        ),
        "helicity_m": int(args.helicity_m),
        "helicity_n": int(args.helicity_n),
        "qs_surfaces": parse_float_list(str(args.qs_surfaces)),
        "qs_ntheta": int(args.qs_ntheta),
        "qs_nphi": int(args.qs_nphi),
        "target_aspect": float(args.target_aspect),
        "target_iota": float(args.target_iota),
        "residual_weight": float(args.residual_weight),
        "qs_weight": float(args.qs_weight),
        "aspect_weight": float(args.aspect_weight),
        "iota_weight": float(args.iota_weight),
        "failure_objective": float(args.failure_objective),
    }
    vmec_config = {
        "input_template": args.input,
        "generated_input": input_path,
        "external_field_provider_kind": "direct_coils",
        "mgrid_file": "DIRECT_COILS",
        "uses_generated_mgrid": False,
        "python_provider_required": True,
        "uses_mgrid_file": False,
        "vmec_input_replay": workflow["vmec_input_replay"],
        "mgrid_compatibility_example": workflow["mgrid_compatibility_example"],
        "vmec_max_iter": int(args.vmec_max_iter),
        "ftol": float(args.ftol),
        "ns": int(args.ns),
        "mpol": int(args.mpol),
        "ntor": int(args.ntor),
        "nzeta": int(args.nzeta),
        "beta_percent": float(args.beta),
        "pressure_profile": str(args.pressure_profile),
        "pressure_scale": float(args.pressure_scale),
        "phiedge": float(args.phiedge),
        "activate_fsq": float(args.activate_fsq),
        "jit_forces": bool(args.jit_forces),
    }
    optimizer_config = {"method": "Powell", "max_iter": int(args.max_iter), "max_evals": int(args.max_evals),
                        "xtol": float(args.xtol), "ftol": float(args.optimizer_ftol)}
    return objective_model, vmec_config, optimizer_config


def parse_float_list(text: str) -> list[float]:
    """Parse comma/space-separated floats from a small CLI option."""
    values = [float(part) for part in str(text).replace(",", " ").split() if part]
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


