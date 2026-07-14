"""Public API for straight-axis magnetic-mirror equilibria.

The mirror backend uses a nonperiodic axial coordinate and is separate from
toroidal VMEC.  Its public surface is intentionally small: inputs and state,
fixed/free-boundary solves, continuation, implicit derivatives, MOUT I/O, and
plotting.  Numerical kernels remain available from their owning submodules.
"""

from importlib import import_module as _import_module


# Public names are lazy so importing ``vmec_jax.mirror`` does not initialize
# every exterior-vacuum and solver dependency.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Model and configuration.
    "MIRROR_INPUT_SCHEMA": (".model", "MIRROR_INPUT_SCHEMA"),
    "MIRROR_OUTPUT_SCHEMA": (".model", "MIRROR_OUTPUT_SCHEMA"),
    "EndCondition": (".model", "EndCondition"),
    "MirrorBoundary": (".model", "MirrorBoundary"),
    "MirrorConfig": (".model", "MirrorConfig"),
    "MirrorResolution": (".model", "MirrorResolution"),
    "MirrorState": (".model", "MirrorState"),
    "PressureClosure": (".model", "PressureClosure"),
    "IsotropicPressureClosure": (".model", "IsotropicPressureClosure"),
    "BiMaxwellianPressureClosure": (".model", "BiMaxwellianPressureClosure"),
    "TabulatedPressureClosure": (".model", "TabulatedPressureClosure"),
    "project_fixed_boundary_state": (".model", "project_fixed_boundary_state"),
    # Fixed and free-boundary solves.
    "MirrorConvergenceError": (".solver", "MirrorConvergenceError"),
    "MirrorSolveResult": (".solver", "MirrorSolveResult"),
    "solve_fixed_boundary_cli": (".solver", "solve_fixed_boundary_cli"),
    "solve_anisotropic_fixed_boundary_cli": (
        ".solver",
        "solve_anisotropic_fixed_boundary_cli",
    ),
    "FreeBoundaryMirrorResult": (".free_boundary", "FreeBoundaryMirrorResult"),
    "solve_axisymmetric_free_boundary_cli": (
        ".free_boundary",
        "solve_axisymmetric_free_boundary_cli",
    ),
    "solve_free_boundary_cli": (".free_boundary", "solve_free_boundary_cli"),
    "build_vacuum_grid": (".vacuum", "build_vacuum_grid"),
    # Continuation and compact restarts.
    "interpolate_fixed_boundary_state": (
        ".continuation",
        "interpolate_fixed_boundary_state",
    ),
    "solve_axisymmetric_beta_scan_cli": (
        ".continuation",
        "solve_axisymmetric_beta_scan_cli",
    ),
    "solve_beta_scan_cli": (".continuation", "solve_beta_scan_cli"),
    "FreeBoundaryRestart": (".restart", "FreeBoundaryRestart"),
    "load_free_boundary_restart": (".restart", "load_free_boundary_restart"),
    "save_free_boundary_restart": (".restart", "save_free_boundary_restart"),
    # Supported diagnostics.
    "boundary_fourier_amplitudes": (
        ".diagnostics",
        "boundary_fourier_amplitudes",
    ),
    "boundary_fourier_norms": (".diagnostics", "boundary_fourier_norms"),
    "summarize_axisymmetric_beta_scan": (
        ".diagnostics",
        "summarize_axisymmetric_beta_scan",
    ),
    "summarize_nonaxisymmetric_beta_scan": (
        ".diagnostics",
        "summarize_nonaxisymmetric_beta_scan",
    ),
    # Implicit differentiation.
    "FixedBoundaryImplicitConfig": (
        ".implicit",
        "FixedBoundaryImplicitConfig",
    ),
    "FixedBoundaryParameters": (".implicit", "FixedBoundaryParameters"),
    "MirrorAdjointResult": (".implicit", "MirrorAdjointResult"),
    "SplineFixedBoundaryParameters": (
        ".implicit",
        "SplineFixedBoundaryParameters",
    ),
    "fixed_boundary_adjoint": (".implicit", "fixed_boundary_adjoint"),
    "fixed_boundary_parameters": (".implicit", "fixed_boundary_parameters"),
    "make_fixed_boundary_implicit_config": (
        ".implicit",
        "make_fixed_boundary_implicit_config",
    ),
    "solve_fixed_boundary_implicit": (
        ".implicit",
        "solve_fixed_boundary_implicit",
    ),
    "spline_fixed_boundary_adjoint": (
        ".implicit",
        "spline_fixed_boundary_adjoint",
    ),
    "spline_fixed_boundary_parameters": (
        ".implicit",
        "spline_fixed_boundary_parameters",
    ),
    "FreeBoundaryAdjointConfig": (
        ".free_boundary_implicit",
        "FreeBoundaryAdjointConfig",
    ),
    "FreeBoundaryAdjointResult": (
        ".free_boundary_implicit",
        "FreeBoundaryAdjointResult",
    ),
    "FreeBoundaryParameters": (
        ".free_boundary_implicit",
        "FreeBoundaryParameters",
    ),
    "free_boundary_adjoint": (
        ".free_boundary_implicit",
        "free_boundary_adjoint",
    ),
    "free_boundary_parameters": (
        ".free_boundary_implicit",
        "free_boundary_parameters",
    ),
    # MOUT and plots.
    "MoutData": (".output", "MoutData"),
    "mout_from_result": (".output", "mout_from_result"),
    "read_mout": (".output", "read_mout"),
    "write_mout": (".output", "write_mout"),
    "plot_mout": (".plotting", "plot_mout"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    entry = _LAZY_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = entry
    module = _import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
