"""Compare reduced mirror free-boundary vector LS derivative backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax._compat import enable_x64, jnp
from vmec_jax.mirror import mirror_free_boundary_residual_vector_least_squares_step


VECTOR_LS_BENCHMARK_SCHEMA = "mirror_free_boundary_vector_ls_benchmark"
VECTOR_LS_BENCHMARK_SCHEMA_VERSION = "0.1"
VECTOR_LS_BENCHMARK_ROW_FIELDS = (
    "name",
    "jacobian_backend",
    "jax_mode",
    "accepted",
    "line_search_factor",
    "residual_value_before",
    "residual_value_after",
    "residual_reduction_fraction",
    "predicted_value",
    "coefficient_error_before",
    "coefficient_error_after",
    "coefficient_error_reduction_fraction",
    "coefficients_initial",
    "coefficients_new",
    "raw_step",
    "limited_step",
    "jacobian_shape",
    "wall_time_s",
)


def validate_vector_ls_benchmark_metrics(metrics: dict[str, object]) -> None:
    """Validate the compact benchmark metrics JSON contract."""

    required = {
        "metrics_schema",
        "metrics_schema_version",
        "nxi",
        "slope_weight",
        "finite_difference_step",
        "max_relative_step",
        "xi",
        "target_coefficients",
        "initial_coefficients",
        "target_radius",
        "initial_radius",
        "rows",
        "best_backend_by_residual",
        "best_backend_by_time",
        "figures",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"missing vector LS benchmark fields: {sorted(missing)}")
    if metrics["metrics_schema"] != VECTOR_LS_BENCHMARK_SCHEMA:
        raise ValueError("unexpected vector LS benchmark schema")
    if metrics["metrics_schema_version"] != VECTOR_LS_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unexpected vector LS benchmark schema version")
    rows = metrics["rows"]
    if not isinstance(rows, list) or len(rows) != len(_case_specs()):
        raise ValueError("rows must contain one entry per derivative backend")
    names = {name for name, _backend, _mode in _case_specs()}
    row_names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be a JSON object")
        missing_row = set(VECTOR_LS_BENCHMARK_ROW_FIELDS).difference(row)
        if missing_row:
            raise ValueError(f"row {index} missing fields: {sorted(missing_row)}")
        row_names.add(str(row["name"]))
        if not bool(row["accepted"]):
            raise ValueError(f"row {index} was not accepted")
        if float(row["residual_value_after"]) > float(row["residual_value_before"]):
            raise ValueError(f"row {index} did not reduce the residual")
        if float(row["coefficient_error_after"]) > float(row["coefficient_error_before"]):
            raise ValueError(f"row {index} did not reduce the coefficient error")
    if row_names != names:
        raise ValueError("row names do not match derivative backend cases")
    if metrics["best_backend_by_residual"] not in names:
        raise ValueError("best_backend_by_residual is not a known backend row")
    if metrics["best_backend_by_time"] not in names:
        raise ValueError("best_backend_by_time is not a known backend row")


def _radius_from_coefficients(coefficients, xi):
    r0, a2, a4 = coefficients
    return r0 * (1.0 + a2 * xi**2 + a4 * xi**4)


def _slope_from_coefficients(coefficients, xi):
    r0, a2, a4 = coefficients
    return r0 * (2.0 * a2 * xi + 4.0 * a4 * xi**3)


def _make_residual_function(*, xi, target_coefficients, slope_weight: float):
    xi_jax = jnp.asarray(xi)
    target = jnp.asarray(target_coefficients)
    target_radius = _radius_from_coefficients(target, xi_jax)
    target_slope = _slope_from_coefficients(target, xi_jax)
    radius_scale = jnp.maximum(jnp.sqrt(jnp.mean(target_radius**2)), jnp.finfo(target_radius.dtype).tiny)
    slope_scale = jnp.maximum(jnp.sqrt(jnp.mean(target_slope**2)), jnp.finfo(target_slope.dtype).tiny)

    def residual(coefficients):
        coefficients = jnp.asarray(coefficients)
        radius = _radius_from_coefficients(coefficients, xi_jax)
        slope = _slope_from_coefficients(coefficients, xi_jax)
        radius_residual = (radius - target_radius) / radius_scale
        slope_residual = float(slope_weight) * (slope - target_slope) / slope_scale
        return jnp.concatenate([radius_residual, slope_residual])

    return residual


def _case_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("finite_difference", "finite_difference", "auto"),
        ("jax_forward", "jax", "forward"),
        ("jax_reverse", "jax", "reverse"),
        ("jax_auto", "jax", "auto"),
    )


def _row_from_step(
    *, name: str, backend: str, jax_mode: str, step, elapsed_s: float, target_coefficients
) -> dict[str, object]:
    coefficient_error_before = float(np.linalg.norm(np.asarray(step.coefficients) - np.asarray(target_coefficients)))
    coefficient_error_after = float(np.linalg.norm(np.asarray(step.new_coefficients) - np.asarray(target_coefficients)))
    return {
        "name": name,
        "jacobian_backend": backend,
        "jax_mode": jax_mode if backend == "jax" else None,
        "accepted": bool(step.accepted),
        "line_search_factor": float(step.line_search_factor),
        "residual_value_before": float(step.residual_value),
        "residual_value_after": float(step.trial_value),
        "residual_reduction_fraction": float(1.0 - step.trial_value / max(step.residual_value, 1.0e-300)),
        "predicted_value": float(step.predicted_value),
        "coefficient_error_before": coefficient_error_before,
        "coefficient_error_after": coefficient_error_after,
        "coefficient_error_reduction_fraction": float(
            1.0 - coefficient_error_after / max(coefficient_error_before, 1.0e-300)
        ),
        "coefficients_initial": [float(value) for value in np.asarray(step.coefficients)],
        "coefficients_new": [float(value) for value in np.asarray(step.new_coefficients)],
        "raw_step": [float(value) for value in np.asarray(step.raw_step)],
        "limited_step": [float(value) for value in np.asarray(step.limited_step)],
        "jacobian_shape": [int(value) for value in np.asarray(step.jacobian).shape],
        "wall_time_s": float(elapsed_s),
    }


def _write_plots(metrics: dict[str, object], *, outdir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics["rows"]
    names = [row["name"] for row in rows]
    x = np.arange(len(rows))
    residual_after = np.asarray([row["residual_value_after"] for row in rows], dtype=float)
    error_after = np.asarray([row["coefficient_error_after"] for row in rows], dtype=float)
    wall_time = np.asarray([row["wall_time_s"] for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    axes[0].bar(x, residual_after)
    axes[0].set_ylabel("residual RMS")
    axes[1].bar(x, error_after)
    axes[1].set_ylabel("coefficient error")
    axes[2].bar(x, wall_time)
    axes[2].set_ylabel("wall time [s]")
    axes[2].set_xticks(x, names, rotation=20, ha="right")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("reduced free-boundary vector LS backend comparison", y=0.995)
    fig.tight_layout()
    summary_path = figure_dir / "mirror_free_boundary_vector_ls_backend_summary.png"
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    xi = np.asarray(metrics["xi"], dtype=float)
    target_radius = np.asarray(metrics["target_radius"], dtype=float)
    initial_radius = np.asarray(metrics["initial_radius"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(xi, target_radius, "k-", linewidth=1.8, label="target")
    ax.plot(xi, initial_radius, "k--", linewidth=1.2, label="initial")
    for row in rows:
        radius = _radius_from_coefficients(np.asarray(row["coefficients_new"], dtype=float), xi)
        ax.plot(xi, radius, label=row["name"])
    ax.set_xlabel("xi")
    ax.set_ylabel("radius")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    radius_path = figure_dir / "mirror_free_boundary_vector_ls_radius_profiles.png"
    fig.savefig(radius_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"summary": str(summary_path), "radius_profiles": str(radius_path)}


def run_case(
    outdir: str | Path,
    *,
    nxi: int = 17,
    slope_weight: float = 0.25,
    finite_difference_step: float = 1.0e-6,
    max_relative_step: float = 0.75,
    write_plots: bool = True,
) -> Path:
    """Run the backend comparison and return the metrics JSON path."""

    enable_x64(True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if int(nxi) < 5:
        raise ValueError("nxi must be at least 5")
    if float(slope_weight) < 0.0:
        raise ValueError("slope_weight must be nonnegative")

    xi = np.linspace(-1.0, 1.0, int(nxi))
    target_coefficients = np.asarray([0.32, 0.18, -0.07], dtype=float)
    initial_coefficients = np.asarray([0.24, 0.02, 0.04], dtype=float)
    residual = _make_residual_function(xi=xi, target_coefficients=target_coefficients, slope_weight=slope_weight)
    rows = []
    for name, backend, jax_mode in _case_specs():
        start = perf_counter()
        step = mirror_free_boundary_residual_vector_least_squares_step(
            initial_coefficients,
            residual,
            jacobian_backend=backend,
            jax_mode=jax_mode,
            finite_difference_step=finite_difference_step,
            max_relative_step=max_relative_step,
            line_search_factors=(1.0, 0.5, 0.25),
        )
        elapsed_s = perf_counter() - start
        rows.append(
            _row_from_step(
                name=name,
                backend=backend,
                jax_mode=jax_mode,
                step=step,
                elapsed_s=elapsed_s,
                target_coefficients=target_coefficients,
            )
        )

    best_by_residual = min(rows, key=lambda row: (float(row["residual_value_after"]), float(row["wall_time_s"])))
    best_by_time = min(rows, key=lambda row: float(row["wall_time_s"]))
    metrics: dict[str, object] = {
        "metrics_schema": VECTOR_LS_BENCHMARK_SCHEMA,
        "metrics_schema_version": VECTOR_LS_BENCHMARK_SCHEMA_VERSION,
        "nxi": int(nxi),
        "slope_weight": float(slope_weight),
        "finite_difference_step": float(finite_difference_step),
        "max_relative_step": float(max_relative_step),
        "xi": [float(value) for value in xi],
        "target_coefficients": [float(value) for value in target_coefficients],
        "initial_coefficients": [float(value) for value in initial_coefficients],
        "target_radius": [float(value) for value in _radius_from_coefficients(target_coefficients, xi)],
        "initial_radius": [float(value) for value in _radius_from_coefficients(initial_coefficients, xi)],
        "rows": rows,
        "best_backend_by_residual": best_by_residual["name"],
        "best_backend_by_time": best_by_time["name"],
        "figures": {},
    }
    if write_plots:
        metrics["figures"] = _write_plots(metrics, outdir=outdir)
    path = outdir / "mirror_free_boundary_vector_ls_benchmark_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/free_boundary_vector_ls_benchmark"))
    parser.add_argument("--nxi", type=int, default=17)
    parser.add_argument("--slope-weight", type=float, default=0.25)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-6)
    parser.add_argument("--max-relative-step", type=float, default=0.75)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = run_case(
        args.outdir,
        nxi=args.nxi,
        slope_weight=args.slope_weight,
        finite_difference_step=args.finite_difference_step,
        max_relative_step=args.max_relative_step,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
