"""Compare reduced mirror implicit gradients for source, profile, and boundary parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    PressureProfile,
    PsiPrimeProfile,
    axisym_reduced_implicit_polynomial_boundary_sensitivity_jax,
    axisym_reduced_implicit_polynomial_boundary_state_jax,
    axisym_reduced_implicit_profile_sensitivity_jax,
    axisym_reduced_implicit_profile_state_jax,
    axisym_reduced_implicit_source_state_jax,
    axisym_reduced_implicit_state_sensitivity_jax,
    axisym_reduced_polynomial_boundary_radius_jax,
    axisym_reduced_residual_jacobian_jax,
    axisym_reduced_residual_jax,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import pack_axisym_reduced_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/implicit_parameter_gradients"))
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--state-ridge", type=float, default=1.0e-3)
    parser.add_argument("--root-tol", type=float, default=1.0e-10)
    parser.add_argument("--solve-method", choices=("dense", "matrix_free_cg"), default="dense")
    parser.add_argument(
        "--families",
        default="source,pressure,current,flux,boundary",
        help="comma-separated subset of source, pressure, current, flux, boundary",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _tiny_case():
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=7, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary_coefficients = np.asarray([0.3, 0.04, 0.01], dtype=float)
    boundary = MirrorBoundary.polynomial_radius(
        r0=boundary_coefficients[0],
        a2=boundary_coefficients[1],
        a4=boundary_coefficients[2],
    )
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    state = MirrorStateAxisym(
        a=base.a * (1.0 + 0.01 * s * (1.0 - s) * (1.0 - xi**2)),
        lam=0.005 * s * (xi - np.mean(grid.xi)),
    )
    return grid, boundary, boundary_coefficients, state


def _relative_error(actual: float, expected: float) -> float:
    return float(abs(float(actual) - float(expected)) / max(abs(float(expected)), np.finfo(float).tiny))


def _solve_perturbed_root(vector, tangent, residual, jacobian, *, epsilon: float, root_tol: float):
    solved = optimize.root(
        residual,
        vector + float(epsilon) * tangent,
        jac=jacobian,
        method="hybr",
        options={"xtol": min(1.0e-11, float(root_tol)), "maxfev": 160},
    )
    residual_norm = float(np.linalg.norm(residual(solved.x)))
    return solved, residual_norm


def _directional_row(
    *,
    family: str,
    custom_gradient,
    direction,
    forward_tangent,
    finite_difference_tangent,
    loss_weights,
    perturbed_residual_norm: float,
    root_tol: float,
):
    custom_directional = float(np.vdot(custom_gradient, direction))
    forward_directional = float(np.vdot(loss_weights, forward_tangent))
    finite_difference_directional = float(np.vdot(loss_weights, finite_difference_tangent))
    custom_vs_forward = _relative_error(custom_directional, forward_directional)
    custom_vs_finite_difference = _relative_error(custom_directional, finite_difference_directional)
    return {
        "family": family,
        "custom_vjp_directional": custom_directional,
        "forward_directional": forward_directional,
        "finite_difference_directional": finite_difference_directional,
        "custom_vs_forward_relative_error": custom_vs_forward,
        "custom_vs_finite_difference_relative_error": custom_vs_finite_difference,
        "perturbed_residual_norm": float(perturbed_residual_norm),
        "accepted": bool(
            perturbed_residual_norm < float(root_tol)
            and custom_vs_forward < 1.0e-8
            and custom_vs_finite_difference < 1.0e-3
        ),
    }


def _write_gradient_plot(rows: list[dict[str, float | str | bool]], *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["family"]) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.25
    custom = np.asarray([float(row["custom_vjp_directional"]) for row in rows])
    forward = np.asarray([float(row["forward_directional"]) for row in rows])
    finite_difference = np.asarray([float(row["finite_difference_directional"]) for row in rows])
    errors = np.asarray([float(row["custom_vs_finite_difference_relative_error"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    axes[0].bar(x - width, custom, width, label="custom VJP")
    axes[0].bar(x, forward, width, label="forward sensitivity")
    axes[0].bar(x + width, finite_difference, width, label="finite difference")
    axes[0].set_yscale("symlog", linthresh=1.0e-2)
    axes[0].set_ylabel("directional dL/dp")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x, errors, width=0.5, color="tab:red")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("reduced mirror implicit parameter gradients")
    fig.tight_layout()
    path = outdir / "mirror_implicit_parameter_gradients.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    epsilon: float = 1.0e-5,
    state_ridge: float = 1.0e-3,
    root_tol: float = 1.0e-10,
    solve_method: str = "dense",
    families: tuple[str, ...] = ("source", "pressure", "current", "flux", "boundary"),
    write_plots: bool = True,
) -> Path:
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if float(state_ridge) <= 0.0:
        raise ValueError("state_ridge must be positive")
    if float(root_tol) <= 0.0:
        raise ValueError("root_tol must be positive")
    family_set = {item.strip().lower() for item in families if item.strip()}
    allowed = {"source", "pressure", "current", "flux", "boundary"}
    unknown = sorted(family_set - allowed)
    if unknown:
        raise ValueError(f"unknown gradient families: {unknown}")

    outdir.mkdir(parents=True, exist_ok=True)
    grid, boundary, boundary_coefficients, state = _tiny_case()
    psi_coefficients = np.asarray([0.01, 0.002], dtype=float)
    current_coefficients = np.asarray([0.03, -0.01], dtype=float)
    pressure_coefficients = np.asarray([0.2, -0.08], dtype=float)
    pressure_gamma = 2.0
    psi = PsiPrimeProfile(coefficients=psi_coefficients)
    current = IPrimeProfile(coefficients=current_coefficients)
    pressure = PressureProfile(coefficients=pressure_coefficients, gamma=pressure_gamma)
    boundary_radius = axisym_reduced_polynomial_boundary_radius_jax(boundary_coefficients, grid)
    vector = pack_axisym_reduced_state(state, grid, boundary)
    source0 = np.asarray(
        axisym_reduced_residual_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            boundary_radius=boundary_radius,
            mu0=1.0,
        )
    )
    root_residual = np.asarray(
        axisym_reduced_residual_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            source_vector=source0,
            state_ridge=state_ridge,
            reference_vector=vector,
            boundary_radius=boundary_radius,
            mu0=1.0,
        )
    )
    loss_weights = np.cos(np.linspace(0.2, 1.5, vector.size))
    rows: list[dict[str, float | str | bool]] = []

    if "source" in family_set:
        source_direction = np.sin(np.linspace(0.1, 1.3, vector.size))

        def loss_for_source(source):
            solved_state = axisym_reduced_implicit_source_state_jax(
                jnp.asarray(vector),
                source,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                state_ridge=state_ridge,
                reference_vector=vector,
                boundary_radius=boundary_radius,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
            return jnp.vdot(jnp.asarray(loss_weights, dtype=solved_state.dtype), solved_state)

        custom_gradient = np.asarray(jax.grad(loss_for_source)(jnp.asarray(source0)))
        forward_tangent = np.asarray(
            axisym_reduced_implicit_state_sensitivity_jax(
                vector,
                -source_direction,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source0,
                state_ridge=state_ridge,
                reference_vector=vector,
                boundary_radius=boundary_radius,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
        )
        source_eps = source0 + float(epsilon) * source_direction

        def residual(items):
            return np.asarray(
                axisym_reduced_residual_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi,
                    i_prime=current,
                    pressure=pressure,
                    source_vector=source_eps,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius,
                    mu0=1.0,
                )
            )

        def jacobian(items):
            return np.asarray(
                axisym_reduced_residual_jacobian_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi,
                    i_prime=current,
                    pressure=pressure,
                    source_vector=source_eps,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius,
                    mu0=1.0,
                )
            )

        solved, perturbed_residual_norm = _solve_perturbed_root(
            vector,
            forward_tangent,
            residual,
            jacobian,
            epsilon=epsilon,
            root_tol=root_tol,
        )
        rows.append(
            _directional_row(
                family="source",
                custom_gradient=custom_gradient,
                direction=source_direction,
                forward_tangent=forward_tangent,
                finite_difference_tangent=(solved.x - vector) / float(epsilon),
                loss_weights=loss_weights,
                perturbed_residual_norm=perturbed_residual_norm,
                root_tol=root_tol,
            )
        )

    profile_specs = {
        "pressure": ("pressure", pressure_coefficients, np.asarray([0.4, -0.3], dtype=float)),
        "current": ("i_prime", current_coefficients, np.asarray([0.4, -0.3], dtype=float)),
        "flux": ("psi_prime", psi_coefficients, np.asarray([0.3, -0.2], dtype=float)),
    }
    for family, (profile, coefficients, direction) in profile_specs.items():
        if family not in family_set:
            continue

        def loss_for_profile(items, *, profile=profile):
            solved_state = axisym_reduced_implicit_profile_state_jax(
                jnp.asarray(vector),
                items,
                grid,
                boundary,
                profile=profile,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source0,
                state_ridge=state_ridge,
                reference_vector=vector,
                boundary_radius=boundary_radius,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
            return jnp.vdot(jnp.asarray(loss_weights, dtype=solved_state.dtype), solved_state)

        custom_gradient = np.asarray(jax.grad(loss_for_profile)(jnp.asarray(coefficients)))
        forward_tangent = np.asarray(
            axisym_reduced_implicit_profile_sensitivity_jax(
                vector,
                coefficients,
                grid,
                boundary,
                profile=profile,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source0,
                state_ridge=state_ridge,
                reference_vector=vector,
                boundary_radius=boundary_radius,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
            @ direction
        )
        perturbed = coefficients + float(epsilon) * direction
        psi_eps = PsiPrimeProfile(coefficients=perturbed) if profile == "psi_prime" else psi
        current_eps = IPrimeProfile(coefficients=perturbed) if profile == "i_prime" else current
        pressure_eps = (
            PressureProfile(coefficients=perturbed, gamma=pressure_gamma) if profile == "pressure" else pressure
        )

        def residual(items):
            return np.asarray(
                axisym_reduced_residual_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi_eps,
                    i_prime=current_eps,
                    pressure=pressure_eps,
                    source_vector=source0,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius,
                    mu0=1.0,
                )
            )

        def jacobian(items):
            return np.asarray(
                axisym_reduced_residual_jacobian_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi_eps,
                    i_prime=current_eps,
                    pressure=pressure_eps,
                    source_vector=source0,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius,
                    mu0=1.0,
                )
            )

        solved, perturbed_residual_norm = _solve_perturbed_root(
            vector,
            forward_tangent,
            residual,
            jacobian,
            epsilon=epsilon,
            root_tol=root_tol,
        )
        rows.append(
            _directional_row(
                family=family,
                custom_gradient=custom_gradient,
                direction=direction,
                forward_tangent=forward_tangent,
                finite_difference_tangent=(solved.x - vector) / float(epsilon),
                loss_weights=loss_weights,
                perturbed_residual_norm=perturbed_residual_norm,
                root_tol=root_tol,
            )
        )

    if "boundary" in family_set:
        boundary_direction = np.asarray([0.02, -0.03, 0.01], dtype=float)

        def loss_for_boundary(items):
            solved_state = axisym_reduced_implicit_polynomial_boundary_state_jax(
                jnp.asarray(vector),
                items,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source0,
                state_ridge=state_ridge,
                reference_vector=vector,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
            return jnp.vdot(jnp.asarray(loss_weights, dtype=solved_state.dtype), solved_state)

        custom_gradient = np.asarray(jax.grad(loss_for_boundary)(jnp.asarray(boundary_coefficients)))
        forward_tangent = np.asarray(
            axisym_reduced_implicit_polynomial_boundary_sensitivity_jax(
                vector,
                boundary_coefficients,
                grid,
                boundary,
                psi_prime=psi,
                i_prime=current,
                pressure=pressure,
                source_vector=source0,
                state_ridge=state_ridge,
                reference_vector=vector,
                solve_method=solve_method,
                cg_tol=1.0e-10,
                cg_maxiter=200,
                mu0=1.0,
            )
            @ boundary_direction
        )
        boundary_radius_eps = axisym_reduced_polynomial_boundary_radius_jax(
            boundary_coefficients + float(epsilon) * boundary_direction,
            grid,
        )

        def residual(items):
            return np.asarray(
                axisym_reduced_residual_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi,
                    i_prime=current,
                    pressure=pressure,
                    source_vector=source0,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius_eps,
                    mu0=1.0,
                )
            )

        def jacobian(items):
            return np.asarray(
                axisym_reduced_residual_jacobian_jax(
                    items,
                    grid,
                    boundary,
                    psi_prime=psi,
                    i_prime=current,
                    pressure=pressure,
                    source_vector=source0,
                    state_ridge=state_ridge,
                    reference_vector=vector,
                    boundary_radius=boundary_radius_eps,
                    mu0=1.0,
                )
            )

        solved, perturbed_residual_norm = _solve_perturbed_root(
            vector,
            forward_tangent,
            residual,
            jacobian,
            epsilon=epsilon,
            root_tol=root_tol,
        )
        rows.append(
            _directional_row(
                family="boundary",
                custom_gradient=custom_gradient,
                direction=boundary_direction,
                forward_tangent=forward_tangent,
                finite_difference_tangent=(solved.x - vector) / float(epsilon),
                loss_weights=loss_weights,
                perturbed_residual_norm=perturbed_residual_norm,
                root_tol=root_tol,
            )
        )

    figures: dict[str, str] = {}
    if write_plots and rows:
        figures["directional_gradients"] = str(_write_gradient_plot(rows, outdir=outdir / "figures"))

    metrics = {
        "vector_size": int(vector.size),
        "epsilon": float(epsilon),
        "state_ridge": float(state_ridge),
        "solve_method": str(solve_method),
        "families": [str(row["family"]) for row in rows],
        "root_residual_norm": float(np.linalg.norm(root_residual)),
        "rows": rows,
        "accepted": bool(rows and all(bool(row["accepted"]) for row in rows)),
        "figures": figures,
    }
    path = outdir / "mirror_implicit_parameter_gradients_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def main() -> None:
    args = build_parser().parse_args()
    families = tuple(item.strip().lower() for item in args.families.split(",") if item.strip())
    path = run_case(
        args.outdir,
        epsilon=args.epsilon,
        state_ridge=args.state_ridge,
        root_tol=args.root_tol,
        solve_method=args.solve_method,
        families=families,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
