"""Benchmark dense and matrix-free reduced mirror implicit solves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorStateAxisym,
    PressureProfile,
    PsiPrimeProfile,
    axisym_reduced_implicit_state_sensitivity_jax,
    axisym_reduced_residual_matvec_jax,
)
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import pack_axisym_reduced_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/implicit_solve_benchmark"))
    parser.add_argument("--ns-array", type=str, default="5,7")
    parser.add_argument("--nxi-array", type=str, default="7,9")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--state-ridge", type=float, default=10.0)
    parser.add_argument("--cg-tol", type=float, default=1.0e-8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_int_array(value: str, *, name: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < 3 for item in items):
        raise ValueError(f"{name} entries must be at least 3")
    return items


def _benchmark_state(ns: int, nxi: int):
    config = MirrorConfig(MirrorResolution(ns=int(ns), ntheta=1, nxi=int(nxi), mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary = MirrorBoundary.polynomial_radius(r0=0.3, a2=0.04)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    state = MirrorStateAxisym(
        a=base.a * (1.0 + 0.01 * s * (1.0 - s) * (1.0 - xi**2)),
        lam=0.005 * s * (xi - np.mean(grid.xi)),
    )
    return grid, boundary, pack_axisym_reduced_state(state, grid, boundary)


def _timed_solve(solve):
    tracemalloc.start()
    start = time.perf_counter()
    solution = np.asarray(solve())
    runtime = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return solution, float(runtime), int(peak)


def _write_csv(rows: list[dict[str, object]], *, outdir: Path) -> Path:
    path = outdir / "mirror_implicit_solve_benchmark.csv"
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary_plot(rows: list[dict[str, object]], *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(str(row["method"]) for row in rows))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.asarray([row["vector_size"] for row in method_rows], dtype=float)
        order = np.argsort(x)
        x = x[order]
        runtime = np.asarray([row["runtime_s_mean"] for row in method_rows], dtype=float)[order]
        memory = np.asarray([row["python_peak_bytes_max"] for row in method_rows], dtype=float)[order] / 1024.0
        error = np.asarray([row["relative_error_vs_dense"] for row in method_rows], dtype=float)[order]
        axes[0].plot(x, runtime, marker="o", label=method)
        axes[1].plot(x, memory, marker="o", label=method)
        axes[2].plot(x, np.maximum(error, 1.0e-16), marker="o", label=method)
    axes[0].set_ylabel("runtime [s]")
    axes[1].set_ylabel("Python peak [KiB]")
    axes[2].set_ylabel("relative error")
    axes[2].set_xlabel("reduced vector size")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.suptitle("reduced mirror implicit solve benchmark")
    fig.tight_layout()
    path = outdir / "mirror_implicit_solve_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    ns_array: tuple[int, ...] = (5, 7),
    nxi_array: tuple[int, ...] = (7, 9),
    repeat: int = 2,
    state_ridge: float = 10.0,
    cg_tol: float = 1.0e-8,
    write_plots: bool = True,
) -> Path:
    if int(repeat) < 1:
        raise ValueError("repeat must be at least 1")
    if float(state_ridge) <= 0.0:
        raise ValueError("state_ridge must be positive")
    if float(cg_tol) <= 0.0:
        raise ValueError("cg_tol must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()

    for ns in ns_array:
        for nxi in nxi_array:
            grid, boundary, vector = _benchmark_state(ns, nxi)
            rhs = np.sin(np.linspace(0.2, 1.7, vector.size))
            dense_solution = None
            for method in ("dense", "matrix_free_cg"):
                solve_kwargs = {
                    "psi_prime": psi,
                    "i_prime": current,
                    "pressure": pressure,
                    "state_ridge": float(state_ridge),
                    "reference_vector": vector,
                    "solve_method": method,
                    "cg_tol": float(cg_tol),
                    "cg_maxiter": max(40, 4 * vector.size),
                    "mu0": 1.0,
                }

                def solve_once():
                    return axisym_reduced_implicit_state_sensitivity_jax(
                        vector,
                        -rhs,
                        grid,
                        boundary,
                        **solve_kwargs,
                    )

                np.asarray(solve_once())
                timings = []
                peaks = []
                solution = None
                for _ in range(int(repeat)):
                    solution, runtime, peak = _timed_solve(solve_once)
                    timings.append(runtime)
                    peaks.append(peak)
                assert solution is not None
                if method == "dense":
                    dense_solution = solution
                if dense_solution is None:
                    raise RuntimeError("dense reference must be computed before matrix_free_cg")
                matvec = np.asarray(
                    axisym_reduced_residual_matvec_jax(
                        vector,
                        solution,
                        grid,
                        boundary,
                        psi_prime=psi,
                        i_prime=current,
                        pressure=pressure,
                        state_ridge=float(state_ridge),
                        reference_vector=vector,
                        mu0=1.0,
                    )
                )
                rows.append(
                    {
                        "ns": int(ns),
                        "nxi": int(nxi),
                        "vector_size": int(vector.size),
                        "method": method,
                        "repeat": int(repeat),
                        "runtime_s_min": float(np.min(timings)),
                        "runtime_s_mean": float(np.mean(timings)),
                        "python_peak_bytes_max": int(np.max(peaks)),
                        "relative_error_vs_dense": float(
                            np.linalg.norm(solution - dense_solution)
                            / max(np.linalg.norm(dense_solution), np.finfo(float).tiny)
                        ),
                        "linear_residual_relative": float(
                            np.linalg.norm(matvec - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
                        ),
                        "solution_norm": float(np.linalg.norm(solution)),
                        "state_ridge": float(state_ridge),
                        "cg_tol": float(cg_tol),
                    }
                )

    csv_path = _write_csv(rows, outdir=outdir)
    figures: dict[str, str] = {}
    if write_plots:
        figures["summary"] = str(_write_summary_plot(rows, outdir=outdir / "figures"))
    accepted = all(float(row["relative_error_vs_dense"]) < 1.0e-5 for row in rows if row["method"] == "matrix_free_cg")
    metrics = {
        "accepted": bool(accepted),
        "ns_array": [int(value) for value in ns_array],
        "nxi_array": [int(value) for value in nxi_array],
        "repeat": int(repeat),
        "state_ridge": float(state_ridge),
        "cg_tol": float(cg_tol),
        "csv": str(csv_path),
        "figures": figures,
        "rows": rows,
    }
    path = outdir / "mirror_implicit_solve_benchmark_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def main() -> None:
    args = build_parser().parse_args()
    path = run_case(
        args.outdir,
        ns_array=_parse_int_array(args.ns_array, name="ns_array"),
        nxi_array=_parse_int_array(args.nxi_array, name="nxi_array"),
        repeat=args.repeat,
        state_ridge=args.state_ridge,
        cg_tol=args.cg_tol,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
