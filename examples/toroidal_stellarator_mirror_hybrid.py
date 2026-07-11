"""Coil-informed fixed-boundary stellarator-mirror hybrid equilibrium."""

from dataclasses import replace
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vmec_jax import plot_wout, write_wout  # noqa: E402
from vmec_jax.core.coils import coil_geometry, square_mirror_coils  # noqa: E402
from vmec_jax.core.hybrid import (  # noqa: E402
    stellarator_mirror_hybrid_input,
    trace_square_coil_vacuum_axis,
)
from vmec_jax.core.multigrid import solve_multigrid  # noqa: E402
from vmec_jax.core.plotting import surface_rz  # noqa: E402
from vmec_jax.core.solver import solve  # noqa: E402
from vmec_jax.core.wout import wout_from_state  # noqa: E402

# Inputs: edit these values, then run the file directly.
OUTPUT_DIR = Path("results/toroidal_stellarator_mirror_hybrid")
N_COILS_PER_SIDE = 4
COIL_SIDE_LENGTH = 3.0
COIL_RADIUS = 0.5
COIL_CURRENT = 8.0e5
TOROIDAL_CURRENT = 3.0e3
MPOL, NTOR = 6, 20
NS_ARRAY = (3, 5)
NTHETA, NZETA = 48, 256
PHIEDGE = 0.004
FTOL = 1.0e-8  # current validated hybrid floor; stricter preconditioning remains open
MAX_ITERATIONS = 5000
SHAPING_STAGES = np.linspace(0.0, 1.0, 11)
SIDE_ELONGATION = 0.10
CORNER_ELLIPTICITY = 0.08
CORNER_ROTATION = 0.20

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

coils = square_mirror_coils(
    n_per_side=N_COILS_PER_SIDE,
    side_length=COIL_SIDE_LENGTH,
    semi_major=COIL_RADIUS,
    semi_minor=COIL_RADIUS,
    current=COIL_CURRENT,
    n_segments=96,
    regularization_epsilon=5.0e-7,
)
vacuum_axis = trace_square_coil_vacuum_axis(coils, side_length=COIL_SIDE_LENGTH, n_steps=4096, nzeta=NZETA)


def input_for(shaping: float, *, multigrid: bool):
    stages = NS_ARRAY if multigrid else (NS_ARRAY[-1],)
    return replace(
        stellarator_mirror_hybrid_input(
            mpol=MPOL,
            ntor=NTOR,
            ns_array=stages,
            ftol_array=tuple(FTOL for _ in stages),
            niter_array=tuple(MAX_ITERATIONS for _ in stages),
            phiedge=PHIEDGE,
            curtor=TOROIDAL_CURRENT,
            ntheta=NTHETA,
            nzeta=NZETA,
            axis_radius_samples=vacuum_axis.radius,
            minor_radius_samples=vacuum_axis.flux_tube_scale,
            minor_radius=0.10,
            side_elongation=shaping * SIDE_ELONGATION,
            corner_ellipticity=shaping * CORNER_ELLIPTICITY,
            corner_rotation=shaping * CORNER_ROTATION,
        ),
        delt=0.02,
    )


print(f"Vacuum axis closure={vacuum_axis.closure_error:.3e} m, planarity={vacuum_axis.planarity_error:.3e} m")
results = [solve_multigrid(input_for(0.0, multigrid=True))]
for shaping in SHAPING_STAGES[1:]:
    result = solve(
        input_for(float(shaping), multigrid=False),
        initial_state=results[-1].state,
    )
    results.append(result)
    print(
        f"shape={shaping:4.2f} iterations={result.iterations:4d} "
        f"fsq=({result.fsqr:.3e},{result.fsqz:.3e},{result.fsql:.3e})"
    )

final_input = input_for(1.0, multigrid=False)
final = results[-1]
indata_path = final_input.to_indata(OUTPUT_DIR / "input.stellarator_mirror_hybrid")
wout = wout_from_state(
    inp=final_input,
    state=final.state,
    fsqr=final.fsqr,
    fsqz=final.fsqz,
    fsql=final.fsql,
    niter=final.iterations,
)
wout_path = write_wout(OUTPUT_DIR / "wout_stellarator_mirror_hybrid.nc", wout)
plot_wout(wout, OUTPUT_DIR, name="stellarator_mirror_hybrid")

fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
for shaping, result in zip(SHAPING_STAGES, results, strict=True):
    history = np.asarray(result.fsq_history)
    residual = np.max(history[:, :3], axis=1)
    ax.semilogy(np.maximum(residual, 1.0e-18), lw=1.4, label=f"{shaping:.1f}")
ax.axhline(FTOL, color="black", ls="--", lw=1.0)
ax.set(xlabel="Iteration", ylabel="Maximum force residual", title="Hybrid shaping continuation")
ax.grid(alpha=0.25)
ax.legend(title="Shaping", ncol=3, fontsize=7)
fig.savefig(OUTPUT_DIR / "hybrid_convergence.png", dpi=140)
plt.close(fig)

theta = np.linspace(0.0, 2.0 * np.pi, 65)
phi = np.linspace(0.0, 2.0 * np.pi, 145)
radius, height = surface_rz(wout, s_index=-1, theta=theta, phi=phi)
x = radius * np.cos(phi)[None]
y = radius * np.sin(phi)[None]
gamma = np.asarray(coil_geometry(coils)[0])
fig = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
ax = fig.add_subplot(projection="3d")
ax.plot_surface(x, y, height, color="#5DA5DA", alpha=0.24, linewidth=0)
for curve in gamma:
    closed = np.vstack([curve, curve[0]])
    ax.plot(*closed.T, color="#C44E52", lw=1.3)
ax.plot(*vacuum_axis.xyz.T, color="#00A6A6", lw=2.0, label="coil vacuum axis")
for theta0 in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
    line_phi = np.linspace(0.0, 2.0 * np.pi, 420)
    line_theta = theta0 + float(wout.iotaf[-1]) * line_phi
    line_r, line_z = surface_rz(wout, s_index=-1, theta=line_theta, phi=line_phi)
    diagonal = np.arange(line_phi.size)
    rr, zz = line_r[diagonal, diagonal], line_z[diagonal, diagonal]
    ax.plot(rr * np.cos(line_phi), rr * np.sin(line_phi), zz, color="#111111", lw=1.0)
ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="16 coils, solved LCFS, and field lines")
ax.set_box_aspect((1.0, 1.0, 0.35))
ax.view_init(elev=28, azim=-48)
fig.savefig(OUTPUT_DIR / "hybrid_coils_fieldlines.png", dpi=140)
plt.close(fig)

print(f"Wrote {indata_path}, {wout_path}, and plots under {OUTPUT_DIR}")
