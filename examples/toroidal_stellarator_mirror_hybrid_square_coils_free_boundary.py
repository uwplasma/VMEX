"""Reduced free-boundary beta scan for a square-coil stellarator mirror hybrid.

This example builds a closed square array of mirror-like coil stacks.  Each
side of the square has ``N`` circular coils on a straight line; by default
``N=4``, giving 16 coils.  The coil planes are perpendicular to the local side
axis, so each straight side behaves like a short mirror segment and the four
segments close through corner regions.

The workflow is a reduced free-boundary planning fixture.  It samples the
vacuum field from the coils, applies a local pressure-balance-inspired LCFS
response, and scans the boundary from beta=0 to beta=10%.  It is intended for
geometry, coil-layout, plotting, and residual-vector development before the
full toroidal free-boundary solve/adjoint path is promoted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax._compat import jnp
from vmec_jax.external_fields import CoilFieldParams, build_coil_field_geometry, sample_coil_field_xyz_from_geometry
from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d


SCHEMA = "toroidal_stellarator_mirror_hybrid_square_coils_free_boundary_beta_scan"
SCHEMA_VERSION = "0.1"
MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class SquareCoilArray:
    """Square array of circular coils and its direct-coil field parameters."""

    params: CoilFieldParams
    centers: np.ndarray
    tangents: np.ndarray
    inward_normals: np.ndarray
    currents: np.ndarray
    side_index: np.ndarray
    side_coordinate: np.ndarray


@dataclass(frozen=True)
class HybridBoundarySurface:
    """Sampled square-torus LCFS for one beta value."""

    beta_percent: float
    theta: np.ndarray
    alpha: np.ndarray
    axis: np.ndarray
    xyz: np.ndarray
    radial_semiaxis: np.ndarray
    vertical_semiaxis: np.ndarray
    scale: np.ndarray
    response: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/toroidal_stellarator_mirror_hybrid_square_coils"))
    parser.add_argument("--n-per-side", type=int, default=4)
    parser.add_argument("--betas", type=str, default="0,1,3,10", help="Comma-separated nominal beta percentages.")
    parser.add_argument("--side-length", type=float, default=3.0)
    parser.add_argument("--axis-half-width", type=float, default=1.0)
    parser.add_argument("--axis-square-power", type=float, default=5.0)
    parser.add_argument("--coil-radius", type=float, default=0.36)
    parser.add_argument("--coil-current", type=float, default=8.0e5)
    parser.add_argument("--n-segments", type=int, default=96)
    parser.add_argument("--minor-radius", type=float, default=0.16)
    parser.add_argument("--side-elongation", type=float, default=0.30)
    parser.add_argument("--side-minor-modulation", type=float, default=0.08)
    parser.add_argument("--corner-ellipticity", type=float, default=0.18)
    parser.add_argument("--corner-amplitude", type=float, default=0.025)
    parser.add_argument("--corner-rotation", type=float, default=0.35)
    parser.add_argument("--corner-helicity", type=int, default=1)
    parser.add_argument("--beta-expansion", type=float, default=0.18)
    parser.add_argument("--response-min", type=float, default=0.35)
    parser.add_argument("--response-max", type=float, default=2.5)
    parser.add_argument("--ntheta", type=int, default=48)
    parser.add_argument("--nalpha", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--field-line-count", type=int, default=8)
    parser.add_argument("--field-line-steps", type=int, default=120)
    parser.add_argument("--field-line-step-size", type=float, default=0.025)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_float_list(value: str) -> tuple[float, ...]:
    items = [item.strip() for item in str(value).replace(",", " ").split() if item.strip()]
    if not items:
        raise ValueError("betas must contain at least one value")
    betas = tuple(float(item) for item in items)
    if any(beta < 0.0 for beta in betas):
        raise ValueError("betas must be nonnegative percentages")
    return betas


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("zero vector cannot be normalized")
    return vector / norm


def build_square_mirror_coils(
    *,
    n_per_side: int = 4,
    side_length: float = 3.0,
    coil_radius: float = 0.36,
    current: float = 8.0e5,
    n_segments: int = 96,
    chunk_size: int | None = 512,
) -> SquareCoilArray:
    """Return 4*N circular coils centered on a square in the ``z=0`` plane."""

    n_per_side = int(n_per_side)
    if n_per_side < 1:
        raise ValueError("n_per_side must be positive")
    if side_length <= 0.0:
        raise ValueError("side_length must be positive")
    if coil_radius <= 0.0:
        raise ValueError("coil_radius must be positive")
    if n_segments < 16:
        raise ValueError("n_segments must be at least 16")

    half = 0.5 * float(side_length)
    coordinates = np.linspace(-half, half, n_per_side + 2, dtype=float)[1:-1]
    vertical = np.asarray([0.0, 0.0, 1.0])
    side_specs = (
        # center coordinate builder, tangent along the side, inward normal.
        (lambda q: np.asarray([q, half, 0.0]), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, -1.0, 0.0])),
        (lambda q: np.asarray([half, -q, 0.0]), np.asarray([0.0, -1.0, 0.0]), np.asarray([-1.0, 0.0, 0.0])),
        (lambda q: np.asarray([-q, -half, 0.0]), np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
        (lambda q: np.asarray([-half, q, 0.0]), np.asarray([0.0, 1.0, 0.0]), np.asarray([1.0, 0.0, 0.0])),
    )

    centers: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    inward_normals: list[np.ndarray] = []
    side_index: list[int] = []
    side_coordinate: list[float] = []
    dofs = np.zeros((4 * n_per_side, 3, 3), dtype=float)
    coil_index = 0
    for sidx, (center_for, tangent, inward) in enumerate(side_specs):
        tangent = _unit(tangent)
        inward = _unit(inward)
        for q in coordinates:
            center = np.asarray(center_for(float(q)), dtype=float)
            centers.append(center)
            tangents.append(tangent)
            inward_normals.append(inward)
            side_index.append(sidx)
            side_coordinate.append(float(q))
            dofs[coil_index, :, 0] = center
            # gamma = center + coil_radius * (sin(t) * inward + cos(t) * z).
            # The oriented area normal is z x inward, matching the side tangent.
            dofs[coil_index, :, 1] = float(coil_radius) * inward
            dofs[coil_index, :, 2] = float(coil_radius) * vertical
            coil_index += 1

    currents = np.full(4 * n_per_side, float(current), dtype=float)
    params = CoilFieldParams(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray(currents),
        n_segments=int(n_segments),
        nfp=1,
        stellsym=False,
        current_scale=1.0,
        regularization_epsilon=1.0e-6 * float(coil_radius),
        chunk_size=None if chunk_size is None else int(chunk_size),
    )
    return SquareCoilArray(
        params=params,
        centers=np.asarray(centers, dtype=float),
        tangents=np.asarray(tangents, dtype=float),
        inward_normals=np.asarray(inward_normals, dtype=float),
        currents=currents,
        side_index=np.asarray(side_index, dtype=int),
        side_coordinate=np.asarray(side_coordinate, dtype=float),
    )


def _superellipse_axis(alpha: np.ndarray, *, half_width: float, power: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth square-like centerline and radial outward frame."""

    c = np.cos(alpha)
    s = np.sin(alpha)
    exponent = 2.0 / float(power)
    x = float(half_width) * np.sign(c) * np.abs(c) ** exponent
    y = float(half_width) * np.sign(s) * np.abs(s) ** exponent
    axis = np.stack([x, y, np.zeros_like(x)], axis=-1)
    radial = np.stack([x, y, np.zeros_like(x)], axis=-1)
    radial_norm = np.linalg.norm(radial, axis=1)
    radial = radial / np.maximum(radial_norm[:, None], np.finfo(float).tiny)
    tangent = np.gradient(axis, alpha, axis=0, edge_order=2)
    tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=1)[:, None], np.finfo(float).tiny)
    return axis, radial, tangent


def _surface_from_scale(
    *,
    theta: np.ndarray,
    alpha: np.ndarray,
    axis: np.ndarray,
    radial_out: np.ndarray,
    scale: np.ndarray,
    minor_radius: float,
    side_elongation: float,
    side_minor_modulation: float,
    corner_ellipticity: float,
    corner_amplitude: float,
    corner_rotation: float,
    corner_helicity: int,
    beta_percent: float,
    response: np.ndarray,
) -> HybridBoundarySurface:
    theta2, alpha2 = np.meshgrid(theta, alpha, indexing="ij")
    side_weight = np.abs(np.cos(2.0 * alpha)) ** 1.5
    corner_weight = np.abs(np.sin(2.0 * alpha)) ** 1.5
    minor = float(minor_radius) * (1.0 + float(side_minor_modulation) * side_weight)
    radial_semiaxis = minor * (1.0 + float(corner_ellipticity) * corner_weight)
    vertical_semiaxis = minor * (1.0 + float(side_elongation) * side_weight) * (
        1.0 - 0.5 * float(corner_ellipticity) * corner_weight
    )
    phase = 2.0 * theta2 - float(int(corner_helicity)) * alpha2
    r_local = radial_semiaxis[None, :] * np.cos(theta2)
    z_local = vertical_semiaxis[None, :] * np.sin(theta2)
    r_local = r_local + float(corner_amplitude) * corner_weight[None, :] * np.cos(phase)
    z_local = z_local + float(corner_amplitude) * corner_weight[None, :] * np.sin(phase)
    r_local = scale * r_local
    z_local = scale * z_local

    tilt = float(corner_rotation) * corner_weight * np.sin(float(int(corner_helicity)) * alpha)
    radial_component = r_local * np.cos(tilt)[None, :] - z_local * np.sin(tilt)[None, :]
    vertical_component = r_local * np.sin(tilt)[None, :] + z_local * np.cos(tilt)[None, :]
    vertical = np.asarray([0.0, 0.0, 1.0])
    xyz = (
        axis[None, :, :]
        + radial_component[:, :, None] * radial_out[None, :, :]
        + vertical_component[:, :, None] * vertical[None, None, :]
    )
    return HybridBoundarySurface(
        beta_percent=float(beta_percent),
        theta=theta,
        alpha=alpha,
        axis=axis,
        xyz=np.asarray(xyz, dtype=float),
        radial_semiaxis=radial_semiaxis,
        vertical_semiaxis=vertical_semiaxis,
        scale=scale,
        response=response,
    )


def _sample_bfield(surface: HybridBoundarySurface, geometry, *, regularization_epsilon: float, chunk_size: int | None):
    points = jnp.asarray(surface.xyz.reshape((-1, 3)))
    field = sample_coil_field_xyz_from_geometry(
        geometry,
        points,
        regularization_epsilon=regularization_epsilon,
        chunk_size=chunk_size,
    )
    field = np.asarray(field).reshape(surface.xyz.shape)
    bmag = np.linalg.norm(field, axis=-1)
    return field, bmag


def _surface_normals(surface: HybridBoundarySurface) -> np.ndarray:
    dtheta = np.gradient(surface.xyz, surface.theta, axis=0, edge_order=2)
    dalpha = np.gradient(surface.xyz, surface.alpha, axis=1, edge_order=2)
    normal = np.cross(dalpha, dtheta)
    norm = np.linalg.norm(normal, axis=-1)
    return normal / np.maximum(norm[..., None], np.finfo(float).tiny)


def make_beta_scan(
    *,
    coil_array: SquareCoilArray,
    betas: tuple[float, ...],
    axis_half_width: float,
    axis_square_power: float,
    minor_radius: float,
    side_elongation: float,
    side_minor_modulation: float,
    corner_ellipticity: float,
    corner_amplitude: float,
    corner_rotation: float,
    corner_helicity: int,
    beta_expansion: float,
    response_min: float,
    response_max: float,
    ntheta: int,
    nalpha: int,
    chunk_size: int | None,
) -> tuple[list[dict[str, Any]], list[HybridBoundarySurface], dict[str, np.ndarray]]:
    """Build beta-dependent surfaces and compact diagnostics."""

    if int(ntheta) < 8 or int(nalpha) < 16:
        raise ValueError("ntheta must be >= 8 and nalpha must be >= 16")
    if not (0.0 < float(response_min) < float(response_max)):
        raise ValueError("response_min and response_max must satisfy 0 < response_min < response_max")
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    alpha = np.linspace(0.0, 2.0 * np.pi, int(nalpha), endpoint=False)
    axis, radial_out, tangent = _superellipse_axis(alpha, half_width=axis_half_width, power=axis_square_power)
    unit_scale = np.ones((theta.size, alpha.size), dtype=float)
    unit_response = np.ones_like(unit_scale)
    base = _surface_from_scale(
        theta=theta,
        alpha=alpha,
        axis=axis,
        radial_out=radial_out,
        scale=unit_scale,
        minor_radius=minor_radius,
        side_elongation=side_elongation,
        side_minor_modulation=side_minor_modulation,
        corner_ellipticity=corner_ellipticity,
        corner_amplitude=corner_amplitude,
        corner_rotation=corner_rotation,
        corner_helicity=corner_helicity,
        beta_percent=0.0,
        response=unit_response,
    )
    geometry = build_coil_field_geometry(coil_array.params)
    base_field, base_bmag = _sample_bfield(
        base,
        geometry,
        regularization_epsilon=coil_array.params.regularization_epsilon,
        chunk_size=chunk_size,
    )
    base_b2 = np.maximum(base_bmag**2, np.finfo(float).tiny)
    response = np.mean(base_b2) / base_b2
    response = np.clip(response, float(response_min), float(response_max))
    response = response / np.mean(response)
    response = np.clip(response, float(response_min), float(response_max))
    response = response / np.mean(response)
    b2_pressure_floor = float(max(np.percentile(base_b2, 5.0), np.finfo(float).tiny))
    max_beta = max(max(float(beta) for beta in betas), 1.0)

    rows: list[dict[str, Any]] = []
    surfaces: list[HybridBoundarySurface] = []
    fields = {"base_bmag": base_bmag, "base_field": base_field, "response": response}
    for beta in betas:
        beta_fraction = float(beta) / max_beta
        scale = 1.0 + float(beta_expansion) * beta_fraction * response
        surface = _surface_from_scale(
            theta=theta,
            alpha=alpha,
            axis=axis,
            radial_out=radial_out,
            scale=scale,
            minor_radius=minor_radius,
            side_elongation=side_elongation,
            side_minor_modulation=side_minor_modulation,
            corner_ellipticity=corner_ellipticity,
            corner_amplitude=corner_amplitude,
            corner_rotation=corner_rotation,
            corner_helicity=corner_helicity,
            beta_percent=float(beta),
            response=response,
        )
        field, bmag = _sample_bfield(
            surface,
            geometry,
            regularization_epsilon=coil_array.params.regularization_epsilon,
            chunk_size=chunk_size,
        )
        normals = _surface_normals(surface)
        bnormal = np.sum(field * normals, axis=-1)
        ds = np.linalg.norm(np.gradient(axis, alpha, axis=0, edge_order=2), axis=1)
        area = np.pi * surface.radial_semiaxis * surface.vertical_semiaxis * np.mean(scale**2, axis=0)
        volume_proxy = float(np.trapezoid(area * ds, alpha))
        pressure = (float(beta) / 100.0) * float(np.mean(base_b2)) / (2.0 * MU0)
        magnetic_pressure = np.maximum(bmag**2, b2_pressure_floor) / (2.0 * MU0)
        pressure_balance_proxy = pressure / magnetic_pressure
        rows.append(
            {
                "beta_percent": float(beta),
                "beta_fraction_of_scan_max": beta_fraction,
                "response_min": float(np.min(response)),
                "response_mean": float(np.mean(response)),
                "response_max": float(np.max(response)),
                "boundary_scale_min": float(np.min(scale)),
                "boundary_scale_mean": float(np.mean(scale)),
                "boundary_scale_max": float(np.max(scale)),
                "relative_volume_proxy": volume_proxy,
                "bmag_min": float(np.min(bmag)),
                "bmag_mean": float(np.mean(bmag)),
                "bmag_max": float(np.max(bmag)),
                "external_bnormal_rms": float(np.sqrt(np.mean(bnormal**2))),
                "pressure_balance_proxy_rms": float(np.sqrt(np.mean(pressure_balance_proxy**2))),
                "pressure_balance_proxy_max": float(np.max(pressure_balance_proxy)),
            }
        )
        surfaces.append(surface)
        fields[f"bmag_beta_{float(beta):g}"] = bmag
        fields[f"bnormal_beta_{float(beta):g}"] = bnormal
    if rows:
        base_volume = rows[0]["relative_volume_proxy"]
        for row in rows:
            row["relative_volume_proxy"] = float(row["relative_volume_proxy"] / base_volume)
    return rows, surfaces, fields


def _import_matplotlib():
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    return plt, Normalize, ScalarMappable


def _closed_curve(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[:1]]


def _write_coils_json(path: Path, coil_array: SquareCoilArray) -> Path:
    data = {
        "format": "vmec_jax_square_mirror_hybrid_circular_fourier_coils",
        "coil_count": int(coil_array.centers.shape[0]),
        "centers": coil_array.centers.tolist(),
        "tangents": coil_array.tangents.tolist(),
        "inward_normals": coil_array.inward_normals.tolist(),
        "currents": coil_array.currents.tolist(),
        "side_index": coil_array.side_index.tolist(),
        "side_coordinate": coil_array.side_coordinate.tolist(),
        "base_curve_dofs": np.asarray(coil_array.params.base_curve_dofs).tolist(),
        "n_segments": int(coil_array.params.n_segments),
        "nfp": int(coil_array.params.nfp),
        "stellsym": bool(coil_array.params.stellsym),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "beta_percent",
        "response_min",
        "response_mean",
        "response_max",
        "boundary_scale_min",
        "boundary_scale_mean",
        "boundary_scale_max",
        "relative_volume_proxy",
        "bmag_min",
        "bmag_mean",
        "bmag_max",
        "external_bnormal_rms",
        "pressure_balance_proxy_rms",
        "pressure_balance_proxy_max",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
    return path


def _trace_vacuum_field_lines(
    coil_array: SquareCoilArray,
    surface: HybridBoundarySurface,
    *,
    n_lines: int = 8,
    n_steps: int = 120,
    step_size: float = 0.025,
    chunk_size: int | None = 512,
) -> np.ndarray:
    """Trace short vacuum-field lines for a visual pitch/orientation check."""

    geometry = build_coil_field_geometry(coil_array.params)
    alpha_indices = np.linspace(0, surface.alpha.size, int(n_lines), endpoint=False, dtype=int)
    theta_index = surface.theta.size // 4
    points = np.asarray(surface.xyz[theta_index, alpha_indices, :], dtype=float)
    lines = np.empty((int(n_steps) + 1, points.shape[0], 3), dtype=float)
    lines[0] = points
    for step in range(1, int(n_steps) + 1):
        field = sample_coil_field_xyz_from_geometry(
            geometry,
            jnp.asarray(points),
            regularization_epsilon=coil_array.params.regularization_epsilon,
            chunk_size=chunk_size,
        )
        field = np.asarray(field, dtype=float)
        direction = field / np.maximum(np.linalg.norm(field, axis=1)[:, None], np.finfo(float).tiny)
        points = points + float(step_size) * direction
        lines[step] = points
    return lines


def _write_geometry_plot(
    outdir: Path,
    coil_array: SquareCoilArray,
    surfaces: list[HybridBoundarySurface],
    *,
    field_line_count: int,
    field_line_steps: int,
    field_line_step_size: float,
) -> Path:
    plt, Normalize, ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    geometry = build_coil_field_geometry(coil_array.params)
    gamma = np.asarray(geometry[0])
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for coil in gamma:
        closed = np.vstack([coil, coil[:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="tab:orange", linewidth=1.0)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(surface.beta_percent for surface in surfaces), vmax=max(surface.beta_percent for surface in surfaces))
    for surface in (surfaces[0], surfaces[-1]):
        color = cmap(norm(surface.beta_percent))
        ax.plot_surface(
            surface.xyz[:, :, 0],
            surface.xyz[:, :, 1],
            surface.xyz[:, :, 2],
            color=color,
            alpha=0.42 if surface is surfaces[0] else 0.58,
            linewidth=0,
            shade=False,
        )
    field_lines = _trace_vacuum_field_lines(
        coil_array,
        surfaces[0],
        n_lines=field_line_count,
        n_steps=field_line_steps,
        step_size=field_line_step_size,
    )
    for line in np.moveaxis(field_lines, 0, 1):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="tab:blue", linewidth=0.9, alpha=0.75)
    ax.set_title("square-coil hybrid LCFS, coils, and field lines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fix_matplotlib_3d(ax)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, pad=0.08, label="beta [%]")
    path = outdir / "square_coil_hybrid_geometry_3d.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_top_view_plot(outdir: Path, coil_array: SquareCoilArray, surfaces: list[HybridBoundarySurface]) -> Path:
    plt, Normalize, ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    geometry = build_coil_field_geometry(coil_array.params)
    gamma = np.asarray(geometry[0])
    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    for coil in gamma:
        closed = np.vstack([coil, coil[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="tab:orange", linewidth=0.9, alpha=0.75)
    ax.plot(coil_array.centers[:, 0], coil_array.centers[:, 1], "o", color="tab:orange", markersize=4.0, label="coil centers")
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(surface.beta_percent for surface in surfaces), vmax=max(surface.beta_percent for surface in surfaces))
    for surface in surfaces:
        outer = surface.xyz[0]
        ax.plot(
            _closed_curve(outer[:, 0]),
            _closed_curve(outer[:, 1]),
            linewidth=1.4,
            color=cmap(norm(surface.beta_percent)),
            label=f"beta={surface.beta_percent:g}%",
        )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("top view boundary response")
    ax.legend(fontsize="x-small", ncols=2)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.78, pad=0.02, label="beta [%]")
    path = outdir / "square_coil_hybrid_top_view_beta_scan.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_cross_section_plot(outdir: Path, surfaces: list[HybridBoundarySurface]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    alpha = surfaces[0].alpha
    side_index = int(np.argmin(np.abs(np.mod(alpha, 2.0 * np.pi) - 0.0)))
    corner_index = int(np.argmin(np.abs(np.mod(alpha, 2.0 * np.pi) - 0.25 * np.pi)))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for ax, idx, title in zip(axes, (side_index, corner_index), ("side section", "corner section")):
        center = surfaces[0].axis[idx]
        for surface in surfaces:
            section = surface.xyz[:, idx, :] - center[None, :]
            horizontal = np.linalg.norm(section[:, :2], axis=1) * np.sign(np.sum(section[:, :2] * center[:2], axis=1))
            ax.plot(_closed_curve(horizontal), _closed_curve(section[:, 2]), label=f"{surface.beta_percent:g}%")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("radial displacement")
        ax.set_ylabel("z")
        ax.set_title(title)
    axes[0].legend(fontsize="small")
    path = outdir / "square_coil_hybrid_cross_sections_beta_scan.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_bmag_plot(outdir: Path, surfaces: list[HybridBoundarySurface], fields: dict[str, np.ndarray]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True, sharey=True)
    plotted_bmag = [fields[f"bmag_beta_{surface.beta_percent:g}"] for surface in (surfaces[0], surfaces[-1])]
    vmin = float(min(np.min(bmag) for bmag in plotted_bmag))
    vmax = float(max(np.max(bmag) for bmag in plotted_bmag))
    mesh = None
    for ax, surface in zip(axes, (surfaces[0], surfaces[-1])):
        bmag = fields[f"bmag_beta_{surface.beta_percent:g}"]
        mesh = ax.pcolormesh(surface.alpha, surface.theta, bmag, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"beta={surface.beta_percent:g}%")
        ax.set_xlabel("square angle")
    axes[0].set_ylabel("theta")
    fig.colorbar(mesh, ax=axes, label="|B|")
    path = outdir / "square_coil_hybrid_boundary_bmag_beta_scan.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary_plot(outdir: Path, rows: list[dict[str, Any]]) -> Path:
    plt, _Normalize, _ScalarMappable = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    beta = np.asarray([row["beta_percent"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(beta, [row["boundary_scale_mean"] for row in rows], "o-", label="mean")
    axes[0].plot(beta, [row["boundary_scale_max"] for row in rows], "s--", label="max")
    axes[0].set_ylabel("LCFS scale")
    axes[0].legend(fontsize="small")
    axes[1].plot(beta, [row["relative_volume_proxy"] for row in rows], "o-", color="tab:green")
    axes[1].set_ylabel("relative volume")
    axes[2].plot(beta, [row["pressure_balance_proxy_rms"] for row in rows], "o-", color="tab:red", label="pressure")
    axes[2].plot(beta, [row["external_bnormal_rms"] for row in rows], "s--", color="tab:blue", label="B.n")
    axes[2].set_ylabel("proxy RMS")
    axes[2].set_xlabel("beta [%]")
    axes[2].legend(fontsize="small")
    fig.suptitle("square-coil free-boundary beta response")
    path = outdir / "square_coil_hybrid_beta_scan_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_case(
    outdir: Path,
    *,
    n_per_side: int = 4,
    betas: tuple[float, ...] = (0.0, 1.0, 3.0, 10.0),
    side_length: float = 3.0,
    axis_half_width: float = 1.0,
    axis_square_power: float = 5.0,
    coil_radius: float = 0.36,
    coil_current: float = 8.0e5,
    n_segments: int = 96,
    minor_radius: float = 0.16,
    side_elongation: float = 0.30,
    side_minor_modulation: float = 0.08,
    corner_ellipticity: float = 0.18,
    corner_amplitude: float = 0.025,
    corner_rotation: float = 0.35,
    corner_helicity: int = 1,
    beta_expansion: float = 0.18,
    response_min: float = 0.35,
    response_max: float = 2.5,
    ntheta: int = 48,
    nalpha: int = 96,
    chunk_size: int | None = 512,
    field_line_count: int = 8,
    field_line_steps: int = 120,
    field_line_step_size: float = 0.025,
    write_plots: bool = True,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coil_array = build_square_mirror_coils(
        n_per_side=n_per_side,
        side_length=side_length,
        coil_radius=coil_radius,
        current=coil_current,
        n_segments=n_segments,
        chunk_size=chunk_size,
    )
    rows, surfaces, fields = make_beta_scan(
        coil_array=coil_array,
        betas=tuple(float(beta) for beta in betas),
        axis_half_width=axis_half_width,
        axis_square_power=axis_square_power,
        minor_radius=minor_radius,
        side_elongation=side_elongation,
        side_minor_modulation=side_minor_modulation,
        corner_ellipticity=corner_ellipticity,
        corner_amplitude=corner_amplitude,
        corner_rotation=corner_rotation,
        corner_helicity=corner_helicity,
        beta_expansion=beta_expansion,
        response_min=response_min,
        response_max=response_max,
        ntheta=ntheta,
        nalpha=nalpha,
        chunk_size=chunk_size,
    )
    coils_json = _write_coils_json(outdir / "square_mirror_hybrid_coils.json", coil_array)
    summary_csv = _write_summary_csv(outdir / "square_coil_hybrid_beta_scan_summary.csv", rows)
    figures: dict[str, str] = {}
    if write_plots:
        figure_dir = outdir / "figures"
        figures["geometry_3d"] = str(
            _write_geometry_plot(
                figure_dir,
                coil_array,
                surfaces,
                field_line_count=field_line_count,
                field_line_steps=field_line_steps,
                field_line_step_size=field_line_step_size,
            )
        )
        figures["top_view"] = str(_write_top_view_plot(figure_dir, coil_array, surfaces))
        figures["cross_sections"] = str(_write_cross_section_plot(figure_dir, surfaces))
        figures["boundary_bmag"] = str(_write_bmag_plot(figure_dir, surfaces, fields))
        figures["beta_scan_summary"] = str(_write_summary_plot(figure_dir, rows))

    metrics = {
        "metrics_schema": SCHEMA,
        "metrics_schema_version": SCHEMA_VERSION,
        "workflow_status": "reduced_free_boundary_beta_scan",
        "free_boundary_solve_status": "reduced_pressure_balance_proxy_not_vmec_solve",
        "hybrid_fixture_kind": "toroidal_stellarator_mirror_hybrid_square_coils",
        "production_free_boundary_claim": False,
        "coil_count": int(coil_array.centers.shape[0]),
        "n_per_side": int(n_per_side),
        "side_length": float(side_length),
        "coil_radius": float(coil_radius),
        "coil_current": float(coil_current),
        "n_segments": int(n_segments),
        "axis_half_width": float(axis_half_width),
        "axis_square_power": float(axis_square_power),
        "minor_radius": float(minor_radius),
        "beta_expansion": float(beta_expansion),
        "response_min": float(response_min),
        "response_max": float(response_max),
        "field_line_count": int(field_line_count),
        "field_line_steps": int(field_line_steps),
        "field_line_step_size": float(field_line_step_size),
        "betas_percent": [float(beta) for beta in betas],
        "rows": rows,
        "coils_json": str(coils_json),
        "summary_csv": str(summary_csv),
        "figures": figures,
    }
    metrics_path = outdir / "square_coil_hybrid_free_boundary_beta_scan_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run_case(
        args.outdir,
        n_per_side=args.n_per_side,
        betas=_parse_float_list(args.betas),
        side_length=args.side_length,
        axis_half_width=args.axis_half_width,
        axis_square_power=args.axis_square_power,
        coil_radius=args.coil_radius,
        coil_current=args.coil_current,
        n_segments=args.n_segments,
        minor_radius=args.minor_radius,
        side_elongation=args.side_elongation,
        side_minor_modulation=args.side_minor_modulation,
        corner_ellipticity=args.corner_ellipticity,
        corner_amplitude=args.corner_amplitude,
        corner_rotation=args.corner_rotation,
        corner_helicity=args.corner_helicity,
        beta_expansion=args.beta_expansion,
        response_min=args.response_min,
        response_max=args.response_max,
        ntheta=args.ntheta,
        nalpha=args.nalpha,
        chunk_size=args.chunk_size,
        field_line_count=args.field_line_count,
        field_line_steps=args.field_line_steps,
        field_line_step_size=args.field_line_step_size,
        write_plots=not args.no_plots,
    )
    print(metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
