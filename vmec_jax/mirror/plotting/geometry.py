"""Geometry plot-data helpers for mirror output files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput


@dataclass(frozen=True)
class MirrorSurfacesRZData:
    """Axisymmetric nested-surface data in the physical ``r-z`` plane."""

    z: np.ndarray
    radii: np.ndarray
    surface_indices: np.ndarray
    boundary_radius: np.ndarray


@dataclass(frozen=True)
class MirrorBoundary3DData:
    """Side-boundary surface data for 3-D plotting."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorCrossSectionsData:
    """Nested-surface cross sections at selected axial positions."""

    z: np.ndarray
    z_indices: np.ndarray
    surface_indices: np.ndarray
    x: np.ndarray
    y: np.ndarray


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def _import_matplotlib():
    try:
        from vmec_jax.plotting import prepare_matplotlib_3d

        prepare_matplotlib_3d()
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for mirror plotting helpers") from exc


def _plot_name(output: MirrorOutput, name: str | None) -> str:
    if name is not None:
        return str(name)
    if output.path is None:
        return "mirror"
    stem = output.path.stem
    return stem[5:] if stem.startswith("mout_") else stem


def mirror_surfaces_rz_data(output_or_path, *, num_surfaces: int = 7) -> MirrorSurfacesRZData:
    """Return nested ``r-z`` surface data from a mirror output."""
    output = _as_output(output_or_path)
    count = max(2, min(int(num_surfaces), output.ns))
    indices = np.unique(np.round(np.linspace(0, output.ns - 1, count)).astype(int))
    return MirrorSurfacesRZData(
        z=np.asarray(output.z),
        radii=np.asarray(output.geometry.r[indices, 0, :]),
        surface_indices=indices,
        boundary_radius=np.asarray(output.geometry.boundary_r[0]),
    )


def mirror_boundary_3d_data(output_or_path, *, ntheta_axisym: int = 64) -> MirrorBoundary3DData:
    """Return side-boundary surface data, revolving axisymmetric output if needed."""
    output = _as_output(output_or_path)
    if output.ntheta > 1:
        return MirrorBoundary3DData(
            x=np.asarray(output.geometry.x[-1]),
            y=np.asarray(output.geometry.y[-1]),
            z=np.asarray(output.geometry.z[-1]),
            bmag=np.asarray(output.field.bmag[-1]),
        )
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta_axisym), endpoint=True)
    radius = np.asarray(output.geometry.boundary_r[0])
    x = radius[None, :] * np.cos(theta)[:, None]
    y = radius[None, :] * np.sin(theta)[:, None]
    z = np.broadcast_to(np.asarray(output.z)[None, :], x.shape).copy()
    bmag = np.broadcast_to(np.asarray(output.field.bmag[-1, 0, :])[None, :], x.shape).copy()
    return MirrorBoundary3DData(x=x, y=y, z=z, bmag=bmag)


def mirror_cross_sections_data(
    output_or_path,
    *,
    num_sections: int = 5,
    num_surfaces: int = 7,
    ntheta_axisym: int = 96,
) -> MirrorCrossSectionsData:
    """Return nested ``x-y`` cross sections at selected axial nodes."""
    output = _as_output(output_or_path)
    section_count = max(2, min(int(num_sections), output.nxi))
    surface_count = max(2, min(int(num_surfaces), output.ns))
    z_indices = np.unique(np.round(np.linspace(0, output.nxi - 1, section_count)).astype(int))
    surface_indices = np.unique(np.round(np.linspace(0, output.ns - 1, surface_count)).astype(int))
    if output.ntheta > 1:
        x = np.asarray(output.geometry.x[np.ix_(surface_indices, np.arange(output.ntheta), z_indices)])
        y = np.asarray(output.geometry.y[np.ix_(surface_indices, np.arange(output.ntheta), z_indices)])
        x = np.moveaxis(x, -1, 0)
        y = np.moveaxis(y, -1, 0)
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta_axisym), endpoint=True)
        radii = np.asarray(output.geometry.r[np.ix_(surface_indices, [0], z_indices)])[:, 0, :]
        radii = radii.T[:, :, None]
        x = radii * np.cos(theta)[None, None, :]
        y = radii * np.sin(theta)[None, None, :]
    return MirrorCrossSectionsData(
        z=np.asarray(output.z)[z_indices],
        z_indices=z_indices,
        surface_indices=surface_indices,
        x=x,
        y=y,
    )


def write_mirror_surfaces_rz(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the nested ``r-z`` surface plot for a mirror output."""
    output = _as_output(output_or_path)
    data = mirror_surfaces_rz_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, radius in zip(data.surface_indices, data.radii):
        ax.plot(data.z, radius, label=f"s={output.s[idx]:.2f}")
        ax.plot(data.z, -radius, color=ax.lines[-1].get_color())
    ax.plot(data.z, data.boundary_radius, "k--", linewidth=1.0)
    ax.plot(data.z, -data.boundary_radius, "k--", linewidth=1.0)
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize="x-small", ncols=2)
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_surfaces_rz.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_cross_sections(
    output_or_path,
    *,
    outdir: str | Path,
    name: str | None = None,
) -> Path:
    """Write nested cross sections at selected axial positions."""
    output = _as_output(output_or_path)
    data = mirror_cross_sections_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cols = min(3, data.z.size)
    rows = int(np.ceil(data.z.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    extent = float(max(np.max(np.abs(data.x)), np.max(np.abs(data.y)), np.finfo(float).tiny))
    for section_index, z_value in enumerate(data.z):
        ax = axes.flat[section_index]
        for surface_index, s_idx in enumerate(data.surface_indices):
            color = "k" if int(s_idx) == output.ns - 1 else None
            linewidth = 1.4 if int(s_idx) == output.ns - 1 else 0.85
            ax.plot(
                data.x[section_index, surface_index],
                data.y[section_index, surface_index],
                color=color,
                linewidth=linewidth,
            )
        ax.set_title(f"z={float(z_value):.3g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05 * extent, 1.05 * extent)
        ax.set_ylim(-1.05 * extent, 1.05 * extent)
    for ax in axes.flat[data.z.size :]:
        ax.axis("off")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_cross_sections.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_boundary_3d(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write a 3-D side-boundary plot for a mirror output."""
    output = _as_output(output_or_path)
    data = mirror_boundary_3d_data(output)
    from .bfield import mirror_boundary_field_line_data

    lines = mirror_boundary_field_line_data(output)
    plt = _import_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bmin = float(np.min(data.bmag))
    bmax = float(np.max(data.bmag))
    norm = Normalize(vmin=bmin, vmax=bmax if bmax > bmin else bmin + 1.0)
    fig = plt.figure(figsize=(6.25, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(data.z, data.x, data.y, facecolors=cm.viridis(norm(data.bmag)), linewidth=0.0, alpha=0.86)
    line_scale = 1.025
    for line_index in range(lines.z.shape[0]):
        ax.plot(
            lines.z[line_index],
            line_scale * lines.x[line_index],
            line_scale * lines.y[line_index],
            color="white",
            linewidth=2.4,
            solid_capstyle="round",
        )
        ax.plot(
            lines.z[line_index],
            line_scale * lines.x[line_index],
            line_scale * lines.y[line_index],
            color="tab:red",
            linewidth=1.4,
            solid_capstyle="round",
        )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_box_aspect([max(1.0, float(np.ptp(data.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="|B|")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_boundary_3d.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
