"""Magnetic-field plot-data helpers for mirror output files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.mout import load_mirror_output
from ..io.schema import MirrorOutput
from .geometry import _import_matplotlib, _plot_name


@dataclass(frozen=True)
class MirrorBmagSXiData:
    """Theta-averaged ``|B|(s, xi)`` data."""

    s: np.ndarray
    xi: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorBmagBoundaryData:
    """Boundary ``|B|(theta, xi)`` data."""

    theta: np.ndarray
    xi: np.ndarray
    bmag: np.ndarray


@dataclass(frozen=True)
class MirrorBfieldBoundaryData:
    """Boundary magnetic-field vector data for 3-D plotting."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    bmag: np.ndarray


def _as_output(output_or_path) -> MirrorOutput:
    return output_or_path if isinstance(output_or_path, MirrorOutput) else load_mirror_output(output_or_path)


def mirror_bmag_sxi_data(output_or_path) -> MirrorBmagSXiData:
    """Return theta-averaged ``|B|`` over ``(s, xi)``."""
    output = _as_output(output_or_path)
    return MirrorBmagSXiData(
        s=np.asarray(output.s),
        xi=np.asarray(output.xi),
        bmag=np.mean(np.asarray(output.field.bmag), axis=1),
    )


def mirror_bmag_boundary_data(output_or_path) -> MirrorBmagBoundaryData:
    """Return boundary ``|B|`` over ``(theta, xi)``."""
    output = _as_output(output_or_path)
    return MirrorBmagBoundaryData(
        theta=np.asarray(output.theta),
        xi=np.asarray(output.xi),
        bmag=np.asarray(output.field.bmag[-1]),
    )


def mirror_bfield_boundary_data(output_or_path, *, stride_theta: int = 2, stride_xi: int = 2) -> MirrorBfieldBoundaryData:
    """Return boundary magnetic-field vectors subsampled for 3-D quiver plots."""
    output = _as_output(output_or_path)
    stride_theta = max(1, int(stride_theta))
    stride_xi = max(1, int(stride_xi))
    theta_slice = slice(None, None, stride_theta)
    xi_slice = slice(None, None, stride_xi)
    return MirrorBfieldBoundaryData(
        x=np.asarray(output.geometry.x[-1, theta_slice, xi_slice]),
        y=np.asarray(output.geometry.y[-1, theta_slice, xi_slice]),
        z=np.asarray(output.geometry.z[-1, theta_slice, xi_slice]),
        bx=np.asarray(output.field.b_x[-1, theta_slice, xi_slice]),
        by=np.asarray(output.field.b_y[-1, theta_slice, xi_slice]),
        bz=np.asarray(output.field.b_z[-1, theta_slice, xi_slice]),
        bmag=np.asarray(output.field.bmag[-1, theta_slice, xi_slice]),
    )


def write_mirror_bmag_sxi(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the theta-averaged ``|B|(s, xi)`` map."""
    output = _as_output(output_or_path)
    data = mirror_bmag_sxi_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(data.xi, data.s, data.bmag, shading="auto")
    ax.set_xlabel("xi")
    ax.set_ylabel("s")
    fig.colorbar(mesh, ax=ax, label="|B|")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bmag_sxi.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_bmag_boundary(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write the boundary ``|B|(theta, xi)`` map."""
    output = _as_output(output_or_path)
    data = mirror_bmag_boundary_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if data.theta.size == 1:
        ax.plot(data.xi, data.bmag[0], ".-")
        ax.set_ylabel("|B| at boundary")
    else:
        mesh = ax.pcolormesh(data.xi, data.theta, data.bmag, shading="auto")
        fig.colorbar(mesh, ax=ax, label="|B|")
        ax.set_ylabel("theta")
    ax.set_xlabel("xi")
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bmag_boundary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_mirror_bfield_boundary(output_or_path, *, outdir: str | Path, name: str | None = None) -> Path:
    """Write a 3-D boundary magnetic-field vector plot."""
    output = _as_output(output_or_path)
    data = mirror_bfield_boundary_data(output)
    plt = _import_matplotlib()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scale = np.maximum(data.bmag, np.finfo(float).tiny)
    fig = plt.figure(figsize=(5.5, 4.25))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        np.asarray(output.geometry.x[-1]),
        np.asarray(output.geometry.y[-1]),
        np.asarray(output.geometry.z[-1]),
        color="lightgray",
        alpha=0.25,
        linewidth=0.0,
    )
    ax.quiver(
        data.x,
        data.y,
        data.z,
        data.bx / scale,
        data.by / scale,
        data.bz / scale,
        length=0.14,
        normalize=False,
        color="tab:blue",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("boundary B direction")
    ax.set_box_aspect([1, 1, max(1.0, float(np.ptp(output.z)))])
    fig.tight_layout()
    path = outdir / f"{_plot_name(output, name)}_mirror_bfield_boundary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
