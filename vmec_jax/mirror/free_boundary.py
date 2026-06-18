"""Small free-boundary helpers for mirror coil-field studies.

This module is a bridge, not a free-boundary equilibrium solver.  It provides
ESSOS-compatible circular-loop coil parameters and field sampling on mirror
grids so later lanes can build LCFS and beta-scan drivers on tested pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vmec_jax._compat import jax, jnp
from vmec_jax.external_fields import CoilFieldParams, sample_external_field_cylindrical

from .core.boundary import MirrorBoundary
from .core.grids import MirrorGrid
from .validation.coils import mirror_boundary_from_on_axis_bz


@dataclass(frozen=True)
class MirrorCircularCoils:
    """Axisymmetric circular coils for mirror free-boundary studies."""

    radii_m: Any
    z_centers_m: Any
    currents_a: Any
    n_segments: int = 128
    regularization_epsilon: float = 0.0
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        radii = np.asarray(self.radii_m, dtype=float)
        z_centers = np.asarray(self.z_centers_m, dtype=float)
        currents = np.asarray(self.currents_a, dtype=float)
        if radii.ndim != 1 or z_centers.ndim != 1 or currents.ndim != 1:
            raise ValueError("radii_m, z_centers_m, and currents_a must be one-dimensional")
        if not (radii.shape == z_centers.shape == currents.shape):
            raise ValueError("radii_m, z_centers_m, and currents_a must have the same shape")
        if radii.size == 0:
            raise ValueError("at least one circular coil is required")
        if np.any(radii <= 0.0):
            raise ValueError("coil radii must be positive")
        if int(self.n_segments) < 8:
            raise ValueError("n_segments must be at least 8")
        if float(self.regularization_epsilon) < 0.0:
            raise ValueError("regularization_epsilon must be nonnegative")
        if self.chunk_size is not None and int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive when provided")
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "z_centers_m", z_centers)
        object.__setattr__(self, "currents_a", currents)
        object.__setattr__(self, "n_segments", int(self.n_segments))
        object.__setattr__(self, "regularization_epsilon", float(self.regularization_epsilon))
        object.__setattr__(self, "chunk_size", None if self.chunk_size is None else int(self.chunk_size))

    @classmethod
    def symmetric_pair(
        cls,
        *,
        coil_radius_m: float,
        separation_m: float,
        current_a: float,
        center_z_m: float = 0.0,
        n_segments: int = 128,
        regularization_epsilon: float = 0.0,
        chunk_size: int | None = None,
    ) -> "MirrorCircularCoils":
        """Return two equal circular coils centered about ``center_z_m``."""

        separation = float(separation_m)
        if separation <= 0.0:
            raise ValueError("separation_m must be positive")
        half = 0.5 * separation
        return cls(
            radii_m=np.asarray([coil_radius_m, coil_radius_m], dtype=float),
            z_centers_m=np.asarray([float(center_z_m) - half, float(center_z_m) + half], dtype=float),
            currents_a=np.asarray([current_a, current_a], dtype=float),
            n_segments=n_segments,
            regularization_epsilon=regularization_epsilon,
            chunk_size=chunk_size,
        )

    def to_direct_coil_params(self) -> CoilFieldParams:
        """Return ESSOS-compatible direct-coil parameters."""

        return mirror_circular_coils_to_direct_params(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "radii_m": self.radii_m.tolist(),
            "z_centers_m": self.z_centers_m.tolist(),
            "currents_a": self.currents_a.tolist(),
            "n_segments": int(self.n_segments),
            "regularization_epsilon": float(self.regularization_epsilon),
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorCircularCoils":
        """Build circular coils from a JSON-friendly mapping."""

        return cls(
            radii_m=data["radii_m"],
            z_centers_m=data["z_centers_m"],
            currents_a=data["currents_a"],
            n_segments=int(data.get("n_segments", 128)),
            regularization_epsilon=float(data.get("regularization_epsilon", 0.0)),
            chunk_size=data.get("chunk_size"),
        )


@dataclass(frozen=True)
class MirrorExternalFieldSample:
    """External-field values sampled on a mirror axis or boundary grid."""

    r: Any
    theta: Any
    z: Any
    br: Any
    btheta: Any
    bz: Any
    bmag: Any


@dataclass(frozen=True)
class MirrorFreeBoundaryBetaCase:
    """One planned beta point for mirror free-boundary scans."""

    beta_percent: float
    beta_fraction: float
    pressure_scale: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly representation."""

        return {
            "beta_percent": float(self.beta_percent),
            "beta_fraction": float(self.beta_fraction),
            "pressure_scale": float(self.pressure_scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorFreeBoundaryBetaCase":
        """Build a beta case from a JSON-friendly mapping."""

        beta_percent = float(data["beta_percent"])
        beta_fraction = float(data.get("beta_fraction", 0.01 * beta_percent))
        pressure_scale = float(data["pressure_scale"])
        return cls(beta_percent=beta_percent, beta_fraction=beta_fraction, pressure_scale=pressure_scale)


@dataclass(frozen=True)
class MirrorFreeBoundaryCircularCoilScan:
    """Serializable setup for circular-coil mirror beta scans."""

    coils: MirrorCircularCoils
    beta_cases: tuple[MirrorFreeBoundaryBetaCase, ...]

    def __post_init__(self) -> None:
        if not self.beta_cases:
            raise ValueError("at least one beta case is required")
        object.__setattr__(self, "beta_cases", tuple(self.beta_cases))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "coils": self.coils.to_dict(),
            "beta_cases": [case.to_dict() for case in self.beta_cases],
            "status": "setup_only_no_lcfs_solve",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorFreeBoundaryCircularCoilScan":
        """Build a scan setup from a JSON-friendly mapping."""

        return cls(
            coils=MirrorCircularCoils.from_dict(data["coils"]),
            beta_cases=tuple(MirrorFreeBoundaryBetaCase.from_dict(case) for case in data["beta_cases"]),
        )


@dataclass(frozen=True)
class MirrorLCFSDiagnostic:
    """Side-boundary diagnostic for future mirror free-boundary updates."""

    theta: Any
    z: Any
    boundary_r: Any
    boundary_dr_dz: Any
    external_bnormal: Any
    external_bnormal_rms: float
    external_bnormal_max: float
    pressure_balance: Any
    pressure_balance_rms: float
    pressure_balance_max: float
    internal_bmag: Any
    external_bmag: Any
    edge_pressure: float


@dataclass(frozen=True)
class MirrorLCFSMerit:
    """Dimensionless merit for accepting LCFS pilot updates."""

    value: float
    pressure_balance_rms: float
    external_bnormal_rms: float
    external_bmag_rms: float
    pressure_scale: float
    bnormal_scale: float
    bnormal_weight: float


@dataclass(frozen=True)
class MirrorLCFSResidual:
    """Normalized LCFS residual vector for coupled free-boundary solves."""

    vector: Any
    pressure_component: Any
    bnormal_component: Any
    value: float
    pressure_balance_rms: float
    external_bnormal_rms: float
    external_bmag_rms: float
    pressure_scale: float
    bnormal_scale: float
    bnormal_weight: float


@dataclass(frozen=True)
class MirrorFreeBoundaryResidual:
    """Combined residual vector for mirror free-boundary solve prototypes."""

    vector: Any
    equilibrium_component: Any
    lcfs_component: Any
    value: float
    equilibrium_rms: float
    lcfs_value: float
    equilibrium_scale: float
    equilibrium_weight: float
    lcfs_weight: float


@dataclass(frozen=True)
class MirrorFreeBoundaryLeastSquaresStep:
    """One linearized free-boundary least-squares boundary update."""

    coefficients: Any
    residual: MirrorFreeBoundaryResidual
    jacobian: Any
    finite_difference_steps: Any
    raw_step: Any
    limited_step: Any
    line_search_factor: float
    new_coefficients: Any
    trial_residual: MirrorFreeBoundaryResidual
    predicted_vector: Any
    predicted_value: float
    accepted: bool
    damping: float
    max_relative_step: float
    ridge: float
    rcond: float | None


@dataclass(frozen=True)
class MirrorFreeBoundaryVectorLeastSquaresStep:
    """One least-squares update for a reduced residual-vector function."""

    coefficients: Any
    residual_vector: Any
    jacobian: Any
    raw_step: Any
    limited_step: Any
    line_search_factor: float
    new_coefficients: Any
    trial_vector: Any
    residual_value: float
    trial_value: float
    predicted_vector: Any
    predicted_value: float
    accepted: bool
    jacobian_backend: str
    damping: float
    max_relative_step: float
    ridge: float
    rcond: float | None


@dataclass(frozen=True)
class MirrorFreeBoundaryVectorLeastSquaresSolveRow:
    """One row from a reduced residual-vector least-squares solve loop."""

    step: int
    ls_step: MirrorFreeBoundaryVectorLeastSquaresStep
    residual_value_before: float
    residual_value_after: float
    accepted: bool
    improvement_fraction: float | None
    stop_reason: str | None


@dataclass(frozen=True)
class MirrorFreeBoundaryVectorLeastSquaresSolveResult:
    """Result of a reduced residual-vector least-squares solve loop."""

    initial_coefficients: Any
    final_coefficients: Any
    initial_residual_value: float
    final_residual_value: float
    rows: tuple[MirrorFreeBoundaryVectorLeastSquaresSolveRow, ...]
    accepted_steps: int
    stop_reason: str
    converged: bool


@dataclass(frozen=True)
class MirrorFreeBoundaryLoopState:
    """State passed between guarded free-boundary least-squares steps."""

    coefficients: Any
    residual: MirrorFreeBoundaryResidual
    merit: float
    equilibrium_value: float | None = None
    payload: Any = None


@dataclass(frozen=True)
class MirrorFreeBoundaryLoopRow:
    """One attempted step in a guarded free-boundary least-squares loop."""

    step: int
    state_before: MirrorFreeBoundaryLoopState
    ls_step: MirrorFreeBoundaryLeastSquaresStep
    trial_state: MirrorFreeBoundaryLoopState | None
    accepted: bool
    status: str
    rejection_reason: str | None
    stop_reason: str | None
    merit_improvement_fraction: float | None
    equilibrium_growth_ratio: float | None


@dataclass(frozen=True)
class MirrorFreeBoundaryLoopResult:
    """Result of a guarded free-boundary least-squares loop."""

    initial_state: MirrorFreeBoundaryLoopState
    final_state: MirrorFreeBoundaryLoopState
    rows: tuple[MirrorFreeBoundaryLoopRow, ...]
    accepted_steps: int
    stop_reason: str | None
    converged: bool


@dataclass(frozen=True)
class MirrorLCFSUpdateProposal:
    """One damped axisymmetric side-boundary update proposal."""

    z: Any
    xi: Any
    old_radius: Any
    new_radius: Any
    delta_radius: Any
    pressure_response: Any
    pressure_balance_before: Any
    pressure_balance_predicted: Any
    pressure_balance_rms_before: float
    pressure_balance_rms_predicted: float
    damping: float
    max_relative_step: float
    cap_taper_power: float
    smoothing_passes: int
    preserve_caps: bool
    boundary: MirrorBoundary
    strategy: str


def mirror_circular_coils_to_direct_params(coils: MirrorCircularCoils) -> CoilFieldParams:
    """Convert circular mirror coils to ESSOS-convention Fourier coil params."""

    radii = np.asarray(coils.radii_m, dtype=float)
    z_centers = np.asarray(coils.z_centers_m, dtype=float)
    dofs = np.zeros((radii.size, 3, 3), dtype=float)
    dofs[:, 0, 2] = radii
    dofs[:, 1, 1] = radii
    dofs[:, 2, 0] = z_centers
    return CoilFieldParams(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray(coils.currents_a),
        n_segments=coils.n_segments,
        nfp=1,
        stellsym=False,
        current_scale=1.0,
        regularization_epsilon=coils.regularization_epsilon,
        chunk_size=coils.chunk_size,
    )


def _as_direct_coil_params(provider_params: Any) -> Any:
    if isinstance(provider_params, MirrorCircularCoils):
        return provider_params.to_direct_coil_params()
    return provider_params


def _sample_field(
    *,
    r: Any,
    theta: Any,
    z: Any,
    provider_params: Any,
    provider_kind: str,
    provider_static: Any | None,
) -> MirrorExternalFieldSample:
    br, btheta, bz = sample_external_field_cylindrical(
        provider_kind,
        provider_static,
        _as_direct_coil_params(provider_params),
        r,
        z,
        theta,
    )
    bmag = jnp.sqrt(br * br + btheta * btheta + bz * bz)
    return MirrorExternalFieldSample(r=r, theta=theta, z=z, br=br, btheta=btheta, bz=bz, bmag=bmag)


def sample_mirror_axis_external_field(
    grid: MirrorGrid,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
) -> MirrorExternalFieldSample:
    """Sample an external field on the mirror axis nodes."""

    z = jnp.asarray(grid.z)
    r = jnp.zeros_like(z)
    theta = jnp.zeros_like(z)
    return _sample_field(
        r=r,
        theta=theta,
        z=z,
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )


def sample_mirror_boundary_external_field(
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
) -> MirrorExternalFieldSample:
    """Sample an external field on the mirror side boundary."""

    r = jnp.asarray(boundary.radius_on_grid_3d(grid))
    theta = jnp.broadcast_to(jnp.asarray(grid.theta)[:, None], r.shape)
    z = jnp.broadcast_to(jnp.asarray(grid.z)[None, :], r.shape)
    return _sample_field(
        r=r,
        theta=theta,
        z=z,
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )


def make_mirror_free_boundary_beta_cases(
    beta_percent: tuple[float, ...] = (1.0, 3.0, 10.0),
    *,
    pressure_scale_for_one_percent: float = 1.0,
) -> tuple[MirrorFreeBoundaryBetaCase, ...]:
    """Return planned pressure scales for a mirror beta scan."""

    scale = float(pressure_scale_for_one_percent)
    if scale <= 0.0:
        raise ValueError("pressure_scale_for_one_percent must be positive")
    cases = []
    for value in beta_percent:
        beta = float(value)
        if beta < 0.0:
            raise ValueError("beta_percent values must be nonnegative")
        cases.append(
            MirrorFreeBoundaryBetaCase(
                beta_percent=beta,
                beta_fraction=0.01 * beta,
                pressure_scale=scale * beta,
            )
        )
    return tuple(cases)


def make_mirror_free_boundary_circular_coil_scan(
    coils: MirrorCircularCoils,
    beta_percent: tuple[float, ...] = (1.0, 3.0, 10.0),
    *,
    pressure_scale_for_one_percent: float = 1.0,
) -> MirrorFreeBoundaryCircularCoilScan:
    """Return a serializable circular-coil beta-scan setup."""

    return MirrorFreeBoundaryCircularCoilScan(
        coils=coils,
        beta_cases=make_mirror_free_boundary_beta_cases(
            beta_percent,
            pressure_scale_for_one_percent=pressure_scale_for_one_percent,
        ),
    )


def write_mirror_free_boundary_circular_coil_scan(
    path: str | Path,
    scan: MirrorFreeBoundaryCircularCoilScan,
) -> Path:
    """Write a circular-coil beta-scan setup to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan.to_dict(), indent=2) + "\n")
    return path


def load_mirror_free_boundary_circular_coil_scan(path: str | Path) -> MirrorFreeBoundaryCircularCoilScan:
    """Load a circular-coil beta-scan setup from JSON."""

    return MirrorFreeBoundaryCircularCoilScan.from_dict(json.loads(Path(path).read_text()))


def mirror_boundary_from_external_axis_field(
    grid: MirrorGrid,
    axis_bz: Any,
    *,
    midplane_radius: float,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build an initial fixed boundary from sampled external on-axis ``Bz``."""

    z = np.asarray(grid.z, dtype=float)
    bz = np.asarray(axis_bz, dtype=float)
    if bz.shape != z.shape:
        raise ValueError(f"axis_bz must have shape {z.shape}, got {bz.shape}")
    radius = float(midplane_radius)
    if radius <= 0.0:
        raise ValueError("midplane_radius must be positive")
    midplane_bmag = float(np.interp(0.0, z, np.abs(bz)))
    if midplane_bmag <= 0.0:
        raise ValueError("midplane external field must be nonzero")
    psi_value = 0.5 * midplane_bmag * radius**2
    return mirror_boundary_from_on_axis_bz(psi_value, z, bz, radius_floor=radius_floor)


def initial_mirror_boundary_from_circular_coil_scan(
    grid: MirrorGrid,
    scan: MirrorFreeBoundaryCircularCoilScan,
    *,
    midplane_radius: float,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build the fixed-boundary baseline from a circular-coil scan setup."""

    axis_sample = sample_mirror_axis_external_field(grid, scan.coils)
    return mirror_boundary_from_external_axis_field(
        grid,
        axis_sample.bz,
        midplane_radius=midplane_radius,
        radius_floor=radius_floor,
    )


def mirror_lcfs_diagnostic_from_arrays(
    *,
    theta: Any,
    z: Any,
    boundary_r: Any,
    edge_internal_bmag: Any,
    external_sample: MirrorExternalFieldSample,
    edge_pressure: float,
    mu0: float = 1.0,
) -> MirrorLCFSDiagnostic:
    """Return LCFS diagnostics from arrays sampled on the side boundary."""

    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)
    boundary_r = np.asarray(boundary_r, dtype=float)
    if boundary_r.shape != (theta.size, z.size):
        raise ValueError(f"boundary_r must have shape {(theta.size, z.size)}, got {boundary_r.shape}")
    if z.size < 2:
        raise ValueError("at least two axial nodes are required for LCFS diagnostics")
    edge_internal_bmag = np.asarray(edge_internal_bmag, dtype=float)
    external_br = np.asarray(external_sample.br, dtype=float)
    external_bz = np.asarray(external_sample.bz, dtype=float)
    external_bmag = np.asarray(external_sample.bmag, dtype=float)
    expected_shape = boundary_r.shape
    if (
        external_br.shape != expected_shape
        or external_bz.shape != expected_shape
        or external_bmag.shape != expected_shape
    ):
        raise ValueError("external field sample must have shape (ntheta, nxi)")
    if edge_internal_bmag.shape != expected_shape:
        raise ValueError("internal edge |B| must have shape (ntheta, nxi)")
    if float(mu0) <= 0.0:
        raise ValueError("mu0 must be positive")
    edge_pressure = float(edge_pressure)

    external_bnormal, dr_dz = mirror_external_bnormal(boundary_r, z, external_sample, return_dr_dz=True)
    pressure_balance = edge_pressure + (edge_internal_bmag**2 - external_bmag**2) / (2.0 * float(mu0))
    return MirrorLCFSDiagnostic(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        boundary_dr_dz=dr_dz,
        external_bnormal=external_bnormal,
        external_bnormal_rms=float(np.sqrt(np.mean(external_bnormal**2))),
        external_bnormal_max=float(np.max(np.abs(external_bnormal))),
        pressure_balance=pressure_balance,
        pressure_balance_rms=float(np.sqrt(np.mean(pressure_balance**2))),
        pressure_balance_max=float(np.max(np.abs(pressure_balance))),
        internal_bmag=edge_internal_bmag,
        external_bmag=external_bmag,
        edge_pressure=edge_pressure,
    )


def mirror_lcfs_diagnostic(
    output: Any, external_sample: MirrorExternalFieldSample, *, mu0: float = 1.0
) -> MirrorLCFSDiagnostic:
    """Return side-boundary normal-field and total-pressure diagnostics.

    The diagnostic target is intentionally local to the side boundary.  It
    gives later LCFS update work a tested quantity to reduce without pretending
    that the current fixed-boundary baseline is already a free-boundary solve.
    """

    return mirror_lcfs_diagnostic_from_arrays(
        theta=output.theta,
        z=output.z,
        boundary_r=output.geometry.boundary_r,
        edge_internal_bmag=np.asarray(output.field.bmag)[-1],
        external_sample=external_sample,
        edge_pressure=float(np.asarray(output.profiles.pressure, dtype=float)[-1]),
        mu0=mu0,
    )


def mirror_external_bnormal(
    boundary_r: Any,
    z: Any,
    external_sample: MirrorExternalFieldSample,
    *,
    return_dr_dz: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return external normal field on a mirror side boundary."""

    boundary_r = np.asarray(boundary_r, dtype=float)
    z = np.asarray(z, dtype=float)
    external_br = np.asarray(external_sample.br, dtype=float)
    external_bz = np.asarray(external_sample.bz, dtype=float)
    if boundary_r.ndim != 2:
        raise ValueError("boundary_r must have shape (ntheta, nxi)")
    if z.ndim != 1 or z.size != boundary_r.shape[-1]:
        raise ValueError("z must be one-dimensional with length nxi")
    if z.size < 2:
        raise ValueError("at least two axial nodes are required")
    if external_br.shape != boundary_r.shape or external_bz.shape != boundary_r.shape:
        raise ValueError("external field sample must have shape (ntheta, nxi)")
    edge_order = 2 if z.size > 2 else 1
    dr_dz = np.gradient(boundary_r, z, axis=-1, edge_order=edge_order)
    normal_scale = np.sqrt(1.0 + dr_dz**2)
    external_bnormal = external_br / normal_scale - external_bz * dr_dz / normal_scale
    if return_dr_dz:
        return external_bnormal, dr_dz
    return external_bnormal


def mirror_external_pressure_balance_response(
    diagnostic: MirrorLCFSDiagnostic,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
    radius_step_fraction: float = 1.0e-3,
    radius_step_min: float = 1.0e-5,
    radius_floor: float = 1.0e-6,
    mu0: float = 1.0,
) -> np.ndarray:
    """Estimate ``d(pressure_balance)/dr`` from external magnetic pressure.

    This keeps the internal fixed-boundary equilibrium frozen and only measures
    how the external coil magnetic pressure changes when the side boundary is
    moved radially.  The resulting response is suitable for a damped first
    LCFS proposal, not for claiming a converged free-boundary equilibrium.
    """

    if float(mu0) <= 0.0:
        raise ValueError("mu0 must be positive")
    if float(radius_step_fraction) <= 0.0:
        raise ValueError("radius_step_fraction must be positive")
    if float(radius_step_min) <= 0.0:
        raise ValueError("radius_step_min must be positive")
    if float(radius_floor) <= 0.0:
        raise ValueError("radius_floor must be positive")

    radius = np.asarray(diagnostic.boundary_r, dtype=float)
    step = np.maximum(float(radius_step_fraction) * np.maximum(radius, float(radius_floor)), float(radius_step_min))
    r_plus = radius + step
    r_minus = np.maximum(radius - step, float(radius_floor))
    denominator = r_plus - r_minus
    if np.any(denominator <= 0.0):
        raise ValueError("finite-difference radius step collapsed")

    plus = _sample_field(
        r=jnp.asarray(r_plus),
        theta=jnp.asarray(diagnostic.theta),
        z=jnp.asarray(diagnostic.z),
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )
    minus = _sample_field(
        r=jnp.asarray(r_minus),
        theta=jnp.asarray(diagnostic.theta),
        z=jnp.asarray(diagnostic.z),
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )
    bmag_plus = np.asarray(plus.bmag, dtype=float)
    bmag_minus = np.asarray(minus.bmag, dtype=float)
    return -(bmag_plus**2 - bmag_minus**2) / (2.0 * float(mu0) * denominator)


def mirror_lcfs_residual(
    diagnostic: MirrorLCFSDiagnostic,
    *,
    pressure_scale: float | None = None,
    bnormal_scale: float | None = None,
    bnormal_weight: float = 1.0,
) -> MirrorLCFSResidual:
    """Return the normalized LCFS residual vector used by coupled solves."""

    pressure_rms = float(diagnostic.pressure_balance_rms)
    bnormal_rms = float(diagnostic.external_bnormal_rms)
    external_bmag_rms = float(np.sqrt(np.mean(np.asarray(diagnostic.external_bmag, dtype=float) ** 2)))
    pressure_scale = pressure_rms if pressure_scale is None else float(pressure_scale)
    bnormal_scale = external_bmag_rms if bnormal_scale is None else float(bnormal_scale)
    bnormal_weight = float(bnormal_weight)
    if pressure_scale <= 0.0:
        raise ValueError("pressure_scale must be positive")
    if bnormal_scale <= 0.0:
        raise ValueError("bnormal_scale must be positive")
    if bnormal_weight < 0.0:
        raise ValueError("bnormal_weight must be nonnegative")
    pressure_balance = np.asarray(getattr(diagnostic, "pressure_balance", pressure_rms), dtype=float)
    external_bnormal = np.asarray(getattr(diagnostic, "external_bnormal", bnormal_rms), dtype=float)
    pressure_component = pressure_balance / pressure_scale
    bnormal_component = np.sqrt(bnormal_weight) * external_bnormal / bnormal_scale
    vector = np.concatenate([pressure_component.ravel(), bnormal_component.ravel()])
    value = float(np.sqrt(np.mean(pressure_component**2) + np.mean(bnormal_component**2)))
    return MirrorLCFSResidual(
        vector=vector,
        pressure_component=pressure_component,
        bnormal_component=bnormal_component,
        value=value,
        pressure_balance_rms=pressure_rms,
        external_bnormal_rms=bnormal_rms,
        external_bmag_rms=external_bmag_rms,
        pressure_scale=float(pressure_scale),
        bnormal_scale=float(bnormal_scale),
        bnormal_weight=float(bnormal_weight),
    )


def mirror_lcfs_merit(
    diagnostic: MirrorLCFSDiagnostic,
    *,
    pressure_scale: float | None = None,
    bnormal_scale: float | None = None,
    bnormal_weight: float = 1.0,
) -> MirrorLCFSMerit:
    """Return a dimensionless side-boundary merit for LCFS pilot steps."""

    residual = mirror_lcfs_residual(
        diagnostic,
        pressure_scale=pressure_scale,
        bnormal_scale=bnormal_scale,
        bnormal_weight=bnormal_weight,
    )
    return MirrorLCFSMerit(
        value=residual.value,
        pressure_balance_rms=residual.pressure_balance_rms,
        external_bnormal_rms=residual.external_bnormal_rms,
        external_bmag_rms=residual.external_bmag_rms,
        pressure_scale=residual.pressure_scale,
        bnormal_scale=residual.bnormal_scale,
        bnormal_weight=residual.bnormal_weight,
    )


def mirror_free_boundary_residual(
    equilibrium_residual: Any,
    lcfs_residual: MirrorLCFSResidual,
    *,
    equilibrium_scale: float | None = None,
    equilibrium_weight: float = 1.0,
    lcfs_weight: float = 1.0,
) -> MirrorFreeBoundaryResidual:
    """Combine equilibrium and LCFS blocks for free-boundary least squares."""

    equilibrium = np.asarray(equilibrium_residual, dtype=float).ravel()
    if equilibrium.size == 0:
        raise ValueError("equilibrium_residual must contain at least one value")
    if not np.all(np.isfinite(equilibrium)):
        raise ValueError("equilibrium_residual must be finite")
    equilibrium_rms = float(np.sqrt(np.mean(equilibrium**2)))
    equilibrium_scale = equilibrium_rms if equilibrium_scale is None else float(equilibrium_scale)
    equilibrium_weight = float(equilibrium_weight)
    lcfs_weight = float(lcfs_weight)
    if equilibrium_scale <= 0.0:
        raise ValueError("equilibrium_scale must be positive")
    if equilibrium_weight < 0.0:
        raise ValueError("equilibrium_weight must be nonnegative")
    if lcfs_weight < 0.0:
        raise ValueError("lcfs_weight must be nonnegative")
    lcfs_vector = np.asarray(lcfs_residual.vector, dtype=float).ravel()
    if lcfs_vector.size == 0:
        raise ValueError("lcfs_residual.vector must contain at least one value")
    if not np.all(np.isfinite(lcfs_vector)):
        raise ValueError("lcfs_residual.vector must be finite")
    equilibrium_component = np.sqrt(equilibrium_weight) * equilibrium / equilibrium_scale
    lcfs_component = np.sqrt(lcfs_weight) * lcfs_vector
    vector = np.concatenate([equilibrium_component, lcfs_component])
    value = float(np.sqrt(np.mean(equilibrium_component**2) + np.mean(lcfs_component**2)))
    return MirrorFreeBoundaryResidual(
        vector=vector,
        equilibrium_component=equilibrium_component,
        lcfs_component=lcfs_component,
        value=value,
        equilibrium_rms=equilibrium_rms,
        lcfs_value=float(lcfs_residual.value),
        equilibrium_scale=float(equilibrium_scale),
        equilibrium_weight=equilibrium_weight,
        lcfs_weight=lcfs_weight,
    )


def _as_free_boundary_residual(value: Any) -> MirrorFreeBoundaryResidual:
    if not isinstance(value, MirrorFreeBoundaryResidual):
        raise TypeError("residual_function must return MirrorFreeBoundaryResidual")
    vector = np.asarray(value.vector, dtype=float).ravel()
    if vector.size == 0:
        raise ValueError("residual_function returned an empty residual vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual_function returned a non-finite residual vector")
    return value


def mirror_free_boundary_residual_jacobian_finite_difference(
    coefficients: Any,
    residual_function: Callable[[np.ndarray], MirrorFreeBoundaryResidual],
    *,
    finite_difference_step: float = 1.0e-6,
    residual: MirrorFreeBoundaryResidual | None = None,
) -> tuple[MirrorFreeBoundaryResidual, np.ndarray, np.ndarray]:
    """Return a central-difference Jacobian for a combined residual function.

    This helper is intended for CLI diagnostics and early coupled-solve
    prototypes where the residual builder may include non-JAX pieces.  JAX or
    implicit derivatives should replace it when the full residual path is
    differentiable.
    """

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    finite_difference_step = float(finite_difference_step)
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")

    base = _as_free_boundary_residual(residual_function(coefficients) if residual is None else residual)
    base_vector = np.asarray(base.vector, dtype=float).ravel()
    steps = finite_difference_step * np.maximum(1.0, np.abs(coefficients))
    jacobian = np.empty((base_vector.size, coefficients.size), dtype=float)
    for index, step in enumerate(steps):
        direction = np.zeros_like(coefficients)
        direction[index] = step
        plus = _as_free_boundary_residual(residual_function(coefficients + direction))
        minus = _as_free_boundary_residual(residual_function(coefficients - direction))
        plus_vector = np.asarray(plus.vector, dtype=float).ravel()
        minus_vector = np.asarray(minus.vector, dtype=float).ravel()
        if plus_vector.shape != base_vector.shape or minus_vector.shape != base_vector.shape:
            raise ValueError("residual_function must return vectors with a fixed shape")
        jacobian[:, index] = (plus_vector - minus_vector) / (2.0 * step)
    return base, jacobian, steps


def mirror_free_boundary_residual_vector_jacobian_jax(
    coefficients: Any,
    residual_vector_function: Callable[[Any], Any],
    *,
    mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a JAX Jacobian for a differentiable residual-vector function.

    This helper is for reduced free-boundary prototypes whose residual can be
    written as a pure JAX vector function of boundary parameters.  Host-side
    CLI workflows with file I/O or fixed-boundary trial solves should keep using
    finite differences or an adjoint/implicit wrapper around the differentiable
    pieces.
    """

    if jax is None:
        raise ImportError("JAX is required for mirror_free_boundary_residual_vector_jacobian_jax")
    coefficients = jnp.asarray(coefficients).ravel()
    if int(coefficients.size) == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(np.asarray(coefficients, dtype=float))):
        raise ValueError("coefficients must be finite")

    def vector_function(items):
        return jnp.ravel(jnp.asarray(residual_vector_function(items)))

    vector = vector_function(coefficients)
    if int(vector.size) == 0:
        raise ValueError("residual_vector_function must return at least one value")
    mode = str(mode).lower()
    if mode == "auto":
        mode = "forward" if int(coefficients.size) <= int(vector.size) else "reverse"
    if mode in ("forward", "fwd", "jacfwd"):
        jacobian = jax.jacfwd(vector_function)(coefficients)
    elif mode in ("reverse", "rev", "jacrev"):
        jacobian = jax.jacrev(vector_function)(coefficients)
    else:
        raise ValueError("mode must be 'auto', 'forward', or 'reverse'")

    vector_np = np.asarray(vector, dtype=float)
    jacobian_np = np.asarray(jacobian, dtype=float)
    if not np.all(np.isfinite(vector_np)):
        raise ValueError("residual_vector_function returned non-finite values")
    if not np.all(np.isfinite(jacobian_np)):
        raise ValueError("residual_vector_function Jacobian is non-finite")
    return vector_np, jacobian_np


def _residual_vector_numpy(
    residual_vector_function: Callable[[np.ndarray], Any], coefficients: np.ndarray
) -> np.ndarray:
    vector = np.asarray(residual_vector_function(coefficients), dtype=float).ravel()
    if vector.size == 0:
        raise ValueError("residual_vector_function must return at least one value")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual_vector_function returned non-finite values")
    return vector


def mirror_free_boundary_residual_vector_jacobian_finite_difference(
    coefficients: Any,
    residual_vector_function: Callable[[np.ndarray], Any],
    *,
    finite_difference_step: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a central-difference Jacobian for a residual-vector function."""

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    finite_difference_step = float(finite_difference_step)
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    base_vector = _residual_vector_numpy(residual_vector_function, coefficients)
    steps = finite_difference_step * np.maximum(1.0, np.abs(coefficients))
    jacobian = np.empty((base_vector.size, coefficients.size), dtype=float)
    for index, step in enumerate(steps):
        direction = np.zeros_like(coefficients)
        direction[index] = step
        plus = _residual_vector_numpy(residual_vector_function, coefficients + direction)
        minus = _residual_vector_numpy(residual_vector_function, coefficients - direction)
        if plus.shape != base_vector.shape or minus.shape != base_vector.shape:
            raise ValueError("residual_vector_function must return vectors with a fixed shape")
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return base_vector, jacobian, steps


def mirror_free_boundary_residual_vector_least_squares_step(
    coefficients: Any,
    residual_vector_function: Callable[[Any], Any],
    *,
    jacobian_backend: str = "finite_difference",
    jax_mode: str = "auto",
    finite_difference_step: float = 1.0e-6,
    damping: float = 1.0,
    max_relative_step: float = 0.25,
    ridge: float = 0.0,
    rcond: float | None = 1.0e-12,
    line_search_factors: Any = (1.0, 0.5, 0.25, 0.125),
    accept_tolerance: float = 1.0e-12,
) -> MirrorFreeBoundaryVectorLeastSquaresStep:
    """Return one LS update for a reduced free-boundary residual vector."""

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    backend = str(jacobian_backend).lower()
    if backend in ("finite_difference", "finite-difference", "fd"):
        residual_vector, jacobian, _steps = mirror_free_boundary_residual_vector_jacobian_finite_difference(
            coefficients,
            residual_vector_function,
            finite_difference_step=finite_difference_step,
        )
        backend = "finite_difference"
    elif backend in ("jax", "ad", "automatic"):
        residual_vector, jacobian = mirror_free_boundary_residual_vector_jacobian_jax(
            coefficients,
            residual_vector_function,
            mode=jax_mode,
        )
        backend = "jax"
    else:
        raise ValueError("jacobian_backend must be 'finite_difference' or 'jax'")

    damping = float(damping)
    max_relative_step = float(max_relative_step)
    ridge = float(ridge)
    accept_tolerance = float(accept_tolerance)
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if accept_tolerance < 0.0:
        raise ValueError("accept_tolerance must be nonnegative")
    factors = np.asarray(tuple(line_search_factors), dtype=float)
    if factors.ndim != 1 or factors.size == 0:
        raise ValueError("line_search_factors must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("line_search_factors must be finite and positive")

    if ridge > 0.0:
        lhs = np.vstack([jacobian, np.sqrt(ridge) * np.eye(coefficients.size)])
        rhs = np.concatenate([-residual_vector, np.zeros(coefficients.size)])
    else:
        lhs = jacobian
        rhs = -residual_vector
    raw_step, *_ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
    if not np.all(np.isfinite(raw_step)):
        raise ValueError("least-squares step is not finite")

    step_limit = max_relative_step * np.maximum(1.0, np.abs(coefficients))
    limited_step = np.clip(damping * raw_step, -step_limit, step_limit)
    predicted_vector = residual_vector + jacobian @ limited_step
    residual_value = float(np.sqrt(np.mean(residual_vector**2)))
    predicted_value = float(np.sqrt(np.mean(predicted_vector**2)))

    best_factor = float(factors[0])
    best_coefficients = coefficients + best_factor * limited_step
    best_vector = _residual_vector_numpy(residual_vector_function, best_coefficients)
    best_value = float(np.sqrt(np.mean(best_vector**2)))
    accepted = best_value <= residual_value + accept_tolerance
    for factor in factors[1:]:
        trial_coefficients = coefficients + float(factor) * limited_step
        trial_vector = _residual_vector_numpy(residual_vector_function, trial_coefficients)
        trial_value = float(np.sqrt(np.mean(trial_vector**2)))
        trial_accepted = trial_value <= residual_value + accept_tolerance
        if trial_accepted and not accepted:
            best_factor = float(factor)
            best_coefficients = trial_coefficients
            best_vector = trial_vector
            best_value = trial_value
            accepted = True
        elif trial_accepted == accepted and trial_value < best_value:
            best_factor = float(factor)
            best_coefficients = trial_coefficients
            best_vector = trial_vector
            best_value = trial_value
            accepted = trial_accepted

    if not accepted:
        best_coefficients = coefficients.copy()
        best_factor = 0.0
        best_vector = residual_vector
        best_value = residual_value

    return MirrorFreeBoundaryVectorLeastSquaresStep(
        coefficients=coefficients,
        residual_vector=residual_vector,
        jacobian=jacobian,
        raw_step=raw_step,
        limited_step=limited_step,
        line_search_factor=best_factor,
        new_coefficients=best_coefficients,
        trial_vector=best_vector,
        residual_value=residual_value,
        trial_value=best_value,
        predicted_vector=predicted_vector,
        predicted_value=predicted_value,
        accepted=bool(accepted),
        jacobian_backend=backend,
        damping=damping,
        max_relative_step=max_relative_step,
        ridge=ridge,
        rcond=rcond,
    )


def mirror_free_boundary_residual_vector_least_squares_solve(
    coefficients: Any,
    residual_vector_function: Callable[[Any], Any],
    *,
    max_steps: int = 8,
    target_residual: float = 1.0e-10,
    stagnation_rtol: float = 0.0,
    jacobian_backend: str = "finite_difference",
    jax_mode: str = "auto",
    finite_difference_step: float = 1.0e-6,
    damping: float = 1.0,
    max_relative_step: float = 0.25,
    ridge: float = 0.0,
    rcond: float | None = 1.0e-12,
    line_search_factors: Any = (1.0, 0.5, 0.25, 0.125),
    accept_tolerance: float = 1.0e-12,
) -> MirrorFreeBoundaryVectorLeastSquaresSolveResult:
    """Solve a reduced free-boundary residual-vector problem by guarded LS steps.

    This nonlinear loop is for compact residual-vector prototypes. It does not
    run host-side fixed-boundary trial solves; those remain in the circular-coil
    CLI workflow until the full coupled residual path is promoted.
    """

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    max_steps = int(max_steps)
    target_residual = float(target_residual)
    stagnation_rtol = float(stagnation_rtol)
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if target_residual < 0.0:
        raise ValueError("target_residual must be nonnegative")
    if stagnation_rtol < 0.0:
        raise ValueError("stagnation_rtol must be nonnegative")

    current_coefficients = coefficients.copy()
    initial_vector = _residual_vector_numpy(residual_vector_function, current_coefficients)
    current_value = float(np.sqrt(np.mean(initial_vector**2)))
    initial_value = current_value
    rows: list[MirrorFreeBoundaryVectorLeastSquaresSolveRow] = []
    accepted_steps = 0
    stop_reason = "target_residual" if current_value <= target_residual else "max_steps"
    converged = current_value <= target_residual

    for step_index in range(1, max_steps + 1):
        if current_value <= target_residual:
            stop_reason = "target_residual"
            converged = True
            break
        ls_step = mirror_free_boundary_residual_vector_least_squares_step(
            current_coefficients,
            residual_vector_function,
            jacobian_backend=jacobian_backend,
            jax_mode=jax_mode,
            finite_difference_step=finite_difference_step,
            damping=damping,
            max_relative_step=max_relative_step,
            ridge=ridge,
            rcond=rcond,
            line_search_factors=line_search_factors,
            accept_tolerance=accept_tolerance,
        )
        if not bool(ls_step.accepted):
            stop_reason = "ls_step_not_accepted"
            rows.append(
                MirrorFreeBoundaryVectorLeastSquaresSolveRow(
                    step=step_index,
                    ls_step=ls_step,
                    residual_value_before=current_value,
                    residual_value_after=ls_step.trial_value,
                    accepted=False,
                    improvement_fraction=None,
                    stop_reason=stop_reason,
                )
            )
            converged = current_value <= target_residual
            break

        denominator = max(abs(current_value), np.finfo(float).tiny)
        improvement_fraction = float((current_value - ls_step.trial_value) / denominator)
        current_coefficients = np.asarray(ls_step.new_coefficients, dtype=float).ravel()
        current_value = float(ls_step.trial_value)
        accepted_steps += 1
        row_stop_reason: str | None = None
        if current_value <= target_residual:
            stop_reason = "target_residual"
            row_stop_reason = stop_reason
            converged = True
        elif stagnation_rtol > 0.0 and improvement_fraction <= stagnation_rtol:
            stop_reason = "stagnation"
            row_stop_reason = stop_reason
            converged = False
        elif step_index == max_steps:
            stop_reason = "max_steps"
            row_stop_reason = stop_reason
            converged = False
        rows.append(
            MirrorFreeBoundaryVectorLeastSquaresSolveRow(
                step=step_index,
                ls_step=ls_step,
                residual_value_before=float(ls_step.residual_value),
                residual_value_after=current_value,
                accepted=True,
                improvement_fraction=improvement_fraction,
                stop_reason=row_stop_reason,
            )
        )
        if row_stop_reason is not None:
            break

    return MirrorFreeBoundaryVectorLeastSquaresSolveResult(
        initial_coefficients=coefficients,
        final_coefficients=current_coefficients,
        initial_residual_value=initial_value,
        final_residual_value=current_value,
        rows=tuple(rows),
        accepted_steps=accepted_steps,
        stop_reason=stop_reason,
        converged=bool(converged),
    )


def mirror_free_boundary_least_squares_step(
    coefficients: Any,
    residual_function: Callable[[np.ndarray], MirrorFreeBoundaryResidual],
    *,
    finite_difference_step: float = 1.0e-6,
    damping: float = 1.0,
    max_relative_step: float = 0.25,
    ridge: float = 0.0,
    rcond: float | None = 1.0e-12,
    line_search_factors: Any = (1.0, 0.5, 0.25, 0.125),
    accept_tolerance: float = 1.0e-12,
) -> MirrorFreeBoundaryLeastSquaresStep:
    """Return one damped least-squares update of boundary coefficients.

    The step solves ``min ||J dx + F||`` for the current combined residual
    vector ``F`` and finite-difference Jacobian ``J``.  It then tries a small
    backtracking list and reports the best non-increasing trial. If every trial
    increases the residual, the returned step is marked unaccepted and keeps
    the original coefficients.
    """

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError("coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite")
    damping = float(damping)
    max_relative_step = float(max_relative_step)
    ridge = float(ridge)
    accept_tolerance = float(accept_tolerance)
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if accept_tolerance < 0.0:
        raise ValueError("accept_tolerance must be nonnegative")
    factors = np.asarray(tuple(line_search_factors), dtype=float)
    if factors.ndim != 1 or factors.size == 0:
        raise ValueError("line_search_factors must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("line_search_factors must be finite and positive")

    residual, jacobian, steps = mirror_free_boundary_residual_jacobian_finite_difference(
        coefficients,
        residual_function,
        finite_difference_step=finite_difference_step,
    )
    vector = np.asarray(residual.vector, dtype=float).ravel()
    if ridge > 0.0:
        lhs = np.vstack([jacobian, np.sqrt(ridge) * np.eye(coefficients.size)])
        rhs = np.concatenate([-vector, np.zeros(coefficients.size)])
    else:
        lhs = jacobian
        rhs = -vector
    raw_step, *_ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
    if not np.all(np.isfinite(raw_step)):
        raise ValueError("least-squares step is not finite")

    step_limit = max_relative_step * np.maximum(1.0, np.abs(coefficients))
    limited_step = np.clip(damping * raw_step, -step_limit, step_limit)
    predicted_vector = vector + jacobian @ limited_step
    predicted_value = float(np.sqrt(np.mean(predicted_vector**2)))

    best_factor = float(factors[0])
    best_coefficients = coefficients + best_factor * limited_step
    best_residual = _as_free_boundary_residual(residual_function(best_coefficients))
    best_value = float(best_residual.value)
    accepted = best_value <= float(residual.value) + accept_tolerance
    for factor in factors[1:]:
        trial_coefficients = coefficients + float(factor) * limited_step
        trial_residual = _as_free_boundary_residual(residual_function(trial_coefficients))
        trial_value = float(trial_residual.value)
        trial_accepted = trial_value <= float(residual.value) + accept_tolerance
        if trial_accepted and not accepted:
            best_factor = float(factor)
            best_coefficients = trial_coefficients
            best_residual = trial_residual
            best_value = trial_value
            accepted = True
        elif trial_accepted == accepted and trial_value < best_value:
            best_factor = float(factor)
            best_coefficients = trial_coefficients
            best_residual = trial_residual
            best_value = trial_value
            accepted = trial_accepted

    if not accepted:
        best_coefficients = coefficients.copy()
        best_factor = 0.0
        best_residual = residual

    return MirrorFreeBoundaryLeastSquaresStep(
        coefficients=coefficients,
        residual=residual,
        jacobian=jacobian,
        finite_difference_steps=steps,
        raw_step=raw_step,
        limited_step=limited_step,
        line_search_factor=best_factor,
        new_coefficients=best_coefficients,
        trial_residual=best_residual,
        predicted_vector=predicted_vector,
        predicted_value=predicted_value,
        accepted=accepted,
        damping=damping,
        max_relative_step=max_relative_step,
        ridge=ridge,
        rcond=rcond,
    )


def _as_loop_state(value: Any, *, label: str) -> MirrorFreeBoundaryLoopState:
    if not isinstance(value, MirrorFreeBoundaryLoopState):
        raise TypeError(f"{label} must be a MirrorFreeBoundaryLoopState")
    coefficients = np.asarray(value.coefficients, dtype=float).ravel()
    if coefficients.size == 0:
        raise ValueError(f"{label}.coefficients must contain at least one value")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{label}.coefficients must be finite")
    _as_free_boundary_residual(value.residual)
    merit = float(value.merit)
    if not np.isfinite(merit) or merit < 0.0:
        raise ValueError(f"{label}.merit must be finite and nonnegative")
    equilibrium_value = value.equilibrium_value
    if equilibrium_value is not None and not np.isfinite(float(equilibrium_value)):
        raise ValueError(f"{label}.equilibrium_value must be finite when provided")
    return MirrorFreeBoundaryLoopState(
        coefficients=coefficients,
        residual=value.residual,
        merit=merit,
        equilibrium_value=None if equilibrium_value is None else float(equilibrium_value),
        payload=value.payload,
    )


def mirror_free_boundary_guarded_least_squares_loop(
    initial_state: MirrorFreeBoundaryLoopState,
    residual_function: Callable[[MirrorFreeBoundaryLoopState, np.ndarray], MirrorFreeBoundaryResidual],
    *,
    trial_function: Callable[
        [MirrorFreeBoundaryLoopState, MirrorFreeBoundaryLeastSquaresStep],
        MirrorFreeBoundaryLoopState,
    ]
    | None = None,
    max_steps: int = 1,
    target_merit: float = 0.0,
    stagnation_rtol: float = 0.0,
    equilibrium_growth_limit: float = 0.0,
    equilibrium_growth_reference: float | None = None,
    finite_difference_step: float = 1.0e-6,
    damping: float = 1.0,
    max_relative_step: float = 0.25,
    ridge: float = 0.0,
    rcond: float | None = 1.0e-12,
    line_search_factors: Any = (1.0, 0.5, 0.25, 0.125),
    accept_tolerance: float = 1.0e-12,
) -> MirrorFreeBoundaryLoopResult:
    """Run a guarded loop of free-boundary least-squares updates.

    The residual callback builds the linearized residual around the current
    state.  The trial callback can then run an expensive realized solve and
    return the next state.  If no trial callback is supplied, the loop uses the
    LS trial residual directly, which is useful for small reduced prototypes.
    """

    current = _as_loop_state(initial_state, label="initial_state")
    max_steps = int(max_steps)
    target_merit = float(target_merit)
    stagnation_rtol = float(stagnation_rtol)
    equilibrium_growth_limit = float(equilibrium_growth_limit)
    accept_tolerance = float(accept_tolerance)
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_merit < 0.0:
        raise ValueError("target_merit must be nonnegative")
    if stagnation_rtol < 0.0:
        raise ValueError("stagnation_rtol must be nonnegative")
    if equilibrium_growth_limit < 0.0:
        raise ValueError("equilibrium_growth_limit must be nonnegative")
    if accept_tolerance < 0.0:
        raise ValueError("accept_tolerance must be nonnegative")

    growth_reference = equilibrium_growth_reference
    if growth_reference is None:
        growth_reference = current.equilibrium_value
    if growth_reference is not None:
        growth_reference = max(abs(float(growth_reference)), 1.0e-300)

    rows: list[MirrorFreeBoundaryLoopRow] = []
    final_state = current
    stop_reason: str | None = None
    for step_index in range(1, max_steps + 1):
        state_before = current

        def local_residual_function(coefficients: np.ndarray) -> MirrorFreeBoundaryResidual:
            return residual_function(state_before, coefficients)

        ls_step = mirror_free_boundary_least_squares_step(
            state_before.coefficients,
            local_residual_function,
            finite_difference_step=finite_difference_step,
            damping=damping,
            max_relative_step=max_relative_step,
            ridge=ridge,
            rcond=rcond,
            line_search_factors=line_search_factors,
            accept_tolerance=accept_tolerance,
        )
        if not bool(ls_step.accepted):
            stop_reason = "ls_step_not_accepted"
            rows.append(
                MirrorFreeBoundaryLoopRow(
                    step=step_index,
                    state_before=state_before,
                    ls_step=ls_step,
                    trial_state=None,
                    accepted=False,
                    status="skipped",
                    rejection_reason="ls_step_not_accepted",
                    stop_reason=stop_reason,
                    merit_improvement_fraction=None,
                    equilibrium_growth_ratio=None,
                )
            )
            break

        if trial_function is None:
            trial_state = MirrorFreeBoundaryLoopState(
                coefficients=ls_step.new_coefficients,
                residual=ls_step.trial_residual,
                merit=float(ls_step.trial_residual.value),
                equilibrium_value=state_before.equilibrium_value,
            )
        else:
            trial_state = trial_function(state_before, ls_step)
        trial_state = _as_loop_state(trial_state, label="trial_state")

        merit_improvement_fraction = float(1.0 - float(trial_state.merit) / max(float(state_before.merit), 1.0e-300))
        equilibrium_growth_ratio = None
        if growth_reference is not None and trial_state.equilibrium_value is not None:
            equilibrium_growth_ratio = float(trial_state.equilibrium_value) / growth_reference

        accepted = True
        status = "accepted"
        rejection_reason = None
        row_stop_reason = None
        if float(trial_state.merit) > float(state_before.merit) + accept_tolerance:
            accepted = False
            status = "rejected"
            rejection_reason = "merit_increase"
            row_stop_reason = "rejected_merit_increase"
        elif (
            equilibrium_growth_ratio is not None
            and equilibrium_growth_limit > 0.0
            and equilibrium_growth_ratio > equilibrium_growth_limit
        ):
            accepted = False
            status = "rejected"
            rejection_reason = "equilibrium_growth_guard"
            row_stop_reason = "equilibrium_growth_guard"
        elif float(trial_state.merit) <= target_merit:
            row_stop_reason = "target_merit"
        elif merit_improvement_fraction <= stagnation_rtol:
            row_stop_reason = "merit_stagnation"
        elif step_index == max_steps:
            row_stop_reason = "max_steps"

        rows.append(
            MirrorFreeBoundaryLoopRow(
                step=step_index,
                state_before=state_before,
                ls_step=ls_step,
                trial_state=trial_state,
                accepted=accepted,
                status=status,
                rejection_reason=rejection_reason,
                stop_reason=row_stop_reason,
                merit_improvement_fraction=merit_improvement_fraction,
                equilibrium_growth_ratio=equilibrium_growth_ratio,
            )
        )
        if not accepted:
            stop_reason = row_stop_reason
            break

        current = trial_state
        final_state = trial_state
        if row_stop_reason is not None:
            stop_reason = row_stop_reason
            break

    return MirrorFreeBoundaryLoopResult(
        initial_state=initial_state,
        final_state=final_state,
        rows=tuple(rows),
        accepted_steps=sum(bool(row.accepted) for row in rows),
        stop_reason=stop_reason,
        converged=stop_reason == "target_merit",
    )


def propose_axisymmetric_mirror_lcfs_update(
    diagnostic: MirrorLCFSDiagnostic,
    pressure_response: Any,
    *,
    damping: float = 0.25,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
    preserve_caps: bool = True,
    cap_taper_power: float = 2.0,
    smoothing_passes: int = 1,
) -> MirrorLCFSUpdateProposal:
    """Return a damped axisymmetric radius proposal from pressure imbalance.

    The update is a clipped Newton step for the theta-averaged side-boundary
    pressure-balance residual.  Cap radii are preserved by default, and the
    update is tapered smoothly toward the caps to avoid creating large axial
    side-boundary slopes.
    """

    damping = float(damping)
    max_relative_step = float(max_relative_step)
    radius_floor = float(radius_floor)
    cap_taper_power = float(cap_taper_power)
    smoothing_passes = int(smoothing_passes)
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if radius_floor <= 0.0:
        raise ValueError("radius_floor must be positive")
    if cap_taper_power < 0.0:
        raise ValueError("cap_taper_power must be nonnegative")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be nonnegative")

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    response = np.asarray(pressure_response, dtype=float)
    if response.shape == np.asarray(diagnostic.boundary_r).shape:
        response = np.mean(response, axis=0)
    elif response.shape != radius.shape:
        raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    if not (np.all(np.isfinite(radius)) and np.all(np.isfinite(residual)) and np.all(np.isfinite(response))):
        raise ValueError("LCFS update inputs must be finite")

    raw_delta = np.zeros_like(radius)
    active = np.abs(response) > np.finfo(float).eps
    raw_delta[active] = -residual[active] / response[active]
    limit = max_relative_step * np.maximum(radius, radius_floor)
    delta = np.clip(damping * raw_delta, -limit, limit)
    if preserve_caps and cap_taper_power > 0.0:
        normalized_z = (z - z[0]) / (z[-1] - z[0])
        delta *= np.sin(np.pi * normalized_z) ** cap_taper_power
    for _ in range(smoothing_passes):
        if delta.size > 2:
            smoothed = delta.copy()
            smoothed[1:-1] = 0.25 * delta[:-2] + 0.5 * delta[1:-1] + 0.25 * delta[2:]
            delta = np.clip(smoothed, -limit, limit)
    if preserve_caps:
        delta[0] = 0.0
        delta[-1] = 0.0
    new_radius = np.maximum(radius + delta, radius_floor)
    delta = new_radius - radius
    predicted = residual + response * delta
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=damping,
        max_relative_step=max_relative_step,
        cap_taper_power=cap_taper_power,
        smoothing_passes=smoothing_passes,
        preserve_caps=bool(preserve_caps),
        boundary=MirrorBoundary.tabulated_radius(xi, new_radius),
        strategy="local_pressure",
    )


def propose_axisymmetric_mirror_lcfs_scale_update(
    diagnostic: MirrorLCFSDiagnostic,
    pressure_response: Any,
    *,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
) -> MirrorLCFSUpdateProposal:
    """Return a shape-preserving radius-scale proposal.

    This candidate keeps the axial boundary shape smooth and changes the flux
    tube radius by one global scale factor chosen from the linearized pressure
    response.  It is useful as a normal-field-friendly alternative to nodal
    pressure updates.
    """

    max_relative_step = float(max_relative_step)
    radius_floor = float(radius_floor)
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if radius_floor <= 0.0:
        raise ValueError("radius_floor must be positive")

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    response = np.asarray(pressure_response, dtype=float)
    if response.shape == np.asarray(diagnostic.boundary_r).shape:
        response = np.mean(response, axis=0)
    elif response.shape != radius.shape:
        raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    if not (np.all(np.isfinite(radius)) and np.all(np.isfinite(residual)) and np.all(np.isfinite(response))):
        raise ValueError("LCFS scale-update inputs must be finite")

    direction = radius.copy()
    linear_response = response * direction
    denom = float(np.dot(linear_response, linear_response))
    scale_step = 0.0 if denom <= np.finfo(float).eps else -float(np.dot(residual, linear_response)) / denom
    scale_step = float(np.clip(scale_step, -max_relative_step, max_relative_step))
    delta = scale_step * direction
    new_radius = np.maximum(radius + delta, radius_floor)
    delta = new_radius - radius
    predicted = residual + response * delta
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=abs(scale_step),
        max_relative_step=max_relative_step,
        cap_taper_power=0.0,
        smoothing_passes=0,
        preserve_caps=False,
        boundary=MirrorBoundary.tabulated_radius(xi, new_radius),
        strategy="scale_pressure",
    )


def propose_axisymmetric_mirror_lcfs_noop_update(
    diagnostic: MirrorLCFSDiagnostic,
    pressure_response: Any | None = None,
) -> MirrorLCFSUpdateProposal:
    """Return an explicit no-op LCFS proposal."""

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    if pressure_response is None:
        response = np.zeros_like(radius)
    else:
        response = np.asarray(pressure_response, dtype=float)
        if response.shape == np.asarray(diagnostic.boundary_r).shape:
            response = np.mean(response, axis=0)
        elif response.shape != radius.shape:
            raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=radius.copy(),
        delta_radius=np.zeros_like(radius),
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=residual.copy(),
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(residual**2))),
        damping=0.0,
        max_relative_step=0.0,
        cap_taper_power=0.0,
        smoothing_passes=0,
        preserve_caps=True,
        boundary=MirrorBoundary.tabulated_radius(xi, radius),
        strategy="noop",
    )


def propose_axisymmetric_mirror_lcfs_bnormal_update(
    diagnostic: MirrorLCFSDiagnostic,
    external_sample: MirrorExternalFieldSample,
    pressure_response: Any | None = None,
    *,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
    slope_limit: float = 5.0,
    smoothing_passes: int = 1,
) -> MirrorLCFSUpdateProposal:
    """Return a radius proposal that moves toward ``B_ext.n = 0``.

    The target shape is obtained by integrating the axisymmetric field-line
    slope ``dr/dz ~= B_r / B_z`` from the current midplane radius, then taking
    a clipped step toward that smooth target.  This is a candidate direction
    for pilot line searches, not a complete free-boundary solve.
    """

    max_relative_step = float(max_relative_step)
    radius_floor = float(radius_floor)
    slope_limit = float(slope_limit)
    smoothing_passes = int(smoothing_passes)
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if radius_floor <= 0.0:
        raise ValueError("radius_floor must be positive")
    if slope_limit <= 0.0:
        raise ValueError("slope_limit must be positive")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be nonnegative")

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    external_br = np.mean(np.asarray(external_sample.br, dtype=float), axis=0)
    external_bz = np.mean(np.asarray(external_sample.bz, dtype=float), axis=0)
    if external_br.shape != radius.shape or external_bz.shape != radius.shape:
        raise ValueError("external field sample must have shape (ntheta, nxi)")
    slope = np.zeros_like(radius)
    active = np.abs(external_bz) > np.finfo(float).eps
    slope[active] = external_br[active] / external_bz[active]
    slope = np.clip(slope, -slope_limit, slope_limit)

    center = int(np.argmin(np.abs(z)))
    target = np.empty_like(radius)
    target[center] = radius[center]
    for index in range(center + 1, z.size):
        dz = z[index] - z[index - 1]
        target[index] = target[index - 1] + 0.5 * (slope[index - 1] + slope[index]) * dz
    for index in range(center - 1, -1, -1):
        dz = z[index + 1] - z[index]
        target[index] = target[index + 1] - 0.5 * (slope[index + 1] + slope[index]) * dz

    delta = target - radius
    limit = max_relative_step * np.maximum(radius, radius_floor)
    delta = np.clip(delta, -limit, limit)
    for _ in range(smoothing_passes):
        if delta.size > 2:
            smoothed = delta.copy()
            smoothed[1:-1] = 0.25 * delta[:-2] + 0.5 * delta[1:-1] + 0.25 * delta[2:]
            delta = np.clip(smoothed, -limit, limit)
    new_radius = np.maximum(radius + delta, radius_floor)
    delta = new_radius - radius
    if pressure_response is None:
        response = np.zeros_like(radius)
    else:
        response = np.asarray(pressure_response, dtype=float)
        if response.shape == np.asarray(diagnostic.boundary_r).shape:
            response = np.mean(response, axis=0)
        elif response.shape != radius.shape:
            raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    predicted = residual + response * delta
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=1.0,
        max_relative_step=max_relative_step,
        cap_taper_power=0.0,
        smoothing_passes=smoothing_passes,
        preserve_caps=False,
        boundary=MirrorBoundary.tabulated_radius(xi, new_radius),
        strategy="bnormal_slope",
    )


def propose_axisymmetric_mirror_lcfs_mixed_update(
    diagnostic: MirrorLCFSDiagnostic,
    external_sample: MirrorExternalFieldSample,
    pressure_response: Any,
    *,
    scale_fractions: Any = (0.25, 0.5, 0.75, 1.0),
    bnormal_fractions: Any = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
    slope_limit: float = 5.0,
    smoothing_passes: int = 1,
    bnormal_weight: float = 1.0,
    bnormal_nonincrease_tolerance: float = 1.0e-14,
) -> MirrorLCFSUpdateProposal:
    """Search a small scale/normal-field basis for an LCFS radius proposal.

    The search combines the smooth pressure scale direction with the
    field-line-slope direction and keeps the best locally predicted candidate
    that improves the combined pressure/normal-field merit without increasing
    ``B_ext.n``.  If no locally valid candidate exists, the best mixed
    candidate is still returned so callers can score it against an explicit
    no-op fallback using their preferred field model.
    """

    scale_values = np.asarray(tuple(scale_fractions), dtype=float)
    bnormal_values = np.asarray(tuple(bnormal_fractions), dtype=float)
    if scale_values.ndim != 1 or scale_values.size == 0:
        raise ValueError("scale_fractions must be a nonempty one-dimensional sequence")
    if bnormal_values.ndim != 1 or bnormal_values.size == 0:
        raise ValueError("bnormal_fractions must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0.0):
        raise ValueError("scale_fractions must be finite and positive")
    if not np.all(np.isfinite(bnormal_values)) or np.any(bnormal_values <= 0.0):
        raise ValueError("bnormal_fractions must be finite and positive")
    bnormal_weight = float(bnormal_weight)
    if bnormal_weight < 0.0:
        raise ValueError("bnormal_weight must be nonnegative")
    tolerance = float(bnormal_nonincrease_tolerance)
    if tolerance < 0.0:
        raise ValueError("bnormal_nonincrease_tolerance must be nonnegative")

    scale = propose_axisymmetric_mirror_lcfs_scale_update(
        diagnostic,
        pressure_response,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
    )
    bnormal = propose_axisymmetric_mirror_lcfs_bnormal_update(
        diagnostic,
        external_sample,
        pressure_response,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
        slope_limit=slope_limit,
        smoothing_passes=smoothing_passes,
    )
    z = scale.z
    radius = scale.old_radius
    residual = scale.pressure_balance_before
    response = scale.pressure_response
    boundary_shape = np.asarray(diagnostic.boundary_r).shape
    current_bnormal = mirror_external_bnormal(np.broadcast_to(radius[None, :], boundary_shape), z, external_sample)
    baseline_pressure_rms = float(np.sqrt(np.mean(residual**2)))
    baseline_bnormal_rms = float(np.sqrt(np.mean(np.asarray(current_bnormal, dtype=float) ** 2)))
    external_bmag_scale = float(np.sqrt(np.mean(np.asarray(external_sample.bmag, dtype=float) ** 2)))
    pressure_scale = max(baseline_pressure_rms, np.finfo(float).tiny)
    bnormal_scale = max(external_bmag_scale, np.finfo(float).tiny)
    baseline_score = float(
        np.sqrt(
            (baseline_pressure_rms / pressure_scale) ** 2 + bnormal_weight * (baseline_bnormal_rms / bnormal_scale) ** 2
        )
    )
    limit = float(max_relative_step) * np.maximum(radius, float(radius_floor))
    best_allowed: tuple[float, float, np.ndarray, np.ndarray, float, float] | None = None
    best_any: tuple[float, float, np.ndarray, np.ndarray, float, float] | None = None
    for scale_fraction in scale_values:
        for bnormal_fraction in bnormal_values:
            delta = scale_fraction * scale.delta_radius + bnormal_fraction * bnormal.delta_radius
            delta = np.clip(delta, -limit, limit)
            new_radius = np.maximum(radius + delta, float(radius_floor))
            delta = new_radius - radius
            predicted_pressure = residual + response * delta
            pressure_rms = float(np.sqrt(np.mean(predicted_pressure**2)))
            bnormal_predicted = mirror_external_bnormal(
                np.broadcast_to(new_radius[None, :], boundary_shape),
                z,
                external_sample,
            )
            bnormal_rms = float(np.sqrt(np.mean(np.asarray(bnormal_predicted, dtype=float) ** 2)))
            score = float(
                np.sqrt((pressure_rms / pressure_scale) ** 2 + bnormal_weight * (bnormal_rms / bnormal_scale) ** 2)
            )
            row = (score, bnormal_rms, new_radius, predicted_pressure, float(scale_fraction), float(bnormal_fraction))
            if best_any is None or row[:2] < best_any[:2]:
                best_any = row
            if bnormal_rms <= baseline_bnormal_rms + tolerance and score <= baseline_score + tolerance:
                if best_allowed is None or row[:2] < best_allowed[:2]:
                    best_allowed = row

    if best_any is None:
        raise ValueError("at least one mixed LCFS candidate is required")
    _, _, new_radius, predicted, scale_fraction, _bnormal_fraction = best_allowed or best_any
    delta = new_radius - radius
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=scale.xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=baseline_pressure_rms,
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=scale_fraction,
        max_relative_step=float(max_relative_step),
        cap_taper_power=0.0,
        smoothing_passes=int(smoothing_passes),
        preserve_caps=False,
        boundary=MirrorBoundary.tabulated_radius(scale.xi, new_radius),
        strategy="mixed_scale_bnormal",
    )


def propose_axisymmetric_mirror_lcfs_candidate_set(
    diagnostic: MirrorLCFSDiagnostic,
    external_sample: MirrorExternalFieldSample,
    pressure_response: Any,
    *,
    damping: float = 0.25,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
    preserve_caps: bool = True,
    cap_taper_power: float = 2.0,
    smoothing_passes: int = 1,
    bnormal_weight: float = 1.0,
) -> tuple[MirrorLCFSUpdateProposal, ...]:
    """Return the standard axisymmetric LCFS proposal candidates."""

    local = propose_axisymmetric_mirror_lcfs_update(
        diagnostic,
        pressure_response,
        damping=damping,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
        preserve_caps=preserve_caps,
        cap_taper_power=cap_taper_power,
        smoothing_passes=smoothing_passes,
    )
    scale = propose_axisymmetric_mirror_lcfs_scale_update(
        diagnostic,
        pressure_response,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
    )
    bnormal = propose_axisymmetric_mirror_lcfs_bnormal_update(
        diagnostic,
        external_sample,
        pressure_response,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
        smoothing_passes=smoothing_passes,
    )
    mixed = propose_axisymmetric_mirror_lcfs_mixed_update(
        diagnostic,
        external_sample,
        pressure_response,
        max_relative_step=max_relative_step,
        radius_floor=radius_floor,
        smoothing_passes=smoothing_passes,
        bnormal_weight=bnormal_weight,
    )
    noop = propose_axisymmetric_mirror_lcfs_noop_update(diagnostic, pressure_response)
    return (local, scale, bnormal, mixed, noop)
