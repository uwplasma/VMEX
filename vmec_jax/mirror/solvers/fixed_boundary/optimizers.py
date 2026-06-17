"""Small projected optimizers for fixed-boundary mirror solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmec_jax._compat import jax, jnp

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from ...kernels.forces import (
    axisym_energy_value_and_gradient,
    axisym_projected_energy_residual,
    axisym_total_energy_jax,
    energy_value_and_gradient_3d,
    projected_energy_residual_3d,
)
from ...kernels.geometry import evaluate_axisym_geometry, evaluate_geometry_3d


@dataclass(frozen=True)
class OptimizerOptions:
    """Numerical options for fixed-boundary optimizer stages."""

    optimizer: str = "gradient_descent"
    maxiter: int = 50
    tolerance: float = 1.0e-8
    step_size: float = 1.0e-3
    min_step_size: float = 1.0e-12
    ftol: float | None = None
    line_search_steps: int = 16
    reduced_coordinate_scaling: str = "geometry"
    residual_linear_maxiter: int = 16
    residual_preconditioner: str = "radial_xi_tridi"
    residual_radial_alpha: float = 0.5
    residual_lambda_alpha: float = 0.5
    residual_xi_alpha: float = 0.2
    mu0: float = 4.0e-7 * np.pi


@dataclass(frozen=True)
class OptimizerStep:
    """Accepted optimizer step payload."""

    state: MirrorStateAxisym | MirrorState3D
    energy: float
    residual_norm: float
    step_size: float
    accepted: bool


@dataclass(frozen=True)
class OptimizerRun:
    """Multi-step optimizer payload."""

    state: MirrorStateAxisym | MirrorState3D
    steps: tuple[OptimizerStep, ...]
    success: bool = False
    status: int = 0
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    accepted: bool = True
    rejection_reason: str = ""
    candidate_energy_total: float | None = None
    candidate_residual_norm: float | None = None
    candidate_min_a: float | None = None
    candidate_min_sqrtg: float | None = None
    candidate_energy_improved: bool | None = None
    candidate_positive_radius: bool | None = None
    candidate_positive_jacobian: bool | None = None


@dataclass(frozen=True)
class _CandidateDiagnostics:
    """Acceptance diagnostics for the raw optimizer candidate."""

    accepted: bool
    reason: str
    min_a: float
    min_sqrtg: float
    energy_improved: bool
    positive_radius: bool
    positive_jacobian: bool


def _scaling_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "identity", "off", "false"}:
        return "none"
    if key in {"geometry", "vmec", "vmec_like", "diagonal"}:
        return "geometry"
    raise ValueError(f"unsupported mirror reduced-coordinate scaling {value!r}")


def _residual_preconditioner_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "identity", "off", "false"}:
        return "none"
    if key in {"radial_tridi", "radial", "vmec", "vmec_like"}:
        return "radial_tridi"
    if key in {"radial_xi_tridi", "open_xi_tridi", "radial_tridi_xi"}:
        return "radial_xi_tridi"
    raise ValueError(f"unsupported mirror residual preconditioner {value!r}")


def _positive_radius(state: MirrorStateAxisym | MirrorState3D, floor: float = 1.0e-10) -> bool:
    return bool(np.all(np.asarray(state.a) > floor))


def _positive_jacobian(state: MirrorStateAxisym | MirrorState3D, grid: MirrorGrid, floor: float = 1.0e-10) -> bool:
    geometry = (
        evaluate_geometry_3d(state, grid) if np.asarray(state.a).ndim == 3 else evaluate_axisym_geometry(state, grid)
    )
    return bool(np.all(np.asarray(geometry.sqrtg) > floor))


def _admissible_state(state: MirrorStateAxisym | MirrorState3D, grid: MirrorGrid) -> bool:
    return _positive_radius(state) and _positive_jacobian(state, grid)


def _candidate_diagnostics(
    step: OptimizerStep,
    grid: MirrorGrid,
    *,
    initial_energy: float,
    floor: float = 1.0e-10,
) -> _CandidateDiagnostics:
    energy = float(step.energy)
    finite_energy = bool(np.isfinite(energy))
    energy_improved = bool(finite_energy and energy <= float(initial_energy))
    min_a = float(np.min(np.asarray(step.state.a, dtype=float)))
    geometry = (
        evaluate_geometry_3d(step.state, grid)
        if np.asarray(step.state.a).ndim == 3
        else evaluate_axisym_geometry(step.state, grid)
    )
    min_sqrtg = float(np.min(np.asarray(geometry.sqrtg, dtype=float)))
    positive_radius = bool(np.isfinite(min_a) and min_a > floor)
    positive_jacobian = bool(np.isfinite(min_sqrtg) and min_sqrtg > floor)

    failures: list[str] = []
    if not finite_energy:
        failures.append("nonfinite_energy")
    elif not energy_improved:
        failures.append("energy_increase")
    if not positive_radius:
        failures.append("nonpositive_radius")
    if not positive_jacobian:
        failures.append("nonpositive_jacobian")
    return _CandidateDiagnostics(
        accepted=not failures,
        reason="accepted" if not failures else ",".join(failures),
        min_a=min_a,
        min_sqrtg=min_sqrtg,
        energy_improved=energy_improved,
        positive_radius=positive_radius,
        positive_jacobian=positive_jacobian,
    )


def _sanitize_scale(scale, *, expected_size: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size != int(expected_size):
        raise ValueError(f"scale vector has size {scale.size}, expected {int(expected_size)}")
    scale = np.abs(scale)
    scale[(~np.isfinite(scale)) | (scale <= np.finfo(float).tiny)] = 1.0
    return scale


def _validate_smoothing_alpha(alpha: float, *, name: str) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return alpha


def _tridiagonal_smooth_zero_dirichlet(values: np.ndarray, *, alpha: float, axis: int = 0) -> np.ndarray:
    """Apply a symmetric zero-Dirichlet tridiagonal smoother along one axis."""
    alpha = _validate_smoothing_alpha(alpha, name="alpha")
    values = np.asarray(values, dtype=float)
    if alpha == 0.0 or values.size == 0:
        return values.copy()
    moved = np.moveaxis(values, axis, 0)
    original_shape = moved.shape
    n = int(original_shape[0])
    if n == 0:
        return values.copy()
    rhs = moved.reshape(n, -1).copy()
    if n == 1:
        solved = rhs / (1.0 + 2.0 * alpha)
        return np.moveaxis(solved.reshape(original_shape), 0, axis)

    lower = np.full(n - 1, -alpha, dtype=float)
    diag = np.full(n, 1.0 + 2.0 * alpha, dtype=float)
    upper = np.full(n - 1, -alpha, dtype=float)

    # Thomas elimination for many right-hand sides sharing one SPD tridiagonal matrix.
    c_prime = np.empty_like(upper)
    d_prime = np.empty_like(rhs)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for row in range(1, n):
        denom = diag[row] - lower[row - 1] * c_prime[row - 1]
        if row < n - 1:
            c_prime[row] = upper[row] / denom
        d_prime[row] = (rhs[row] - lower[row - 1] * d_prime[row - 1]) / denom

    solved = np.empty_like(rhs)
    solved[-1] = d_prime[-1]
    for row in range(n - 2, -1, -1):
        solved[row] = d_prime[row] - c_prime[row] * solved[row + 1]
    return np.moveaxis(solved.reshape(original_shape), 0, axis)


def axisym_reduced_a_mask(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary axisymmetric solves."""
    mask = np.zeros((grid.ns, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, 1:-1] = True
    return mask


def axisym_reduced_coordinate_scale(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal reduced-coordinate scale used by mirror L-BFGS-B.

    This mirrors the ``x_scale`` convention used by the toroidal optimization
    code: SciPy optimizes ``y = x / scale`` and gradients are transformed as
    ``grad_y = grad_x * scale``.  Radius DOFs use the local fixed-boundary
    radius; lambda DOFs use the median boundary-radius scale until a dedicated
    VMEC-style radial/lambda preconditioner is promoted for mirror states.
    """
    vector_size = pack_axisym_reduced_state(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.nxi)
    a_scale = np.broadcast_to(radius_scale[None, :], (grid.ns, grid.nxi))[axisym_reduced_a_mask(grid)]
    lambda_scale = np.full(grid.ns * (grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def axisym_reduced_residual_preconditioner(
    vector: np.ndarray,
    grid: MirrorGrid,
    *,
    kind: str = "radial_xi_tridi",
    radial_alpha: float = 0.5,
    lambda_alpha: float = 0.5,
    xi_alpha: float = 0.2,
) -> np.ndarray:
    """Apply a VMEC-like reduced-coordinate residual preconditioner.

    The mirror reduced vector contains interior radius ``a`` nodes followed by
    gauge-fixed ``lambda`` nodes.  This applies a symmetric tridiagonal
    smoother to the reduced coordinates, matching the regular VMEC residual
    iteration idea of radial tridiagonal preconditioning while respecting the
    mirror fixed-boundary packing.  The optional ``radial_xi_tridi`` mode also
    smooths radius updates along the open-ended axial direction with
    zero-Dirichlet ghost caps.
    """
    key = _residual_preconditioner_key(kind)
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if key == "none":
        return vector.copy()
    radial_alpha = _validate_smoothing_alpha(radial_alpha, name="radial_alpha")
    lambda_alpha = _validate_smoothing_alpha(lambda_alpha, name="lambda_alpha")
    xi_alpha = _validate_smoothing_alpha(xi_alpha, name="xi_alpha")

    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    expected_size = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected_size:
        raise ValueError(f"preconditioner vector has size {vector.size}, expected {expected_size}")

    a_values = vector[:num_a]
    lam_values = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    if num_a:
        a_values = a_values.reshape(grid.ns - 2, grid.nxi - 2)
        if radial_alpha > 0.0:
            a_values = _tridiagonal_smooth_zero_dirichlet(a_values, alpha=radial_alpha, axis=0)
        if key == "radial_xi_tridi" and xi_alpha > 0.0:
            a_values = _tridiagonal_smooth_zero_dirichlet(a_values, alpha=xi_alpha, axis=1)
        a_values = a_values.ravel()
    if lambda_alpha > 0.0:
        lam_values = _tridiagonal_smooth_zero_dirichlet(lam_values, alpha=lambda_alpha, axis=0)
    return np.concatenate([a_values, lam_values.ravel()])


def reduced_a_mask_3d(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary 3D solves."""
    mask = np.zeros((grid.ns, grid.ntheta, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, :, 1:-1] = True
    return mask


def reduced_coordinate_scale_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal reduced-coordinate scale used by 3D mirror L-BFGS-B."""
    vector_size = pack_reduced_state_3d(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid_3d(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.ntheta * grid.nxi).reshape(
        grid.ntheta,
        grid.nxi,
    )
    a_scale = np.broadcast_to(radius_scale[None, :, :], (grid.ns, grid.ntheta, grid.nxi))[reduced_a_mask_3d(grid)]
    lambda_scale = np.full(grid.ns * (grid.ntheta * grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def pack_axisym_reduced_state(state: MirrorStateAxisym, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_axisym_state(state, grid, boundary)
    a_values = projected.a[axisym_reduced_a_mask(grid)]
    lam_values = np.asarray(projected.lam[:, :-1], dtype=float).ravel()
    return np.concatenate([a_values, lam_values])


def axisym_reduced_bounds(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for axisymmetric reduced coordinates."""
    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    num_lam = grid.ns * (grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def scale_reduced_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    """Convert reduced-coordinate bounds into scaled optimizer coordinates."""
    scale = _sanitize_scale(scale, expected_size=len(bounds))
    scaled: list[tuple[float | None, float | None]] = []
    for (lower, upper), item_scale in zip(bounds, scale, strict=True):
        scaled.append(
            (
                None if lower is None else float(lower) / float(item_scale),
                None if upper is None else float(upper) / float(item_scale),
            )
        )
    return scaled


def _scaled_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    return scale_reduced_bounds(bounds, scale)


def unpack_axisym_reduced_state(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorStateAxisym:
    """Reconstruct a projected axisymmetric state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = axisym_reduced_a_mask(grid)
    num_a = int(np.count_nonzero(mask))
    expected = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid(grid)
    a = np.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    lam[:, -1] = -np.einsum("j,ij->i", grid.w_xi[:-1], lam[:, :-1]) / float(grid.w_xi[-1])
    return project_axisym_state(MirrorStateAxisym(a=a, lam=lam), grid, boundary)


def _unpack_axisym_reduced_state_jax(vector, grid: MirrorGrid, boundary: MirrorBoundary):
    boundary_radius = jnp.asarray(boundary.radius_on_grid(grid), dtype=jnp.asarray(vector).dtype)
    mask_i, mask_j = np.nonzero(axisym_reduced_a_mask(grid))
    num_a = int(mask_i.size)

    a = jnp.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi))
    a = a.at[(mask_i, mask_j)].set(vector[:num_a])
    a = a.at[0, :].set(a[1, :])
    a = a.at[0, 0].set(boundary_radius[0])
    a = a.at[0, -1].set(boundary_radius[-1])

    lam_inner = jnp.reshape(vector[num_a:], (grid.ns, grid.nxi - 1))
    w_xi = jnp.asarray(grid.w_xi, dtype=jnp.asarray(vector).dtype)
    lam_last = -jnp.einsum("j,ij->i", w_xi[:-1], lam_inner) / w_xi[-1]
    lam = jnp.concatenate([lam_inner, lam_last[:, None]], axis=1)
    return a, lam


def _axisym_reduced_energy_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float,
):
    a, lam = _unpack_axisym_reduced_state_jax(vector, grid, boundary)
    return axisym_total_energy_jax(
        a,
        lam,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def pack_reduced_state_3d(state: MirrorState3D, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent 3D ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_state_3d(state, grid, boundary)
    a_values = projected.a[reduced_a_mask_3d(grid)]
    lam_values = np.asarray(projected.lam[:, :, :], dtype=float).reshape(grid.ns, -1)[:, :-1].ravel()
    return np.concatenate([a_values, lam_values])


def reduced_bounds_3d(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for 3D reduced coordinates."""
    num_a = int(np.count_nonzero(reduced_a_mask_3d(grid)))
    num_lam = grid.ns * (grid.ntheta * grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def unpack_reduced_state_3d(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorState3D:
    """Reconstruct a projected 3D state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = reduced_a_mask_3d(grid)
    num_a = int(np.count_nonzero(mask))
    num_lam_surface = grid.ntheta * grid.nxi - 1
    expected = num_a + grid.ns * num_lam_surface
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid_3d(grid)
    a = np.broadcast_to(boundary_radius[None, :, :], (grid.ns, grid.ntheta, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.ntheta * grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, num_lam_surface)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam[:, -1] = -np.einsum("j,ij->i", flat_weights[:-1], lam[:, :-1]) / float(flat_weights[-1])
    lam = lam.reshape(grid.ns, grid.ntheta, grid.nxi)
    return project_state_3d(MirrorState3D(a=a, lam=lam), grid, boundary)


def pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid: MirrorGrid) -> np.ndarray:
    """Pack full-state axisymmetric gradients into reduced fixed-boundary coordinates."""
    mask = axisym_reduced_a_mask(grid)
    grad_a = np.asarray(grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :] += grad_a[0, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(grad_lam, dtype=float)
    lam_values = grad_lam[:, :-1] - (grid.w_xi[:-1] / grid.w_xi[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def _pack_axisym_reduced_gradient(gradient, grid: MirrorGrid) -> np.ndarray:
    return pack_axisym_reduced_gradient_components(gradient.grad_a, gradient.grad_lam, grid)


def _pack_reduced_gradient_3d(gradient, grid: MirrorGrid) -> np.ndarray:
    mask = reduced_a_mask_3d(grid)
    grad_a = np.asarray(gradient.grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :, :] += grad_a[0, :, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(gradient.grad_lam, dtype=float).reshape(grid.ns, -1)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam_values = grad_lam[:, :-1] - (flat_weights[:-1] / flat_weights[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def reduced_axisym_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_a=None,
    source_lam=None,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return energy and exact reduced-coordinate gradient."""
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    energy = float(gradient.energy)
    grad_a = np.asarray(gradient.grad_a, dtype=float)
    grad_lam = np.asarray(gradient.grad_lam, dtype=float)
    if source_a is not None:
        source_a = np.asarray(source_a, dtype=float)
        if source_a.shape != state.a.shape:
            raise ValueError(f"source_a shape {source_a.shape} does not match state shape {state.a.shape}")
        energy -= float(np.sum(source_a * state.a))
        grad_a = grad_a - source_a
    if source_lam is not None:
        source_lam = np.asarray(source_lam, dtype=float)
        if source_lam.shape != state.lam.shape:
            raise ValueError(f"source_lam shape {source_lam.shape} does not match state shape {state.lam.shape}")
        energy -= float(np.sum(source_lam * state.lam))
        grad_lam = grad_lam - source_lam
    return energy, pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid)


def reduced_3d_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return 3D energy and exact reduced-coordinate gradient."""
    state = unpack_reduced_state_3d(vector, grid, boundary)
    gradient = energy_value_and_gradient_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return gradient.energy, _pack_reduced_gradient_3d(gradient, grid)


def projected_gradient_step(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerStep:
    """Take one projected gradient step with backtracking line search."""
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if residual.norm <= options.tolerance:
        return OptimizerStep(
            state=state,
            energy=residual.energy,
            residual_norm=residual.norm,
            step_size=0.0,
            accepted=True,
        )

    step = float(options.step_size)
    for _ in range(int(options.line_search_steps)):
        trial = MirrorStateAxisym(
            a=state.a - step * residual.projected_a,
            lam=state.lam - step * residual.projected_lam,
        )
        trial = project_axisym_state(trial, grid, boundary)
        if _admissible_state(trial, grid):
            trial_residual = axisym_projected_energy_residual(
                trial,
                grid,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                mu0=options.mu0,
            )
            if np.isfinite(trial_residual.energy) and trial_residual.energy <= residual.energy:
                return OptimizerStep(
                    state=trial,
                    energy=trial_residual.energy,
                    residual_norm=trial_residual.norm,
                    step_size=step,
                    accepted=True,
                )
        step *= 0.5
        if step < options.min_step_size:
            break

    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=0.0,
        accepted=False,
    )


def projected_gradient_step_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerStep:
    """Take one projected 3D gradient step with backtracking line search."""
    residual = projected_energy_residual_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if residual.norm <= options.tolerance:
        return OptimizerStep(
            state=state,
            energy=residual.energy,
            residual_norm=residual.norm,
            step_size=0.0,
            accepted=True,
        )

    step = float(options.step_size)
    for _ in range(int(options.line_search_steps)):
        trial = MirrorState3D(
            a=state.a - step * residual.projected_a,
            lam=state.lam - step * residual.projected_lam,
        )
        trial = project_state_3d(trial, grid, boundary)
        if _admissible_state(trial, grid):
            trial_residual = projected_energy_residual_3d(
                trial,
                grid,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                mu0=options.mu0,
            )
            if np.isfinite(trial_residual.energy) and trial_residual.energy <= residual.energy:
                return OptimizerStep(
                    state=trial,
                    energy=trial_residual.energy,
                    residual_norm=trial_residual.norm,
                    step_size=step,
                    accepted=True,
                )
        step *= 0.5
        if step < options.min_step_size:
            break

    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _reduced_step_payload(
    vector,
    previous_vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
    accepted: bool,
) -> OptimizerStep:
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    step_size = float(np.linalg.norm(np.asarray(vector, dtype=float) - np.asarray(previous_vector, dtype=float)))
    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=step_size,
        accepted=accepted,
    )


def _reduced_step_payload_3d(
    vector,
    previous_vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
    accepted: bool,
) -> OptimizerStep:
    state = unpack_reduced_state_3d(vector, grid, boundary)
    residual = projected_energy_residual_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    step_size = float(np.linalg.norm(np.asarray(vector, dtype=float) - np.asarray(previous_vector, dtype=float)))
    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=step_size,
        accepted=accepted,
    )


def _rejected_lbfgs_step(initial_state: MirrorStateAxisym, initial_residual) -> OptimizerStep:
    return OptimizerStep(
        state=initial_state,
        energy=initial_residual.energy,
        residual_norm=initial_residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _rejected_lbfgs_step_3d(initial_state: MirrorState3D, initial_residual) -> OptimizerStep:
    return OptimizerStep(
        state=initial_state,
        energy=initial_residual.energy,
        residual_norm=initial_residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _lbfgs_options(options: OptimizerOptions) -> dict[str, float | int]:
    ftol = float(options.ftol) if options.ftol is not None else float(max(options.min_step_size, np.finfo(float).eps))
    return {
        "maxiter": int(options.maxiter),
        "gtol": float(options.tolerance),
        "maxls": int(options.line_search_steps),
        "ftol": ftol,
    }


def _optimizer_run_from_result(
    *,
    state: MirrorStateAxisym | MirrorState3D,
    steps: tuple[OptimizerStep, ...],
    result,
    accepted: bool = True,
    candidate_step: OptimizerStep | None = None,
    candidate_diagnostics: _CandidateDiagnostics | None = None,
) -> OptimizerRun:
    return OptimizerRun(
        state=state,
        steps=steps,
        success=bool(getattr(result, "success", False)),
        status=int(getattr(result, "status", 0)),
        message=str(getattr(result, "message", "")),
        nit=int(getattr(result, "nit", len(steps))),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        accepted=bool(accepted),
        rejection_reason="" if candidate_diagnostics is None else str(candidate_diagnostics.reason),
        candidate_energy_total=None if candidate_step is None else float(candidate_step.energy),
        candidate_residual_norm=None if candidate_step is None else float(candidate_step.residual_norm),
        candidate_min_a=None if candidate_diagnostics is None else float(candidate_diagnostics.min_a),
        candidate_min_sqrtg=None if candidate_diagnostics is None else float(candidate_diagnostics.min_sqrtg),
        candidate_energy_improved=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.energy_improved),
        candidate_positive_radius=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.positive_radius),
        candidate_positive_jacobian=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.positive_jacobian),
    )


def projected_lbfgs_solve(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a reduced-coordinate L-BFGS-B fixed-boundary solve."""
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("mirror optimizer='lbfgs' requires scipy.optimize.minimize") from exc

    initial_state = project_axisym_state(state, grid, boundary)
    x0 = pack_axisym_reduced_state(initial_state, grid, boundary)
    initial_residual = axisym_projected_energy_residual(
        initial_state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if initial_residual.norm <= options.tolerance:
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    steps: list[OptimizerStep] = []
    previous_x = x0.copy()
    x_scale = axisym_reduced_coordinate_scale(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y0 = x0 / x_scale

    def _x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def objective(vector_y):
        vector = _x_from_y(vector_y)
        value, gradient = reduced_axisym_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
        return value, np.asarray(gradient, dtype=float) * x_scale

    def record_step(vector_y, *, accepted: bool = True) -> OptimizerStep:
        nonlocal previous_x
        vector = _x_from_y(vector_y)
        step = _reduced_step_payload(
            vector,
            previous_x,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            accepted=accepted,
        )
        previous_x = np.asarray(vector, dtype=float).copy()
        return step

    def callback(vector_y):
        steps.append(record_step(vector_y))

    result = minimize(
        objective,
        y0,
        jac=True,
        method="L-BFGS-B",
        bounds=_scaled_bounds(axisym_reduced_bounds(grid), x_scale),
        callback=callback,
        options=_lbfgs_options(options),
    )
    final_step = record_step(np.asarray(result.x, dtype=float), accepted=bool(np.isfinite(result.fun)))
    if not steps or final_step.step_size > 0.0 or abs(final_step.energy - steps[-1].energy) > 1.0e-14:
        steps.append(final_step)

    final = steps[-1]
    candidate_diagnostics = _candidate_diagnostics(final, grid, initial_energy=initial_residual.energy)
    if not candidate_diagnostics.accepted:
        rejected_steps = (_rejected_lbfgs_step(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=initial_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=final,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=final.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=final,
        candidate_diagnostics=candidate_diagnostics,
    )


def projected_residual_newton_solve(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a matrix-free damped Newton solve on reduced fixed-boundary residuals."""
    if jax is None:
        raise RuntimeError("optimizer='residual_newton' requires JAX")
    try:
        from scipy.optimize import OptimizeResult
        from scipy.sparse.linalg import LinearOperator, lsmr
    except Exception as exc:  # pragma: no cover
        raise ImportError("optimizer='residual_newton' requires scipy.sparse.linalg.lsmr") from exc

    initial_state = project_axisym_state(state, grid, boundary)
    x0 = pack_axisym_reduced_state(initial_state, grid, boundary)
    initial_residual = axisym_projected_energy_residual(
        initial_state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if initial_residual.norm <= options.tolerance:
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    x_scale = axisym_reduced_coordinate_scale(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y = x0 / x_scale
    scale_jax = jnp.asarray(x_scale, dtype=jnp.asarray(x0).dtype)
    preconditioner_kind = _residual_preconditioner_key(options.residual_preconditioner)

    def objective_x(vector):
        return _axisym_reduced_energy_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )

    grad_fun = jax.grad(objective_x)

    def x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def reduced_gradient_x(vector: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fun(jnp.asarray(vector, dtype=scale_jax.dtype)), dtype=float)

    def step_payload(vector: np.ndarray, previous_vector: np.ndarray, *, accepted: bool = True) -> OptimizerStep:
        return _reduced_step_payload(
            vector,
            previous_vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            accepted=accepted,
        )

    steps: list[OptimizerStep] = []
    current_x = x0.copy()
    current_step = step_payload(current_x, current_x)
    nfev = 0
    njev = 0
    success = False
    status = 0
    message = "maximum iterations reached"
    ftol = float(options.ftol) if options.ftol is not None else float(options.min_step_size)

    for _iteration in range(1, int(options.maxiter) + 1):
        grad_x = reduced_gradient_x(current_x)
        nfev += 1
        reduced_norm = float(np.linalg.norm(grad_x))
        if current_step.residual_norm <= options.tolerance or reduced_norm <= options.tolerance:
            success = True
            status = 1
            message = "`gtol` termination condition is satisfied."
            break

        x_jax = jnp.asarray(current_x, dtype=scale_jax.dtype)

        def matvec_y(vector_y):
            direction_x = scale_jax * jnp.asarray(vector_y, dtype=scale_jax.dtype)
            _, hvp = jax.jvp(grad_fun, (x_jax,), (direction_x,))
            return np.asarray(hvp * scale_jax, dtype=float)

        def precondition_y(vector_y):
            return axisym_reduced_residual_preconditioner(
                vector_y,
                grid,
                kind=preconditioner_kind,
                radial_alpha=options.residual_radial_alpha,
                lambda_alpha=options.residual_lambda_alpha,
                xi_alpha=options.residual_xi_alpha,
            )

        size = int(y.size)
        if preconditioner_kind == "none":
            operator = LinearOperator((size, size), matvec=matvec_y, rmatvec=matvec_y, dtype=float)
        else:

            def matvec_preconditioned(vector_z):
                return matvec_y(precondition_y(vector_z))

            def rmatvec_preconditioned(vector_y):
                return precondition_y(matvec_y(vector_y))

            operator = LinearOperator(
                (size, size),
                matvec=matvec_preconditioned,
                rmatvec=rmatvec_preconditioned,
                dtype=float,
            )
        rhs = -grad_x * x_scale
        linear_maxiter = max(1, int(options.residual_linear_maxiter))
        linear_result = lsmr(
            operator,
            rhs,
            atol=min(1.0e-10, max(options.tolerance, np.finfo(float).eps)),
            btol=min(1.0e-10, max(options.tolerance, np.finfo(float).eps)),
            maxiter=linear_maxiter,
        )
        njev += int(linear_result[2])
        step_y_raw = np.asarray(linear_result[0], dtype=float)
        step_y = step_y_raw if preconditioner_kind == "none" else precondition_y(step_y_raw)
        if not np.all(np.isfinite(step_y)):
            message = "non-finite reduced-Newton step"
            break

        step_x_norm = float(np.linalg.norm(step_y * x_scale))
        if step_x_norm <= ftol * max(1.0, float(np.linalg.norm(current_x))):
            success = True
            status = 3
            message = "`xtol` termination condition is satisfied."
            break

        accepted = False
        alpha = 1.0
        for _ in range(int(options.line_search_steps)):
            trial_y = y + alpha * step_y
            trial_x = x_from_y(trial_y)
            trial_state = unpack_axisym_reduced_state(trial_x, grid, boundary)
            if _admissible_state(trial_state, grid):
                trial_step = step_payload(trial_x, current_x)
                energy_ok = np.isfinite(trial_step.energy) and trial_step.energy <= current_step.energy + 1.0e-12
                residual_ok = np.isfinite(trial_step.residual_norm) and (
                    trial_step.residual_norm < current_step.residual_norm
                )
                if energy_ok and residual_ok:
                    y = trial_y
                    current_x = trial_x
                    current_step = trial_step
                    steps.append(trial_step)
                    accepted = True
                    if trial_step.residual_norm <= options.tolerance:
                        success = True
                        status = 1
                        message = "`gtol` termination condition is satisfied."
                    break
            alpha *= 0.5
        if success:
            break
        if not accepted:
            message = "line search failed to reduce the projected residual"
            rejected = OptimizerStep(
                state=current_step.state,
                energy=current_step.energy,
                residual_norm=current_step.residual_norm,
                step_size=0.0,
                accepted=False,
            )
            steps.append(rejected)
            break

    final_step = current_step if not steps else steps[-1]
    final_state = final_step.state if final_step.accepted else current_step.state
    result = OptimizeResult(
        success=success,
        status=status,
        message=message,
        nit=sum(1 for step in steps if step.accepted),
        nfev=nfev,
        njev=njev,
    )
    candidate_diagnostics = _candidate_diagnostics(current_step, grid, initial_energy=initial_residual.energy)
    accepted_run = bool(steps and steps[-1].accepted and candidate_diagnostics.accepted)
    if not accepted_run:
        rejected_steps = tuple(steps) if steps else (_rejected_lbfgs_step(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=final_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=current_step,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=current_step.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=current_step,
        candidate_diagnostics=candidate_diagnostics,
    )


def projected_lbfgs_solve_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a reduced-coordinate L-BFGS-B fixed-boundary solve for 3D states."""
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("mirror optimizer='lbfgs' requires scipy.optimize.minimize") from exc

    initial_state = project_state_3d(state, grid, boundary)
    x0 = pack_reduced_state_3d(initial_state, grid, boundary)
    initial_residual = projected_energy_residual_3d(
        initial_state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if initial_residual.norm <= options.tolerance:
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    steps: list[OptimizerStep] = []
    previous_x = x0.copy()
    x_scale = reduced_coordinate_scale_3d(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y0 = x0 / x_scale

    def _x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def objective(vector_y):
        vector = _x_from_y(vector_y)
        value, gradient = reduced_3d_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
        return value, np.asarray(gradient, dtype=float) * x_scale

    def record_step(vector_y, *, accepted: bool = True) -> OptimizerStep:
        nonlocal previous_x
        vector = _x_from_y(vector_y)
        step = _reduced_step_payload_3d(
            vector,
            previous_x,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            accepted=accepted,
        )
        previous_x = np.asarray(vector, dtype=float).copy()
        return step

    def callback(vector_y):
        steps.append(record_step(vector_y))

    result = minimize(
        objective,
        y0,
        jac=True,
        method="L-BFGS-B",
        bounds=_scaled_bounds(reduced_bounds_3d(grid), x_scale),
        callback=callback,
        options=_lbfgs_options(options),
    )
    final_step = record_step(np.asarray(result.x, dtype=float), accepted=bool(np.isfinite(result.fun)))
    if not steps or final_step.step_size > 0.0 or abs(final_step.energy - steps[-1].energy) > 1.0e-14:
        steps.append(final_step)

    final = steps[-1]
    candidate_diagnostics = _candidate_diagnostics(final, grid, initial_energy=initial_residual.energy)
    if not candidate_diagnostics.accepted:
        rejected_steps = (_rejected_lbfgs_step_3d(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=initial_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=final,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=final.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=final,
        candidate_diagnostics=candidate_diagnostics,
    )
