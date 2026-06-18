"""Fixed side-boundary parameterizations for mirror geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kernels.chebyshev import interpolate_chebyshev_values


@dataclass(frozen=True)
class MirrorBoundary:
    """Fixed side boundary radius ``r_b(xi)`` or ``r_b(theta, xi)``."""

    kind: str
    r0: float | None = None
    a2: float = 0.0
    a4: float = 0.0
    epsilon: float = 0.0
    theta_mode: int = 0
    rotation_angle: float = 0.0
    stellarator_fraction: float = 1.0
    xi: np.ndarray | None = None
    radius_values: np.ndarray | None = None

    @classmethod
    def constant_radius(cls, radius: float) -> "MirrorBoundary":
        """Return a cylindrical side boundary."""
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("boundary radius must be positive")
        return cls(kind="polynomial_radius", r0=radius)

    @classmethod
    def polynomial_radius(cls, *, r0: float, a2: float = 0.0, a4: float = 0.0) -> "MirrorBoundary":
        """Return ``r_b(xi) = r0 * (1 + a2*xi**2 + a4*xi**4)``."""
        r0 = float(r0)
        if r0 <= 0.0:
            raise ValueError("r0 must be positive")
        return cls(kind="polynomial_radius", r0=r0, a2=float(a2), a4=float(a4))

    @classmethod
    def tabulated_radius(cls, xi, radius_values) -> "MirrorBoundary":
        """Return a boundary interpolated from nodal radius values."""
        xi = np.asarray(xi, dtype=float)
        radius_values = np.asarray(radius_values, dtype=float)
        if xi.ndim != 1 or radius_values.ndim != 1:
            raise ValueError("xi and radius_values must be one-dimensional")
        if xi.size != radius_values.size:
            raise ValueError("xi and radius_values must have the same length")
        if xi.size < 2:
            raise ValueError("at least two boundary nodes are required")
        if not np.all(np.diff(xi) > 0.0):
            raise ValueError("xi nodes must be strictly increasing")
        if np.any(radius_values <= 0.0):
            raise ValueError("boundary radius values must be positive")
        return cls(kind="tabulated_radius", xi=xi, radius_values=radius_values)

    @classmethod
    def cosine_modulated_radius(
        cls,
        *,
        r0: float,
        a2: float = 0.0,
        a4: float = 0.0,
        epsilon: float,
        theta_mode: int = 2,
    ) -> "MirrorBoundary":
        """Return ``r0 * (1 + a2*xi**2 + a4*xi**4) * (1 + epsilon*cos(m*theta))``."""
        r0 = float(r0)
        epsilon = float(epsilon)
        theta_mode = int(theta_mode)
        if r0 <= 0.0:
            raise ValueError("r0 must be positive")
        if theta_mode <= 0:
            raise ValueError("theta_mode must be positive for a nonaxisymmetric boundary")
        if abs(epsilon) >= 1.0:
            raise ValueError("abs(epsilon) must be less than one so the boundary stays positive")
        return cls(
            kind="cosine_modulated_radius",
            r0=r0,
            a2=float(a2),
            a4=float(a4),
            epsilon=epsilon,
            theta_mode=theta_mode,
        )

    @classmethod
    def rotating_ellipse_mirror_hybrid(
        cls,
        *,
        r0: float,
        a2: float = 0.0,
        a4: float = 0.0,
        epsilon: float = 0.12,
        rotation_angle: float = np.pi,
        stellarator_fraction: float = 0.6,
    ) -> "MirrorBoundary":
        """Return a straight-axis mirror with a central rotating ellipse.

        The elliptical deformation is strongest at the midplane, rotates by
        ``rotation_angle`` across the active central segment, and tapers
        smoothly to circular mirror end sections.  The profile is up-down
        symmetric in the sense that ``r(theta, xi) = r(-theta, -xi)``.
        """

        r0 = float(r0)
        epsilon = float(epsilon)
        stellarator_fraction = float(stellarator_fraction)
        if r0 <= 0.0:
            raise ValueError("r0 must be positive")
        if abs(epsilon) >= 1.0:
            raise ValueError("abs(epsilon) must be less than one so the boundary stays positive")
        if not (0.0 < stellarator_fraction <= 1.0):
            raise ValueError("stellarator_fraction must be in (0, 1]")
        if not np.isfinite(rotation_angle):
            raise ValueError("rotation_angle must be finite")
        return cls(
            kind="rotating_ellipse_mirror_hybrid",
            r0=r0,
            a2=float(a2),
            a4=float(a4),
            epsilon=epsilon,
            rotation_angle=float(rotation_angle),
            stellarator_fraction=stellarator_fraction,
        )

    @property
    def is_axisymmetric(self) -> bool:
        """Return whether this boundary is independent of theta."""
        return self.kind in {"polynomial_radius", "tabulated_radius"}

    def _axial_radius(self, xi, *, dtype: Any | None = None) -> np.ndarray:
        xi = np.asarray(xi, dtype=dtype or float)
        if self.kind in {"polynomial_radius", "cosine_modulated_radius", "rotating_ellipse_mirror_hybrid"}:
            radius = float(self.r0) * (1.0 + self.a2 * xi**2 + self.a4 * xi**4)
        elif self.kind == "tabulated_radius":
            radius = interpolate_chebyshev_values(self.radius_values, self.xi, xi)
        else:
            raise ValueError(f"unsupported mirror boundary kind {self.kind!r}")
        return np.asarray(radius, dtype=dtype or float)

    def _rotating_ellipse_factor(self, theta, xi, *, dtype: Any | None = None) -> np.ndarray:
        xi = np.asarray(xi, dtype=dtype or float)
        theta = np.asarray(theta, dtype=dtype or float)
        normalized = np.abs(xi) / float(self.stellarator_fraction)
        envelope = np.where(normalized < 1.0, (1.0 - normalized**2) ** 2, 0.0)
        ellipticity = float(self.epsilon) * envelope
        phase = 0.5 * float(self.rotation_angle) * np.clip(xi / float(self.stellarator_fraction), -1.0, 1.0)
        major = 1.0 + ellipticity
        minor = 1.0 - ellipticity
        alpha = theta[:, None] - phase[None, :]
        denominator = np.sqrt((minor[None, :] * np.cos(alpha)) ** 2 + (major[None, :] * np.sin(alpha)) ** 2)
        return major[None, :] * minor[None, :] / denominator

    def radius(self, xi, *, theta=None, dtype: Any | None = None) -> np.ndarray:
        """Evaluate the boundary radius on axial nodes."""
        radius = self._axial_radius(xi, dtype=dtype)
        if self.kind == "cosine_modulated_radius":
            if theta is None:
                raise ValueError("theta nodes are required for a nonaxisymmetric boundary")
            theta = np.asarray(theta, dtype=dtype or float)
            radius = (1.0 + self.epsilon * np.cos(self.theta_mode * theta[:, None])) * radius[None, :]
        elif self.kind == "rotating_ellipse_mirror_hybrid":
            if theta is None:
                raise ValueError("theta nodes are required for a rotating-ellipse hybrid boundary")
            radius = radius[None, :] * self._rotating_ellipse_factor(theta, xi, dtype=dtype)
        radius = np.asarray(radius, dtype=dtype or float)
        if np.any(radius <= 0.0):
            raise ValueError("boundary radius must be positive on the requested grid")
        return radius

    def radius_on_grid(self, grid) -> np.ndarray:
        """Evaluate the boundary radius on a ``MirrorGrid`` axial grid."""
        if not self.is_axisymmetric:
            raise ValueError("use radius_on_grid_3d for theta-dependent boundaries")
        return self.radius(grid.xi, dtype=grid.xi.dtype)

    def radius_on_grid_3d(self, grid) -> np.ndarray:
        """Evaluate the side-boundary radius on ``(theta, xi)`` grid nodes."""
        if self.is_axisymmetric:
            return np.broadcast_to(self.radius_on_grid(grid)[None, :], (grid.ntheta, grid.nxi)).copy()
        return self.radius(grid.xi, theta=grid.theta, dtype=grid.xi.dtype)
