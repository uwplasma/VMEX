"""Manufactured-solution helpers for mirror variational tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmec_jax._compat import jax, jnp

from .energy import MU0
from .forces import axisym_energy_value_and_gradient, axisym_total_energy_jax


@dataclass(frozen=True)
class ManufacturedAxisymCase:
    """Axisymmetric manufactured mirror case with exact stationary source terms."""

    name: str
    grid: object
    state: object
    boundary: object
    psi_prime: object
    i_prime: object
    pressure: object
    source_a: np.ndarray
    source_lam: np.ndarray


def build_axisym_manufactured_case(
    *,
    name: str,
    grid,
    state,
    boundary,
    psi_prime,
    i_prime,
    pressure,
    mu0: float = MU0,
) -> ManufacturedAxisymCase:
    """Build MMS sources by differentiating the unforced energy at the exact state."""
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return ManufacturedAxisymCase(
        name=name,
        grid=grid,
        state=state,
        boundary=boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        source_a=gradient.grad_a,
        source_lam=gradient.grad_lam,
    )


def axisym_mms_objective_jax(a, lam, case: ManufacturedAxisymCase, *, mu0: float = MU0):
    """Return ``W - <source_a,a> - <source_lam,lambda>`` for an MMS case."""
    energy = axisym_total_energy_jax(
        a,
        lam,
        case.grid,
        psi_prime=case.psi_prime,
        i_prime=case.i_prime,
        pressure=case.pressure,
        mu0=mu0,
    )
    source_a = jnp.asarray(case.source_a, dtype=jnp.asarray(a).dtype)
    source_lam = jnp.asarray(case.source_lam, dtype=jnp.asarray(lam).dtype)
    return energy - jnp.sum(source_a * a) - jnp.sum(source_lam * lam)


def axisym_mms_gradient(case: ManufacturedAxisymCase, *, mu0: float = MU0) -> tuple[np.ndarray, np.ndarray]:
    """Return AD gradient of the manufactured objective at the exact state."""
    if jax is None:
        raise RuntimeError("JAX is required for manufactured mirror gradients")

    def objective(a, lam):
        return axisym_mms_objective_jax(a, lam, case, mu0=mu0)

    grad_a, grad_lam = jax.grad(objective, argnums=(0, 1))(case.state.a, case.state.lam)
    return np.asarray(grad_a), np.asarray(grad_lam)
