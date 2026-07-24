"""Matrix-free bordered linear algebra for a coupled plasma/NESTOR root."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

Array = Any
MatVec = Callable[[Array], Array]

__all__ = ["NestorBorderedOperator"]


@dataclass(frozen=True)
class NestorBorderedOperator:
    """The coupled linearization ``[[A, B], [C, D]]``.

    ``A`` is the plasma-force linearization, ``D`` the NESTOR potential
    equation, and ``B``/``C`` their edge coupling. All four blocks are
    matrix-free callables; only their static input sizes are stored.
    """

    plasma: MatVec
    vacuum_to_plasma: MatVec
    plasma_to_vacuum: MatVec
    vacuum: MatVec
    plasma_size: int
    vacuum_size: int

    @property
    def shape(self) -> tuple[int, int]:
        size = self.plasma_size + self.vacuum_size
        return size, size

    def __call__(self, value: Array) -> Array:
        x = value[:self.plasma_size]
        q = value[self.plasma_size:]
        return jnp.concatenate((
            self.plasma(x) + self.vacuum_to_plasma(q),
            self.plasma_to_vacuum(x) + self.vacuum(q),
        ))

    def transpose(self, value: Array) -> Array:
        """Apply the exact transpose, generated from the four linear blocks."""
        template = jnp.zeros(
            (self.plasma_size + self.vacuum_size,), dtype=jnp.asarray(value).dtype
        )
        return jax.linear_transpose(self, template)(value)[0]

    def schur(self, plasma_solve: MatVec) -> MatVec:
        """Return ``q -> (D - C A^-1 B) q`` without materializing blocks."""
        return lambda q: self.vacuum(q) - self.plasma_to_vacuum(
            plasma_solve(self.vacuum_to_plasma(q))
        )

    def preconditioner(
        self, plasma_solve: MatVec, schur_solve: MatVec
    ) -> MatVec:
        """Exact block inverse when both supplied solves are exact."""
        def apply(rhs):
            rx = rhs[:self.plasma_size]
            rq = rhs[self.plasma_size:]
            ax = plasma_solve(rx)
            q = schur_solve(rq - self.plasma_to_vacuum(ax))
            x = plasma_solve(rx - self.vacuum_to_plasma(q))
            return jnp.concatenate((x, q))

        return apply
