"""Optional adapter from ESSOS coils to vmec_jax direct-coil params."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jnp

from .coils_jax import CoilFieldParams


def from_essos_coils(coils: Any, regularization_epsilon: float = 0.0, chunk_size: int | None = None) -> CoilFieldParams:
    """Convert an ESSOS ``Coils`` object into ``CoilFieldParams``.

    ESSOS is intentionally not imported at module import time.  The adapter
    works with objects exposing the ESSOS ``Coils`` attributes:
    ``dofs_curves``, ``dofs_currents``, ``currents_scale``, ``n_segments``,
    ``nfp``, and ``stellsym``.

    Raises
    ------
    ImportError
        If the supplied object does not expose the expected ESSOS attributes.
    """

    required = ("dofs_curves", "dofs_currents", "currents_scale", "n_segments", "nfp", "stellsym")
    missing = [name for name in required if not hasattr(coils, name)]
    if missing:
        raise ImportError(
            "Cannot convert ESSOS coils: object is missing "
            f"{', '.join(missing)}. Install/import ESSOS and pass an essos.coils.Coils instance."
        )
    return CoilFieldParams(
        base_curve_dofs=jnp.asarray(coils.dofs_curves),
        base_currents=jnp.asarray(coils.dofs_currents),
        n_segments=int(coils.n_segments),
        nfp=int(coils.nfp),
        stellsym=bool(coils.stellsym),
        current_scale=float(coils.currents_scale),
        regularization_epsilon=float(regularization_epsilon),
        chunk_size=chunk_size,
    )
