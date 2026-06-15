"""Residual-style diagnostics for mirror field kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import divergence_free_numerator


@dataclass(frozen=True)
class MirrorFieldDiagnostics:
    """Compact scalar diagnostics for mirror field checks."""

    min_bmag: float
    max_bmag: float
    mirror_ratio: float
    max_divergence_numerator: float


def field_diagnostics(field, grid, *, fluxes=None) -> MirrorFieldDiagnostics:
    """Return basic field and divergence diagnostics."""
    bmag = np.asarray(field.bmag)
    min_bmag = float(np.min(bmag))
    max_bmag = float(np.max(bmag))
    if min_bmag <= 0.0:
        mirror_ratio = float("inf")
    else:
        mirror_ratio = max_bmag / min_bmag
    if fluxes is None:
        max_divergence = 0.0
    else:
        max_divergence = float(np.max(np.abs(divergence_free_numerator(fluxes, grid))))
    return MirrorFieldDiagnostics(
        min_bmag=min_bmag,
        max_bmag=max_bmag,
        mirror_ratio=mirror_ratio,
        max_divergence_numerator=max_divergence,
    )
