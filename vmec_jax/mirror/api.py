"""Small public surface for experimental mirror-geometry primitives."""

from .core.basis import ChebyshevLobattoBasis, ThetaFourierBasis
from .core.boundary import MirrorBoundary
from .core.config import MirrorConfig, MirrorResolution
from .core.grids import MirrorGrid, make_mirror_grid
from .core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from .core.state import MirrorStateAxisym
from .io.mout import is_mirror_output, load_mirror_output, read_mirror_output, write_mirror_output
from .io.schema import MirrorOutput
from .plotting.export import mirror_axisym_slice_to_csv, mirror_output_to_npz, plot_mirror_output
from .solvers.fixed_boundary.api import MirrorFixedBoundaryResult, MirrorSolveOptions, run_mirror_fixed_boundary
from .validation.wham import (
    build_wham_loop_table,
    load_wham_fixture,
    mirror_boundary_from_vacuum_flux_tube,
    wham_on_axis_mirror_ratio,
    wham_reference_field,
    wham_vacuum_field_rz,
)


__all__ = [
    "ChebyshevLobattoBasis",
    "IPrimeProfile",
    "MirrorBoundary",
    "MirrorConfig",
    "MirrorFixedBoundaryResult",
    "MirrorGrid",
    "MirrorOutput",
    "MirrorResolution",
    "MirrorSolveOptions",
    "MirrorStateAxisym",
    "PressureProfile",
    "PsiPrimeProfile",
    "ThetaFourierBasis",
    "is_mirror_output",
    "load_mirror_output",
    "load_wham_fixture",
    "make_mirror_grid",
    "mirror_axisym_slice_to_csv",
    "mirror_boundary_from_vacuum_flux_tube",
    "mirror_output_to_npz",
    "plot_mirror_output",
    "read_mirror_output",
    "run_mirror_fixed_boundary",
    "build_wham_loop_table",
    "wham_on_axis_mirror_ratio",
    "wham_reference_field",
    "wham_vacuum_field_rz",
    "write_mirror_output",
]
