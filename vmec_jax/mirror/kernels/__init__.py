"""Numerical kernels for mirror coordinates."""

from .chebyshev import (
    apply_chebyshev_filter,
    chebyshev_interpolation_matrix,
    chebyshev_lobatto_derivative_matrix,
    chebyshev_lobatto_nodes,
    chebyshev_values_to_coefficients,
    clenshaw_curtis_weights,
    interpolate_chebyshev_values,
)
from .fourier import (
    evaluate_real_fourier,
    evaluate_real_fourier_derivative,
    fourier_derivative,
    real_fourier_modes,
    theta_nodes,
    theta_weights,
)
from .geometry import AxisymMirrorGeometry, evaluate_axisym_geometry
from .constraints import (
    lambda_surface_average_axisym,
    project_axisym_state,
    project_lambda_gauge_axisym,
)
from .energy import MU0, MirrorEnergy, magnetic_energy_axisym, pressure_energy_axisym, total_energy_axisym
from .fields import (
    AxisymMirrorField,
    MirrorContravariantFluxes,
    contravariant_fluxes_from_lambda,
    divergence_free_numerator,
    evaluate_axisym_field,
)
from .residuals import MirrorFieldDiagnostics, field_diagnostics

__all__ = [
    "AxisymMirrorGeometry",
    "AxisymMirrorField",
    "apply_chebyshev_filter",
    "chebyshev_interpolation_matrix",
    "chebyshev_lobatto_derivative_matrix",
    "chebyshev_lobatto_nodes",
    "chebyshev_values_to_coefficients",
    "clenshaw_curtis_weights",
    "contravariant_fluxes_from_lambda",
    "divergence_free_numerator",
    "evaluate_real_fourier",
    "evaluate_real_fourier_derivative",
    "fourier_derivative",
    "interpolate_chebyshev_values",
    "evaluate_axisym_geometry",
    "evaluate_axisym_field",
    "field_diagnostics",
    "lambda_surface_average_axisym",
    "magnetic_energy_axisym",
    "MirrorContravariantFluxes",
    "MirrorEnergy",
    "MirrorFieldDiagnostics",
    "MU0",
    "pressure_energy_axisym",
    "project_axisym_state",
    "project_lambda_gauge_axisym",
    "real_fourier_modes",
    "theta_nodes",
    "theta_weights",
    "total_energy_axisym",
]
