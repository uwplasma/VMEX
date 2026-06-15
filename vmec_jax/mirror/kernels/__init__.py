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
from .forces import (
    AxisymEnergyGradient,
    AxisymProjectedResidual,
    axisym_energy_value_and_gradient,
    axisym_flat_state_energy_jax,
    axisym_projected_energy_residual,
    axisym_total_energy_jax,
    central_difference_energy_component,
    project_axisym_residual,
    radial_derivative_matrix,
)
from .manufactured import (
    ManufacturedAxisymCase,
    axisym_mms_gradient,
    axisym_mms_objective_jax,
    build_axisym_manufactured_case,
)
from .residuals import MirrorFieldDiagnostics, field_diagnostics

__all__ = [
    "AxisymEnergyGradient",
    "AxisymMirrorGeometry",
    "AxisymMirrorField",
    "AxisymProjectedResidual",
    "apply_chebyshev_filter",
    "axisym_energy_value_and_gradient",
    "axisym_flat_state_energy_jax",
    "axisym_mms_gradient",
    "axisym_mms_objective_jax",
    "axisym_projected_energy_residual",
    "axisym_total_energy_jax",
    "build_axisym_manufactured_case",
    "central_difference_energy_component",
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
    "ManufacturedAxisymCase",
    "MU0",
    "pressure_energy_axisym",
    "project_axisym_residual",
    "project_axisym_state",
    "project_lambda_gauge_axisym",
    "radial_derivative_matrix",
    "real_fourier_modes",
    "theta_nodes",
    "theta_weights",
    "total_energy_axisym",
]
