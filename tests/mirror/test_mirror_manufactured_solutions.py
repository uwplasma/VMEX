from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import MirrorResolution
from vmec_jax.mirror.kernels.forces import central_difference_energy_component
from vmec_jax.mirror.kernels.manufactured import axisym_mms_gradient
from vmec_jax.mirror.validation.manufactured import make_mms_case

pytestmark = pytest.mark.mirror


def test_axisym_mms_exact_state_is_stationary_for_polynomial_case():
    case = make_mms_case("axisym_flared_polynomial", MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0))
    grad_a, grad_lam = axisym_mms_gradient(case)
    assert np.max(np.abs(grad_a)) < 2.0e-9
    assert np.max(np.abs(grad_lam)) < 2.0e-9


def test_axisym_mms_exact_state_is_stationary_with_lambda_and_pressure():
    for name in ["axisym_lambda", "axisym_finite_pressure"]:
        case = make_mms_case(name, MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0))
        grad_a, grad_lam = axisym_mms_gradient(case)
        assert np.max(np.abs(grad_a)) < 5.0e-9
        assert np.max(np.abs(grad_lam)) < 5.0e-9


def test_axisym_mms_source_matches_central_finite_difference_gradient():
    case = make_mms_case("axisym_lambda", MirrorResolution(ns=6, ntheta=1, nxi=11, mpol=0), mu0=1.0)
    for component, index, expected in [
        ("a", (2, 5), case.source_a[2, 5]),
        ("lam", (2, 4), case.source_lam[2, 4]),
    ]:
        finite_difference = central_difference_energy_component(
            case.state,
            case.grid,
            psi_prime=case.psi_prime,
            i_prime=case.i_prime,
            pressure=case.pressure,
            component=component,
            index=index,
            step=2.0e-6,
            mu0=1.0,
        )
        assert np.isfinite(finite_difference)
        assert np.isclose(expected, finite_difference, rtol=3.0e-5, atol=1.0e-7)


def test_axisym_mms_stationarity_holds_across_resolutions():
    for nxi in [9, 13, 17]:
        case = make_mms_case("axisym_flared_polynomial", MirrorResolution(ns=7, ntheta=1, nxi=nxi, mpol=0))
        grad_a, grad_lam = axisym_mms_gradient(case)
        residual_norm = np.sqrt(np.sum(grad_a**2) + np.sum(grad_lam**2))
        assert residual_norm < 1.0e-7
