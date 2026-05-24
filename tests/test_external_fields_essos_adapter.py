from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import from_essos_coils
from vmec_jax.external_fields.coils_jax import biot_savart_xyz, expanded_coil_geometry


def test_from_essos_coils_extracts_expected_attributes_from_duck_type():
    from vmec_jax._compat import jnp

    enable_x64(True)
    dofs = jnp.zeros((1, 3, 3), dtype=float).at[0, 0, 2].set(1.0).at[0, 1, 1].set(1.0)
    coils = SimpleNamespace(
        dofs_curves=dofs,
        dofs_currents=jnp.asarray([2.0]),
        currents_scale=5.0,
        n_segments=24,
        nfp=2,
        stellsym=True,
    )

    params = from_essos_coils(coils, regularization_epsilon=1.0e-4, chunk_size=7)

    assert params.base_curve_dofs.shape == (1, 3, 3)
    np.testing.assert_allclose(params.base_currents, [2.0])
    assert params.current_scale == 5.0
    assert params.n_segments == 24
    assert params.nfp == 2
    assert params.stellsym is True
    assert params.regularization_epsilon == 1.0e-4
    assert params.chunk_size == 7


def test_from_essos_coils_reports_missing_attributes():
    with pytest.raises(ImportError, match="missing"):
        from_essos_coils(SimpleNamespace(dofs_curves=np.zeros((1, 3, 3))))


def test_essos_biot_savart_parity_when_essos_is_installed():
    pytest.importorskip("essos")
    from vmec_jax._compat import jnp
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    enable_x64(True)
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 0].set(0.2)
    dofs = dofs.at[0, 0, 2].set(1.1)
    dofs = dofs.at[0, 1, 1].set(0.9)
    dofs = dofs.at[0, 2, 2].set(0.1)
    essos_coils = Coils(Curves(dofs, n_segments=48, nfp=1, stellsym=False), jnp.asarray([3.0]))
    params = from_essos_coils(essos_coils)
    gamma, gamma_dash, _gamma_dashdash, currents = expanded_coil_geometry(params)

    points = jnp.asarray(
        [
            [0.1, 0.2, 0.4],
            [0.4, -0.2, 0.3],
            [-0.2, 0.1, -0.3],
        ]
    )
    essos_field = BiotSavart(essos_coils)
    expected = jnp.stack([essos_field.B(point) for point in points])
    actual = biot_savart_xyz(points, gamma, gamma_dash, currents)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-17)
