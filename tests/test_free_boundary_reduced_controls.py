from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solvers.free_boundary import (
    FreeBoundaryNativeSplineForce,
    FreeBoundaryNativeSplineState,
    FreeBoundaryNativeSplineUpdate,
    FreeBoundaryNativeSplineUnknownVector,
    FreeBoundaryNativeSplineVectorStep,
    FreeBoundaryReducedEdgeState,
    ReducedControlMap,
    ReducedControlState,
    free_boundary_reduced_edge_state_from_vmec_state,
    free_boundary_reduced_edge_state_to_vmec_state,
    free_boundary_native_spline_unknown_vector_from_vmec_state,
    free_boundary_native_spline_project_vmec_delta_jax,
    free_boundary_native_spline_vector_projected_residual_jax,
    free_boundary_native_spline_vector_residual_jax,
    free_boundary_native_spline_vector_to_vmec_state_jax,
    free_boundary_native_spline_vector_edge_step,
    reduced_control_decode,
    reduced_control_least_squares_step,
    reduced_control_pullback,
)
from vmec_jax.config import VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.solvers.free_boundary.control import (
    _freeb_edge_control_native_coordinate_step,
    _prepare_freeb_edge_control_projection,
)
from vmec_jax.state import StateLayout, VMECState, pack_state
from vmec_jax.static import build_static


def test_reduced_control_least_squares_step_is_public() -> None:
    import vmec_jax as vj
    import vmec_jax.api as public_api

    assert vj.ReducedControlMap is ReducedControlMap
    assert vj.ReducedControlState is ReducedControlState
    assert vj.FreeBoundaryNativeSplineForce is FreeBoundaryNativeSplineForce
    assert vj.FreeBoundaryNativeSplineState is FreeBoundaryNativeSplineState
    assert vj.FreeBoundaryNativeSplineUpdate is FreeBoundaryNativeSplineUpdate
    assert vj.FreeBoundaryReducedEdgeState is FreeBoundaryReducedEdgeState
    assert public_api.ReducedControlMap is ReducedControlMap
    assert public_api.ReducedControlState is ReducedControlState
    assert public_api.FreeBoundaryNativeSplineForce is FreeBoundaryNativeSplineForce
    assert public_api.FreeBoundaryNativeSplineState is FreeBoundaryNativeSplineState
    assert public_api.FreeBoundaryNativeSplineUpdate is FreeBoundaryNativeSplineUpdate
    assert public_api.FreeBoundaryNativeSplineUnknownVector is FreeBoundaryNativeSplineUnknownVector
    assert public_api.FreeBoundaryNativeSplineVectorStep is FreeBoundaryNativeSplineVectorStep
    assert public_api.FreeBoundaryReducedEdgeState is FreeBoundaryReducedEdgeState
    assert (
        vj.free_boundary_native_spline_unknown_vector_from_vmec_state
        is free_boundary_native_spline_unknown_vector_from_vmec_state
    )
    assert (
        public_api.free_boundary_native_spline_unknown_vector_from_vmec_state
        is free_boundary_native_spline_unknown_vector_from_vmec_state
    )
    assert (
        vj.free_boundary_native_spline_project_vmec_delta_jax
        is free_boundary_native_spline_project_vmec_delta_jax
    )
    assert (
        public_api.free_boundary_native_spline_project_vmec_delta_jax
        is free_boundary_native_spline_project_vmec_delta_jax
    )
    assert (
        vj.free_boundary_native_spline_vector_projected_residual_jax
        is free_boundary_native_spline_vector_projected_residual_jax
    )
    assert (
        public_api.free_boundary_native_spline_vector_projected_residual_jax
        is free_boundary_native_spline_vector_projected_residual_jax
    )
    assert vj.free_boundary_native_spline_vector_to_vmec_state_jax is free_boundary_native_spline_vector_to_vmec_state_jax
    assert (
        public_api.free_boundary_native_spline_vector_to_vmec_state_jax
        is free_boundary_native_spline_vector_to_vmec_state_jax
    )
    assert vj.free_boundary_native_spline_vector_residual_jax is free_boundary_native_spline_vector_residual_jax
    assert (
        public_api.free_boundary_native_spline_vector_residual_jax
        is free_boundary_native_spline_vector_residual_jax
    )
    assert vj.free_boundary_native_spline_vector_edge_step is free_boundary_native_spline_vector_edge_step
    assert public_api.free_boundary_native_spline_vector_edge_step is free_boundary_native_spline_vector_edge_step
    assert (
        vj.free_boundary_reduced_edge_state_from_vmec_state
        is free_boundary_reduced_edge_state_from_vmec_state
    )
    assert (
        public_api.free_boundary_reduced_edge_state_from_vmec_state
        is free_boundary_reduced_edge_state_from_vmec_state
    )
    assert (
        vj.free_boundary_reduced_edge_state_to_vmec_state
        is free_boundary_reduced_edge_state_to_vmec_state
    )
    assert (
        public_api.free_boundary_reduced_edge_state_to_vmec_state
        is free_boundary_reduced_edge_state_to_vmec_state
    )
    assert vj.reduced_control_decode is reduced_control_decode
    assert public_api.reduced_control_decode is reduced_control_decode
    assert vj.reduced_control_least_squares_step is reduced_control_least_squares_step
    assert public_api.reduced_control_least_squares_step is reduced_control_least_squares_step
    assert vj.reduced_control_pullback is reduced_control_pullback
    assert public_api.reduced_control_pullback is reduced_control_pullback


def test_reduced_control_least_squares_step_reports_exact_and_uncontrolled_parts() -> None:
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    target = np.asarray([3.0, 4.0, 5.0])

    step = reduced_control_least_squares_step(jacobian, target, labels=("side", "corner"))

    np.testing.assert_allclose(step.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(step.predicted_delta, [3.0, 4.0, 0.0])
    np.testing.assert_allclose(step.residual_after, [0.0, 0.0, 5.0])
    assert step.control_delta_by_label == {"side": 3.0, "corner": 2.0}
    assert step.rank == 2
    assert step.condition_number == pytest.approx(2.0)
    assert step.residual_l2 == pytest.approx(5.0)
    assert step.residual_rel == pytest.approx(5.0 / np.sqrt(50.0))
    assert step.to_dict()["control_delta_by_label"] == {"side": 3.0, "corner": 2.0}


def test_reduced_control_least_squares_step_supports_ridge_damping() -> None:
    step = reduced_control_least_squares_step([[1.0]], [2.0], ridge=3.0)

    np.testing.assert_allclose(step.control_delta, [0.5])
    np.testing.assert_allclose(step.predicted_delta, [0.5])
    np.testing.assert_allclose(step.residual_after, [1.5])
    assert step.ridge == pytest.approx(3.0)
    assert step.trust_scale == pytest.approx(1.0)


def test_reduced_control_least_squares_step_supports_trust_radius() -> None:
    step = reduced_control_least_squares_step(np.eye(2), [3.0, 4.0], trust_radius=2.5)

    np.testing.assert_allclose(step.control_delta, [1.5, 2.0])
    np.testing.assert_allclose(step.predicted_delta, [1.5, 2.0])
    assert step.control_l2 == pytest.approx(2.5)
    assert step.trust_scale == pytest.approx(0.5)


def test_reduced_control_map_encodes_decodes_and_projects_boundary_values() -> None:
    control_map = ReducedControlMap(
        initial=np.asarray([10.0, -1.0, 0.5]),
        jacobian=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        ),
        labels=("side", "corner"),
        rcond=1.0e-12,
    )
    full_values = np.asarray([13.0, 3.0, 7.5])

    step = control_map.encode(full_values)
    decoded = control_map.decode(step.control_delta)
    projected = control_map.project(full_values)
    pulled = control_map.pullback([1.0, 2.0, 3.0])
    payload = control_map.to_dict()

    np.testing.assert_allclose(step.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(decoded, [13.0, 3.0, 0.5])
    np.testing.assert_allclose(projected, decoded)
    np.testing.assert_allclose(step.residual_after, [0.0, 0.0, 7.0])
    assert step.control_delta_by_label == {"side": 3.0, "corner": 2.0}
    assert control_map.full_size == 3
    assert control_map.control_count == 2
    np.testing.assert_allclose(pulled, [1.0, 4.0])
    assert payload["rank"] == 2
    assert payload["rank_deficient"] is False
    assert payload["labels"] == ["side", "corner"]


def test_reduced_control_state_tracks_native_coordinates() -> None:
    control_map = ReducedControlMap(
        initial=np.asarray([10.0, -1.0, 0.5]),
        jacobian=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        ),
        labels=("side", "corner"),
        rcond=1.0e-12,
    )

    state = ReducedControlState.from_full_values(control_map, [13.0, 3.0, 7.5])
    updated = state.update([0.25, -0.5])
    payload = updated.to_dict()

    np.testing.assert_allclose(state.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(state.decode(), [13.0, 3.0, 0.5])
    np.testing.assert_allclose(updated.control_delta, [3.25, 1.5])
    np.testing.assert_allclose(updated.decode(), [13.25, 2.0, 0.5])
    np.testing.assert_allclose(np.asarray(updated.decode_jax()), updated.decode())
    assert updated.control_delta_by_label == {"side": 3.25, "corner": 1.5}
    assert payload["reduced_unknown_size"] == 2
    assert payload["unknown_by_label"] == {"side": 3.25, "corner": 1.5}


def test_free_boundary_reduced_edge_state_encodes_vmec_lcfs_edge_and_pullback() -> None:
    import jax

    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    zeros = np.zeros((3, static.modes.K), dtype=float)
    anchor_rcos = zeros.copy()
    anchor_rcos[-1, 0] = 3.0
    state0 = VMECState(
        layout=layout,
        Rcos=anchor_rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    indata = InData(
        scalars={"MPOL": 2, "NTOR": 0, "NS_ARRAY": [3], "NFP": 1, "LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 3.0}},
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[0, 0] = 1.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["R00"],
            "control_jacobian": jacobian,
            "update_mode": "native_coordinate",
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )

    rcos = zeros.copy()
    rcos[-1, 0] = 3.25
    rcos[-1, 1] = 0.5
    state = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )

    reduced_edge = free_boundary_reduced_edge_state_from_vmec_state(state, projection)
    native_spline_state = FreeBoundaryNativeSplineState.from_vmec_state(state, projection)
    decoded_state = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge,
        state,
        projection,
        host_update=True,
    )
    decoded_state_jax = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge,
        state,
        projection,
        host_update=False,
    )
    updated_state = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge.update([0.5]),
        state,
        projection,
        host_update=True,
    )
    native_decoded_state = native_spline_state.to_vmec_state(host_update=True)
    native_decoded_state_jax = native_spline_state.to_vmec_state(host_update=False)
    native_updated_state = native_spline_state.update_edge([0.5]).to_vmec_state(host_update=True)
    force_dR = zeros.copy()
    force_dR[-1, 0] = 2.0
    force_deltas = (
        force_dR,
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )
    native_update = native_spline_state.apply_edge_control_update(
        [0.5],
        update_deltas=force_deltas,
        host_update=True,
    )
    native_force = native_spline_state.force_from_delta_tuple(force_deltas)
    full_adjoint = jnp.asarray([2.0, 7.0, *([0.0] * (4 * static.modes.K - 2))])
    _decoded, vjp_fun = jax.vjp(
        lambda values: reduced_edge.control_state.control_map.decode_jax(values),
        jnp.asarray(reduced_edge.control_delta),
    )

    assert projection["enabled"] is True
    assert native_spline_state.control_delta_by_label == {"R00": pytest.approx(0.25)}
    assert native_spline_state.to_dict()["mode"] == "free_boundary_native_spline_state"
    assert native_spline_state.to_dict()["full_edge_size"] == 4 * static.modes.K
    assert reduced_edge.control_delta_by_label == {"R00": pytest.approx(0.25)}
    assert reduced_edge.fit_residual_linf > 0.0
    assert reduced_edge.to_dict()["mode"] == "native_reduced_lcfs_edge_state"
    np.testing.assert_allclose(reduced_edge.decode_edge_values()[0], 3.25)
    np.testing.assert_allclose(np.asarray(reduced_edge.decode_edge_values_jax())[0], 3.25)
    assert np.asarray(decoded_state.Rcos)[-1, 0] == pytest.approx(3.25)
    assert np.asarray(decoded_state.Rcos)[-1, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(decoded_state_jax.Rcos), np.asarray(decoded_state.Rcos))
    assert np.asarray(updated_state.Rcos)[-1, 0] == pytest.approx(3.75)
    np.testing.assert_allclose(np.asarray(native_decoded_state.Rcos), np.asarray(decoded_state.Rcos))
    np.testing.assert_allclose(
        np.asarray(native_decoded_state_jax.Rcos),
        np.asarray(decoded_state.Rcos),
    )
    assert np.asarray(native_updated_state.Rcos)[-1, 0] == pytest.approx(3.75)
    assert isinstance(native_update, FreeBoundaryNativeSplineUpdate)
    assert np.asarray(native_update.state.Rcos)[-1, 0] == pytest.approx(3.75)
    assert np.asarray(native_update.update_deltas[0])[-1, 0] == pytest.approx(0.5)
    assert native_update.native_state.control_delta_by_label == {"R00": pytest.approx(0.75)}
    assert native_update.decoded_edge_update_l2 == pytest.approx(0.5)
    assert native_update.source_edge_update_l2 == pytest.approx(2.0)
    assert native_update.source_update_residual_l2 == pytest.approx(1.5)
    assert native_update.source_update_residual_linf == pytest.approx(1.5)
    assert native_update.source_update_residual_rel == pytest.approx(0.75)
    assert native_update.source_update_captured_fraction == pytest.approx(0.25)
    assert native_update.to_dict()["mode"] == "free_boundary_native_spline_update"
    assert isinstance(native_force, FreeBoundaryNativeSplineForce)
    np.testing.assert_allclose(native_force.control_force, [2.0])
    assert native_force.metric == "pullback"
    assert native_force.control_force_l2 == pytest.approx(2.0)
    assert native_force.source_edge_force_l2 == pytest.approx(2.0)
    assert native_force.to_dict()["mode"] == "free_boundary_native_spline_force"
    np.testing.assert_allclose(native_spline_state.pullback_delta_tuple(force_deltas), [2.0])
    np.testing.assert_allclose(
        np.asarray(reduced_edge.pullback_jax(full_adjoint)),
        np.asarray(vjp_fun(full_adjoint)[0]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(reduced_edge.pullback(np.asarray(full_adjoint)), [2.0])


def test_native_spline_unknown_vector_packs_edge_controls_not_fourier_edge() -> None:
    import jax

    cfg = VMECConfig(
        mpol=1,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    assert static.modes.K == 1
    zeros = np.zeros((3, static.modes.K), dtype=float)
    state0 = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0], [2.0], [3.0]]),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    indata = InData(
        scalars={"MPOL": 1, "NTOR": 0, "NS_ARRAY": [3], "NFP": 1, "LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 3.0}},
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[0, 0] = 2.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["edge_radius"],
            "control_jacobian": jacobian,
            "update_mode": "native_coordinate",
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )
    state = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0], [2.0], [3.5]]),
        Rsin=np.asarray([[0.1], [0.2], [0.0]]),
        Zcos=np.asarray([[0.3], [0.4], [0.0]]),
        Zsin=np.asarray([[0.5], [0.6], [0.0]]),
        Lcos=np.asarray([[0.7], [0.8], [0.9]]),
        Lsin=np.asarray([[1.0], [1.1], [1.2]]),
    )

    unknowns = free_boundary_native_spline_unknown_vector_from_vmec_state(state, projection)
    decoded = unknowns.to_vmec_state()
    decoded_jax = free_boundary_native_spline_vector_to_vmec_state_jax(
        jnp.asarray(unknowns.vector),
        state,
        projection,
    )
    payload = unknowns.to_dict()

    assert isinstance(unknowns, FreeBoundaryNativeSplineUnknownVector)
    assert unknowns.full_vmec_size == 18
    assert unknowns.native_unknown_size == 15
    assert unknowns.removed_fourier_edge_dofs == 3
    assert payload["schema"] == "FreeBoundaryNativeSplineUnknownVector.v1"
    assert payload["edge_control_size"] == 1
    assert payload["removed_fourier_edge_dofs"] == 3
    assert payload["edge_reconstruction_residual_linf"] == pytest.approx(0.0)
    np.testing.assert_allclose(unknowns.control_delta, [0.25])
    np.testing.assert_allclose(np.asarray(decoded.Rcos), np.asarray(state.Rcos))
    np.testing.assert_allclose(np.asarray(decoded.Rsin[:-1]), np.asarray(state.Rsin[:-1]))
    np.testing.assert_allclose(np.asarray(decoded.Zcos[:-1]), np.asarray(state.Zcos[:-1]))
    np.testing.assert_allclose(np.asarray(decoded.Zsin[:-1]), np.asarray(state.Zsin[:-1]))
    np.testing.assert_allclose(np.asarray(decoded.Lcos), np.asarray(state.Lcos))
    np.testing.assert_allclose(np.asarray(decoded.Lsin), np.asarray(state.Lsin))
    np.testing.assert_allclose(np.asarray(pack_state(decoded_jax)), np.asarray(pack_state(decoded)))
    edge_grad = jax.grad(
        lambda vector: free_boundary_native_spline_vector_to_vmec_state_jax(
            vector,
            state,
            projection,
        ).Rcos[-1, 0]
    )(jnp.asarray(unknowns.vector))
    np.testing.assert_allclose(np.asarray(edge_grad[:-1]), np.zeros(unknowns.native_unknown_size - 1))
    np.testing.assert_allclose(np.asarray(edge_grad[-1:]), [2.0])
    residual_jac = jax.jacfwd(
        lambda vector: free_boundary_native_spline_vector_residual_jax(
            vector,
            state,
            projection,
            lambda decoded_state: jnp.asarray([decoded_state.Rcos[0, 0], decoded_state.Rcos[-1, 0]]),
        )
    )(jnp.asarray(unknowns.vector))
    np.testing.assert_allclose(np.asarray(residual_jac[0, 0]), 1.0)
    np.testing.assert_allclose(np.asarray(residual_jac[0, 1:]), np.zeros(unknowns.native_unknown_size - 1))
    np.testing.assert_allclose(np.asarray(residual_jac[1, :-1]), np.zeros(unknowns.native_unknown_size - 1))
    np.testing.assert_allclose(np.asarray(residual_jac[1, -1]), 2.0)
    with pytest.raises(TypeError, match="residual_fn"):
        free_boundary_native_spline_vector_residual_jax(unknowns.vector, state, projection, object())

    dR = np.asarray([[10.0], [20.0], [8.0]])
    dR_sin = np.asarray([[0.1], [0.2], [0.0]])
    dZ_cos = np.asarray([[0.3], [0.4], [0.0]])
    dZ = np.asarray([[0.5], [0.6], [0.0]])
    dL_cos = np.asarray([[0.7], [0.8], [0.9]])
    dL = np.asarray([[1.0], [1.1], [1.2]])
    deltas = (dR, dR_sin, dZ_cos, dZ, dL_cos, dL)

    projected_vector = unknowns.vector_from_delta_tuple(deltas, edge_metric="least_squares")
    pullback_vector = unknowns.vector_from_delta_tuple(deltas, edge_metric="pullback")
    decoded_deltas = unknowns.delta_tuple_from_vector(projected_vector)
    projected_vector_jax = free_boundary_native_spline_project_vmec_delta_jax(
        deltas,
        state,
        projection,
        edge_metric="least_squares",
    )
    pullback_vector_jax = free_boundary_native_spline_project_vmec_delta_jax(
        deltas,
        state,
        projection,
        edge_metric="pullback",
    )
    projected_residual_jax = free_boundary_native_spline_vector_projected_residual_jax(
        unknowns.vector,
        state,
        projection,
        lambda _decoded_state: deltas,
        edge_metric="pullback",
    )

    np.testing.assert_allclose(projected_vector[-1:], [4.0])
    np.testing.assert_allclose(pullback_vector[-1:], [16.0])
    np.testing.assert_allclose(np.asarray(projected_vector_jax), projected_vector)
    np.testing.assert_allclose(np.asarray(pullback_vector_jax), pullback_vector)
    np.testing.assert_allclose(np.asarray(projected_residual_jax), pullback_vector)
    np.testing.assert_allclose(decoded_deltas[0][:-1], dR[:-1])
    np.testing.assert_allclose(decoded_deltas[0][-1], dR[-1])
    np.testing.assert_allclose(decoded_deltas[1][:-1], dR_sin[:-1])
    np.testing.assert_allclose(decoded_deltas[2][:-1], dZ_cos[:-1])
    np.testing.assert_allclose(decoded_deltas[3][:-1], dZ[:-1])
    np.testing.assert_allclose(decoded_deltas[4], dL_cos)
    np.testing.assert_allclose(decoded_deltas[5], dL)

    with pytest.raises(ValueError, match="edge_metric"):
        unknowns.vector_from_delta_tuple(deltas, edge_metric="bad")
    with pytest.raises(TypeError, match="residual_fn"):
        free_boundary_native_spline_vector_projected_residual_jax(
            unknowns.vector,
            state,
            projection,
            object(),
        )


def test_native_spline_vector_edge_step_matches_existing_edge_bridge() -> None:
    cfg = VMECConfig(
        mpol=1,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    zeros = np.zeros((3, static.modes.K), dtype=float)
    state0 = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0], [2.0], [3.0]]),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    indata = InData(
        scalars={"MPOL": 1, "NTOR": 0, "NS_ARRAY": [3], "NFP": 1, "LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 3.0}},
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[0, 0] = 2.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["edge_radius"],
            "control_jacobian": jacobian,
            "update_mode": "native_coordinate",
            "native_force_metric": "least_squares",
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )
    state_current = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0], [2.0], [3.5]]),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    state_candidate = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.5], [2.5], [9.0]]),
        Rsin=np.asarray([[0.1], [0.2], [0.3]]),
        Zcos=np.asarray([[0.4], [0.5], [0.6]]),
        Zsin=np.asarray([[0.7], [0.8], [0.9]]),
        Lcos=np.asarray([[1.0], [1.1], [1.2]]),
        Lsin=np.asarray([[1.3], [1.4], [1.5]]),
    )
    update_deltas = (
        np.asarray([[0.5], [0.5], [4.0]]),
        np.asarray([[0.1], [0.2], [0.3]]),
        np.asarray([[0.4], [0.5], [0.6]]),
        np.asarray([[0.7], [0.8], [0.9]]),
        np.asarray([[1.0], [1.1], [1.2]]),
        np.asarray([[1.3], [1.4], [1.5]]),
    )
    force_deltas = (
        np.asarray([[0.0], [0.0], [8.0]]),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )
    kwargs = {
        "dt_eff": 0.1,
        "b1": 0.8,
        "fac": 1.0,
        "force_scale": 1.0,
        "flip_sign": 1.0,
    }

    legacy = _freeb_edge_control_native_coordinate_step(
        state_current=state_current,
        state_candidate=state_candidate,
        update_deltas=update_deltas,
        force_deltas=force_deltas,
        projection=projection,
        control_velocity=None,
        edge_state=None,
        host_update=True,
        **kwargs,
    )
    vector_step = free_boundary_native_spline_vector_edge_step(
        state_current=state_current,
        state_candidate=state_candidate,
        update_deltas=update_deltas,
        force_deltas=force_deltas,
        projection=projection,
        control_velocity=None,
        **kwargs,
    )

    assert isinstance(vector_step, FreeBoundaryNativeSplineVectorStep)
    assert vector_step.to_dict()["mode"] == "free_boundary_native_spline_vector_step"
    assert vector_step.force_metric == legacy.force_metric == "least_squares"
    np.testing.assert_allclose(vector_step.control_force, legacy.control_force)
    np.testing.assert_allclose(vector_step.control_update, legacy.control_update)
    np.testing.assert_allclose(vector_step.control_velocity, legacy.control_velocity)
    np.testing.assert_allclose(np.asarray(vector_step.state.Rcos), np.asarray(legacy.state.Rcos))
    np.testing.assert_allclose(np.asarray(vector_step.state.Rsin), np.asarray(legacy.state.Rsin))
    np.testing.assert_allclose(np.asarray(vector_step.state.Zcos), np.asarray(legacy.state.Zcos))
    np.testing.assert_allclose(np.asarray(vector_step.state.Zsin), np.asarray(legacy.state.Zsin))
    for actual, expected in zip(vector_step.update_deltas, legacy.update_deltas, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_native_spline_vector_residual_jax_respects_mode_scale() -> None:
    import jax

    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    assert static.modes.K >= 2
    zeros = np.zeros((3, static.modes.K), dtype=float)
    rcos = zeros.copy()
    rcos[-1, 0] = 3.0
    state0 = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    indata = InData(
        scalars={"MPOL": 2, "NTOR": 0, "NS_ARRAY": [3], "NFP": 1, "LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 3.0}},
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[1, 0] = 3.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["scaled_edge_mode"],
            "control_jacobian": jacobian,
            "update_mode": "native_coordinate",
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )
    unknowns = free_boundary_native_spline_unknown_vector_from_vmec_state(state0, projection)
    vector = np.array(unknowns.vector, dtype=float, copy=True)
    vector[-1] = 0.2
    scale = float(projection["mode_scale_np"][1])

    def residual(values):
        return free_boundary_native_spline_vector_residual_jax(
            values,
            state0,
            projection,
            lambda decoded_state: decoded_state.Rcos[-1, 1],
        )

    grad = jax.grad(residual)(jnp.asarray(vector))
    projected_grad = jax.grad(
        lambda values: free_boundary_native_spline_vector_projected_residual_jax(
            values,
            state0,
            projection,
            lambda decoded_state: VMECState(
                layout=decoded_state.layout,
                Rcos=decoded_state.Rcos,
                Rsin=jnp.zeros_like(decoded_state.Rsin),
                Zcos=jnp.zeros_like(decoded_state.Zcos),
                Zsin=jnp.zeros_like(decoded_state.Zsin),
                Lcos=jnp.zeros_like(decoded_state.Lcos),
                Lsin=jnp.zeros_like(decoded_state.Lsin),
            ),
            edge_metric="pullback",
        )[-1]
    )(jnp.asarray(vector))
    eps = 1.0e-6
    vector_plus = vector.copy()
    vector_minus = vector.copy()
    vector_plus[-1] += eps
    vector_minus[-1] -= eps
    finite_difference = float((residual(vector_plus) - residual(vector_minus)) / (2.0 * eps))

    np.testing.assert_allclose(np.asarray(grad[:-1]), np.zeros(unknowns.native_unknown_size - 1), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(grad[-1]), 3.0 / scale, rtol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(projected_grad[:-1]),
        np.zeros(unknowns.native_unknown_size - 1),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(np.asarray(projected_grad[-1]), 9.0, rtol=1.0e-12)
    assert finite_difference == pytest.approx(3.0 / scale, rel=1.0e-8)


def test_reduced_control_decode_matches_host_map_and_jacobian() -> None:
    import jax

    initial = np.asarray([10.0, -1.0, 0.5])
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    controls = jnp.asarray([3.0, 2.0])
    control_map = ReducedControlMap(initial=initial, jacobian=jacobian, labels=("side", "corner"))

    decoded = reduced_control_decode(initial, jacobian, controls)
    decoded_from_map = control_map.decode_jax(controls)
    derivative = jax.jacfwd(lambda values: reduced_control_decode(initial, jacobian, values))(controls)

    np.testing.assert_allclose(np.asarray(decoded), control_map.decode(np.asarray(controls)), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(decoded_from_map), np.asarray(decoded), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(derivative), jacobian, atol=1.0e-14)


def test_reduced_control_pullback_matches_decode_vjp() -> None:
    import jax

    initial = np.asarray([10.0, -1.0, 0.5])
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.5, -1.0],
        ]
    )
    controls = jnp.asarray([3.0, 2.0])
    full_adjoint = jnp.asarray([7.0, 11.0, 13.0])
    control_map = ReducedControlMap(initial=initial, jacobian=jacobian, labels=("side", "corner"))

    pulled = reduced_control_pullback(jacobian, full_adjoint)
    pulled_from_map = control_map.pullback_jax(full_adjoint)
    _decoded, vjp_fun = jax.vjp(lambda values: reduced_control_decode(initial, jacobian, values), controls)
    pulled_from_vjp = vjp_fun(full_adjoint)[0]

    expected = jacobian.T @ np.asarray(full_adjoint)
    np.testing.assert_allclose(control_map.pullback(np.asarray(full_adjoint)), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled_from_map), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled_from_vjp), expected, atol=1.0e-14)


@pytest.mark.parametrize(
    ("jacobian", "target", "kwargs", "match"),
    [
        ([[1.0, 2.0]], [1.0], {"labels": ("a",)}, "labels length"),
        ([[1.0, 2.0]], [1.0, 2.0], {}, "row count"),
        ([[1.0, np.nan]], [1.0], {}, "finite"),
        ([[1.0]], [1.0], {"ridge": -1.0}, "ridge"),
        ([[1.0]], [1.0], {"rcond": -1.0}, "rcond"),
        ([[1.0]], [1.0], {"trust_radius": 0.0}, "trust_radius"),
    ],
)
def test_reduced_control_least_squares_step_rejects_invalid_inputs(
    jacobian, target, kwargs, match
) -> None:
    with pytest.raises(ValueError, match=match):
        reduced_control_least_squares_step(jacobian, target, **kwargs)


@pytest.mark.parametrize(
    ("initial", "jacobian", "kwargs", "match"),
    [
        ([0.0], [[[1.0]]], {}, "two-dimensional"),
        ([0.0, 0.0], [[1.0]], {}, "row count"),
        ([0.0], [[]], {}, "at least one control"),
        ([np.nan], [[1.0]], {}, "finite"),
        ([0.0], [[1.0]], {"labels": ("a", "b")}, "labels length"),
        ([0.0], [[1.0]], {"rcond": -1.0}, "rcond"),
    ],
)
def test_reduced_control_map_rejects_invalid_inputs(initial, jacobian, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ReducedControlMap(initial=initial, jacobian=jacobian, **kwargs)


def test_reduced_control_map_rejects_mismatched_decode_and_encode_sizes() -> None:
    control_map = ReducedControlMap(initial=[0.0, 0.0], jacobian=np.eye(2))

    with pytest.raises(ValueError, match="full_values size"):
        control_map.encode([1.0])
    with pytest.raises(ValueError, match="control_delta size"):
        control_map.decode([1.0])
    with pytest.raises(ValueError, match="full_values size"):
        control_map.pullback([1.0])
    with pytest.raises(ValueError, match="finite"):
        control_map.decode([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        control_map.pullback([1.0, np.nan])
    with pytest.raises(ValueError, match="control_delta size"):
        reduced_control_decode([0.0, 0.0], np.eye(2), [1.0])
    with pytest.raises(ValueError, match="full_values size"):
        reduced_control_pullback(np.eye(2), [1.0])
    with pytest.raises(ValueError, match="control_delta size"):
        ReducedControlState(control_map=control_map, control_delta=[1.0])
    with pytest.raises(ValueError, match="control_update size"):
        ReducedControlState(control_map=control_map, control_delta=[1.0, 2.0]).update([1.0])
