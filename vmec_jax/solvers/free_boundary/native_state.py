"""Native spline-control unknown vectors for free-boundary VMEC solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vmec_jax.state import VMECState

from .control import (
    FreeBoundaryReducedEdgeState,
    _freeb_edge_control_delta_tuple_target,
    _freeb_edge_control_project_vector_np,
    _freeb_edge_control_reduced_map,
    _freeb_edge_control_state_edge_values,
    free_boundary_reduced_edge_state_from_vmec_state,
    free_boundary_reduced_edge_state_to_vmec_state,
)
from .reduced_controls import ReducedControlState


def _finite_vector(values: Any, *, name: str, size: int | None = None) -> np.ndarray:
    """Return a validated one-dimensional host vector."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    if size is not None and vector.size != int(size):
        raise ValueError(f"{name} must have size {int(size)}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _native_spline_interior_size(state: VMECState) -> int:
    """Return the native vector size excluding reduced LCFS controls."""

    ns = int(state.layout.ns)
    k = int(state.layout.K)
    return int(4 * max(ns - 1, 0) * k + 2 * ns * k)


def _pack_vmec_interior_without_edge(state: VMECState) -> np.ndarray:
    """Pack VMEC unknowns while omitting the LCFS R/Z Fourier edge rows."""

    return np.concatenate(
        [
            np.asarray(state.Rcos, dtype=float)[:-1].reshape(-1),
            np.asarray(state.Rsin, dtype=float)[:-1].reshape(-1),
            np.asarray(state.Zcos, dtype=float)[:-1].reshape(-1),
            np.asarray(state.Zsin, dtype=float)[:-1].reshape(-1),
            np.asarray(state.Lcos, dtype=float).reshape(-1),
            np.asarray(state.Lsin, dtype=float).reshape(-1),
        ],
        axis=0,
    )


def _unpack_vmec_interior_without_edge(
    vector: Any,
    template_state: VMECState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack VMEC interior rows using ``template_state`` for LCFS R/Z slots."""

    ns = int(template_state.layout.ns)
    k = int(template_state.layout.K)
    interior_size = _native_spline_interior_size(template_state)
    values = _finite_vector(vector, name="interior_vector", size=interior_size)
    pos = 0
    interior_block = max(ns - 1, 0) * k
    lambda_block = ns * k

    def take(count: int) -> np.ndarray:
        nonlocal pos
        out = values[pos : pos + count]
        pos += count
        return out

    Rcos = np.array(template_state.Rcos, dtype=float, copy=True)
    Rsin = np.array(template_state.Rsin, dtype=float, copy=True)
    Zcos = np.array(template_state.Zcos, dtype=float, copy=True)
    Zsin = np.array(template_state.Zsin, dtype=float, copy=True)
    Lcos = np.array(template_state.Lcos, dtype=float, copy=True)
    Lsin = np.array(template_state.Lsin, dtype=float, copy=True)
    if ns > 1:
        Rcos[:-1] = take(interior_block).reshape((ns - 1, k))
        Rsin[:-1] = take(interior_block).reshape((ns - 1, k))
        Zcos[:-1] = take(interior_block).reshape((ns - 1, k))
        Zsin[:-1] = take(interior_block).reshape((ns - 1, k))
    else:
        take(0)
        take(0)
        take(0)
        take(0)
    Lcos[:] = take(lambda_block).reshape((ns, k))
    Lsin[:] = take(lambda_block).reshape((ns, k))
    return Rcos, Rsin, Zcos, Zsin, Lcos, Lsin


def _edge_update_delta_tuple(
    template_state: VMECState,
    projection: dict[str, Any],
    control_update: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a reduced edge-control update into physical VMEC delta rows."""

    k = int(projection["mode_count"])
    control = _finite_vector(
        control_update,
        name="control_update",
        size=int(_freeb_edge_control_reduced_map(projection).control_count),
    )
    decoded = np.asarray(projection["jacobian_np"], dtype=float) @ control
    if decoded.size != 4 * k:
        raise ValueError("decoded edge-control update has the wrong size")
    scale = np.asarray(projection["mode_scale_np"], dtype=float)
    dR = np.zeros_like(np.asarray(template_state.Rcos, dtype=float))
    dR_sin = np.zeros_like(np.asarray(template_state.Rsin, dtype=float))
    dZ_cos = np.zeros_like(np.asarray(template_state.Zcos, dtype=float))
    dZ = np.zeros_like(np.asarray(template_state.Zsin, dtype=float))
    dR[-1] = decoded[0:k] / scale
    dR_sin[-1] = decoded[k : 2 * k] / scale
    dZ_cos[-1] = decoded[2 * k : 3 * k] / scale
    dZ[-1] = decoded[3 * k : 4 * k] / scale
    return dR, dR_sin, dZ_cos, dZ


@dataclass(frozen=True)
class FreeBoundaryNativeSplineUnknownVector:
    """VMEC interior unknowns plus reduced spline controls for the LCFS edge.

    The vector drops the full Fourier R/Z edge rows from the nonlinear state and
    stores only reduced spline-control coordinates for that edge.  Decoding
    recreates a full ``VMECState`` for the existing force kernels.
    """

    template_state: VMECState
    edge_state: FreeBoundaryReducedEdgeState
    vector: np.ndarray
    projection: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.template_state, VMECState):
            raise TypeError("template_state must be a VMECState")
        if not isinstance(self.edge_state, FreeBoundaryReducedEdgeState):
            raise TypeError("edge_state must be a FreeBoundaryReducedEdgeState")
        if not bool(self.projection.get("enabled", False)):
            raise ValueError("projection must be enabled")
        expected = self.native_unknown_size
        vector = _finite_vector(self.vector, name="vector", size=expected)
        object.__setattr__(self, "vector", vector)

    @classmethod
    def from_vmec_state(
        cls,
        state: VMECState,
        projection: dict[str, Any],
    ) -> "FreeBoundaryNativeSplineUnknownVector":
        """Encode ``state`` as a native spline-control unknown vector."""

        edge_state = free_boundary_reduced_edge_state_from_vmec_state(state, projection)
        vector = np.concatenate(
            [
                _pack_vmec_interior_without_edge(state),
                np.asarray(edge_state.control_delta, dtype=float).reshape(-1),
            ],
            axis=0,
        )
        return cls(
            template_state=state,
            edge_state=edge_state,
            vector=vector,
            projection=projection,
        )

    @property
    def interior_size(self) -> int:
        """Number of VMEC-basis unknowns retained in the native vector."""

        return _native_spline_interior_size(self.template_state)

    @property
    def edge_control_size(self) -> int:
        """Number of reduced LCFS spline controls."""

        return int(self.edge_state.control_state.control_map.control_count)

    @property
    def native_unknown_size(self) -> int:
        """Total size of the reduced native unknown vector."""

        return int(self.interior_size + self.edge_control_size)

    @property
    def full_vmec_size(self) -> int:
        """Total size of the equivalent full VMEC state vector."""

        return int(self.template_state.layout.size)

    @property
    def removed_fourier_edge_dofs(self) -> int:
        """Number of full Fourier LCFS R/Z coefficients removed."""

        return int(4 * self.template_state.layout.K - self.edge_control_size)

    @property
    def interior_vector(self) -> np.ndarray:
        """Packed VMEC interior/Lambda portion of the native unknown vector."""

        return np.asarray(self.vector[: self.interior_size], dtype=float)

    @property
    def control_delta(self) -> np.ndarray:
        """Reduced LCFS spline-control coordinates."""

        return np.asarray(self.vector[self.interior_size :], dtype=float)

    def with_vector(self, vector: Any) -> "FreeBoundaryNativeSplineUnknownVector":
        """Return the same native layout with a new packed vector."""

        values = _finite_vector(vector, name="vector", size=self.native_unknown_size)
        control_state = ReducedControlState(
            control_map=self.edge_state.control_state.control_map,
            control_delta=values[self.interior_size :],
        )
        return FreeBoundaryNativeSplineUnknownVector(
            template_state=self.template_state,
            edge_state=FreeBoundaryReducedEdgeState(control_state=control_state),
            vector=values,
            projection=self.projection,
        )

    def to_vmec_state(self) -> VMECState:
        """Decode this native vector to the full VMEC Fourier state."""

        Rcos, Rsin, Zcos, Zsin, Lcos, Lsin = _unpack_vmec_interior_without_edge(
            self.interior_vector,
            self.template_state,
        )
        template = VMECState(
            layout=self.template_state.layout,
            Rcos=Rcos,
            Rsin=Rsin,
            Zcos=Zcos,
            Zsin=Zsin,
            Lcos=Lcos,
            Lsin=Lsin,
        )
        control_state = ReducedControlState(
            control_map=self.edge_state.control_state.control_map,
            control_delta=self.control_delta,
        )
        edge_state = FreeBoundaryReducedEdgeState(control_state=control_state)
        return free_boundary_reduced_edge_state_to_vmec_state(
            edge_state,
            template,
            self.projection,
            host_update=True,
        )

    def vector_from_delta_tuple(self, deltas: Any, *, edge_metric: str = "pullback") -> np.ndarray:
        """Pack a VMEC delta or force tuple in native spline coordinates.

        ``edge_metric="pullback"`` is the adjoint/force transform.  Use
        ``edge_metric="least_squares"`` for a physical update direction that
        should be expressed as reduced edge coordinates.
        """

        dR, dR_sin, dZ_cos, dZ, dL_cos, dL = deltas
        interior = _pack_vmec_interior_without_edge(
            VMECState(
                layout=self.template_state.layout,
                Rcos=dR,
                Rsin=dR_sin,
                Zcos=dZ_cos,
                Zsin=dZ,
                Lcos=dL_cos,
                Lsin=dL,
            )
        )
        metric = str(edge_metric).strip().lower()
        target = _freeb_edge_control_delta_tuple_target(deltas, self.projection)
        if metric in {"pullback", "adjoint", "force", "jtf", "j.t"}:
            edge = self.edge_state.pullback(target)
        elif metric in {"least_squares", "least-squares", "ls", "projection", "coordinate"}:
            edge = _freeb_edge_control_project_vector_np(target, self.projection).control_delta
        else:
            raise ValueError("edge_metric must be 'pullback' or 'least_squares'")
        return np.concatenate([interior, np.asarray(edge, dtype=float).reshape(-1)], axis=0)

    def delta_tuple_from_vector(self, vector: Any) -> tuple[np.ndarray, ...]:
        """Decode a native update vector into a VMEC delta tuple."""

        values = _finite_vector(vector, name="vector", size=self.native_unknown_size)
        Rcos, Rsin, Zcos, Zsin, Lcos, Lsin = _unpack_vmec_interior_without_edge(
            values[: self.interior_size],
            VMECState(
                layout=self.template_state.layout,
                Rcos=np.zeros_like(np.asarray(self.template_state.Rcos, dtype=float)),
                Rsin=np.zeros_like(np.asarray(self.template_state.Rsin, dtype=float)),
                Zcos=np.zeros_like(np.asarray(self.template_state.Zcos, dtype=float)),
                Zsin=np.zeros_like(np.asarray(self.template_state.Zsin, dtype=float)),
                Lcos=np.zeros_like(np.asarray(self.template_state.Lcos, dtype=float)),
                Lsin=np.zeros_like(np.asarray(self.template_state.Lsin, dtype=float)),
            ),
        )
        dR_edge, dR_sin_edge, dZ_cos_edge, dZ_edge = _edge_update_delta_tuple(
            self.template_state,
            self.projection,
            values[self.interior_size :],
        )
        Rcos[-1] = dR_edge[-1]
        Rsin[-1] = dR_sin_edge[-1]
        Zcos[-1] = dZ_cos_edge[-1]
        Zsin[-1] = dZ_edge[-1]
        return (Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)

    def edge_fit_residual(self) -> dict[str, float | None]:
        """Measure how exactly the native vector reconstructs its LCFS edge."""

        decoded = self.to_vmec_state()
        edge_values = _freeb_edge_control_state_edge_values(decoded, self.projection)
        expected = self.edge_state.control_state.control_map.decode(self.control_delta)
        residual = edge_values - expected
        finite = residual[np.isfinite(residual)]
        residual_l2 = float(np.linalg.norm(finite)) if finite.size else 0.0
        target_l2 = float(np.linalg.norm(expected))
        return {
            "l2": residual_l2,
            "linf": float(np.max(np.abs(finite))) if finite.size else 0.0,
            "rel": None if target_l2 <= np.finfo(float).tiny else float(residual_l2 / target_l2),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return compact JSON-friendly native unknown-vector diagnostics."""

        edge_residual = self.edge_fit_residual()
        return {
            "schema": "FreeBoundaryNativeSplineUnknownVector.v1",
            "mode": "free_boundary_native_spline_unknown_vector",
            "host_side": True,
            "full_vmec_size": int(self.full_vmec_size),
            "native_unknown_size": int(self.native_unknown_size),
            "interior_unknown_size": int(self.interior_size),
            "edge_control_size": int(self.edge_control_size),
            "removed_fourier_edge_dofs": int(self.removed_fourier_edge_dofs),
            "unknown_reduction_fraction": float(self.native_unknown_size / self.full_vmec_size),
            "edge_labels": list(self.edge_state.control_state.control_map.labels),
            "edge_control_l2": float(np.linalg.norm(self.control_delta)),
            "edge_control_linf": float(np.max(np.abs(self.control_delta))) if self.control_delta.size else 0.0,
            "edge_reconstruction_residual_l2": edge_residual["l2"],
            "edge_reconstruction_residual_linf": edge_residual["linf"],
            "edge_reconstruction_residual_rel": edge_residual["rel"],
        }


def free_boundary_native_spline_unknown_vector_from_vmec_state(
    state: VMECState,
    projection: dict[str, Any],
) -> FreeBoundaryNativeSplineUnknownVector:
    """Encode ``state`` as native free-boundary spline-control unknowns."""

    return FreeBoundaryNativeSplineUnknownVector.from_vmec_state(state, projection)
