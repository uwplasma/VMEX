"""Mirror-native NetCDF output for solved straight-axis equilibria.

``mout`` is deliberately separate from VMEC's toroidal ``wout`` schema. It
stores physical-grid arrays so a solved open-ended equilibrium can be plotted
or inspected without reconstructing the solver objects.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import contravariant_field, evaluate_geometry, magnetic_field_xyz
from .model import MIRROR_OUTPUT_SCHEMA


@dataclass(frozen=True)
class MoutData:
    """Data-only representation of one straight-axis mirror equilibrium."""

    s: Any
    theta: Any
    xi: Any
    z: Any
    boundary_radius: Any
    radius_scale: Any
    lambda_stream: Any
    mod_b: Any
    b_xyz: Any
    p_perpendicular: Any
    p_parallel: Any
    history: Any
    coil_xyz: Any
    ftol: float
    iterations: int
    converged: bool
    mass_scale: float
    variational_max: float
    normal_stress_rms: float
    b_normal_rms: float
    closure: str = "unknown"
    message: str = ""
    schema: str = MIRROR_OUTPUT_SCHEMA


def mout_from_result(
    result: Any,
    grid: Any,
    config: Any,
    *,
    axial_flux_derivative: Any,
    current_derivative: Any = 0.0,
    parallel_pressure: Any | None = None,
    coil_xyz: Any | None = None,
    closure: str = "unknown",
) -> MoutData:
    """Collect a solved mirror result and reproducible plotting fields."""

    geometry = evaluate_geometry(result.plasma_state, grid)
    field = contravariant_field(
        result.plasma_state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    perpendicular = np.asarray(result.perpendicular_pressure)
    parallel = (
        np.full(perpendicular.shape, np.nan)
        if parallel_pressure is None
        else np.asarray(parallel_pressure)
    )
    if parallel.shape != perpendicular.shape:
        raise ValueError("parallel_pressure must match perpendicular_pressure")
    coils = np.empty((0, 0, 3)) if coil_xyz is None else np.asarray(coil_xyz)
    if coils.ndim != 3 or coils.shape[-1] != 3:
        raise ValueError("coil_xyz must have shape (ncoil, npoint, 3)")
    return MoutData(
        s=np.asarray(grid.s),
        theta=np.asarray(grid.theta),
        xi=np.asarray(grid.xi),
        z=np.asarray(grid.z),
        boundary_radius=np.asarray(result.boundary.radius_scale),
        radius_scale=np.asarray(result.plasma_state.radius_scale),
        lambda_stream=np.asarray(result.plasma_state.lambda_stream),
        mod_b=np.sqrt(np.asarray(result.plasma_b_squared)),
        b_xyz=np.asarray(magnetic_field_xyz(field, geometry)),
        p_perpendicular=perpendicular,
        p_parallel=parallel,
        history=np.asarray(result.history),
        coil_xyz=coils,
        ftol=float(config.ftol),
        iterations=int(result.iterations),
        converged=bool(result.converged),
        mass_scale=float(result.mass_scale),
        variational_max=float(result.variational_max),
        normal_stress_rms=float(result.interface.normal_stress_rms),
        b_normal_rms=float(result.interface.vacuum_b_normal_rms),
        closure=str(closure),
        message=str(result.message),
    )


def _finite_shape(data: MoutData) -> tuple[int, int, int]:
    shape = np.asarray(data.radius_scale).shape
    if len(shape) != 3:
        raise ValueError("radius_scale must have shape (ns, ntheta, nxi)")
    ns, ntheta, nxi = shape
    expected = {
        "boundary_radius": (ntheta, nxi),
        "lambda_stream": shape,
        "mod_b": shape,
        "b_xyz": (*shape, 3),
        "p_perpendicular": shape,
        "p_parallel": shape,
        "s": (ns,),
        "theta": (ntheta,),
        "xi": (nxi,),
        "z": (nxi,),
    }
    for name, wanted in expected.items():
        if np.asarray(getattr(data, name)).shape != wanted:
            raise ValueError(f"{name} must have shape {wanted}")
    return ns, ntheta, nxi


def write_mout(path: str | Path, data: MoutData, *, overwrite: bool = True) -> Path:
    """Write a compact mirror-native NetCDF file."""

    import netCDF4

    path = Path(path)
    if path.suffix.lower() != ".nc":
        path = path.with_suffix(".nc")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, ntheta, nxi = _finite_shape(data)
    history = np.asarray(data.history)
    if history.ndim != 2:
        raise ValueError("history must be a two-dimensional table")
    coils = np.asarray(data.coil_xyz)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncattr("schema", data.schema)
        for name in (
            "ftol", "iterations", "converged", "mass_scale", "variational_max",
            "normal_stress_rms", "b_normal_rms", "closure", "message",
        ):
            value = getattr(data, name)
            dataset.setncattr(name, int(value) if isinstance(value, (bool, np.bool_)) else value)
        for name, size in (
            ("s", ns), ("theta", ntheta), ("xi", nxi), ("xyz", 3),
            ("history_row", history.shape[0]), ("history_column", history.shape[1]),
            ("coil", coils.shape[0]), ("coil_point", coils.shape[1]),
        ):
            dataset.createDimension(name, size)
        variables = {
            "s": (("s",), data.s),
            "theta": (("theta",), data.theta),
            "xi": (("xi",), data.xi),
            "z": (("xi",), data.z),
            "boundary_radius": (("theta", "xi"), data.boundary_radius),
            "radius_scale": (("s", "theta", "xi"), data.radius_scale),
            "lambda_stream": (("s", "theta", "xi"), data.lambda_stream),
            "mod_b": (("s", "theta", "xi"), data.mod_b),
            "b_xyz": (("s", "theta", "xi", "xyz"), data.b_xyz),
            "p_perpendicular": (("s", "theta", "xi"), data.p_perpendicular),
            "p_parallel": (("s", "theta", "xi"), data.p_parallel),
            "history": (("history_row", "history_column"), history),
            "coil_xyz": (("coil", "coil_point", "xyz"), coils),
        }
        for name, (dimensions, values) in variables.items():
            variable = dataset.createVariable(name, "f8", dimensions, zlib=True, complevel=4)
            variable[:] = np.asarray(values)
    return path


def read_mout(path: str | Path) -> MoutData:
    """Read a :class:`MoutData` file and validate its schema."""

    import netCDF4

    with netCDF4.Dataset(Path(path)) as dataset:
        schema = str(dataset.getncattr("schema"))
        if schema != MIRROR_OUTPUT_SCHEMA:
            raise ValueError(f"unsupported mirror output schema: {schema}")
        arrays = {name: np.asarray(dataset[name][:]) for name in (
            "s", "theta", "xi", "z", "boundary_radius", "radius_scale",
            "lambda_stream", "mod_b", "b_xyz", "p_perpendicular", "p_parallel",
            "history", "coil_xyz",
        )}
        attributes = {
            field.name: dataset.getncattr(field.name)
            for field in fields(MoutData)
            if field.name not in arrays and field.name != "schema"
        }
    attributes["converged"] = bool(attributes["converged"])
    attributes["iterations"] = int(attributes["iterations"])
    data = MoutData(**arrays, **attributes, schema=schema)
    _finite_shape(data)
    return data


__all__ = ["MoutData", "mout_from_result", "read_mout", "write_mout"]
