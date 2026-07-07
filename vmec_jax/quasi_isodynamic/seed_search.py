"""QI seed-search primitives for basin surveys and local landscape scans.

These helpers are production package code because staged QI optimization uses
them to build bounded candidate jumps before local differentiable refinement.
Developer diagnostics may wrap these functions in CLI reports, but installed
vmec_jax must not depend on repo-local ``tools/`` modules for this path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from vmec_jax.optimization import BoundaryParamSpec, boundary_param_names


LANDSCAPE_SURFACES = (0.35, 0.65)
LANDSCAPE_METRICS = (
    ("qi_smooth_total", "QI residual"),
    ("qi_mirror_ratio_max", "Mirror ratio"),
    ("qi_max_elongation", "Max elongation"),
    ("aspect", "Aspect"),
    ("mean_iota", "Mean iota"),
)
BASIN_SUMMARY_FIELDS = (
    "rank",
    "label",
    "kind",
    "radius",
    "score",
    "qi_smooth_total",
    "qi_legacy_total",
    "qi_mirror_ratio_max",
    "qi_max_elongation",
    "mean_iota",
    "aspect",
    "input_path",
    "error",
)


@dataclass(frozen=True)
class ScanAxis:
    """One scanned boundary parameter and its increment values."""

    dof: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class SurveyTargets:
    """Acceptance and ranking targets for far-seed QI basin surveys."""

    smooth_qi_max: float = 2.0e-3
    legacy_qi_max: float = 2.0e-3
    mirror_ratio_max: float = 0.35
    max_elongation: float = 8.0
    abs_iota_min: float = 0.41
    target_aspect: float = 10.0
    aspect_tolerance: float = 2.0


@dataclass(frozen=True)
class BasinCandidate:
    """One boundary perturbation candidate in VMEC optimization coordinates."""

    label: str
    kind: str
    radius: float
    params: tuple[float, ...]
    dominant_dof: str | None = None

    def as_record(self, names: Sequence[str]) -> dict[str, Any]:
        """Return a JSON-ready record with only nonzero boundary deltas."""

        params = list(float(v) for v in self.params)
        return {
            "label": self.label,
            "kind": self.kind,
            "radius": float(self.radius),
            "dominant_dof": self.dominant_dof,
            "params": params,
            "deltas": {name: value for name, value in zip(names, params) if abs(value) > 0.0},
        }


def parse_surfaces(raw: str) -> tuple[float, ...]:
    """Parse comma-separated QI diagnostic surfaces in ``(0, 1]``."""

    surfaces = tuple(float(part) for part in raw.split(",") if part.strip())
    if not surfaces:
        raise argparse.ArgumentTypeError("--surfaces must contain at least one value")
    for surface in surfaces:
        if surface <= 0.0 or surface > 1.0:
            raise argparse.ArgumentTypeError("QI landscape surfaces must be in (0, 1]")
    return surfaces


def parse_dofs(raw: str) -> tuple[str, ...]:
    """Parse one or two comma-separated boundary degree-of-freedom names."""

    dofs = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not 1 <= len(dofs) <= 2:
        raise argparse.ArgumentTypeError("--dofs must contain one or two comma-separated boundary DOF names")
    return dofs


def resolve_input_path(path: Path) -> Path:
    """Resolve either a VMEC input file or an optimization output directory."""

    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    candidates = [
        path / "input.final",
        *sorted(path.glob("stage_*_mode*/input.final"), reverse=True),
        path / "input.initial",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No input.final/input.initial file found below {path}")


def finite_float(value: Any) -> float | None:
    """Return the first finite scalar in ``value``, or ``None``."""

    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    out = float(arr.ravel()[0])
    return out if np.isfinite(out) else None


def base_param_values(stage) -> np.ndarray:
    """Return the stage's current boundary parameter vector when available."""

    base_vector = getattr(stage.optimizer, "_base_params_vector", None)
    if callable(base_vector):
        return np.asarray(base_vector(), dtype=float).reshape(-1)
    return np.zeros(len(stage.specs), dtype=float)


def choose_default_dofs(stage, *, count: int) -> tuple[str, ...]:
    """Choose high-amplitude free coefficients when the CLI omits DOFs."""

    names = boundary_param_names(stage.specs)
    if not names:
        raise ValueError("No active boundary DOFs are available for this stage.")
    base = base_param_values(stage)
    order = np.argsort(-np.abs(base)) if base.size == len(names) else np.arange(len(names))
    selected: list[str] = []
    for idx in order:
        name = names[int(idx)]
        if name.lower() == "rc00":
            continue
        selected.append(name)
        if len(selected) == int(count):
            break
    if len(selected) < int(count):
        raise ValueError(f"Requested {count} DOFs but only found {len(selected)} active DOFs.")
    return tuple(selected)


def axis_from_span(dof: str, *, span: float, points: int) -> ScanAxis:
    """Build a symmetric scan axis around one boundary degree of freedom."""

    if int(points) < 2:
        raise ValueError("points must be at least 2")
    return ScanAxis(dof=dof, values=tuple(float(v) for v in np.linspace(-float(span), float(span), int(points))))


def scan_landscape_records(
    *,
    axes: Sequence[ScanAxis],
    specs: Sequence[BoundaryParamSpec],
    evaluate: Callable[[np.ndarray], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate a 1D or 2D QI landscape scan with an injected solver callback."""

    if not 1 <= len(axes) <= 2:
        raise ValueError("Landscape scans support one or two axes.")
    names = boundary_param_names(specs)
    name_to_index = {name: idx for idx, name in enumerate(names)}
    missing = [axis.dof for axis in axes if axis.dof not in name_to_index]
    if missing:
        raise ValueError(f"Unknown boundary DOF(s): {', '.join(missing)}. Available DOFs: {', '.join(names)}")

    records: list[dict[str, Any]] = []
    for point in itertools.product(*(axis.values for axis in axes)):
        params = np.zeros(len(specs), dtype=float)
        deltas = {}
        for axis, value in zip(axes, point):
            params[name_to_index[axis.dof]] = float(value)
            deltas[axis.dof] = float(value)
        diagnostic = dict(evaluate(params))
        metric_values = {key: finite_float(diagnostic.get(key)) for key, _label in LANDSCAPE_METRICS}
        records.append(
            {
                "deltas": deltas,
                "params": params.tolist(),
                "metrics": metric_values,
                "diagnostics": diagnostic,
            }
        )
    return records


def build_stage(
    *,
    input_path: Path,
    max_mode: int,
    min_vmec_mode: int,
    include: Sequence[str],
    fix: Sequence[str],
    project_input_boundary_to_max_mode: bool,
    inner_max_iter: int,
    inner_ftol: float,
    trial_max_iter: int,
    trial_ftol: float,
    solver_device: str | None,
):
    """Build a forward-solve-only fixed-boundary objective stage."""

    from vmec_jax import load_config
    from vmec_jax.config import config_from_indata
    from vmec_jax.optimization_workflow import (
        AspectRatio,
        build_fixed_boundary_objective_stage,
        rebuild_for_optimization_resolution,
    )

    cfg, indata = load_config(str(input_path))
    indata = rebuild_for_optimization_resolution(
        indata,
        max_mode=int(max_mode),
        min_vmec_mode=int(min_vmec_mode),
    )
    cfg = config_from_indata(indata)
    objective = AspectRatio().to_objective_term(target=1.0, residual_weight=1.0)
    return build_fixed_boundary_objective_stage(
        cfg,
        indata,
        stage_mode=int(max_mode),
        objectives=[objective],
        include=tuple(include),
        fix=tuple(fix),
        project_input_boundary_to_max_mode=bool(project_input_boundary_to_max_mode),
        min_coeff=0.0,
        inner_max_iter=int(inner_max_iter),
        inner_ftol=float(inner_ftol),
        trial_max_iter=int(trial_max_iter),
        trial_ftol=float(trial_ftol),
        solver_device=solver_device,
    )


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    """Scale a direction so its largest absolute component is one."""

    direction = np.asarray(direction, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(direction))) if direction.size else 0.0
    if max_abs == 0.0 or not np.isfinite(max_abs):
        return np.zeros_like(direction)
    return direction / max_abs


def axis_order(names: Sequence[str], x_scale: np.ndarray) -> list[int]:
    """Order non-axis boundary DOFs by decreasing ESS scale."""

    order = np.argsort(-np.asarray(x_scale, dtype=float))
    return [int(idx) for idx in order if str(names[int(idx)]).lower() != "rc00"]


def generate_basin_candidates(
    *,
    names: Sequence[str],
    x_scale: Sequence[float],
    radii: Sequence[float],
    n_random: int,
    rng_seed: int,
    axis_count: int,
    directions: Sequence[str],
    include_zero: bool = True,
) -> list[BasinCandidate]:
    """Create deterministic large-step candidates in boundary-coordinate space."""

    names = tuple(str(name) for name in names)
    scale = np.asarray(x_scale, dtype=float).reshape(-1)
    if scale.size != len(names):
        raise ValueError("x_scale length must match names length")
    if np.any(~np.isfinite(scale)):
        raise ValueError("x_scale must be finite")
    if any(float(radius) < 0.0 for radius in radii):
        raise ValueError("survey radii must be non-negative")
    direction_set = {str(item).strip().lower() for item in directions if str(item).strip()}
    candidates: list[BasinCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def add(label: str, kind: str, radius: float, direction: np.ndarray, dominant_dof: str | None = None) -> None:
        """Append one unique scaled candidate direction."""

        normed = normalize_direction(direction)
        params = tuple(float(v) for v in (float(radius) * scale * normed))
        key = tuple(round(value, 16) for value in params)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            BasinCandidate(
                label=label,
                kind=kind,
                radius=float(radius),
                params=params,
                dominant_dof=dominant_dof,
            )
        )

    if include_zero:
        add("zero", "baseline", 0.0, np.zeros(len(names), dtype=float))

    axis_indices = axis_order(names, scale)[: max(0, int(axis_count))]
    if "axes" in direction_set:
        for radius in radii:
            for idx in axis_indices:
                direction = np.zeros(len(names), dtype=float)
                direction[idx] = 1.0
                add(f"axis+:{names[idx]}:{radius:g}", "axis_positive", float(radius), direction, names[idx])
                add(f"axis-:{names[idx]}:{radius:g}", "axis_negative", float(radius), -direction, names[idx])

    rng = np.random.default_rng(int(rng_seed))
    for radius in radii:
        if "rademacher" in direction_set:
            for sample in range(max(0, int(n_random))):
                direction = rng.choice(np.asarray([-1.0, 1.0]), size=len(names))
                add(f"rademacher:{sample:03d}:{radius:g}", "rademacher", float(radius), direction)
        if "gaussian" in direction_set:
            for sample in range(max(0, int(n_random))):
                direction = rng.normal(size=len(names))
                add(f"gaussian:{sample:03d}:{radius:g}", "gaussian", float(radius), direction)

    return candidates


def basin_score(metrics: dict[str, Any], targets: SurveyTargets = SurveyTargets()) -> float:
    """Return a finite basin-ranking score where smaller is better."""

    values = {key: finite_float(metrics.get(key)) for key in BASIN_SUMMARY_FIELDS}
    smooth = values.get("qi_smooth_total")
    legacy = values.get("qi_legacy_total")
    mirror = values.get("qi_mirror_ratio_max")
    elongation = values.get("qi_max_elongation")
    iota = values.get("mean_iota")
    aspect = values.get("aspect")
    if any(value is None for value in (smooth, legacy, mirror, elongation, iota, aspect)):
        return 1.0e12
    smooth_score = max(0.0, float(smooth)) / max(float(targets.smooth_qi_max), 1.0e-16)
    legacy_score = max(0.0, float(legacy)) / max(float(targets.legacy_qi_max), 1.0e-16)
    mirror_score = max(0.0, float(mirror) - float(targets.mirror_ratio_max)) / max(
        float(targets.mirror_ratio_max), 1.0e-16
    )
    elongation_score = max(0.0, float(elongation) - float(targets.max_elongation)) / max(
        float(targets.max_elongation), 1.0e-16
    )
    iota_score = max(0.0, float(targets.abs_iota_min) - abs(float(iota))) / max(
        float(targets.abs_iota_min), 1.0e-16
    )
    aspect_score = max(0.0, abs(float(aspect) - float(targets.target_aspect)) - float(targets.aspect_tolerance)) / max(
        float(targets.target_aspect), 1.0e-16
    )
    return float(
        smooth_score
        + legacy_score
        + 2.0 * mirror_score
        + elongation_score
        + 4.0 * iota_score
        + 0.25 * aspect_score
    )


def rank_candidate_records(
    records: Sequence[dict[str, Any]],
    *,
    targets: SurveyTargets = SurveyTargets(),
) -> list[dict[str, Any]]:
    """Attach basin scores/ranks and return records sorted by score."""

    scored = []
    for record in records:
        out = dict(record)
        metrics = dict(out.get("metrics", {}))
        score = 1.0e12 if out.get("error") else basin_score(metrics, targets=targets)
        out["score"] = float(score)
        scored.append(out)
    scored.sort(key=lambda item: (float(item["score"]), str(item.get("label", ""))))
    for rank, record in enumerate(scored, start=1):
        record["rank"] = rank
    return scored


def write_basin_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Write ranked basin records to a compact CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASIN_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            metrics = dict(record.get("metrics", {}))
            row = {field: record.get(field) for field in BASIN_SUMMARY_FIELDS}
            row.update({key: metrics.get(key) for key in BASIN_SUMMARY_FIELDS if key in metrics})
            writer.writerow(row)
