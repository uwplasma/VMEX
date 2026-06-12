#!/usr/bin/env python
"""Bounded command/summary harness for small QI optimization probes.

This diagnostic builds a lightweight Cartesian matrix for
``examples/optimization/QI_optimization.py`` without launching optimization
jobs by default.  It is intended for short, auditable QI parameter searches
over knobs such as ``max_mode``, ``stage_mode_policy``, ``stage_repeats``,
Boozer/QI resolution, internal VMEC ``MPOL``/``NTOR`` floors, ``max_nfev``,
optimizer ``method``, ESS ``alpha``, accepted/trial VMEC tolerances and
iteration caps, and supported objective weights.  The harness writes a
``plan.json`` plus command summaries that can be run manually or by a scheduler.

Boundary ``max_m``/``max_n`` are emitted through per-case
``--stage-mode-limits-json`` files, so probes can vary active poloidal and
toroidal boundary mode rectangles independently.  Unsupported weight keys are
preserved in the plan and reported as non-command metadata.

The same module can parse existing output directories containing
``diagnostics.json`` and/or ``history.json`` into compact JSON/CSV summaries.
It does not import ``vmec_jax`` and should be safe to use in dry-run tooling.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import shlex
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.QI_stel_seed_3127"
DEFAULT_SCRIPT = REPO_ROOT / "examples" / "optimization" / "QI_optimization.py"
DEFAULT_OUT_ROOT = Path("results/diagnostics/qi_parameter_probe_harness")

CSV_FIELDS = (
    "case_id",
    "max_mode",
    "stage_mode_policy",
    "stage_repeats",
    "boundary_max_m",
    "boundary_max_n",
    "min_vmec_mode",
    "vmec_mpol",
    "vmec_ntor",
    "qi_mboz",
    "qi_nboz",
    "qi_nphi",
    "qi_nalpha",
    "qi_n_bounce",
    "max_nfev",
    "method",
    "ess_alpha",
    "inner_ftol",
    "inner_max_iter",
    "trial_ftol",
    "trial_max_iter",
    "weights",
    "output_dir",
    "stage_mode_limits_json",
    "command",
    "unsupported_weight_keys",
)

SUMMARY_FIELDS = (
    "case_id",
    "output_dir",
    "selection",
    "selection_reason",
    "smooth_qi",
    "legacy_qi",
    "mirror",
    "elongation",
    "iota",
    "aspect",
    "objective_final",
    "wall_time_s",
    "diagnostics_json",
    "history_json",
    "error",
)

SUPPORTED_WEIGHT_FLAGS = {
    "mirror": "--mirror-weight",
    "mirror_weight": "--mirror-weight",
    "elongation": "--elongation-weight",
    "elongation_weight": "--elongation-weight",
}

SUPPORTED_METHODS = (
    "auto",
    "auto_scalar",
    "gauss_newton",
    "scipy",
    "scipy_matrix_free",
    "lbfgs_adjoint",
    "scalar_trust",
)


@dataclass(frozen=True)
class ProbeCase:
    """One generated QI probe command plus metadata."""

    case_id: str
    max_mode: int
    stage_mode_policy: str
    stage_repeats: int
    boundary_max_m: int | None
    boundary_max_n: int | None
    min_vmec_mode: int | None
    vmec_mpol: int | None
    vmec_ntor: int | None
    qi_mboz: int
    qi_nboz: int
    qi_nphi: int
    qi_nalpha: int
    qi_n_bounce: int
    max_nfev: int
    method: str
    ess_alpha: float
    inner_ftol: float
    inner_max_iter: int
    trial_ftol: float
    trial_max_iter: int
    weights: dict[str, float]
    output_dir: str
    command: list[str]
    stage_mode_limits_json: str | None = None
    stage_mode_limits: tuple[dict[str, Any], ...] = ()
    unsupported_weight_keys: tuple[str, ...] = ()

    def command_text(self) -> str:
        return shlex.join(self.command)

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["command"] = self.command_text()
        record["unsupported_weight_keys"] = list(self.unsupported_weight_keys)
        return record


def _split_csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _int_values(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in _split_csv(raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {raw!r}") from exc
    return values


def _float_values(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in _split_csv(raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {raw!r}") from exc
    return values


def _method_values(raw: str) -> tuple[str, ...]:
    values = _split_csv(raw)
    unknown = [value for value in values if value not in SUPPORTED_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported method(s): {', '.join(unknown)}; expected one of {', '.join(SUPPORTED_METHODS)}"
        )
    return values


def _optional_int_values(raw: str) -> tuple[int | None, ...]:
    values: list[int | None] = []
    for part in _split_csv(raw):
        if part.lower() in {"none", "null", "-"}:
            values.append(None)
        else:
            try:
                values.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid optional integer list: {raw!r}") from exc
    return tuple(values)


def parse_weight_set(raw: str) -> dict[str, float]:
    """Parse one ``key=value`` objective-weight set."""

    weights: dict[str, float] = {}
    if not str(raw).strip():
        return weights
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"weight entries must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        if not key:
            raise argparse.ArgumentTypeError(f"weight key cannot be empty in {item!r}")
        try:
            weights[key] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid weight value in {item!r}") from exc
    return weights


def _case_token(value: Any) -> str:
    text = str(value).replace("-", "m").replace(".", "p").replace("+", "")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)


def _case_id(
    *,
    max_mode: int,
    stage_mode_policy: str,
    stage_repeats: int,
    method: str,
    max_nfev: int,
    ess_alpha: float,
    weight_index: int,
) -> str:
    return (
        f"m{max_mode}_{_case_token(stage_mode_policy)}_r{stage_repeats}_"
        f"{_case_token(method)}_nfev{max_nfev}_a{_case_token(ess_alpha)}_w{weight_index:02d}"
    )


def _stage_mode_sequence(max_mode: int, *, policy: str, repeats: int, continuation_nfev: int = 1) -> list[int]:
    """Mirror the QI stage policies without importing vmec_jax."""

    max_mode = int(max_mode)
    repeats = max(1, int(repeats))
    key = str(policy).strip().lower().replace("_", "-")
    if max_mode <= 1 or int(continuation_nfev) <= 0:
        return [max_mode]
    if key in {"lower", "lower-mode", "mode", "qs"}:
        out: list[int] = []
        for mode in range(1, max_mode + 1):
            out.extend([mode] * (2 if mode == 1 else 3))
        return out
    if key in {"lower-repeat", "ladder-repeat", "repeat-lower", "rung-repeat"}:
        return [mode for mode in range(1, max_mode + 1) for _ in range(repeats)]
    if key in {"repeat", "same", "same-mode", "same-mode-repeat"}:
        return [max_mode] * repeats
    raise ValueError(f"Unsupported QI stage mode policy: {policy!r}")


def _stage_mode_limits_payload(
    *,
    max_mode: int,
    policy: str,
    repeats: int,
    boundary_max_m: int | None,
    boundary_max_n: int | None,
) -> tuple[dict[str, Any], ...]:
    """Return anisotropic stage limits, or an empty tuple for scalar mode stages."""

    if boundary_max_m is None and boundary_max_n is None:
        return ()
    max_mode = int(max_mode)
    global_m = max_mode if boundary_max_m is None else int(boundary_max_m)
    global_n = max_mode if boundary_max_n is None else int(boundary_max_n)
    payload = []
    for stage_mode in _stage_mode_sequence(max_mode, policy=policy, repeats=repeats):
        max_m = min(abs(global_m), int(stage_mode))
        max_n = min(abs(global_n), int(stage_mode))
        payload.append(
            {
                "mode": max(max_m, max_n),
                "max_m": max_m,
                "max_n": max_n,
                "label": f"m{max_m}_n{max_n}",
            }
        )
    return tuple(payload)


def build_command(
    *,
    python_executable: str,
    script: Path,
    input_file: Path,
    output_dir: Path,
    max_mode: int,
    stage_mode_policy: str,
    stage_repeats: int,
    min_vmec_mode: int | None,
    vmec_mpol: int | None,
    vmec_ntor: int | None,
    qi_mboz: int,
    qi_nboz: int,
    qi_nphi: int,
    qi_nalpha: int,
    qi_n_bounce: int,
    max_nfev: int,
    method: str,
    ess_alpha: float,
    inner_ftol: float,
    inner_max_iter: int,
    trial_ftol: float,
    trial_max_iter: int,
    weights: dict[str, float],
    stage_mode_limits_json: Path | None = None,
) -> tuple[list[str], tuple[str, ...]]:
    """Return a supported QI example command and unsupported weight metadata."""

    command = [
        python_executable,
        str(script),
        "--input-file",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--max-mode",
        str(int(max_mode)),
        "--stage-mode-policy",
        str(stage_mode_policy),
        "--stage-repeats",
        str(int(stage_repeats)),
        "--qi-mboz",
        str(int(qi_mboz)),
        "--qi-nboz",
        str(int(qi_nboz)),
        "--qi-nphi",
        str(int(qi_nphi)),
        "--qi-nalpha",
        str(int(qi_nalpha)),
        "--qi-n-bounce",
        str(int(qi_n_bounce)),
        "--audit-qi-mboz",
        str(int(qi_mboz)),
        "--audit-qi-nboz",
        str(int(qi_nboz)),
        "--audit-qi-nphi",
        str(int(qi_nphi)),
        "--audit-qi-nalpha",
        str(int(qi_nalpha)),
        "--audit-qi-n-bounce",
        str(int(qi_n_bounce)),
        "--max-nfev",
        str(int(max_nfev)),
        "--method",
        str(method),
        "--ess-alpha",
        f"{float(ess_alpha):.17g}",
        "--inner-ftol",
        f"{float(inner_ftol):.17g}",
        "--inner-max-iter",
        str(int(inner_max_iter)),
        "--trial-ftol",
        f"{float(trial_ftol):.17g}",
        "--trial-max-iter",
        str(int(trial_max_iter)),
        "--no-make-plots",
    ]
    if min_vmec_mode is not None:
        command.extend(["--min-vmec-mode", str(int(min_vmec_mode))])
    if vmec_mpol is not None:
        command.extend(["--vmec-mpol", str(int(vmec_mpol))])
    if vmec_ntor is not None:
        command.extend(["--vmec-ntor", str(int(vmec_ntor))])
    if stage_mode_limits_json is not None:
        command.extend(["--stage-mode-limits-json", str(stage_mode_limits_json)])
    unsupported: list[str] = []
    for key, value in sorted(weights.items()):
        flag = SUPPORTED_WEIGHT_FLAGS.get(key)
        if flag is None:
            unsupported.append(key)
            continue
        command.extend([flag, f"{float(value):.17g}"])
    return command, tuple(unsupported)


def generate_cases(
    *,
    input_file: Path,
    out_root: Path,
    script: Path,
    python_executable: str,
    max_modes: Sequence[int],
    stage_mode_policies: Sequence[str],
    stage_repeats: Sequence[int],
    boundary_max_ms: Sequence[int | None],
    boundary_max_ns: Sequence[int | None],
    min_vmec_modes: Sequence[int | None],
    vmec_mpol_values: Sequence[int | None],
    vmec_ntor_values: Sequence[int | None],
    qi_mboz_values: Sequence[int],
    qi_nboz_values: Sequence[int],
    qi_nphi_values: Sequence[int],
    qi_nalpha_values: Sequence[int],
    qi_n_bounce_values: Sequence[int],
    max_nfev_values: Sequence[int],
    methods: Sequence[str],
    ess_alpha_values: Sequence[float],
    inner_ftol_values: Sequence[float],
    inner_max_iter_values: Sequence[int],
    trial_ftol_values: Sequence[float],
    trial_max_iter_values: Sequence[int],
    weight_sets: Sequence[dict[str, float]],
) -> list[ProbeCase]:
    """Build the bounded Cartesian matrix of QI probe commands."""

    cases: list[ProbeCase] = []
    for values in itertools.product(
        max_modes,
        stage_mode_policies,
        stage_repeats,
        boundary_max_ms,
        boundary_max_ns,
        min_vmec_modes,
        vmec_mpol_values,
        vmec_ntor_values,
        qi_mboz_values,
        qi_nboz_values,
        qi_nphi_values,
        qi_nalpha_values,
        qi_n_bounce_values,
        max_nfev_values,
        methods,
        ess_alpha_values,
        inner_ftol_values,
        inner_max_iter_values,
        trial_ftol_values,
        trial_max_iter_values,
        enumerate(weight_sets),
    ):
        (
            max_mode,
            policy,
            repeats,
            boundary_max_m,
            boundary_max_n,
            min_vmec_mode,
            vmec_mpol,
            vmec_ntor,
            qi_mboz,
            qi_nboz,
            qi_nphi,
            qi_nalpha,
            qi_n_bounce,
            max_nfev,
            method,
            ess_alpha,
            inner_ftol,
            inner_max_iter,
            trial_ftol,
            trial_max_iter,
            weight_pair,
        ) = values
        weight_index, weights = weight_pair
        stage_mode_limits = _stage_mode_limits_payload(
            max_mode=int(max_mode),
            policy=str(policy),
            repeats=int(repeats),
            boundary_max_m=None if boundary_max_m is None else int(boundary_max_m),
            boundary_max_n=None if boundary_max_n is None else int(boundary_max_n),
        )
        case_id = _case_id(
            max_mode=int(max_mode),
            stage_mode_policy=str(policy),
            stage_repeats=int(repeats),
            method=str(method),
            max_nfev=int(max_nfev),
            ess_alpha=float(ess_alpha),
            weight_index=int(weight_index),
        )
        output_dir = out_root / "runs" / case_id
        stage_mode_limits_json = output_dir / "stage_mode_limits.json" if stage_mode_limits else None
        command, unsupported = build_command(
            python_executable=python_executable,
            script=script,
            input_file=input_file,
            output_dir=output_dir,
            max_mode=int(max_mode),
            stage_mode_policy=str(policy),
            stage_repeats=int(repeats),
            min_vmec_mode=None if min_vmec_mode is None else int(min_vmec_mode),
            vmec_mpol=None if vmec_mpol is None else int(vmec_mpol),
            vmec_ntor=None if vmec_ntor is None else int(vmec_ntor),
            qi_mboz=int(qi_mboz),
            qi_nboz=int(qi_nboz),
            qi_nphi=int(qi_nphi),
            qi_nalpha=int(qi_nalpha),
            qi_n_bounce=int(qi_n_bounce),
            max_nfev=int(max_nfev),
            method=str(method),
            ess_alpha=float(ess_alpha),
            inner_ftol=float(inner_ftol),
            inner_max_iter=int(inner_max_iter),
            trial_ftol=float(trial_ftol),
            trial_max_iter=int(trial_max_iter),
            weights=dict(weights),
            stage_mode_limits_json=stage_mode_limits_json,
        )
        cases.append(
            ProbeCase(
                case_id=case_id,
                max_mode=int(max_mode),
                stage_mode_policy=str(policy),
                stage_repeats=int(repeats),
                boundary_max_m=None if boundary_max_m is None else int(boundary_max_m),
                boundary_max_n=None if boundary_max_n is None else int(boundary_max_n),
                min_vmec_mode=None if min_vmec_mode is None else int(min_vmec_mode),
                vmec_mpol=None if vmec_mpol is None else int(vmec_mpol),
                vmec_ntor=None if vmec_ntor is None else int(vmec_ntor),
                qi_mboz=int(qi_mboz),
                qi_nboz=int(qi_nboz),
                qi_nphi=int(qi_nphi),
                qi_nalpha=int(qi_nalpha),
                qi_n_bounce=int(qi_n_bounce),
                max_nfev=int(max_nfev),
                method=str(method),
                ess_alpha=float(ess_alpha),
                inner_ftol=float(inner_ftol),
                inner_max_iter=int(inner_max_iter),
                trial_ftol=float(trial_ftol),
                trial_max_iter=int(trial_max_iter),
                weights=dict(weights),
                output_dir=str(output_dir),
                command=command,
                stage_mode_limits_json=None if stage_mode_limits_json is None else str(stage_mode_limits_json),
                stage_mode_limits=stage_mode_limits,
                unsupported_weight_keys=unsupported,
            )
        )
    return cases


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _history_final(history: dict[str, Any], key: str) -> float | None:
    value = history.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return _finite_float(value[-1])
    return _finite_float(value)


def summarize_output_dir(path: Path) -> dict[str, Any]:
    """Summarize one QI output directory without requiring vmec_jax imports."""

    path = Path(path).expanduser()
    diagnostics_path = path / "diagnostics.json"
    history_path = path / "history.json"
    diagnostics = _read_json(diagnostics_path)
    history = _read_json(history_path)
    error = None
    if not diagnostics and not history:
        error = "missing diagnostics.json and history.json"
    failure_reasons = diagnostics.get("qi_failure_reasons", [])
    if isinstance(failure_reasons, list):
        selection_reason = "; ".join(str(item) for item in failure_reasons)
    else:
        selection_reason = str(failure_reasons) if failure_reasons else ""
    selected = diagnostics.get("qi_engineering_gate_passed")
    selection = "selected" if selected is True else "rejected" if selected is False else None
    return {
        "case_id": path.name,
        "output_dir": str(path),
        "selection": selection,
        "selection_reason": selection_reason,
        "smooth_qi": _finite_float(diagnostics.get("qi_smooth_total")),
        "legacy_qi": _finite_float(diagnostics.get("qi_legacy_total")),
        "mirror": _finite_float(diagnostics.get("qi_mirror_ratio_max")),
        "elongation": _finite_float(diagnostics.get("qi_max_elongation")),
        "iota": _finite_float(diagnostics.get("mean_iota")),
        "aspect": _finite_float(diagnostics.get("aspect")),
        "objective_final": _history_final(history, "objective_final"),
        "wall_time_s": _finite_float(history.get("total_wall_time_s") or history.get("wall_time_s")),
        "diagnostics_json": str(diagnostics_path) if diagnostics_path.exists() else None,
        "history_json": str(history_path) if history_path.exists() else None,
        "error": error,
    }


def discover_output_dirs(roots: Iterable[Path]) -> list[Path]:
    """Find result directories below roots that contain QI output JSON files."""

    found: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file():
            root = root.parent
        if (root / "diagnostics.json").exists() or (root / "history.json").exists():
            found.add(root)
        if root.is_dir():
            for json_path in itertools.chain(root.rglob("diagnostics.json"), root.rglob("history.json")):
                found.add(json_path.parent)
    return sorted(found)


def write_case_outputs(cases: Sequence[ProbeCase], out_root: Path, *, command_only: bool) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    records = [case.as_record() for case in cases]
    plan = {
        "kind": "qi_parameter_probe_harness",
        "execute": False,
        "command_only": bool(command_only),
        "case_count": len(records),
        "note": "Commands are generated only; this harness does not launch QI optimization jobs.",
        "cases": records,
    }
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    with (out_root / "commands.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in CSV_FIELDS}
            row["weights"] = json.dumps(record.get("weights", {}), sort_keys=True)
            row["unsupported_weight_keys"] = ",".join(record.get("unsupported_weight_keys", []))
            writer.writerow(row)
    for case in cases:
        if case.stage_mode_limits_json is None:
            continue
        path = Path(case.stage_mode_limits_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(case.stage_mode_limits), indent=2, sort_keys=True) + "\n")
    commands_text = "\n".join(case.command_text() for case in cases)
    (out_root / "commands.sh").write_text(commands_text + ("\n" if commands_text else ""))


def write_summary(records: Sequence[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(list(records), indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--max-mode", type=_int_values, default=(3,))
    parser.add_argument("--stage-mode-policy", type=_split_csv, default=("lower-repeat",))
    parser.add_argument("--stage-repeats", type=_int_values, default=(1,))
    parser.add_argument("--boundary-max-m", type=_optional_int_values, default=(None,))
    parser.add_argument("--boundary-max-n", type=_optional_int_values, default=(None,))
    parser.add_argument("--min-vmec-mode", type=_optional_int_values, default=(None,))
    parser.add_argument("--vmec-mpol", type=_optional_int_values, default=(None,))
    parser.add_argument("--vmec-ntor", type=_optional_int_values, default=(None,))
    parser.add_argument("--qi-mboz", type=_int_values, default=(8,))
    parser.add_argument("--qi-nboz", type=_int_values, default=(8,))
    parser.add_argument("--qi-nphi", type=_int_values, default=(41,))
    parser.add_argument("--qi-nalpha", type=_int_values, default=(9,))
    parser.add_argument("--qi-n-bounce", type=_int_values, default=(11,))
    parser.add_argument("--max-nfev", type=_int_values, default=(2,))
    parser.add_argument("--method", type=_method_values, default=("scipy",))
    parser.add_argument("--ess-alpha", type=_float_values, default=(1.2,))
    parser.add_argument("--inner-ftol", type=_float_values, default=(1.0e-8,))
    parser.add_argument("--inner-max-iter", type=_int_values, default=(80,))
    parser.add_argument("--trial-ftol", type=_float_values, default=(1.0e-8,))
    parser.add_argument("--trial-max-iter", type=_int_values, default=(50,))
    parser.add_argument(
        "--weights",
        action="append",
        type=parse_weight_set,
        default=None,
        help="One weight set as key=value[,key=value]. Repeat for multiple weight cases.",
    )
    parser.add_argument("--limit", type=int, default=64, help="Safety cap on generated cases.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated commands after writing the plan.")
    parser.add_argument(
        "--command-only",
        action="store_true",
        help="Only write command artifacts; same non-executing behavior, with a plan marker for schedulers.",
    )
    parser.add_argument(
        "--parse-output-dir",
        action="append",
        type=Path,
        default=[],
        help="Existing output root/case directory to summarize. Repeat for multiple roots.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    weight_sets = args.weights if args.weights is not None else [{}]
    cases = generate_cases(
        input_file=args.input_file.expanduser(),
        out_root=args.out_root.expanduser(),
        script=args.script.expanduser(),
        python_executable=str(args.python),
        max_modes=args.max_mode,
        stage_mode_policies=args.stage_mode_policy,
        stage_repeats=args.stage_repeats,
        boundary_max_ms=args.boundary_max_m,
        boundary_max_ns=args.boundary_max_n,
        min_vmec_modes=args.min_vmec_mode,
        vmec_mpol_values=args.vmec_mpol,
        vmec_ntor_values=args.vmec_ntor,
        qi_mboz_values=args.qi_mboz,
        qi_nboz_values=args.qi_nboz,
        qi_nphi_values=args.qi_nphi,
        qi_nalpha_values=args.qi_nalpha,
        qi_n_bounce_values=args.qi_n_bounce,
        max_nfev_values=args.max_nfev,
        methods=args.method,
        ess_alpha_values=args.ess_alpha,
        inner_ftol_values=args.inner_ftol,
        inner_max_iter_values=args.inner_max_iter,
        trial_ftol_values=args.trial_ftol,
        trial_max_iter_values=args.trial_max_iter,
        weight_sets=weight_sets,
    )
    if len(cases) > int(args.limit):
        raise SystemExit(f"Refusing to generate {len(cases)} cases; raise --limit if this is intentional.")
    write_case_outputs(cases, args.out_root.expanduser(), command_only=bool(args.command_only))
    print(f"Wrote QI parameter probe plan: {args.out_root / 'plan.json'}")
    print(f"Wrote {len(cases)} generated command(s): {args.out_root / 'commands.sh'}")
    if args.dry_run:
        for case in cases:
            print(case.command_text())

    if args.parse_output_dir:
        output_dirs = discover_output_dirs(args.parse_output_dir)
        records = [summarize_output_dir(path) for path in output_dirs]
        summary_json = args.summary_json or args.out_root / "summary.json"
        summary_csv = args.summary_csv or args.out_root / "summary.csv"
        write_summary(records, summary_json, summary_csv)
        print(f"Wrote QI parameter probe summary: {summary_json}")
        print(f"Wrote QI parameter probe CSV summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
