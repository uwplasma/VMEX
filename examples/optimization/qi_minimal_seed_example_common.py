"""Shared runner for the per-NFP minimal-seed QI example scripts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
QI_DRIVER = REPO_ROOT / "examples" / "optimization" / "QI_optimization.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optimization.qi_optimization_cases import QI_CASES


@dataclass(frozen=True)
class MinimalSeedQIExample:
    """Editable controls for one public minimal-seed QI optimization example."""

    nfp: int
    policy_case: str
    input_file: Path
    reference_input: Path
    output_dir: Path
    max_mode: int = 5
    min_vmec_mode: int = 8
    method: str = "auto"
    max_nfev: int = 70
    continuation_nfev: int = 20
    inner_max_iter: int = 550
    inner_ftol: float = 1.0e-10
    trial_max_iter: int = 550
    trial_ftol: float = 1.0e-10
    ess_alpha: float = 1.2
    target_aspect: float = 6.0
    target_abs_iota_min: float = 0.41
    max_mirror_ratio: float = 0.35
    max_elongation: float = 10.0
    make_plots: bool = True
    dry_run: bool = False


def _path_arg(path: Path) -> str:
    """Return a repository-relative path when possible."""

    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _path_arg(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(f"{float(value):.16g}" for value in values)


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def policy_boundary_reference(example: MinimalSeedQIExample) -> dict[str, Any]:
    """Return the same-NFP reference-family preconditioner for this example."""

    case = QI_CASES[example.policy_case]
    boundary = dict(case.get("boundary_reference_preconditioner", {}))
    boundary.update(
        enabled=True,
        reference_input=example.reference_input,
        max_mode=example.max_mode,
        target_aspect=example.target_aspect,
        abs_iota_min=example.target_abs_iota_min,
        max_mirror_ratio=example.max_mirror_ratio,
        max_elongation=example.max_elongation,
        accept_as_baseline=True,
    )
    return _jsonable(boundary)


def policy_mirror_ramp_stages(example: MinimalSeedQIExample) -> list[dict[str, Any]]:
    """Return the guarded QI/mirror cleanup stages for this example."""

    return [_jsonable(dict(stage)) for stage in QI_CASES[example.policy_case].get("mirror_ramp_stages", ())]


def write_policy_files(example: MinimalSeedQIExample, directory: Path) -> tuple[Path, Path]:
    """Write portable JSON policy files consumed by ``QI_optimization.py``."""

    boundary_path = directory / "boundary_reference.json"
    stages_path = directory / "mirror_ramp_stages.json"
    boundary_path.write_text(json.dumps(policy_boundary_reference(example), indent=2, sort_keys=True) + "\n")
    stages_path.write_text(json.dumps(policy_mirror_ramp_stages(example), indent=2, sort_keys=True) + "\n")
    return boundary_path, stages_path


def build_qi_optimization_command(
    example: MinimalSeedQIExample,
    boundary_reference_json: Path,
    mirror_ramp_stages_json: Path,
) -> list[str]:
    """Build the delegated QI command with all scientific controls explicit."""

    boundary = policy_boundary_reference(example)
    lambdas = tuple(float(value) for value in boundary.get("lambdas", (1.0,)))
    command = [
        sys.executable,
        _path_arg(QI_DRIVER),
        "--input-file",
        _path_arg(example.input_file),
        "--output-dir",
        str(example.output_dir),
        "--max-mode",
        str(example.max_mode),
        "--min-vmec-mode",
        str(example.min_vmec_mode),
        "--method",
        example.method,
        "--target-aspect",
        f"{example.target_aspect:.16g}",
        "--target-abs-iota-min",
        f"{example.target_abs_iota_min:.16g}",
        "--max-mirror-ratio",
        f"{example.max_mirror_ratio:.16g}",
        "--max-elongation",
        f"{example.max_elongation:.16g}",
        "--use-simple-seed",
        "--use-target-helicity-seed",
        "--use-reference-family-seed",
        "--reference-input",
        _path_arg(example.reference_input),
        "--reference-lambdas",
        _csv(lambdas),
        "--boundary-reference-json",
        str(boundary_reference_json),
        "--mirror-ramp-stages-json",
        str(mirror_ramp_stages_json),
        "--accept-boundary-reference-baseline",
        "--use-mode-continuation",
        "--stage-mode-policy",
        "lower-repeat",
        "--stage-repeats",
        "2",
        "--max-nfev",
        str(example.max_nfev),
        "--continuation-nfev",
        str(example.continuation_nfev),
        "--inner-max-iter",
        str(example.inner_max_iter),
        "--inner-ftol",
        f"{example.inner_ftol:.16g}",
        "--trial-max-iter",
        str(example.trial_max_iter),
        "--trial-ftol",
        f"{example.trial_ftol:.16g}",
        "--ess-alpha",
        f"{example.ess_alpha:.16g}",
        "--use-ess",
        _bool_flag("make-plots", example.make_plots),
    ]
    return command


def run_minimal_seed_qi_example(example: MinimalSeedQIExample) -> int:
    """Run one per-NFP QI example through the editable QI driver."""

    with tempfile.TemporaryDirectory(prefix=f"vmec_jax_qi_nfp{example.nfp}_") as tmpdir:
        boundary_json, stages_json = write_policy_files(example, Path(tmpdir))
        command = build_qi_optimization_command(example, boundary_json, stages_json)
        print(f"Running NFP={example.nfp} minimal-seed QI optimization.")
        print(f"  raw seed:     {_path_arg(example.input_file)}")
        print(f"  QI reference: {_path_arg(example.reference_input)}")
        print(f"  policy case:  {example.policy_case}")
        print(f"  output dir:   {example.output_dir}")
        print("  command:")
        print("    " + " ".join(shlex.quote(item) for item in command))
        if example.dry_run:
            return 0
        sys.stdout.flush()
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0
