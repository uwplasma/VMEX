"""Generate a small 3-way runtime comparison (VMEC2000 vs vmec_jax vs VMEC++) for README cases.

This script targets the same two fixed-boundary cases featured in the README
fsq_total trace figure:

- ITERModel
- LandremanPaul2021_QA_lowres

Runtimes are measured using the user-facing CLIs:

- VMEC2000: xvmec2000
- vmec_jax: vmec_jax <inputfile> (no flags)
- VMEC++: vmec_standalone <input.json>

Note: VMEC++'s C++ standalone executable consumes VMEC++ JSON input. We convert
VMEC2000 INDATA -> JSON using the Fortran `indata2json` tool shipped with VMEC++.
The reported VMEC++ runtime is solver-only (conversion time is reported separately).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_vmecpp_tools(vmecpp_root: Path | None) -> tuple[Path, Path]:
    # Prefer explicit root, then common local layout: ../external/vmecpp (next to vmec_jax repo).
    roots: list[Path] = []
    if vmecpp_root is not None:
        roots.append(vmecpp_root.expanduser().resolve())
    roots.append(REPO_ROOT.parent / "external" / "vmecpp")
    roots.append(REPO_ROOT / "external" / "vmecpp")

    for root in roots:
        standalone = root / "build" / "vmec_standalone"
        i2j = root / "build" / "_deps" / "indata2json-build" / "indata2json"
        if standalone.exists() and i2j.exists():
            return standalone, i2j

    raise FileNotFoundError(
        "VMEC++ tools not found. Provide --vmecpp-root pointing at a vmecpp checkout with a built `build/vmec_standalone`."
    )


def _run_vmec_jax_cli(input_path: Path, *, workdir: Path, timeout_s: float) -> float:
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["vmec_jax", str(input_path)],
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=float(timeout_s),
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"vmec_jax failed for {input_path}:\n{proc.stderr}")
    return float(dt)


def _run_vmecpp_standalone(
    *,
    indata_path: Path,
    vmecpp_standalone: Path,
    indata2json: Path,
    workdir: Path,
    timeout_s: float,
) -> tuple[float, float, Path]:
    """Return (conversion_s, runtime_s, json_path)."""
    workdir.mkdir(parents=True, exist_ok=True)
    local_input = workdir / indata_path.name
    shutil.copy2(indata_path, local_input)

    # indata2json writes <case>.json in cwd, where case is derived from input.<case>.
    t0 = time.perf_counter()
    subprocess.run([str(indata2json), local_input.name], cwd=str(workdir), check=True, stdout=subprocess.DEVNULL)
    conv_s = time.perf_counter() - t0

    case = indata_path.name[len("input.") :] if indata_path.name.startswith("input.") else indata_path.stem
    json_path = workdir / f"{case}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"indata2json did not produce expected {json_path}")

    t1 = time.perf_counter()
    subprocess.run([str(vmecpp_standalone), str(json_path)], cwd=str(workdir), check=True, stdout=subprocess.DEVNULL, timeout=float(timeout_s))
    rt_s = time.perf_counter() - t1
    return float(conv_s), float(rt_s), json_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs-dir",
        type=Path,
        default=REPO_ROOT / "examples_single_grid" / "data",
        help="Directory containing input.* files.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "docs" / "_static" / "figures",
        help="Where to write the PNG figure.",
    )
    p.add_argument(
        "--workdir",
        type=Path,
        default=REPO_ROOT / "outputs" / "readme_vmecpp_runtime_two_cases_work",
        help="Scratch directory for run artifacts (ignored by git).",
    )
    p.add_argument(
        "--reuse-workdir",
        action="store_true",
        help="Reuse existing results.json under --workdir (skip rerunning solvers).",
    )
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--vmecpp-root", type=Path, default=None, help="Path to a vmecpp checkout (with build/).")
    args = p.parse_args()

    vmec_exec = find_vmec2000_exec(root=REPO_ROOT.parent)
    if vmec_exec is None:
        raise SystemExit("VMEC2000 executable not found. Set VMEC2000_EXEC or ensure STELLOPT/VMEC2000 is available.")

    vmecpp_standalone, indata2json = _find_vmecpp_tools(args.vmecpp_root)

    inputs_dir = args.inputs_dir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    workdir = args.workdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    cases = {
        "ITERModel": inputs_dir / "input.ITERModel",
        "LandremanPaul2021_QA_lowres": inputs_dir / "input.LandremanPaul2021_QA_lowres",
    }
    for name, path in cases.items():
        if not path.exists():
            raise FileNotFoundError(f"missing input for {name}: {path}")

    results_path = workdir / "results.json"
    if bool(args.reuse_workdir) and results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results: dict[str, dict[str, float]] = {}
        for name, input_path in cases.items():
            case_work = workdir / name
            shutil.rmtree(case_work, ignore_errors=True)
            case_work.mkdir(parents=True, exist_ok=True)

            # VMEC2000 runtime
            vmec = run_xvmec2000(
                input_path,
                exec_path=Path(vmec_exec),
                workdir=case_work / "vmec2000",
                timeout_s=float(args.timeout_s),
            )
            vmec_rt = float(vmec.runtime_s)

            # vmec_jax runtime (CLI; no flags)
            jax_rt = _run_vmec_jax_cli(input_path, workdir=case_work / "vmec_jax", timeout_s=float(args.timeout_s))

            # VMEC++ runtime (C++ standalone + converter)
            conv_s, vmecpp_rt, _ = _run_vmecpp_standalone(
                indata_path=input_path,
                vmecpp_standalone=vmecpp_standalone,
                indata2json=indata2json,
                workdir=case_work / "vmecpp",
                timeout_s=float(args.timeout_s),
            )

            results[name] = {
                "vmec2000_runtime_s": vmec_rt,
                "vmec_jax_runtime_s": jax_rt,
                "vmecpp_runtime_s": vmecpp_rt,
                "vmecpp_conversion_s": conv_s,
            }

        # Write results JSON for reproducibility (under outputs/, ignored).
        results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    # Plot.
    labels = list(cases.keys())
    vmec2000 = [results[k]["vmec2000_runtime_s"] for k in labels]
    vmec_jax = [results[k]["vmec_jax_runtime_s"] for k in labels]
    vmecpp = [results[k]["vmecpp_runtime_s"] for k in labels]

    x = list(range(len(labels)))
    w = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.8))
    ax.bar([v - w for v in x], vmec2000, width=w, label="VMEC2000", color="#1f77b4")
    ax.bar(x, vmec_jax, width=w, label="vmec_jax", color="#ff7f0e")
    ax.bar([v + w for v in x], vmecpp, width=w, label="VMEC++ (solver only)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yscale("log")
    ax.set_ylabel("runtime (s, log scale)")
    ax.set_title("")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Single-grid fixed-boundary runtime\n(NS_ARRAY=151, NITER_ARRAY=5000, FTOL_ARRAY=1e-14, NSTEP=500)",
        fontsize=14,
        y=0.99,
    )
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    out_png = outdir / "readme_runtime_two_cases_vmecpp.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
