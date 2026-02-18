"""Generate README runtime comparison (VMEC2000 vs vmec_jax)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000


def _run_vmec2000(input_path: Path, *, niter: int, ftol: float, workdir: Path) -> float:
    exe = find_vmec2000_exec()
    if exe is None:
        raise SystemExit("xvmec2000 executable not found")
    cfg, _ = load_config(str(input_path))
    vmec = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=workdir,
        timeout_s=600.0,
        indata_updates={
            "NITER": str(niter),
            "NSTEP": "1",
            "NS_ARRAY": f"{int(cfg.ns)}",
            "NITER_ARRAY": f"{niter}",
            "FTOL": f"{float(ftol):.3e}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
        keep_workdir=True,
    )
    return float(vmec.runtime_s)


def _run_vmec_jax(input_path: Path, *, niter: int, ftol: float, workdir: Path) -> float:
    patched = _patch_indata(
        input_path.read_text(),
        updates={
            "NITER": str(niter),
            "NSTEP": "1",
            "FTOL": f"{float(ftol):.3e}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
    )
    tmp_input = workdir / f"input_patched_{input_path.name}"
    tmp_input.write_text(patched)
    t0 = time.perf_counter()
    _ = run_fixed_boundary(
        str(tmp_input),
        solver="vmec2000_iter",
        max_iter=int(niter),
        multigrid=False,
        multigrid_use_input_niter=False,
        verbose=False,
        performance_mode=False,
    )
    return float(time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--axisym-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.shaped_tokamak_pressure"),
    )
    p.add_argument(
        "--stellarator-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.n3are_R7.75B5.7_lowres"),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument("--niter", type=int, default=250)
    p.add_argument("--ftol", type=float, default=1e-14)
    args = p.parse_args()

    axisym_input = Path(args.axisym_input).expanduser().resolve()
    stellarator_input = Path(args.stellarator_input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    axisym_work = outdir / "readme_axisym_vmec2000_runtime"
    st_work = outdir / "readme_stellarator_vmec2000_runtime"
    axisym_work.mkdir(parents=True, exist_ok=True)
    st_work.mkdir(parents=True, exist_ok=True)

    t_vmec_a = _run_vmec2000(axisym_input, niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work)
    t_vmec_s = _run_vmec2000(stellarator_input, niter=int(args.niter), ftol=float(args.ftol), workdir=st_work)

    t_jax_a_cold = _run_vmec_jax(axisym_input, niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work)
    t_jax_a_warm = _run_vmec_jax(axisym_input, niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work)

    t_jax_s_cold = _run_vmec_jax(stellarator_input, niter=int(args.niter), ftol=float(args.ftol), workdir=st_work)
    t_jax_s_warm = _run_vmec_jax(stellarator_input, niter=int(args.niter), ftol=float(args.ftol), workdir=st_work)

    cases = ["axisym", "n3are"]
    vmec = [t_vmec_a, t_vmec_s]
    jax_cold = [t_jax_a_cold, t_jax_s_cold]
    jax_warm = [t_jax_a_warm, t_jax_s_warm]

    x = np.arange(len(cases))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - width, vmec, width, label="VMEC2000")
    ax.bar(x, jax_cold, width, label="vmec_jax (cold)")
    ax.bar(x + width, jax_warm, width, label="vmec_jax (warm)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("runtime (s)")
    ax.set_title(f"Runtime comparison (NITER={int(args.niter)}, FTOL={float(args.ftol):.1e})")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    outpath = outdir / "readme_runtime_compare.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
