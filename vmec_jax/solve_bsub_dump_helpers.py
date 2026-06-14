"""Covariant-field debug dump helpers for VMEC solve diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._solve_runtime import _parse_iter_list


def maybe_dump_bsube(*, bc, static, iter_idx: int) -> None:
    """Optionally dump scaled full-mesh covariant ``B_u/B_v`` fields."""

    env = os.getenv("VMEC_JAX_DUMP_BSUBE", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(bc.bsubu_e_scaled)
    bsubv = np.asarray(bc.bsubv_e_scaled)
    ns, ntheta, nzeta = bsubu.shape

    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube dump (scaled)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write(f"lamscale={float(np.asarray(bc.lamscale)):.16e}\n")
        f.write("columns: js lt lz bsubu_e bsubv_e\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")


def maybe_dump_bsube_terms(*, bc, static, iter_idx: int) -> None:
    """Optionally dump terms entering scaled full-mesh covariant fields."""

    env = os.getenv("VMEC_JAX_DUMP_BSUBE_TERMS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_terms_ns{ns}_iter{int(iter_idx)}.dat"

    lvv_sh = np.asarray(getattr(bc, "lvv_sh"))
    lu0 = np.asarray(getattr(bc, "lu0_force"))
    lu1 = np.asarray(getattr(bc, "lu1_full"))
    phip = np.asarray(getattr(bc, "phip_internal"))
    bsubu_tmp = np.asarray(getattr(bc, "bsubu_tmp"))
    bsubv_pre = np.asarray(getattr(bc, "bsubv_preblend"))

    ns, ntheta, nzeta = lvv_sh.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube terms dump\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz lvv_sh lu0 lu1 phipf bsubu_tmp bsubv_pre\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{lvv_sh[js, lt, lz]:24.16e}{lu0[js, lt, lz]:24.16e}{lu1[js, lt, lz]:24.16e}"
                        f"{phip[js]:24.16e}{bsubu_tmp[js, lt, lz]:24.16e}{bsubv_pre[js, lt, lz]:24.16e}\n"
                    )


def maybe_dump_bsubh(*, bc, static, iter_idx: int) -> None:
    """Optionally dump half-mesh covariant ``B_u/B_v`` fields."""

    env = os.getenv("VMEC_JAX_DUMP_BSUBH", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsubh_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(getattr(bc, "bsubu"))
    bsubv = np.asarray(getattr(bc, "bsubv"))

    ns, ntheta, nzeta = bsubu.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsubh dump (half mesh)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz bsubuh bsubvh\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")
