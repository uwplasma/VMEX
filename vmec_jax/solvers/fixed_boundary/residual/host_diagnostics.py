"""Host-side VMEC2000 print/diagnostic contexts for residual iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Vmec2000PrintContext:
    """Bound host-side VMEC2000 row-print helpers for one residual solve."""

    nstep_screen: int
    print_iter_row: Callable[..., None]
    should_print: Callable[[int, int], bool]


def resolve_vmec2000_print_context(
    *,
    cfg: Any,
    indata: Any,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    getenv: Callable[[str, str], str],
    resolve_debug_print_config: Callable[..., Any],
    resolve_nstep_screen: Callable[..., int],
    emit_iter_row: Callable[..., None],
    should_print_row: Callable[..., bool],
    print_row: Callable[..., None],
) -> Vmec2000PrintContext:
    """Resolve row-printing policy and return bound print/cadence helpers."""

    debug_print_config = resolve_debug_print_config(
        print_env=getenv("VMEC_JAX_SCAN_PRINT", "1"),
        mode_env=getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_print"),
        ordered_env=getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
    )
    scan_print_mode = debug_print_config.mode
    scan_print_ordered = debug_print_config.ordered
    print_live = debug_print_config.print_live
    jax_debug = None
    io_callback = None
    if print_live:
        try:
            from jax import debug as jax_debug  # type: ignore[assignment]
        except Exception:
            jax_debug = None
    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as io_callback  # type: ignore[assignment]
        except Exception:
            scan_print_mode = resolve_debug_print_config(
                print_env="1",
                mode_env=scan_print_mode,
                ordered_env="0",
                io_callback_available=False,
            ).mode
            io_callback = None

    nstep_screen = resolve_nstep_screen(
        indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
        override_env=getenv("VMEC_JAX_NSTEP_OVERRIDE", ""),
    )

    def print_iter_row(
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        fsqr1: float,
        fsqz1: float,
        fsql1: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        del fsqr1, fsqz1, fsql1
        emit_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(cfg.lasym),
            z00=z00,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            print_live=bool(print_live),
            scan_print_mode=scan_print_mode,
            scan_print_ordered=bool(scan_print_ordered),
            jax_debug=jax_debug,
            io_callback=io_callback,
            print_row=print_row,
        )

    def should_print(iter_idx: int, max_iter: int) -> bool:
        return should_print_row(
            iter_idx=iter_idx,
            max_iter=max_iter,
            nstep_screen=nstep_screen,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
        )

    return Vmec2000PrintContext(
        nstep_screen=int(nstep_screen),
        print_iter_row=print_iter_row,
        should_print=should_print,
    )


__all__ = ["Vmec2000PrintContext", "resolve_vmec2000_print_context"]
