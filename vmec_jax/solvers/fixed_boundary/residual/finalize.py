"""Final result assembly helpers for residual-iteration VMEC solves."""

from __future__ import annotations

from typing import Any, Callable, Mapping
import time

import numpy as np

from ..diagnostics.io import _pack_resume_state_record
from .runtime import (
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _format_residual_iter_timing_message,
)

_EMPTY_HISTORY_KEYS = ("w_history", "fsqr2_history", "fsqz2_history", "fsql2_history", "grad_rms_history", "step_history")
_RESUME_BASE_KEYS = (
    "time_step inv_tau fsq_prev fsq0_prev flip_sign iter1 last_iter2 ijacob "
    "bad_resets res0 res1 prev_rz_fsq bad_growth_streak huge_force_restart_count "
    "vmec2000_cache_valid freeb_ivac freeb_ivacskip freeb_nvacskip freeb_nvskip0 "
    "freeb_last_model freeb_nestor_runtime"
).split()
_RESUME_HEAVY_ARRAY_KEYS = "vRcc vRss vZsc vZcs vLsc vLcs vRsc vRcs vZcc vZss vLcc vLss".split()
_RESUME_HEAVY_OBJECT_KEYS = (
    "state_checkpoint cache_precond_diag cache_tcon cache_norms cache_rz_scale cache_l_scale "
    "cache_rz_norm cache_f_norm1 cache_prec_rz_mats cache_prec_rz_jmax cache_prec_lam_prec "
    "cache_prec_faclam cache_prec_lam_debug cache_constraint_rcon0 cache_constraint_zcon0"
).split()

__all__ = [
    "attach_residual_iter_timing_diagnostics",
    "build_residual_iter_resume_state_payload",
    "build_residual_iter_resume_state_from_namespace",
    "finalize_residual_iter_result",
    "precompile_only_residual_iter_result",
    "vmec2000_state_only_scan_result",
    "vmec2000_traced_scan_result",
]


def attach_residual_iter_timing_diagnostics(
    diagnostics: dict[str, Any],
    timing_stats: dict[str, float],
    *,
    timing_enabled: bool,
    timing_detail_enabled: bool,
    finalize_diag_build_start: float | None,
    iteration_loop_start: float | None,
    finalize_start: float | None,
    solve_wall_start: float,
    print_timing: bool = True,
) -> dict[str, Any]:
    """Attach residual-iteration timing diagnostics to ``diagnostics``."""

    if not bool(timing_enabled):
        return diagnostics
    if finalize_diag_build_start is not None:
        timing_stats["finalize_diag_build"] += time.perf_counter() - float(finalize_diag_build_start)
    if iteration_loop_start is not None and finalize_start is not None:
        timing_stats["iteration_loop"] = float(finalize_start) - float(iteration_loop_start)
    if finalize_start is not None:
        timing_stats["finalize"] = time.perf_counter() - float(finalize_start)
    timing_report = _build_residual_iter_timing_report(
        timing_stats,
        solve_total_s=float(time.perf_counter() - float(solve_wall_start)),
        timing_detail_enabled=bool(timing_detail_enabled),
    )
    timing_stats["iteration_loop_unattributed"] = float(timing_report["iteration_loop_unattributed_s"])
    diagnostics["timing"] = timing_report
    if bool(print_timing):
        try:
            print(
                _format_residual_iter_timing_message(
                    timing_report,
                    timing_detail_enabled=bool(timing_detail_enabled),
                ),
                flush=True,
            )
        except Exception:
            pass
    return diagnostics


def build_residual_iter_resume_state_payload(
    *,
    resume_state_mode: str,
    base_kwargs: Mapping[str, Any],
    heavy_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build packed resume-state payload for residual iteration."""

    mode = str(resume_state_mode)
    if mode == "none":
        return None
    base = _build_resume_state_base(**dict(base_kwargs))
    heavy = dict(heavy_payload) if mode == "full" and heavy_payload is not None else None
    return _pack_resume_state_record(base=base, heavy=heavy, mode=mode)


def build_residual_iter_resume_state_from_namespace(
    namespace: Mapping[str, Any],
    *,
    resume_state_mode: str,
) -> dict[str, Any] | None:
    """Build a residual-iteration resume payload from selected local values."""

    mode = str(resume_state_mode)
    if mode == "none":
        return None
    base_kwargs = {key: namespace[key] for key in _RESUME_BASE_KEYS}
    heavy = None
    if mode == "full":
        heavy = {key: np.asarray(namespace[key]) for key in _RESUME_HEAVY_ARRAY_KEYS}
        heavy.update({key: namespace[key] for key in _RESUME_HEAVY_OBJECT_KEYS})
    return build_residual_iter_resume_state_payload(
        resume_state_mode=mode,
        base_kwargs=base_kwargs,
        heavy_payload=heavy,
    )


def final_free_boundary_residual_reports_from_namespace(
    namespace: Mapping[str, Any],
    *,
    nestor_external_only_step_func: Callable[..., Any],
    residual_fsq_from_norms_func: Callable[..., Any],
    device_get_floats_func: Callable[..., tuple[float, ...]],
) -> dict[str, Any]:
    """Return final accepted-state free-boundary residual report fields."""

    ns = namespace
    clock = time.perf_counter
    timing_stats, timing_enabled = ns["timing_stats"], bool(ns["timing_enabled"])
    final_bsqvac_half_current = ns["freeb_bsqvac_half_current"]
    report = {
        "final_fsqr_report": float(ns["fsqr_f"]),
        "final_fsqz_report": float(ns["fsqz_f"]),
        "final_fsql_report": float(ns["fsql_f"]),
        "final_residual_recomputed": False,
        "final_nestor_model": str(ns["freeb_last_model"]),
        "final_nestor_diagnostics": dict(ns["freeb_last_diagnostics"]),
        "final_nestor_recompute_attempted": False,
        "final_nestor_recompute_failed": False,
        "final_nestor_sample_time_s": 0.0,
        "final_nestor_solve_time_s": 0.0,
    }
    report["final_vacuum_stub"] = not bool(
        str(report["final_nestor_model"]).strip() and str(report["final_nestor_model"]) != "none"
    )
    if bool(ns["free_boundary_enabled"] and ns["freeb_couple_edge"]) and not report["final_vacuum_stub"]:
        report["final_nestor_recompute_attempted"] = True
        start = clock() if timing_enabled else None
        try:
            nestor_final, _runtime = nestor_external_only_step_func(
                state=ns["state"], static=ns["static"], ivac=1, ivacskip=0, iter_idx=None,
                runtime=ns["freeb_nestor_runtime"], extcur=tuple(getattr(ns["static"], "free_boundary_extcur", ()) or ()),
                plascur=float(ns["freeb_plascur"]),
                external_field_provider_kind=ns["external_field_provider_kind"],
                external_field_provider_static=ns["external_field_provider_static"],
                external_field_provider_params=ns["external_field_provider_params"],
            )
            report["final_nestor_sample_time_s"] = float(getattr(nestor_final, "sample_time_s", 0.0))
            report["final_nestor_solve_time_s"] = float(getattr(nestor_final, "solve_time_s", 0.0))
            report["final_nestor_model"] = str(getattr(nestor_final, "model", report["final_nestor_model"]))
            diag_final = getattr(nestor_final, "diagnostics", None)
            if isinstance(diag_final, dict):
                report["final_nestor_diagnostics"] = dict(diag_final)
            bsqvac_edge_final = np.asarray(nestor_final.vac_total.bsqvac, dtype=float)
            if (
                bsqvac_edge_final.ndim == 2 and int(bsqvac_edge_final.shape[1]) == 1 and int(getattr(ns["static"].cfg, "nzeta", 1)) > 1
            ):
                bsqvac_edge_final = np.repeat(bsqvac_edge_final, int(ns["static"].cfg.nzeta), axis=1)
            final_bsqvac_half_current = bsqvac_edge_final
            report["final_vacuum_stub"] = False
        except Exception:
            report["final_nestor_recompute_failed"] = True
            final_bsqvac_half_current = ns["freeb_bsqvac_half_current"]
        finally:
            if timing_enabled and start is not None:
                timing_stats["finalize_nestor_recompute"] += clock() - float(start)
    if bool(ns["free_boundary_enabled"]) and final_bsqvac_half_current is not None:
        start = clock() if timing_enabled else None
        try:
            _, _, gcr2_final, gcz2_final, gcl2_final, _, _, norms_final = ns["_compute_forces_iter"](
                ns["state"], include_edge=bool(ns["include_edge"]), include_edge_residual=True,
                zero_m1=ns["zero_m1"], freeb_bsqvac_half=final_bsqvac_half_current,
                constraint_precond_diag=ns["constraint_precond_diag"], constraint_tcon=ns["constraint_tcon_override"],
                constraint_precond_active=ns["constraint_precond_active"],
                constraint_tcon_active=ns["constraint_tcon_active"],
                iter2=ns["last_iter2"],
            )
            fsqr_final, fsqz_final, fsql_final = residual_fsq_from_norms_func(
                norms_final, gcr2=gcr2_final, gcz2=gcz2_final, gcl2=gcl2_final,
            )
            get_start = clock() if timing_enabled else None
            report["final_fsqr_report"], report["final_fsqz_report"], report["final_fsql_report"] = device_get_floats_func(fsqr_final, fsqz_final, fsql_final)
            if timing_enabled and get_start is not None:
                timing_stats["finalize_residual_device_get"] += clock() - float(get_start)
            report["final_residual_recomputed"] = True
        except Exception:
            report["final_fsqr_report"] = float(ns["fsqr_f"])
            report["final_fsqz_report"] = float(ns["fsqz_f"])
            report["final_fsql_report"] = float(ns["fsql_f"])
        finally:
            if timing_enabled and start is not None:
                timing_stats["finalize_residual_recompute"] += clock() - float(start)
    return report


def finalize_residual_iter_result(
    *,
    result_type: type,
    state: Any,
    w_history: Any,
    fsqr2_history: Any,
    fsqz2_history: Any,
    fsql2_history: Any,
    grad_rms_history: Any,
    step_history: Any,
    diagnostics: dict[str, Any],
    attach_free_boundary_diagnostics: Callable[[Any], Any],
    return_final_force_payload: bool,
    converged: bool,
    final_force_payload: Any,
) -> Any:
    """Construct the final residual-iteration result object."""

    result = attach_free_boundary_diagnostics(
        result_type(
            state=state,
            n_iter=len(w_history) - 1,
            w_history=np.asarray(w_history, dtype=float),
            fsqr2_history=np.asarray(fsqr2_history, dtype=float),
            fsqz2_history=np.asarray(fsqz2_history, dtype=float),
            fsql2_history=np.asarray(fsql2_history, dtype=float),
            grad_rms_history=np.asarray(grad_rms_history, dtype=float),
            step_history=np.asarray(step_history, dtype=float),
            diagnostics=diagnostics,
        )
    )
    if bool(return_final_force_payload) and bool(converged):
        try:
            object.__setattr__(result, "_final_force_payload", final_force_payload)
        except Exception:
            pass
    return result


def _empty_history_result(
    *,
    result_type: type,
    state: Any,
    n_iter: int,
    diagnostics: dict[str, Any],
    empty_history: Any | None = None,
) -> Any:
    empty = np.zeros((0,), dtype=float) if empty_history is None else empty_history
    return result_type(
        state=state,
        n_iter=int(n_iter),
        diagnostics=diagnostics,
        **dict.fromkeys(_EMPTY_HISTORY_KEYS, empty),
    )


def precompile_only_residual_iter_result(*, result_type: type, state: Any) -> Any:
    return _empty_history_result(result_type=result_type, state=state, n_iter=0, diagnostics={"precompile_only": True})


def vmec2000_state_only_scan_result(
    *,
    result_type: type,
    carry_final: Any,
    empty_history: Any,
    max_iter: int,
    diagnostics: dict[str, Any],
    attach_free_boundary_diagnostics: Callable[[Any], Any],
) -> Any:
    """Construct a VMEC2000 state-only scan result with empty histories."""

    return attach_free_boundary_diagnostics(
        _empty_history_result(result_type=result_type, state=carry_final.state, n_iter=max_iter, diagnostics=diagnostics, empty_history=empty_history)
    )


def vmec2000_traced_scan_result(
    *,
    result_type: type,
    carry_final: Any,
    empty_history: Any,
    max_iter: int,
    resume_state: dict[str, Any],
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    attach_free_boundary_diagnostics: Callable[[Any], Any],
    traced_diagnostics_func: Callable[..., dict[str, Any]],
) -> Any:
    """Construct a traced VMEC2000 scan result with resume diagnostics."""

    diagnostics = traced_diagnostics_func(
        resume_state=resume_state,
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
    )
    return attach_free_boundary_diagnostics(
        _empty_history_result(result_type=result_type, state=carry_final.state, n_iter=max_iter, diagnostics=diagnostics, empty_history=empty_history)
    )
