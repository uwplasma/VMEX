"""JIT payload facade for residual-iteration preconditioned updates."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import has_jax
from vmec_jax.solvers.fixed_boundary.preconditioning import payload as _payload
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _jax_available(has_jax_func=None) -> bool:
    """Evaluate the caller-provided JAX availability hook when supplied."""

    return bool((has_jax if has_jax_func is None else has_jax_func)())


_STRICT_UPDATE_STEP_JIT_CACHE = _payload.STRICT_UPDATE_STEP_JIT_CACHE
_PRECOND_OUTPUT_SCALE_JIT_CACHE = _payload.PRECOND_OUTPUT_SCALE_JIT_CACHE
_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE = _payload.PRECOND_OUTPUT_PAYLOAD_JIT_CACHE
_PRECOND_APPLY_PAYLOAD_JIT_CACHE = _payload.PRECOND_APPLY_PAYLOAD_JIT_CACHE
_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE = _payload.ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE


def _strict_update_step_jit(
    static,
    *,
    limit_update_rms: bool,
    need_update_rms: bool,
    divide_by_scalxc_for_update: bool,
    enforce_edge: bool = True,
    has_jax_func=None,
):
    """Return the cached strict-update JIT, or ``None`` when JAX is unavailable."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.strict_update_step_jit(
        static,
        limit_update_rms=limit_update_rms,
        need_update_rms=need_update_rms,
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
        enforce_edge=enforce_edge,
    )


def _preconditioner_output_scaling_jit(*, apply_lambda_update_scale: bool, has_jax_func=None):
    """Return the cached output-scaling JIT, or ``None`` when JAX is unavailable."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_output_scaling_jit(apply_lambda_update_scale=apply_lambda_update_scale)


def _preconditioner_output_payload_jit(
    *,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    has_jax_func=None,
):
    """Return the cached residual-output payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_output_payload_jit(
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        scaling_func=lambda **kwargs: _preconditioner_output_scaling_jit(
            has_jax_func=has_jax_func,
            **kwargs,
        ),
    )


def _preconditioner_apply_payload_jit(
    *,
    jmax: int,
    lthreed: bool,
    lasym: bool,
    use_precomputed: bool,
    use_lax_tridi: bool,
    has_lax_t: bool,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_frcs: bool,
    has_fzcc: bool,
    has_fzss: bool,
    has_flcs: bool,
    has_flcc: bool,
    has_flss: bool,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool,
    has_jax_func=None,
):
    """Return the cached preconditioner-apply payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_apply_payload_jit(
        jmax=jmax,
        lthreed=lthreed,
        lasym=lasym,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
        has_lax_t=has_lax_t,
        has_frss=has_frss,
        has_fzcs=has_fzcs,
        has_frsc=has_frsc,
        has_frcs=has_frcs,
        has_fzcc=has_fzcc,
        has_fzss=has_fzss,
        has_flcs=has_flcs,
        has_flcc=has_flcc,
        has_flss=has_flss,
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        include_control_ptau=include_control_ptau,
    )


def _accepted_control_payload_jit(*, has_jax_func=None):
    """Return the cached accepted-controller payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.accepted_control_payload_jit()


def _preconditioner_apply_payload_fused(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale_j,
    f_norm1,
    delta_s,
    s,
    use_precomputed: bool | None,
    use_lax_tridi: bool | None,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool = False,
    control_ptau_arrays: tuple[Any, ...] | None = None,
    control_ptau_pshalf: Any = None,
    control_ptau_ohs: Any = None,
):
    """Apply the fused preconditioner payload using the shared JIT factory."""

    return _payload.preconditioner_apply_payload_fused(
        frzl_in=frzl_in,
        mats=mats,
        jmax=jmax,
        cfg=cfg,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        f_norm1=f_norm1,
        delta_s=delta_s,
        s=s,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        include_control_ptau=include_control_ptau,
        control_ptau_arrays=control_ptau_arrays,
        control_ptau_pshalf=control_ptau_pshalf,
        control_ptau_ohs=control_ptau_ohs,
        apply_payload_jit_func=_preconditioner_apply_payload_jit,
    )


_ptau_compute_jit = _payload.ptau_compute_jit


__all__ = [
    "_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE",
    "_PRECOND_APPLY_PAYLOAD_JIT_CACHE",
    "_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE",
    "_PRECOND_OUTPUT_SCALE_JIT_CACHE",
    "_STRICT_UPDATE_STEP_JIT_CACHE",
    "_accepted_control_payload_jit",
    "_preconditioner_apply_payload_fused",
    "_preconditioner_apply_payload_jit",
    "_preconditioner_output_payload_jit",
    "_preconditioner_output_scaling_jit",
    "_ptau_compute_jit",
    "_strict_update_step_jit",
]
