"""Replay and tangent-cache policy helpers for fixed-boundary exact callbacks."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import fields, is_dataclass, replace
import os

import numpy as np


def _env_flag(name: str) -> bool | None:
    """Return a boolean environment override without mutating optimizer state."""

    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _column_chunk_override_metadata() -> tuple[str, int | None]:
    """Return the generic replay-column chunk override for diagnostics only."""

    value = os.getenv("VMEC_JAX_REPLAY_COLUMN_CHUNK")
    if value is None:
        return "unset", None
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return "auto", None
    if text in ("0", "none", "off", "false", "no"):
        return "disabled", None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return "malformed", None
    if parsed <= 0:
        return "disabled", None
    return "active", int(parsed)


def _positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment setting or a safe default."""

    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def optimizer_backend_name(solver_device_name: str | None) -> str:
    """Return the active optimizer backend name without changing device policy."""

    backend = str(solver_device_name or "").strip().lower()
    if backend:
        return backend
    try:
        from ..._compat import jax as _jax

        return str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
    except Exception:
        return "cpu"


def exact_tape_backend_name(optimizer) -> str:
    """Return the backend name used for exact-tape optimization policy."""

    backend = str(getattr(optimizer, "_solver_device_name", None) or "").strip().lower()
    if backend:
        return backend
    return optimizer_backend_name(None)


def env_bool_override(name: str) -> bool | None:
    """Return a boolean optimizer environment override, or ``None`` if unset."""

    value = os.getenv(str(name), "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def resolve_solver_device(solver_device: str | None) -> str | None:
    """Normalize an optimizer device request without wrapping the active backend."""

    name = "auto" if solver_device is None else str(solver_device).strip().lower()
    if name in ("", "none", "auto", "default"):
        return None
    try:
        from ... import _compat as _compat_module

        jax_module = _compat_module.jax
        current_backend = str(jax_module.default_backend()).strip().lower() if jax_module is not None else ""
    except Exception:
        current_backend = ""
    aliases = {
        "gpu": {"gpu", "cuda", "rocm", "tpu"},
        "cuda": {"gpu", "cuda"},
        "rocm": {"gpu", "rocm"},
        "tpu": {"tpu"},
        "cpu": {"cpu"},
    }
    if current_backend in aliases.get(name, {name}):
        return None
    return name


def solver_device_context(optimizer):
    """Return the JAX default-device context for optimizer callbacks."""

    if optimizer._solver_device_name is None:
        return nullcontext()
    try:
        from ... import _compat as _compat_module

        jax_module = _compat_module.jax
        if jax_module is None:
            return nullcontext()
        devices = jax_module.devices(optimizer._solver_device_name)
        if not devices:
            return nullcontext()
        return jax_module.default_device(devices[0])
    except Exception:
        return nullcontext()


def move_to_solver_device(optimizer, value):
    """Move optimizer pytrees to the requested JAX callback device."""

    if optimizer._solver_device_name is None:
        return value
    try:
        from ... import _compat as _compat_module

        jax_module = _compat_module.jax
        if jax_module is None:
            return value
        device = jax_module.devices(optimizer._solver_device_name)[0]
        jax_array_type = jax_module.Array
    except Exception:
        return value

    def _move(obj):
        if obj is None or isinstance(obj, (str, bytes, int, float, complex, bool)):
            return obj
        if isinstance(obj, (np.ndarray, jax_array_type)):
            return jax_module.device_put(obj, device)
        if is_dataclass(obj) and not isinstance(obj, type):
            return replace(
                obj,
                **{field.name: _move(getattr(obj, field.name)) for field in fields(obj)},
            )
        if isinstance(obj, dict):
            return {key: _move(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [_move(item) for item in obj]
        if isinstance(obj, tuple):
            moved = tuple(_move(item) for item in obj)
            if hasattr(obj, "_fields"):
                return type(obj)(*moved)
            return moved
        return obj

    return _move(value)


def run_in_solver_device_context(optimizer, fn, *args, **kwargs):
    """Run a callback in the requested solver-device and FFT policy contexts."""

    if optimizer._solver_device_name is None or optimizer._inside_solver_device_context:
        return fn(*args, **kwargs)
    from ...kernels.tomnsp import tomnsps_fft_policy_override

    backend_name = str(optimizer._solver_device_name).strip().lower()
    tomnsps_fft_override = (
        backend_name in ("gpu", "cuda", "rocm", "tpu")
        if os.getenv("VMEC_JAX_TOMNSPS_FFT") is None
        else None
    )
    with ExitStack() as stack:
        stack.enter_context(optimizer._solver_device_context())
        stack.enter_context(tomnsps_fft_policy_override(tomnsps_fft_override))
        optimizer._inside_solver_device_context = True
        try:
            return fn(*args, **kwargs)
        finally:
            optimizer._inside_solver_device_context = False


def gpu_like_exact_tape_backend(optimizer) -> bool:
    """Return whether exact-tape callbacks target a GPU-like backend."""

    return exact_tape_backend_name(optimizer) in ("gpu", "cuda", "rocm", "tpu", "metal")


def resolve_optimizer_method(optimizer, method: str, scipy_lsmr_maxiter: int | None) -> tuple[str, int | None, str | None]:
    """Resolve public optimizer method aliases and the conservative auto policy."""

    method_key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "matrix_free": "scipy_matrix_free",
        "scipy_mf": "scipy_matrix_free",
        "trf": "scipy",
    }
    method_key = aliases.get(method_key, method_key)
    scalar_auto_requested = method_key in ("auto_scalar", "auto_adjoint", "adaptive_scalar", "adaptive_adjoint")
    if method_key not in ("auto", "adaptive") and not scalar_auto_requested:
        return method_key, scipy_lsmr_maxiter, None

    if optimizer._has_stellarator_asymmetric_configuration():
        prefix = "auto_scalar" if scalar_auto_requested else "auto"
        return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-lasym"

    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    helicity_m = None if optimizer._helicity_m is None else int(optimizer._helicity_m)
    helicity_n = None if optimizer._helicity_n is None else int(optimizer._helicity_n)
    if optimizer._spec_max_mode() >= 3 and optimizer._objective_family in ("qs", "qi"):
        if scalar_auto_requested:
            suffix = f"{backend}-" if backend in ("gpu", "cuda", "rocm", "tpu", "metal") else ""
            return "scalar_trust", scipy_lsmr_maxiter, f"auto_scalar:{suffix}high-mode-scalar-trust"
        if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
            return "scipy", scipy_lsmr_maxiter, f"auto:dense-preserves-{backend}"
        if helicity_m == 1 and helicity_n == 0:
            family = "qa"
        elif helicity_m == 0 and helicity_n not in (None, 0):
            family = "qp"
        elif helicity_m == 1 and helicity_n not in (None, 0):
            family = "qh"
        else:
            family = str(optimizer._objective_family or "qs")
        return "scipy", scipy_lsmr_maxiter, f"auto:{family}-dense-default"

    if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
        prefix = "auto_scalar" if scalar_auto_requested else "auto"
        return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-preserves-{backend}"
    prefix = "auto_scalar" if scalar_auto_requested else "auto"
    return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-default"


def select_exact_path(optimizer) -> str:
    """Choose the accepted-point differentiation path for exact callbacks."""

    requested = getattr(optimizer, "_exact_path_request", None)
    if requested in ("scan", "tape"):
        return str(requested)
    forced = os.getenv("VMEC_JAX_OPT_EXACT_PATH", "").strip().lower()
    if forced in ("scan", "tape"):
        return forced
    return "tape"


def use_precomputed_tridi_for_exact_tape(optimizer) -> bool | None:
    """Return whether accepted exact-tape solves should precompute tridiagonal factors."""

    forced = optimizer._env_bool_override("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE")
    if forced is not None:
        return forced
    backend = optimizer._exact_tape_backend_name()
    if backend not in ("gpu", "cuda", "tpu", "rocm"):
        return None
    try:
        max_dofs = int(os.getenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "48"))
    except ValueError:
        max_dofs = 48
    if max_dofs < 0:
        return False
    return True if len(optimizer._specs) <= max_dofs else None


def trial_solver_scan_policy(optimizer, *, max_nfev: int | None = None) -> tuple[bool, str, str]:
    """Return trial-solve scan policy and provenance for exact optimization."""

    forced = os.getenv("VMEC_JAX_OPT_TRIAL_SCAN", "").strip().lower()
    if forced in ("1", "true", "yes", "on", "scan"):
        return True, "environment", "VMEC_JAX_OPT_TRIAL_SCAN=scan"
    if forced in ("0", "false", "no", "off", "loop", "none"):
        return False, "environment", "VMEC_JAX_OPT_TRIAL_SCAN=loop"
    if max_nfev is not None and int(max_nfev) <= 2:
        return False, "stage_budget", "max_nfev<=2"
    family = str(getattr(optimizer, "_objective_family", "")).strip().lower()
    if family == "qi":
        return False, "objective_family", "qi_trial_loop_default"
    try:
        helicity_m = None if optimizer._helicity_m is None else int(optimizer._helicity_m)
        helicity_n = None if optimizer._helicity_n is None else int(optimizer._helicity_n)
    except Exception:
        helicity_m = None
        helicity_n = None
    if family == "qs" and helicity_m == 0 and helicity_n not in (None, 0):
        return False, "objective_family", "quasi_poloidal_trial_loop_default"
    backend = optimizer._exact_tape_backend_name()
    use_scan = backend in ("gpu", "cuda", "tpu", "rocm")
    detail = f"{backend}_trial_{'scan' if use_scan else 'loop'}_default"
    return use_scan, "backend_default", detail


def use_scan_for_trial_solves(optimizer, *, max_nfev: int | None = None) -> bool:
    """Return whether trial residual solves should use the scan loop."""

    return bool(optimizer._trial_solver_scan_policy(max_nfev=max_nfev)[0])


def ensure_solver_policy_defaults(optimizer) -> None:
    """Install minimal solve-policy dictionaries for lightweight optimizer stubs."""

    if not isinstance(getattr(optimizer, "_exact_solver_kwargs", None), dict):
        optimizer._exact_solver_kwargs = {"use_scan": False, "light_history": True, "resume_state_mode": "none"}
    if not isinstance(getattr(optimizer, "_trial_solver_kwargs", None), dict):
        optimizer._trial_solver_kwargs = {"use_scan": False, "light_history": True, "resume_state_mode": "none"}
    if not hasattr(optimizer, "_trial_solver_scan_policy_source"):
        optimizer._trial_solver_scan_policy_source = "stub_default"
    if not hasattr(optimizer, "_trial_solver_scan_policy_detail"):
        optimizer._trial_solver_scan_policy_detail = "optimizer_run_default"


def _backend_is_accelerator(backend: str) -> bool:
    """Return true for JAX accelerator backends that favor coarse replay buckets."""

    normalized = str(backend).strip().lower()
    return normalized in {"gpu", "cuda", "rocm"} or normalized.startswith(("gpu:", "cuda:", "rocm:"))


def dynamic_replay_bucket_for_backend(backend: str) -> int:
    """Return the dynamic replay bucket used for optimizer provenance.

    The replay implementation defaults to coarser buckets on accelerators and
    smaller buckets on CPU.  This helper mirrors that policy while respecting an
    explicit optimizer device selection, so diagnostics remain meaningful even
    when the outer Python process default backend differs from the callback
    backend.
    """

    default = 128 if _backend_is_accelerator(backend) else 32
    return _positive_int_env("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", default)


def dynamic_replay_mode_from_env() -> str:
    """Return the configured dynamic replay linearization strategy."""

    mode = os.getenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "basepoint").strip().lower()
    if mode in ("whole_scan", "scan", "full_scan"):
        return "whole_scan"
    return "basepoint"


def _optimizer_lasym(optimizer) -> bool:
    """Return LASYM status for real optimizers and lightweight test doubles."""

    static = getattr(optimizer, "_static", None)
    return bool(getattr(getattr(static, "cfg", None), "lasym", False))


def _optimizer_has_stellarator_asymmetry(optimizer) -> bool:
    """Return whether replay policy should use LASYM/asymmetric safeguards.

    The production optimizer exposes ``_has_stellarator_asymmetric_configuration``.
    Some unit tests intentionally use minimal objects to exercise metadata paths
    without constructing a full VMEC static state; for those, fall back to the
    parameter-spec kinds and static ``cfg.lasym`` flag.
    """

    checker = getattr(optimizer, "_has_stellarator_asymmetric_configuration", None)
    if callable(checker):
        try:
            return bool(checker())
        except AttributeError:
            pass
    specs = getattr(optimizer, "_specs", ())
    if any(str(getattr(spec, "kind", "")).lower() in ("rs", "zc") for spec in specs):
        return True
    return _optimizer_lasym(optimizer)


def lasym_replay_column_chunk(optimizer, n_params: int) -> int | None:
    """Replay-column chunk heuristic for dense exact Jacobians."""

    env_override = os.environ.get("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK")
    if env_override is not None:
        from ...discrete_adjoint import _replay_column_chunk_override

        handled, requested = _replay_column_chunk_override(env_override)
        if handled:
            return requested
    if os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK") is not None:
        return None
    backend_name = None
    if optimizer._solver_device_name is not None:
        backend_name = str(optimizer._solver_device_name).lower()
    else:
        try:
            from ..._compat import jax as _jax

            backend_name = str(_jax.default_backend()).lower()
        except Exception:
            backend_name = None
    if backend_name in ("gpu", "cuda", "rocm"):
        if int(n_params) < 24:
            return None
        if _optimizer_lasym(optimizer):
            # LASYM doubles the boundary columns and remains more memory
            # sensitive on GPU; keep the older conservative replay chunks.
            return 8
        if int(n_params) <= 64:
            return int(n_params)
        if int(n_params) <= 128:
            return max(24, int(n_params) // 2)
        return 64
    if backend_name == "tpu":
        return None
    if not _optimizer_lasym(optimizer):
        return None
    if int(n_params) >= 64:
        return 8
    if int(n_params) >= 32:
        return 4
    return None


def precompute_linear_operator_initial_tangents_enabled(optimizer, n_params: int) -> bool:
    """Whether matrix-free operators should cache initial-state tangent columns."""

    if int(n_params) <= 0:
        return False
    flag = os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENTS")
    if flag is not None:
        return flag.strip().lower() not in ("", "0", "false", "no", "off")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
        return False
    if _optimizer_has_stellarator_asymmetry(optimizer):
        return False
    min_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MIN_DOFS", "64"))
    max_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MAX_DOFS", "128"))
    return min_dofs <= int(n_params) <= max_dofs


def scalar_gradient_initial_tangents_enabled(optimizer, n_params: int) -> bool:
    """Whether scalar-adjoint gradients should project cached initial tangents."""

    if int(n_params) <= 0:
        return False
    flag = os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENTS")
    if flag is not None:
        return flag.strip().lower() not in ("", "0", "false", "no", "off")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend not in ("cpu", "gpu", "cuda", "rocm", "tpu", "metal"):
        return False
    if _optimizer_has_stellarator_asymmetry(optimizer):
        return False
    min_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MIN_DOFS", "24"))
    default_max_dofs = "128" if backend == "cpu" else "256"
    max_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MAX_DOFS", default_max_dofs))
    return min_dofs <= int(n_params) <= max_dofs


def projected_replay_residuals_enabled(optimizer, n_params: int | None = None) -> bool:
    """Whether dense Jacobians should project replayed tangents without an intermediate sync."""

    flag = os.getenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes", "on")
    solver_device_name = getattr(optimizer, "_solver_device_name", None)
    if solver_device_name is None:
        try:
            from ..._compat import jax as _jax

            backend = str(_jax.default_backend()).strip().lower() if _jax is not None else ""
        except Exception:
            return False
    else:
        backend = optimizer_backend_name(solver_device_name)
    if backend not in ("cpu", "gpu", "cuda", "rocm"):
        return False
    if n_params is None:
        return False
    static = getattr(optimizer, "_static", None)
    if bool(getattr(getattr(static, "cfg", None), "lasym", False)):
        return False
    # Projected replay avoids materializing full state tangent columns on the
    # host before residual projection.  Recent QA/QH/QP budget probes show this
    # pays off on CPU as well as accelerators once the exact callback has a
    # multi-column dense Jacobian.
    min_params = 8 if backend == "cpu" else 48
    return int(n_params) >= min_params


def fused_projected_replay_enabled() -> bool:
    """Whether projected replay should fuse replay and residual projection when possible."""

    flag = os.getenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "").strip().lower()
    if flag:
        return flag in ("1", "true", "yes", "on")
    return False


def chunked_projected_replay_projection_enabled(
    optimizer,
    column_chunk: int | None,
    n_params: int,
) -> bool:
    """Whether to project residual tangents immediately after each replay chunk."""

    if column_chunk is None:
        return False
    if int(n_params) <= int(column_chunk):
        return False
    flag = os.getenv("VMEC_JAX_OPT_CHUNKED_PROJECTED_REPLAY_PROJECTION", "").strip().lower()
    if flag:
        return flag in ("1", "true", "yes", "on")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend not in ("gpu", "cuda", "rocm"):
        return False
    if _optimizer_lasym(optimizer):
        return False
    return True


def projected_replay_projection_column_chunk(
    optimizer,
    column_chunk: int | None,
    n_params: int,
) -> int | None:
    """Return the chunk size for per-chunk replay projection, if active.

    ``VMEC_JAX_REPLAY_COLUMN_CHUNK`` is a generic replay knob.  The standard
    replay path consumes it internally, but the projected path needs the same
    value before replay starts so it can project residual tangents per chunk
    instead of materializing all final-state tangent columns first.
    """

    active_chunk = None if column_chunk is None else int(column_chunk)
    if active_chunk is None:
        policy, requested = _column_chunk_override_metadata()
        if policy == "active" and requested is not None:
            active_chunk = int(requested)
    if active_chunk is None or int(n_params) <= int(active_chunk):
        return None
    if not chunked_projected_replay_projection_enabled(optimizer, active_chunk, n_params):
        return None
    return int(active_chunk)


def exact_replay_policy_metadata(optimizer, n_params: int | None = None) -> dict[str, object]:
    """Summarize exact-callback replay policy choices without changing them.

    The exact callback has several mathematically equivalent replay routes:
    dense tangent replay, projected replay, chunked projection, scalar-adjoint
    gradients, and accelerator-oriented JVP-only tapes.  This diagnostic record
    exposes the chosen route and its controlling backend/shape conditions so
    optimization histories and benchmark summaries can classify performance
    regressions without storing large Jacobians or mutating profile counters.
    """

    n_params_int = None if n_params is None else int(n_params)
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    static = getattr(optimizer, "_static", None)
    lasym = _optimizer_lasym(optimizer)
    accelerator = backend in ("gpu", "cuda", "rocm", "tpu", "metal")
    gpu_like = backend in ("gpu", "cuda", "rocm", "tpu", "metal")

    jvp_only_override = _env_flag("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE")
    basepoint_override = _env_flag("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
    chunk_override_policy, chunk_override_value = _column_chunk_override_metadata()
    column_chunk = None
    projected = False
    chunked_projection = False
    projection_column_chunk = None
    scalar_initial_tangents = False
    linear_operator_initial_tangents = False
    if n_params_int is not None:
        column_chunk = lasym_replay_column_chunk(optimizer, n_params_int)
        projected = projected_replay_residuals_enabled(optimizer, n_params_int)
        projection_column_chunk = projected_replay_projection_column_chunk(
            optimizer,
            column_chunk,
            n_params_int,
        )
        chunked_projection = projection_column_chunk is not None
        scalar_initial_tangents = scalar_gradient_initial_tangents_enabled(optimizer, n_params_int)
        linear_operator_initial_tangents = precompute_linear_operator_initial_tangents_enabled(
            optimizer,
            n_params_int,
        )

    return {
        "backend": backend,
        "n_parameters": n_params_int,
        "lasym": lasym,
        "projected_replay": bool(projected),
        "projected_replay_reason": "enabled" if projected else "disabled_or_below_threshold",
        "fused_projected_replay": fused_projected_replay_enabled(),
        "column_chunk": None if column_chunk is None else int(column_chunk),
        "projected_replay_projection_column_chunk": projection_column_chunk,
        "requested_replay_column_chunk": chunk_override_value,
        "requested_replay_column_chunk_policy": chunk_override_policy,
        "chunked_projected_replay_projection": bool(chunked_projection),
        "dynamic_replay_mode": dynamic_replay_mode_from_env(),
        "dynamic_replay_bucket": dynamic_replay_bucket_for_backend(backend),
        "scalar_gradient_initial_tangents": bool(scalar_initial_tangents),
        "linear_operator_initial_tangents": bool(linear_operator_initial_tangents),
        "jvp_only_exact_tape": bool(gpu_like if jvp_only_override is None else jvp_only_override),
        "jvp_only_basepoint_carries": bool(gpu_like if basepoint_override is None else basepoint_override),
        "accelerator_backend": bool(accelerator),
    }
