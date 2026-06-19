"""Opt-in JAX NESTOR operator cache used by free-boundary validation paths."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from .types import ExternalBoundarySample, NestorPoissonCache, NestorVmecLikeCache

try:  # pragma: no cover - optional dependency
    from scipy.linalg import lu_factor as _SCIPY_LU_FACTOR  # type: ignore
    from scipy.linalg import lu_solve as _SCIPY_LU_SOLVE  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional at runtime
    _SCIPY_LU_FACTOR = None
    _SCIPY_LU_SOLVE = None


JAX_NESTOR_BASIS_KEYS = (
    "lasym",
    "mf",
    "nf",
    "mn0",
    "mnpd",
    "mnpd2",
    "nu_full",
    "nuv3",
    "nuv_full",
    "onp",
    "cmns",
    "cos_phase",
    "cosmni",
    "imirr",
    "imirr_full",
    "n_raw",
    "sin_phase",
    "sinmni",
    "theta",
    "wint",
    "xmpot",
    "zeta",
)

FREEB_JAX_NESTOR_OPERATOR_FN_CACHE: dict[tuple[Any, ...], Any] = {}


def dense_lu_factor(matrix: np.ndarray) -> Any | None:
    if _SCIPY_LU_FACTOR is None:
        return None
    try:
        return _SCIPY_LU_FACTOR(np.asarray(matrix, dtype=float))
    except Exception:
        return None


def dense_lu_solve(lu_fac: Any | None, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rhs_arr = np.asarray(rhs, dtype=float)
    if lu_fac is not None and _SCIPY_LU_SOLVE is not None:
        try:
            return np.asarray(_SCIPY_LU_SOLVE(lu_fac, rhs_arr), dtype=float)
        except Exception:
            pass
    return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=float), rhs_arr), dtype=float)


def build_vmec_cmns(*, mf: int, nf: int, onp: float) -> np.ndarray:
    """VMEC precal.f cmns(l,m,n) coefficients for the n>=0 block."""

    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lmax = mf + nf
    cmn = np.zeros((lmax + 1, mf + 1, nf + 1), dtype=float)
    for m in range(mf + 1):
        for n in range(nf + 1):
            jmn = m + n
            imn = m - n
            kmn = abs(imn)
            smn = (jmn + kmn) // 2
            f1 = 1.0
            f2 = 1.0
            f3 = 1.0
            for i in range(1, kmn + 1):
                f1 *= float(smn + 1 - i)
                f2 *= float(i)
            for l in range(kmn, jmn + 1, 2):
                cmn[l, m, n] = (f1 / (f2 * f3)) * ((-1.0) ** ((l - imn) // 2))
                f1 = f1 * 0.25 * float((jmn + l + 2) * (jmn - l))
                f2 = f2 * 0.5 * float(l + 2 + kmn)
                f3 = f3 * 0.5 * float(l + 2 - kmn)

    alp = 2.0 * np.pi * float(onp)
    cmns = np.zeros_like(cmn)
    if mf >= 1 and nf >= 1:
        cmns[:, 1 : mf + 1, 1 : nf + 1] = (
            0.5
            * alp
            * (
                cmn[:, 1 : mf + 1, 1 : nf + 1]
                + cmn[:, :mf, 1 : nf + 1]
                + cmn[:, 1 : mf + 1, :nf]
                + cmn[:, :mf, :nf]
            )
        )
    if mf >= 1:
        cmns[:, 1 : mf + 1, 0] = 0.5 * alp * (cmn[:, 1 : mf + 1, 0] + cmn[:, :mf, 0])
    if nf >= 1:
        cmns[:, 0, 1 : nf + 1] = 0.5 * alp * (cmn[:, 0, 1 : nf + 1] + cmn[:, 0, :nf])
    cmns[:, 0, 0] = 0.5 * alp * (cmn[:, 0, 0] + cmn[:, 0, 0])
    return cmns


def build_vmec_mode_basis(
    *,
    ntheta: int,
    nzeta: int,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint: np.ndarray,
) -> dict[str, Any]:
    """Build VMEC-like mode tables and weighted sin/cos basis arrays."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    nfp = max(1, int(nfp))
    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lasym = bool(lasym)

    pi2 = 2.0 * np.pi
    if lasym:
        nu_full = int(ntheta)
    else:
        nu_full = max(int(ntheta), 2 * (int(ntheta) - 1))
    theta = (pi2 / float(max(1, nu_full))) * np.arange(ntheta, dtype=float)
    zeta = (pi2 / float(max(1, nzeta))) * np.arange(nzeta, dtype=float)
    th_grid = np.broadcast_to(theta[:, None], (ntheta, nzeta))
    ze_grid = np.broadcast_to(zeta[None, :], (ntheta, nzeta))
    th = th_grid.reshape(-1)
    ze = ze_grid.reshape(-1)

    w = np.asarray(wint, dtype=float).reshape(-1)
    if w.size != th.size:
        w = np.full((th.size,), 1.0 / float(max(1, th.size)), dtype=float)

    mvals: list[int] = []
    nvals: list[int] = []
    for n in range(-nf, nf + 1):
        for m in range(0, mf + 1):
            mvals.append(int(m))
            nvals.append(int(n))
    xmpot = np.asarray(mvals, dtype=np.int64)
    n_raw = np.asarray(nvals, dtype=np.int64)
    xnpot = np.asarray(n_raw * nfp, dtype=np.int64)
    mnpd = int(xmpot.size)
    mnpd2 = int(mnpd * (2 if lasym else 1))

    phase = (xmpot[None, :] * th[:, None]) - (n_raw[None, :] * ze[:, None])
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    weight = ((pi2 * pi2) * w)[:, None]
    sinmni = weight * sin_phase
    cosmni = weight * cos_phase

    idx = np.arange(th.size, dtype=np.int64)
    lt = idx // max(1, nzeta)
    lz = idx % max(1, nzeta)
    if lasym or (nu_full == ntheta):
        lt_m = (ntheta - lt) % max(1, ntheta)
    else:
        lt_m_full = (nu_full - lt) % max(1, nu_full)
        lt_m = np.minimum(lt_m_full, (nu_full - lt_m_full) % max(1, nu_full))
    lz_m = (nzeta - lz) % max(1, nzeta)
    imirr = (lt_m * nzeta + lz_m).astype(np.int64)
    nuv_full = int(max(1, nu_full) * max(1, nzeta))
    idx_full = np.arange(nuv_full, dtype=np.int64)
    ku_full = idx_full // max(1, nzeta)
    kv_full = idx_full % max(1, nzeta)
    ku_m_full = (nu_full - ku_full) % max(1, nu_full)
    kv_m_full = (nzeta - kv_full) % max(1, nzeta)
    imirr_full = (ku_m_full * nzeta + kv_m_full).astype(np.int64)

    mn0 = 0
    for j in range(mnpd):
        if int(xmpot[j]) == 0 and int(n_raw[j]) == 0:
            mn0 = int(j)
            break

    return {
        "xmpot": xmpot,
        "xnpot": xnpot,
        "n_raw": n_raw,
        "sin_phase": sin_phase,
        "cos_phase": cos_phase,
        "sinmni": sinmni,
        "cosmni": cosmni,
        "wint": w,
        "imirr": imirr,
        "imirr_full": imirr_full,
        "mnpd": mnpd,
        "mnpd2": mnpd2,
        "nuv3": int(th.size),
        "nuv_full": nuv_full,
        "mn0": mn0,
        "onp": 1.0 / float(nfp),
        "nfp": nfp,
        "mf": mf,
        "nf": nf,
        "nu_full": int(nu_full),
        "lasym": lasym,
        "theta": th,
        "zeta": ze,
        "cmns": build_vmec_cmns(mf=mf, nf=nf, onp=1.0 / float(nfp)),
    }


def build_poisson_cache(*, ntheta: int, nzeta: int) -> NestorPoissonCache:
    """Build spectral Laplacian eigenvalues on a periodic ``(theta,zeta)`` grid."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ku2 = ku[:, None] * ku[:, None]
    kv2 = kv[None, :] * kv[None, :]
    lam = ku2 + kv2
    lam[0, 0] = 1.0
    return NestorPoissonCache(ntheta=ntheta, nzeta=nzeta, lam=lam)


def build_vmec_like_cache(
    sample: ExternalBoundarySample,
    *,
    alpha: float,
    dist_eps: float,
    rhs_floor: float,
    diag_coeff: float,
    row_sum_zero: bool,
    singular_diag_scale: float,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint_vmec: np.ndarray | None = None,
    factor_physical_matrix: bool = True,
) -> NestorVmecLikeCache:
    """Build a dense boundary-integral-like operator on the VMEC angular grid."""

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    ntheta, nzeta = R.shape
    npts = int(ntheta * nzeta)
    phi_grid = np.asarray(sample.phi, dtype=float)
    if phi_grid.shape != R.shape:
        phi_grid = np.broadcast_to(phi_grid, R.shape)
    x = R * np.cos(phi_grid)
    y = R * np.sin(phi_grid)
    coords = np.stack([x, y, Z], axis=-1).reshape(npts, 3)
    det = np.asarray(sample.vac_ext.det_guv, dtype=float)
    w = np.sqrt(np.maximum(np.abs(det), 0.0)).reshape(npts)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= rhs_floor:
        w = np.full((npts,), 1.0 / float(max(1, npts)), dtype=float)
    else:
        w = w / w_sum

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + float(dist_eps) ** 2)
    invdist = np.where(dist > 0.0, 1.0 / dist, 0.0)
    np.fill_diagonal(invdist, 0.0)

    kernel = (invdist * w[None, :]) / (4.0 * np.pi)
    if bool(row_sum_zero):
        row_sum = np.sum(kernel, axis=1)
        kernel[np.arange(npts), np.arange(npts)] -= row_sum

    diag_extra = np.zeros((npts,), dtype=float)
    if float(singular_diag_scale) != 0.0:
        dist_nodiag = np.asarray(dist, dtype=float).copy()
        np.fill_diagonal(dist_nodiag, np.inf)
        h = np.minimum(np.min(dist_nodiag, axis=1), 1.0 / float(max(1, npts)))
        h = np.maximum(h, float(dist_eps))
        diag_extra = (float(singular_diag_scale) / (4.0 * np.pi)) * (w / h)

    matrix = float(alpha) * kernel
    matrix[np.arange(npts), np.arange(npts)] += float(diag_coeff) + diag_extra
    rhs_scale = np.where(w > rhs_floor, w, rhs_floor)

    wint_use = np.asarray(wint_vmec, dtype=float) if wint_vmec is not None else np.asarray(w, dtype=float).reshape(ntheta, nzeta)
    mode_basis = build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=int(nfp),
        mf=int(mf),
        nf=int(nf),
        lasym=bool(lasym),
        wint=np.asarray(wint_use, dtype=float),
    )
    sinmni = np.asarray(mode_basis["sinmni"], dtype=float)
    cosmni = np.asarray(mode_basis["cosmni"], dtype=float)
    B = np.concatenate([sinmni, cosmni], axis=1) if bool(lasym) else sinmni
    mode_matrix = B.T @ (matrix @ B)
    mnpd = int(mode_basis["mnpd"])
    if mnpd > 0:
        pi3 = float(4.0 * (np.pi**3))
        mode_matrix[:mnpd, :mnpd][np.diag_indices(mnpd)] += pi3
        if bool(lasym):
            mode_matrix[mnpd:, mnpd:][np.diag_indices(mnpd)] += pi3
            mn0 = int(mode_basis["mn0"])
            if 0 <= mn0 < mnpd:
                mode_matrix[mnpd + mn0, mnpd + mn0] += pi3

    return NestorVmecLikeCache(
        ntheta=ntheta,
        nzeta=nzeta,
        matrix=matrix,
        rhs_scale=rhs_scale,
        mode_basis=mode_basis,
        mode_matrix=mode_matrix,
        matrix_lu=dense_lu_factor(matrix) if bool(factor_physical_matrix) else None,
        mode_matrix_lu=dense_lu_factor(mode_matrix),
    )


def solve_vmec_like_dense(rhs: np.ndarray, cache: NestorVmecLikeCache) -> np.ndarray:
    rhs_flat = np.asarray(rhs, dtype=float).reshape(-1) * np.asarray(cache.rhs_scale, dtype=float)
    phi_flat = dense_lu_solve(cache.matrix_lu, np.asarray(cache.matrix, dtype=float), rhs_flat)
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return phi


def vmec_source_from_gsource(*, gsource: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """VMEC fouri.f source symmetrization from gsource."""

    gsrc = np.asarray(gsource, dtype=float).reshape(-1)
    onp = float(basis["onp"])
    nuv3 = int(basis.get("nuv3", gsrc.size))
    nuv_full = int(basis.get("nuv_full", nuv3))
    if bool(basis["lasym"]):
        src = onp * gsrc[:nuv3]
    elif gsrc.size >= nuv_full and "imirr_full" in basis:
        imirr_full = np.asarray(basis["imirr_full"], dtype=np.int64)
        src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr_full[:nuv3]])
    else:
        imirr = np.asarray(basis["imirr"], dtype=np.int64)
        src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr[:nuv3]])
    return np.asarray(src, dtype=float)


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("", "0", "false", "no")


def digest_array_for_cache(value: Any) -> tuple[tuple[int, ...], str, str]:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return tuple(int(i) for i in arr.shape), str(arr.dtype), digest


def mapping_cache_signature(mapping: dict[str, Any], keys: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    selected = tuple(sorted(mapping)) if keys is None else tuple(key for key in keys if key in mapping)
    signature: list[Any] = []
    for key in selected:
        value = mapping[key]
        if isinstance(value, dict):
            continue
        signature.append((key, digest_array_for_cache(value)))
    return tuple(signature)


def compact_jax_nestor_basis(basis: dict[str, Any]) -> dict[str, Any]:
    return {key: basis[key] for key in JAX_NESTOR_BASIS_KEYS if key in basis}


def jax_nestor_operator_cache_key(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool,
    input_signature: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    return (
        int(signgs),
        int(nvper),
        bool(include_analytic),
        bool(symmetric),
        tuple(input_signature),
        mapping_cache_signature(basis, JAX_NESTOR_BASIS_KEYS),
        mapping_cache_signature(tables),
    )


def jax_nestor_input_signature(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((tuple(int(i) for i in np.asarray(arg).shape), str(np.asarray(arg).dtype)) for arg in args)


def jitted_jax_nestor_operator(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool = False,
    example_args: tuple[Any, ...] = (),
) -> tuple[Any | None, bool]:
    """Return a cached compiled dense JAX NESTOR operator closure.

    The closure bakes mode-basis and kernel-table arrays as static constants so
    the active free-boundary update does not execute the JAX operator as many
    small eager dispatches. This cache is intentionally used only by the opt-in
    research path selected with ``VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR=1``.
    """

    try:
        from ..._compat import jax as _jax
        from ...free_boundary_adjoint import dense_vmec_nestor_mode_solve_jax
    except Exception:
        return None, False
    if _jax is None:
        return None, False
    if bool(getattr(_jax.config, "jax_disable_jit", False)):
        return None, False

    key = jax_nestor_operator_cache_key(
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        include_analytic=bool(include_analytic),
        symmetric=bool(symmetric),
        input_signature=jax_nestor_input_signature(tuple(example_args)),
    )
    cached = FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.get(key)
    if cached is not None:
        return cached, True

    if len(FREEB_JAX_NESTOR_OPERATOR_FN_CACHE) >= 32:
        FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()

    basis_static = compact_jax_nestor_basis(basis)
    tables_static = {key: tables[key] for key in sorted(tables)}

    def _compiled(
        R: Any,
        Z: Any,
        Ru: Any,
        Zu: Any,
        Rv: Any,
        Zv: Any,
        ruu: Any,
        ruv: Any,
        rvv: Any,
        zuu: Any,
        zuv: Any,
        zvv: Any,
        bexni: Any,
    ) -> dict[str, Any]:
        return dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni,
            basis=basis_static,
            tables=tables_static,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            symmetric=bool(symmetric),
        )

    jitted = _jax.jit(_compiled)
    compiled = jitted.lower(*example_args).compile() if example_args else jitted
    FREEB_JAX_NESTOR_OPERATOR_FN_CACHE[key] = compiled
    return compiled, False


def jax_nestor_operator_guard(
    *,
    sample: Any,
    basis: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Return whether the experimental JAX VMEC/NESTOR operator can run safely."""

    if basis is None:
        return False, "missing_mode_basis"
    try:
        from ..._compat import has_jax, x64_enabled

        if not has_jax():
            return False, "jax_unavailable"
        if not x64_enabled():
            return False, "jax_x64_disabled"
    except Exception:
        return False, "jax_unavailable"
    if sample.R.ndim != 2:
        return False, "sample_R_not_2d"
    if int(sample.R.size) != int(basis.get("nuv3", sample.R.size)):
        return False, "requires_active_vmec_grid_points"
    if bool(basis.get("lasym", False)) and int(sample.R.size) != int(basis.get("nuv_full", sample.R.size)):
        return False, "requires_lasym_full_vmec_grid_points"
    if int(sample.R.shape[0]) > int(basis.get("nu_full", sample.R.shape[0])):
        return False, "active_grid_exceeds_full_grid"
    for name in ("Z", "Ru", "Zu", "Rv", "Zv"):
        arr = np.asarray(getattr(sample, name), dtype=float)
        if arr.shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    for name in ("ruu", "ruv", "rvv", "zuu", "zuv", "zvv"):
        arr = getattr(sample, name)
        if arr is None:
            return False, f"missing_{name}"
        if np.asarray(arr).shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    return True, "enabled"


def solve_vmec_like_mode_with_jax_nestor_operator(
    *,
    sample: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
    include_analytic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Run the experimental dense JAX VMEC/NESTOR mode operator."""

    from ...free_boundary_adjoint import dense_vmec_nestor_mode_solve_jax

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    Ru = np.asarray(sample.Ru, dtype=float)
    Zu = np.asarray(sample.Zu, dtype=float)
    Rv = np.asarray(sample.Rv, dtype=float)
    Zv = np.asarray(sample.Zv, dtype=float)
    ruu = np.asarray(sample.ruu, dtype=float)
    ruv = np.asarray(sample.ruv, dtype=float)
    rvv = np.asarray(sample.rvv, dtype=float)
    zuu = np.asarray(sample.zuu, dtype=float)
    zuv = np.asarray(sample.zuv, dtype=float)
    zvv = np.asarray(sample.zvv, dtype=float)
    bexni_arr = np.asarray(bexni, dtype=float)
    operator_args = (R, Z, Ru, Zu, Rv, Zv, ruu, ruv, rvv, zuu, zuv, zvv, bexni_arr)
    compiled = None
    cache_hit = False
    if env_truthy("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", True):
        compiled, cache_hit = jitted_jax_nestor_operator(
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
            example_args=operator_args,
        )
    if compiled is None:
        out = dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni_arr,
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
        )
        jit_used = False
    else:
        out = compiled(*operator_args)
        jit_used = True
    potvac = np.asarray(out["mode_coeffs"], dtype=float)
    rhs_mode = np.asarray(out["rhs_mode"], dtype=float)
    mode_matrix = np.asarray(out["mode_matrix"], dtype=float)
    grpmn = np.asarray(out["grpmn"], dtype=float)
    gsource_nonsing = np.asarray(out["gsource_nonsing"], dtype=float)
    mnpd2 = int(basis["mnpd2"])
    if mode_matrix.shape != (mnpd2, mnpd2):
        raise ValueError("jax_nestor_mode_matrix_shape")
    if rhs_mode.shape != (mnpd2,) or potvac.shape != (mnpd2,):
        raise ValueError("jax_nestor_mode_vector_shape")
    for name, arr in (
        ("rhs_mode", rhs_mode),
        ("mode_matrix", mode_matrix),
        ("mode_coeffs", potvac),
        ("grpmn", grpmn),
        ("gsource_nonsing", gsource_nonsing),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"jax_nestor_nonfinite_{name}")
    residual = mode_matrix @ potvac - rhs_mode
    residual_tol = 1.0e-8 * (1.0 + float(np.linalg.norm(rhs_mode)))
    if float(np.linalg.norm(residual)) > residual_tol:
        raise ValueError("jax_nestor_linear_residual")
    mnpd = int(basis["mnpd"])
    sin_phase = np.asarray(basis["sin_phase"], dtype=float)
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    if bool(basis["lasym"]) and potvac.size >= 2 * mnpd:
        phi_flat = sin_phase @ potvac[:mnpd] + cos_phase @ potvac[mnpd : 2 * mnpd]
    else:
        phi_flat = sin_phase @ potvac[:mnpd]
    phi = np.asarray(phi_flat, dtype=float).reshape(np.asarray(sample.R).shape)
    phi = phi - float(np.mean(phi))
    return (
        phi,
        potvac,
        rhs_mode,
        mode_matrix,
        grpmn,
        gsource_nonsing,
        jit_used,
        cache_hit,
    )


__all__ = [
    "FREEB_JAX_NESTOR_OPERATOR_FN_CACHE",
    "JAX_NESTOR_BASIS_KEYS",
    "build_poisson_cache",
    "build_vmec_cmns",
    "build_vmec_like_cache",
    "build_vmec_mode_basis",
    "compact_jax_nestor_basis",
    "dense_lu_factor",
    "dense_lu_solve",
    "digest_array_for_cache",
    "env_truthy",
    "jax_nestor_input_signature",
    "jax_nestor_operator_cache_key",
    "jax_nestor_operator_guard",
    "jitted_jax_nestor_operator",
    "mapping_cache_signature",
    "solve_vmec_like_dense",
    "solve_vmec_like_mode_with_jax_nestor_operator",
    "vmec_source_from_gsource",
]
