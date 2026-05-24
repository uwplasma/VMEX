"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp


def dense_vacuum_solve_jax(A: Any, b: Any, *, symmetric: bool = False) -> Any:
    """Solve a dense toy vacuum linear system with an implicit adjoint.

    Parameters
    ----------
    A:
        Dense square matrix.
    b:
        Right-hand side vector or matrix.
    symmetric:
        If true, the transpose solve is the same as the primal solve.

    Notes
    -----
    This is a scaffold for small tests and future NESTOR refactoring.  It does
    not imply that the current production NESTOR path is fully differentiable.
    The production path should eventually expose a JAX-native matrix-free
    operator and pass it through ``jax.lax.custom_linear_solve`` or equivalent.
    """

    A_arr = jnp.asarray(A)
    b_arr = jnp.asarray(b)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square dense matrix")
    if b_arr.shape[0] != A_arr.shape[0]:
        raise ValueError(f"b leading dimension {b_arr.shape[0]} does not match A size {A_arr.shape[0]}")

    if jax is None:  # pragma: no cover - dependency fallback.
        return jnp.linalg.solve(A_arr, b_arr)

    def matvec(x):
        return A_arr @ x

    def solve_fn(_matvec, rhs):
        return jnp.linalg.solve(A_arr, rhs)

    def transpose_solve_fn(_matvec, rhs):
        matrix = A_arr if bool(symmetric) else A_arr.T
        return jnp.linalg.solve(matrix, rhs)

    return jax.lax.custom_linear_solve(
        matvec,
        b_arr,
        solve_fn,
        transpose_solve=transpose_solve_fn,
        symmetric=bool(symmetric),
    )


def dense_vacuum_residual(A: Any, x: Any, b: Any) -> Any:
    """Return ``A @ x - b`` for tests and diagnostics."""

    return jnp.asarray(A) @ jnp.asarray(x) - jnp.asarray(b)


def vmec_source_from_gsource_jax(
    gsource: Any,
    *,
    onp: float,
    lasym: bool,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
) -> Any:
    """JAX version of VMEC/NESTOR source symmetrization.

    ``gsource`` is the weighted normal-field source used by the VMEC-like
    NESTOR bridge.  For stellarator-symmetric solves VMEC anti-symmetrizes the
    source with its mirror point before projecting onto sine modes.  For LASYM
    solves it uses the source directly.  This helper is intentionally small and
    side-effect free so the source-to-mode-RHS stage can be differentiated and
    finite-difference checked independently of the current host NESTOR path.
    """

    gsrc = jnp.reshape(jnp.asarray(gsource), (-1,))
    n_source = int(gsrc.shape[0])
    n3 = int(nuv3) if nuv3 is not None else n_source
    nfull = int(nuv_full) if nuv_full is not None else n3

    if bool(lasym):
        return float(onp) * gsrc[:n3]

    if n_source >= nfull and imirr_full is not None:
        mirror = jnp.asarray(imirr_full, dtype=jnp.int32)[:n3]
        mirrored = gsrc[mirror]
    elif imirr is not None:
        mirror = jnp.asarray(imirr, dtype=jnp.int32)[:n3]
        mirrored = gsrc[mirror]
    else:
        raise ValueError("non-LASYM source symmetrization requires imirr or imirr_full")
    return 0.5 * float(onp) * (gsrc[:n3] - mirrored)


def mode_rhs_from_gsource_jax(
    gsource: Any,
    *,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    onp: float,
    lasym: bool,
    cos_basis: Any | None = None,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
) -> Any:
    """Project a VMEC/NESTOR grid source into mode-space RHS coefficients.

    This mirrors the production ``_vmec_bvec_from_gsource`` contract with JAX
    arrays.  It is a validation rung for the future production adjoint:
    differentiable external fields can feed this source projection, then a
    custom-linear-solve vacuum primitive, before the full NESTOR operator is
    ported.
    """

    src = vmec_source_from_gsource_jax(
        gsource,
        onp=float(onp),
        lasym=bool(lasym),
        nuv3=nuv3,
        nuv_full=nuv_full,
        imirr=imirr,
        imirr_full=imirr_full,
    )
    sin = jnp.asarray(sin_basis)
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    bsin = sin.T @ src

    xmpot_arr = jnp.asarray(xmpot)
    n_raw_arr = jnp.asarray(n_raw)
    skip_mask = jnp.logical_and(xmpot_arr == 0, n_raw_arr < 0)
    bsin = jnp.where(skip_mask, 0.0, bsin)

    if not bool(lasym):
        return bsin
    if cos_basis is None:
        raise ValueError("cos_basis is required for LASYM mode RHS projection")
    cos = jnp.asarray(cos_basis)
    if cos.shape != sin.shape:
        raise ValueError("cos_basis must match sin_basis shape")
    bcos = cos.T @ src
    bcos = jnp.where(skip_mask, 0.0, bcos)
    return jnp.concatenate([bsin, bcos], axis=0)


def dense_mode_vacuum_solve_jax(
    mode_matrix: Any,
    rhs_mode: Any,
    sin_basis: Any,
    cos_basis: Any | None = None,
    *,
    symmetric: bool = False,
) -> dict[str, Any]:
    """Solve a dense mode-space vacuum system and reconstruct a grid potential.

    This is the next scaffold between the dense toy solve and the production
    NESTOR path.  The current VMEC-like NESTOR implementation eventually builds
    a dense mode-space matrix and right-hand side before reconstructing a
    scalar potential on the boundary grid.  This helper makes that contract
    JAX-transformable and differentiable while the full source/operator assembly
    remains in the host implementation.

    Parameters
    ----------
    mode_matrix:
        Dense mode-space matrix ``A``.
    rhs_mode:
        Right-hand side vector ``b``.
    sin_basis, cos_basis:
        Flattened boundary-grid basis arrays with shape ``(npoints, nmodes)``.
        For stellarator-symmetric mode vectors pass only ``sin_basis``.  For
        LASYM-style doubled vectors pass both basis blocks; the first block of
        ``mode_coeffs`` multiplies ``sin_basis`` and the second multiplies
        ``cos_basis``.
    symmetric:
        Forwarded to :func:`dense_vacuum_solve_jax`.
    """

    A = jnp.asarray(mode_matrix)
    rhs = jnp.asarray(rhs_mode)
    sin = jnp.asarray(sin_basis)
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    coeffs = dense_vacuum_solve_jax(A, rhs, symmetric=bool(symmetric))

    if cos_basis is None:
        if coeffs.shape[0] != sin.shape[1]:
            raise ValueError("rhs/mode_matrix size must match sin_basis columns")
        phi_flat = sin @ coeffs
    else:
        cos = jnp.asarray(cos_basis)
        if cos.shape != sin.shape:
            raise ValueError("cos_basis must match sin_basis shape")
        nmodes = int(sin.shape[1])
        if coeffs.shape[0] != 2 * nmodes:
            raise ValueError("doubled rhs/mode_matrix size must be 2 * sin_basis columns")
        phi_flat = sin @ coeffs[:nmodes] + cos @ coeffs[nmodes:]

    return {
        "mode_coeffs": coeffs,
        "phi_flat": phi_flat,
        "residual": dense_vacuum_residual(A, coeffs, rhs),
    }


def vacuum_boundary_fields_from_cylindrical_jax(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    det_floor: float = 1.0e-30,
) -> dict[str, Any]:
    """JAX version of the VMEC boundary-field projection scaffold.

    This mirrors ``free_boundary.vacuum_boundary_fields_from_cylindrical`` for
    derivative tests.  It intentionally returns a plain dict rather than the
    NumPy dataclass used by the production bridge, so it can be transformed by
    ``jax.grad``/``jax.jacfwd`` while the full NESTOR path is still being
    ported.
    """

    br_arr = jnp.asarray(br)
    bp_arr = jnp.asarray(bp)
    bz_arr = jnp.asarray(bz)
    R_arr = jnp.asarray(R)
    Ru_arr = jnp.asarray(Ru)
    Zu_arr = jnp.asarray(Zu)
    Rv_arr = jnp.asarray(Rv)
    Zv_arr = jnp.asarray(Zv)

    g_uu = Ru_arr * Ru_arr + Zu_arr * Zu_arr
    g_uv = Ru_arr * Rv_arr + Zu_arr * Zv_arr
    g_vv = R_arr * R_arr + Rv_arr * Rv_arr + Zv_arr * Zv_arr
    det = g_uu * g_vv - g_uv * g_uv
    det_safe = jnp.where(
        jnp.abs(det) >= float(det_floor),
        det,
        jnp.sign(det + 1.0e-300) * float(det_floor),
    )

    bu = br_arr * Ru_arr + bz_arr * Zu_arr
    bv = br_arr * Rv_arr + bp_arr * R_arr + bz_arr * Zv_arr
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)

    n_r = -R_arr * Zu_arr
    n_phi = Zu_arr * Rv_arr - Ru_arr * Zv_arr
    n_z = R_arr * Ru_arr
    bnormal = br_arr * n_r + bp_arr * n_phi + bz_arr * n_z
    n_norm = jnp.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
    bnormal_unit = bnormal / jnp.where(n_norm > 0.0, n_norm, 1.0)

    return {
        "bu": bu,
        "bv": bv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "bsqvac": bsqvac,
        "bnormal": bnormal,
        "bnormal_unit": bnormal_unit,
        "g_uu": g_uu,
        "g_uv": g_uv,
        "g_vv": g_vv,
        "det_guv": det,
    }
