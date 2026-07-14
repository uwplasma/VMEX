# vmec_jax mirror plan

## Purpose

Finish a small, maintainable, research-grade equilibrium package. The core VMEC
backend remains a fast, differentiable toroidal solver. The mirror backend adds
open-field-line equilibria only where it has independent physics validation.
ESSOS owns coils and Biot--Savart; SOLVAX owns generic linear/nonlinear solver
infrastructure. vmec_jax owns equilibrium physics, discretisation, IO, and
user-facing diagnostics.

This is the only active plan. Git history and compact benchmark JSON files are
the completion record; do not recreate long historical plans.

## Architecture rules

1. Public free-boundary inputs are `MgridField` for a forward solve and an
   `xyz -> B` callable for differentiable residuals. ESSOS supplies both.
   No new public `CoilSet`, Biot--Savart, or coil-construction API belongs in
   vmec_jax.
2. Use SOLVAX for generic Krylov, direct structured solves, implicit solves,
   preconditioner primitives, and chunked AD. Keep only VMEC/mirror residuals,
   physics-scalings, and state packing in this repository.
3. Each supported model has one source module family, one top-level example,
   one focused test family, and one compact documentation figure. Generated
   WOUT/MOUT, mgrid, traces, and uncompressed figures stay ignored.
4. A feature is supported only after an analytic or independent-code check,
   resolution study, convergence history, example, and documentation. Failed
   refinement is a deferred result, not a supported capability.
5. Keep public modules focused and documented. Prefer deleting a compatibility
   shim after migration over adding a second abstraction layer.

## Accepted scope

| Model | Status | Required evidence |
|---|---|---|
| Toroidal fixed boundary | Supported | VMEC2000 parity and implicit-gradient tests |
| Toroidal NESTOR free boundary | Supported | VMEC2000/mgrid tests; ESSOS field tabulation |
| Straight axisymmetric mirror | Supported | two-coil analytic field, MMS, finite-beta continuation |
| Straight nonaxisymmetric fixed mirror | Research, active | paraxial rotating-ellipse validation and refinement |
| Straight nonaxisymmetric free mirror | Research, active | fixed-mirror gate plus exterior/refinement convergence |
| Toroidal stellarator-mirror hybrid | Design research | native B-spline geometry; no Fourier boundary promotion |

Profile cubic/Akima splines are already supported. They are unrelated to the
native B-spline *geometry* state required below.

## Native B-spline geometry

The hybrid must represent a closed torus with two long straight mirror legs and
two curved return sections. A Fourier projection is not an acceptable design
representation: it introduces global ringing and cannot locally optimise the
straight-to-curved transitions. Implement a periodic, locally supported
B-spline centerline and cross-section state:

- periodic cubic B-spline centerline in Cartesian space;
- local orthonormal frame, with B-spline semi-axes and ellipse angle along arc
  length;
- exact periodic closure and at least C2 continuity at the four joins;
- knot insertion/refinement without changing the represented geometry;
- JAX pytree controls and `jax.jvp`/`jax.vjp` tests;
- geometry-to-equilibrium adapters. The mirror solver consumes the open straight
  patch directly. The toroidal VMEC solver receives a controlled sampled
  representation only for compatibility studies, never as the optimization
  state.

The first B-spline deliverable is geometry, vacuum field-line, and derivative
validation. A full toroidal free-boundary B-spline equilibrium is promoted only
after its own solver/state formulation passes the gates below; do not disguise a
Fourier projection as native B-spline support.

## Nonaxisymmetric straight-mirror validation

Use the paraxial vacuum construction for a straight axis `z`. Let the leading
cross-section be an ellipse whose principal angle changes smoothly by 90 degrees
from one end to the other. At small radius, validate:

1. flux conservation: `X1c*Y1s - X1s*Y1c = Bbar/B0(z)`;
2. no first-order poloidal field-strength variation: `B1c = B1s = 0`;
3. leading quadrupole: fit `B = B0(z) + r^2 [B20 + B2c cos(2a) +
   B2s sin(2a)] + O(r^3)` and compare fitted coefficients to the paraxial
   construction;
4. a 90-degree rotation changes the quadrupole phase continuously, with no
   artificial `m=1` component;
5. convergence under radial, poloidal, axial, and exterior-panel refinement;
6. finite-beta pressure balance and normal-field residuals on the solved LCFS.

The reference derivation is Appendix C of Rodriguez, Helander & Goodman,
*J. Plasma Phys.* 90 (2024), DOI 10.1017/S0022377824000345. It explicitly
identifies the near-axis approximation as the paraxial mirror approximation,
derives the flux constraint and the quadrupolar leading `|B|` variation. Also
cross-check the two-coil axis field against the standard circular-loop formula.

## Ordered work

### 1. Finish the active nonaxisymmetric mirror lane

1. Add a compact `RotatingEllipseMirror` analytic fixture in the mirror geometry
   package, not in an example. It returns geometry, paraxial coefficients, and
   expected invariants.
2. Add fixed-boundary tests for the six gates above at three resolutions.
3. Add one root example with parameters at top, no parser: fixed solve, residual
   history, horizontal 3-D field lines, `|B|`, cross-sections, fitted paraxial
   coefficients, iota/current diagnostics, and convergence plot.
4. Couple the same fixture to the exterior free-boundary solve. Require a
   resolution-stable LCFS, `B.n`, pressure balance, and beta continuation before
   promotion. If it fails, retain one compact negative benchmark and stop.

### 2. B-spline hybrid geometry

1. Add a small `mirror/splines.py` owning B-spline basis evaluation, periodic
   knot handling, local frame, and validation. Do not duplicate generic sparse
   linear algebra.
2. Build the two-straight/two-return hybrid from B-spline controls and validate
   straightness, curvature, C2 joins, tube clearance, field-line closure, and
   derivatives.
3. Replace `core.hybrid` Fourier-target construction and its square-coil helper
   dependence with this geometry API. Keep a compatibility reader only while
   old benchmark data are migrated, then delete it.
4. Rebuild the fixed and free hybrid examples around ESSOS coil files/objects.
   Forward fields are tabulated to `MgridField`; derivative objectives use an
   ESSOS callable. Do not retain in-tree coil formulas.

### 3. Simplify ownership

1. Migrate the remaining internal-coil consumers (`mirror.vacuum`, hybrid
   helpers, three examples, and their tests) to ESSOS/MGRID/callable inputs.
2. Delete `vmec_jax/core/coils.py` and `tests/test_coils.py` only after external
   ESSOS contract tests replace their physical assertions.
3. Move any generic solver copied locally to SOLVAX only when it has at least two
   users. Current SOLVAX coverage is sufficient for GMRES/GCROT, PCG,
   block-Thomas, banded/tridiagonal, multigrid/preconditioner primitives,
   implicit/root solves, and chunked AD. Do not perform a speculative rewrite
   of VMEC or mirror residual drivers.
4. After each migration, remove obsolete examples, benchmark scripts, imports,
   and documentation. The target is fewer modules and fewer public names, not
   wrappers around old code.

### 4. Documentation and release evidence

1. README: one capabilities table, one toroidal result, one axisymmetric mirror
   result, one rotating-ellipse result, and one B-spline hybrid result.
2. `docs/mirror_geometry.rst`: equations, scope, assumptions, validation plots,
   and honest deferred limits. `docs/architecture.rst`: ESSOS/SOLVAX ownership.
3. Run ruff, strict Sphinx, focused unit tests for each changed module, then the
   relevant CI shard. Do not wait on CI between small commits.

## Completion criteria

- No public coil implementation in vmec_jax.
- Nonaxisymmetric fixed and free straight mirrors satisfy the paraxial and
  refinement gates, or are explicitly deferred with a reproducible failure.
- B-spline hybrid geometry has local controls, C2 periodic joins, JAX derivative
  tests, and an ESSOS-driven example. It is never represented as Fourier-only
  support.
- README/docs figures are compressed, generated reproducibly, and correspond to
  actual solved equilibria.
- The branch decreases source/module count after the ESSOS migration and has no
  stale experimental API or duplicate solver implementation.
