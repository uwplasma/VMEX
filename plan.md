# Mirror equilibrium release plan

Status: final authoritative plan for draft PR #22. This file supersedes the
original `/Users/rogeriojorge/Downloads/plan_mirror.md` and every earlier
version of this plan. Do not create parallel roadmaps. Commits, tests, and the
four compact benchmark JSON files are the execution log.

Audit baseline (2026-07-14, refreshed after the final source review):

- branch `codex/mirror-geometry` at `dac37b42`;
- base `origin/main` at `ed4ac7ac`, zero commits behind and 307 ahead;
- PR #22 is open, draft, and mergeable;
- diff is 51 files, 16,401 insertions, and 1,608 deletions;
- committed `vmec_jax/mirror` contains 13 modules, 7,921 lines, and 20 lazy public
  names;
- mirror tests contain ten substantive owner-aligned files and 4,241 lines;
- after the `mpol` migration, the committed mirror suite passed 104 tests and
  skipped 10 in 303.98 seconds. The audited M1 worktree passes 105 tests and
  skips 10 in 395.44 seconds; strict Sphinx, Ruff, pre-commit, and
  `git diff --check` pass; every required pushed CI check is green;
- public `ntheta` has been removed. `mpol` denotes the highest retained angular
  Fourier mode and the grid derives `ntheta = 2 * mpol + 1` exactly;
- M1 work in the uncommitted tree raises the package temporarily to 8,051 lines.
  Its manufactured cylindrical force test is second order, but the circular
  hybrid first radial row grows from about `1.00` at `ns=5` to `1.37` at
  `ns=9`; this work is not accepted until that near-axis defect is resolved.

The branch contains real equilibrium solvers and useful validation, but it is
not release-ready. The immediate blockers are the independent strong-force
near-axis reconstruction, stale shaped benchmarks predating the
axis and `mpol` corrections, duplicate nodal/spline solver paths, dense
free-boundary Jacobian storage, a preconditioner that omits
radius--stream-function coupling, and incomplete closed-hybrid limiting cases.
The milestones below remove those blockers in dependency order.

## 1. Release contract

PR #22 will support these scalar-pressure equilibrium models:

1. straight-axis, fixed-boundary, axisymmetric mirrors;
2. straight-axis, fixed-boundary, nonaxisymmetric mirrors, including both the
   Agren--Savenko straight-field-line mirror and the independent 90-degree
   rotating-ellipse paraxial case;
3. straight-axis, axisymmetric free-boundary mirrors in a supplied external
   field, with beta continuation through 50%;
4. fixed-boundary toroidal stellarator--mirror hybrids represented by periodic
   cubic B-splines, with two long straight mirror legs connected by two smooth
   stellarator returns.

All four models use coefficient-native cubic B-splines for every optimized
longitudinal geometry and stream-function degree of freedom. The open models
use clamped splines and the closed hybrid uses periodic splines. This is the
single production representation; nodal CGL states remain only as quadrature,
exterior-collocation, manufactured-test, and migration references.

One lane is conditional:

- nonaxisymmetric open free boundary receives one bounded promotion attempt
  after the strong-force and structured-preconditioner milestones. If it
  cannot pass a third-grid refinement gate within the stated resource budget,
  retain a compact negative benchmark, keep it out of the public API, and
  defer it without blocking the four required models.

The following are explicitly deferred to later PRs:

- free-boundary closed hybrids and their coil coupling;
- anisotropic pressure and ANIMEC physics;
- arbitrary curved open axes;
- kinetic end losses, sheaths, transport, stability, islands, and stochastic
  fields;
- mirror Boozer output and use of open mirrors in toroidal WOUT consumers;
- coil construction and Biot--Savart calculations, which belong in ESSOS.

A supported result is a converged nested-surface ideal-MHD equilibrium. A
sampled coil field, prescribed beta-dependent tube, small `B.n` without a
plasma-force solve, optimizer success flag, or Fourier fit of a square is not
an equilibrium result.

### 1.1 Promotion gates

Every supported lane must pass all applicable gates:

- component-wise normalized variational residual at or below `1e-12`;
- a documented double-precision exception no larger than `1e-11` only after a
  resolution study demonstrates the numerical floor;
- an independently assembled staggered weak first variation converging to the
  same floor without differentiating the production energy;
- an independently reconstructed `J x B - grad(p)` residual that passes exact
  manufactured tests and decreases under physical refinement;
- positive Jacobian, nested surfaces, adequate self-clearance, and normalized
  `div(B)` near roundoff;
- physical observables stable on three independently refined grids, assessed
  by observed order or Richardson extrapolation and a predeclared tolerance;
- for open free boundary, separately reported area-weighted `B.n`, total
  pressure jump, and artificial-cap compatibility residuals;
- an analytic, asymptotic, or independent-code comparison;
- forward and reverse implicit derivatives for every advertised control,
  checked against reconverged centered finite differences after the primal
  lane passes all preceding gates;
- one parser-free root example, one compact benchmark record, current docs,
  and compressed publication-quality figures.

The nonlinear variational residual, weak residual, strong force, and boundary
constraints are distinct diagnostics. None may be relabeled as another.

Observable refinement tolerances are fixed before each benchmark run. The
default is an extrapolated relative error below 0.5% for on-axis field and
central radius; a tighter 0.1% target is used when the observed order supports
it. A single pairwise difference, especially a historical 0.05% field target,
is not a valid convergence claim.

## 2. Physical models

### 2.1 Open straight mirrors

The coordinates are

`(s, theta, xi) in [0,1] x [0,2*pi) x [-1,1]`.

The axis is straight, `theta` is periodic, and `xi` is nonperiodic. The
lateral surface `s=1` is the plasma--vacuum interface. The planes at
`xi = +/-1` are fixed computational cuts crossed by magnetic flux; they are
not material interfaces and must not impose `B.n = 0`.

The production representation is Fourier in `theta`, VMEC-like and staggered
in `s`, and coefficient-native clamped cubic B-spline in `xi` for boundary,
interior geometry, and stream function. Chebyshev--Gauss--Lobatto nodes remain
an independent collocation, quadrature, exterior-panel, and validation
representation. “Full B-spline mirror” means that every optimized longitudinal
geometry and stream coefficient uses the same spline space; it does not mean
replacing the periodic poloidal angle or radial nested-surface mesh with
splines.

Regularity is a physical constraint, not post-processing:

- all `m > 0` geometry and stream coefficients vanish with the correct power
  of radius at the magnetic axis;
- scalar axis values are single-valued in `theta`;
- stream-function gauges are removed before optimization and linear solves;
- end values and end derivatives follow the declared fixed-cut conditions.

The free-boundary solve evaluates spline coefficients on the existing CGL and
panel nodes before calling the one cap-aware exterior backend. Shape
derivatives pass through this linear evaluation map. Promotion requires tests
of endpoint value and derivative constraints, cap-rim continuity, knot
refinement, and shape derivatives. Do not create a second spline-specific BIE.

### 2.2 Closed stellarator--mirror hybrid

The hybrid remains toroidal. Its magnetic axis is one smooth periodic curve
with two long nearly straight mirror legs and two curved stellarator returns.
Periodic cubic B-splines describe the centerline, Bishop-frame section shape,
and longitudinal stream coordinates. Fourier modes describe the periodic
cross-section angle; the radial coordinate remains VMEC-like.

The construction must provide:

- periodic position, tangent, frame, and spline derivatives through the joins;
- up--down and leg-exchange symmetries as coefficient maps, not duplicated
  geometry;
- positive section Jacobian and clearance along the entire circuit;
- a circular-axis limit matching ordinary `vmec_jax` and VMEC2000;
- a long-leg local limit matching the fixed open B-spline mirror;
- a rotating noncircular section in the returns capable of generating
  rotational transform.

Fixed-boundary beta scans change the interior equilibrium surfaces while the
LCFS stays fixed by definition. Claims of beta-driven LCFS motion belong only
to a future free-boundary hybrid.

### 2.3 Pressure model

This PR retains scalar pressure `p(s)`. ANIMEC is not equivalent to adding two
pressure arrays: its energy depends on `p_parallel(s, B)`, includes
`p_perpendicular`, anisotropy `sigma`, effective-current terms, fixed-`B`
derivatives, and firehose/mirror constraints. A partial implementation would
be misleading and is removed from this scope.

## 3. Current evidence ledger

### 3.1 Accepted evidence

- Axisymmetric fixed boundary reaches `ftol = 1e-12` and passes cylinder and
  flared-tube analytic checks, an independent weak residual, and implicit
  derivative tests.
- Fixed nonaxisymmetric solves optimize both radius and stream function at
  finite current; open spline/Chebyshev parity improves with refinement.
- Spline reverse derivatives agree with finite differences at about `3e-10`,
  and the forward implicit tangent test passes.
- A finite-beta radial pressure-balance manufactured case shows second-order
  radial convergence. The in-progress VMEC-style half-to-full reconstruction
  also gives second-order convergence for a cylindrical polynomial state with
  nonzero pressure, current, and lambda, but its closed-axis first-row limit is
  still a blocker and is not accepted evidence.
- A nonaxisymmetric shaped coordinate map with uniform Cartesian field has
  force below `1e-12`.
- Axis regularity now enforces a single-valued axis `|B|` and a consistent
  derivative pullback.
- The corrected axisymmetric free-boundary solve reaches a small variational
  and weak residual and shows the expected expansion and field depression.
- Periodic cubic racetrack geometry, Bishop transport, circular geometry, and
  field tracing have focused tests.
- Pleiades data are independently generated from commit
  `0161abb3f9c9e0885f5c739d6afec55cb73de733` and provide low-beta external-code
  evidence at 1%, 3%, and 10%.

### 3.2 Evidence that must be regenerated

All shaped benchmark JSON generated before the axis-regularity correction and
the exact `mpol` semantics is stale. In particular, old records that declared
`mpol = 1` with five or six angular nodes represented a different coefficient
space. Keep the filenames and schemas where possible, but regenerate the
numbers only after the strong-force milestone passes.

### 3.3 Rejected or incomplete evidence

- In the corrected axisymmetric free beta scan, the pointwise normalized force
  at beta 50% changes approximately `0.155 -> 0.197 -> 0.216`; it is not
  converging and blocks promotion despite small variational and boundary
  residuals.
- The medium/fine central radius differs by about 0.083%, while the central
  field differs by about 0.473%. The old 0.05% field target was unjustified;
  the replacement is the observed-order/extrapolation gate in section 1.1.
- Nonaxisymmetric free-boundary local `m=1` coefficients change by 73--81%; a
  medium pair took about 801 seconds and 4.25 GiB. It is a research result, not
  a supported equilibrium.
- The rotating-ellipse `m=2` response remains about 48% above the paraxial
  estimate at the tested knot count and must be rerun after the corrected
  angular semantics.
- The periodic preconditioner stalled at 3,000 GMRES iterations with linear
  residual about 0.136. CG/MINRES reached only about 0.016. The scalable closed
  primal path therefore remains disabled.
- With the in-progress staggered reconstruction, circular-hybrid total strong
  force is about `0.581`, `0.535`, and `0.523` at `ns=5,7,9`, while the
  physical bulk is about `0.581`, `0.074`, and `0.079`. The first active radial
  row instead grows from about `1.00` to `1.37`; no bulk mask may be used to
  declare this converged.
- There is no accepted hybrid VMEC parity case, open-leg limit, release MOUT,
  or hybrid implicit-derivative benchmark.

Failed diagnostics must remain visible. Do not tune tolerances, omit boundary
collars, or change normalizations after seeing results without documenting the
reason and rerunning all comparison grids.

## 4. Conclusions from external sources

### 4.1 VMEC2000, VMEC++, and ANIMEC

VMEC2000's `forces.f`, `residue.f90`, `bcovar.f`, `fbal.f`, and `jxbforce.f`
separate raw force assembly from preconditioned residuals and use the radial
staggering consistently. `precon2d.f` and the block-tridiagonal solver preserve
neighbor coupling instead of reducing the system to scalar diagonal scaling.
These are the primary references for repairing the mirror strong-force
reconstruction and designing the coupled preconditioner.

The concrete `jxbforce.f` pattern is the acceptance reference: covariant field
is stored on half surfaces, radial differences place current on full interior
surfaces, `sqrt(g) B^u` and `sqrt(g) B^v` are averaged before division by an
averaged Jacobian, and pressure uses a half-to-full difference. Axis and edge
values receive explicit one-sided treatment. VMEC's `fsqr/fsqz/fsql` remain
preconditioned variational residuals and are not interchangeable with this
pointwise diagnostic.

VMEC++ adds operational lessons: restartable continuation, explicit residual
history, bounded retries, and broad VMEC2000 parity. It does not validate open
topology. VMEC2000/VMEC++ are independent references only for the circular
closed-hybrid limit and matched toroidal inputs.

Cooper's 1992 variational formulation, the 2009 free-boundary ANIMEC paper, the
ANIMEC reports, and the `_ANIMEC` source paths confirm the scope decision in
section 2.3. The implementation modifies covariant-field/current terms through
`sigma`, evaluates `p_parallel(s,B)` and `p_perpendicular`, and adds fixed-`B`
profile derivatives to radial force balance. ANIMEC remains a future physics
model, not a pressure-array option inside this scalar solver.

### 4.2 DESC branches

The public DESC `mirror` (`0dba071d`) and `mirror_anisotropy` (`805b77fc`)
branches provide useful prototypes for Chebyshev--Zernike coordinates, end-cap
constraints, and continuation. They are not validated production references:

- mirror equilibrium and objective logic is largely duplicated in large
  branch-only modules;
- `FixEndCapR/Z` contains placeholder behavior and explicit mode selection is
  not implemented;
- the nominal mirror boundary-condition test contains imports rather than
  assertions;
- substantial upstream tests are renamed or disabled on the branch;
- notebook and binary artifacts make direct code transfer undesirable.

DESC's newer `dd/cylindrical` branch implements
`DoubleChebyshevFourierBasis`: two nonperiodic Cartesian Chebyshev coordinates
and one periodic Fourier coordinate. It has basis tests but no accepted mirror
equilibrium, end-cap, axis-regularity, or free-boundary validation. The
`finite_element_basis` branches add large experimental scikit-fem paths,
debugging output, and incomplete JAX integration. Use formulas from these
branches as review material only. Neither branch is a reason to replace the
compact Fourier/radial/spline tensor product in this PR.

DESC's released free-boundary work is more relevant conceptually: tangency,
pressure jump, and any sheet-current condition are distinct equations. Its
smooth-toroidal singular quadrature does not remove the open cap-rim
singularity, which is why the current open exterior formulation remains
separate.

### 4.3 Mirror and numerical literature

Ågren and Savenko's straight-field-line-mirror construction supplies exact
paraxial fixtures already represented in `analytic.py`:

- `x = x0 (1 + z/c)` and `y = y0 (1 - z/c)`;
- `B_axis = B0 / (1 - z^2/c^2)`;
- section ellipticity `(1 + |z|/c) / (1 - |z|/c)`;
- straight but nonparallel vacuum field lines, a marginal minimum-`B` field,
  and zero vacuum cross-field drift.

This SFLM section changes ellipticity but does not execute a prescribed
90-degree rigid rotation. The rotating ellipse in `RotatingEllipseParaxial`
comes from the independent near-axis construction in Appendix C of Rodriguez,
Helander, and Goodman. The two cases must have separate names, fixtures, and
error tables.

Related first-order finite-beta SFLM work gives the long-thin trend
`B approximately B_vacuum * sqrt(1 - beta)`. It supports an asymptotic trend
test, not an exact finite-beta equilibrium benchmark. The rotating ellipse
must first match the vacuum paraxial limit as minor radius and beta approach
zero, then be tested away from that limit.

The linked-mirror configuration of Feng et al. supports the closed topology:
two straight mirror sections joined by two half-tori, with nonparallel curved
sections producing transform. It does not prescribe the boundary
representation or establish MHD convergence.

Rodríguez, Helander, and Goodman's maximum-`J` paper contains useful paraxial
near-axis mirror equations in its appendix; it is not itself a
straight-field-line-mirror construction.

GVEC validates several architectural choices--coefficient-native B-splines,
independent quadrature, Fourier periodic angles, and a general transported
G-frame--but its published splines are radial rather than longitudinal. Its
strongly shaped examples show why an axis-following frame can lower Fourier
resolution and avoid invalid initial maps. It supports the Bishop-like hybrid
frame, not direct reuse of a GVEC representation. VEPEC provides historical
precedent for vector-potential variables and divergence-preserving tricubic
spline interpolation in high-beta minimum-`B` mirror studies.

Goodman, Freidberg, and Lane expand simultaneously in beta and the long-thin
parameter and compare section distortions with VEPEC. Their formulas are an
asymptotic low-beta/thin-tube gate, not a beta-50 parity result. Near-unity-beta
diamagnetic-bubble and RealTwin studies generally require anisotropic or
kinetic closures, so they cannot validate the scalar-pressure model
quantitatively.

Pleiades remains the only independent open-mirror numerical comparison in the
current branch. RealTwin's high-field tandem-mirror study and diamagnetic
bubble literature provide qualitative high-beta context, including pressure
anisotropy and difficult near-unity-beta behavior; neither is scalar-pressure
numerical parity. Beta 50% is a demanding validation point, not a claim of a
diamagnetic-bubble model.

### 4.4 SOLVAX and differentiation

Released SOLVAX `v0.7.3` provides block Thomas, banded and periodic-banded
operators, matrix-free and bordered operators, Krylov methods, preconditioner
utilities, implicit differentiation, and chunked AD. `vmec_jax` currently uses
its block-Thomas primitive. Generic linear algebra belongs in SOLVAX, but
mirror-specific geometry, residuals, gauges, coefficient maps, continuation,
and physics preconditioners stay here.

Do not depend on unpublished local SOLVAX commits. A prior direct replacement
of the host GMRES path did not meet runtime/residual gates. Adopt another
released primitive only after a reproducible A/B benchmark shows equal state,
lower memory or runtime, and no loss in true linear residual.

The production derivative strategy remains:

- solve the nonlinear equilibrium to the declared primal tolerance;
- differentiate the converged residual with exact JAX JVP/VJP actions;
- use forward implicit tangents for few controls;
- use reverse adjoints for scalar outputs with many controls;
- solve primal and transpose systems with the same physical preconditioner;
- never reverse-differentiate through the nonlinear iteration history;
- validate against fully reconverged centered finite differences and report
  both nonlinear and linear residuals.

JAX `custom_vjp` and `linearize`, JAXopt `custom_root`, Optimistix, and Lineax
all encode variants of the same implicit-function solve. Lineax offers useful
operator abstractions and transpose-aware solver state, while Optax supplies
first-order optimizers and Equinox supplies PyTree modules; none provides the
missing mirror block physics. The automated spectral-adjoint work of Skene and
Burns supports operator-transpose adjoints, but its symbolic sparse Dedalus
graph is not portable to this code. Add no new optimization or linear-solver
dependency in this PR.

## 5. Architecture and simplification rules

The current physical ownership boundaries are sound:

| Module | Owner |
| --- | --- |
| `model.py`, `basis.py`, `splines.py` | input contract, bases, coefficient maps, regularity |
| `geometry.py`, `analytic.py` | coordinate metrics and validation solutions |
| `forces.py` | energy, weak residual, strong-force diagnostics |
| `solver.py` | fixed-boundary nonlinear solve and continuation |
| `exterior.py`, `exterior_bie.py` | open-vacuum fields and cap-aware shape calculus |
| `free_boundary.py` | coupled open free-boundary solve |
| `implicit.py` | converged-state tangent and adjoint solves |
| `output.py` | MOUT serialization, diagnostics, and plotting data |

Do not collapse unrelated physics into one file. Simplification means removing
duplicated packing, gauges, linear actions, and configuration fields; using
one shared implementation per operation; and deleting failed public
scaffolding. It does not mean hiding distinct equations behind generic names.

The largest avoidable duplication is now explicit. `_MirrorStateVectorizer`
and `_SplineStateVectorizer`, their packed preconditioners, and their
primal/implicit Krylov drivers implement the same constraints twice. Preserve
small representation-specific coefficient maps, but use one solver vector
contract, one radius--lambda block preconditioner, and one SOLVAX-backed linear
solve. Once coefficient-native open fixed and free parity passes, remove the
nodal production solve and its compatibility paths; keep only nodal evaluation
fixtures.

Concrete branch budgets at merge:

- no more than 48 changed files and 13 mirror source modules;
- no increase above 8,000 mirror source lines at any accepted milestone, with
  a merge target below 7,500;
- no mirror source file above 1,000 lines;
- no more than 20 public mirror names, with removals preferred;
- exactly four canonical compact benchmark JSON files;
- at most three root examples and three compressed showcase figures;
- no generated run directories, dense arrays, notebooks, or uncompressed
  raster sequences in git.

Public `ntheta` has been removed from `MirrorResolution`; exterior quadrature
resolution stays independent. Do not add a public dealiasing knob unless a
refinement test demonstrates aliasing; if needed, overintegration is initially
internal and derived from `mpol`.

Docstrings state inputs, units, coordinate location, normalization, and failure
conditions in plain language. Comments explain non-obvious discretization or
physics decisions, not individual assignments.

## 6. Ordered implementation milestones

These milestones are sequential. A failed gate is fixed before downstream
examples or derivatives are promoted. Commit and push after each coherent
substep; inspect CI in batches rather than waiting after every push.

### M0. Evidence reset and API cleanup -- complete

1. Remove redundant public `ntheta`; derive exact collocation from `mpol`.
2. Mark all shaped benchmark records stale in their metadata until regenerated.
3. Add benchmark provenance fields for code SHA, schema, basis, represented
   modes, grid, hardware class, and promotion status.
4. Remove remaining example-only helpers and obsolete compatibility paths when
   a source owner already exists.
5. Run unit, API, import, Ruff, strict Sphinx, and example smoke tests.

Exit: one unambiguous resolution API and no benchmark that silently describes
a different discrete space.

### M1. Repair the independent strong-force diagnostic

1. Trace VMEC2000's half/full radial mesh placement through `forces.f`,
   `bcovar.f`, `residue.f90`, and `jxbforce.f`.
2. Write a short discretization note mapping every mirror field, pressure,
   metric, current, and derivative to its radial and axial location.
3. Replace mixed unstaggered reconstruction with conservative half-to-full
   interpolation and metric-consistent curl and pressure gradient.
4. Add exact polynomial manufactured cases with nonzero pressure, current,
   lambda, and nonaxisymmetric geometry. Add a separate closed-axis regular
   manufactured case; a regular cylinder alone cannot expose the first-row
   toroidal coordinate limit.
5. Report axis, physical interior, first radial row, end collar, and volume
   integral separately. Coordinate-singular endpoint samples must not dominate
   the physical norm, but no region may be silently dropped.
6. Demonstrate expected refinement order in `ns`, `mpol`, and axial knots on
   both open and circular periodic coordinates. The first active row must
   decrease independently; a bulk-only decrease is insufficient.

Exit: manufactured order is established and the physical strong force
decreases on three grids for accepted fixed equilibria.

### M2. Promote fixed-boundary open mirrors

1. Make clamped coefficient-native splines the production open fixed-boundary
   state. Retain CGL evaluation only as the independent parity representation.
2. Regenerate axisymmetric cylinder and flared-tube B-spline/CGL parity studies.
3. Regenerate two separate nonaxisymmetric studies at decreasing minor radius
   and beta: the Agren--Savenko SFLM and the 90-degree rotating ellipse.
   Compare section matrices, ellipticity/orientation, axis field, and field-line
   slopes only with the corresponding exact paraxial fixture.
4. Increase radial surfaces, spline knots, angular modes, and quadrature order
   independently; establish observed convergence rather than fitting one
   resolution.
5. Exercise finite current and lambda, positive Jacobian, nestedness, axis
   regularity, weak residual, and repaired strong force.
6. Revalidate forward and reverse derivatives only after the primal gates pass.
7. Regenerate `mirror_fixed_boundary.json` and the fixed-mirror root example.

Exit: axisymmetric and nonaxisymmetric fixed open mirrors are supported.

### M3. Unify solver maps and build one coupled preconditioner

1. Replace duplicate nodal/spline packing and gauge code with one coefficient
   map protocol used by primal, tangent, and adjoint solves.
2. Freeze a local approximate Hessian in radius and stream function, retaining
   `2 x 2` coupling blocks, neighboring radial coupling, and local axial spline
   support. Diagonalize periodic theta modes where symmetry permits.
3. Use released SOLVAX `v0.7.3` block/banded/operator/Krylov primitives behind
   the existing interface. Keep SciPy's host optimizer for the nondifferentiable
   CLI lane when it is faster; do not move mirror physics into SOLVAX.
4. Use the same operator and scaling for Newton, tangent, and transpose-adjoint
   systems. Apply gauges and fixed-end constraints before factorization.
5. A/B test no preconditioner, the current separable preconditioner, and the
   coupled preconditioner on fixed axisymmetric, fixed nonaxisymmetric, and
   circular periodic cases.
6. Remove the duplicate nodal production solver after spline parity and API
   tests pass. This milestone must reduce, not increase, source lines.

Exit gates: converged state unchanged within tolerance, true linear residual
at or below `1e-8`, bounded iteration growth under refinement, lower peak
memory, and either at least 2x runtime reduction or enabling a case previously
blocked by memory. Remove a new solver path if it cannot meet these gates.

### M4. Promote axisymmetric open free boundary through beta 50%

1. Represent boundary, interior geometry, and lambda with the same clamped
   axial spline coefficients used by fixed boundary. Evaluate them on the
   existing CGL/panel nodes and differentiate through that linear map.
2. Replace materialized full free-boundary Jacobians with JVP/VJP operator
   actions; keep a dense path only for tiny test systems.
3. Preserve the one cap-aware exterior BIE and separately monitor lateral
   tangency, pressure jump, raw/corrected cap compatibility, cap-rim continuity,
   endpoint shape constraints, and shape gauge.
4. Continue one physical equilibrium through beta
   `0, 0.01, 0.03, 0.10, 0.30, 0.50`, warm-starting only between adjacent
   values and recording retries.
5. Run three independently refined radial, axial, exterior, and angular grids;
   refine one family at a time before the combined study.
6. Compare 1%, 3%, and 10% with Pleiades. Compare high-beta radius expansion
   and on-axis field depression only to declared paraxial/diamagnetic trends.
7. Diagnose any beta-insensitive result by checking pressure normalization,
   enclosed volume, boundary work, field-energy balance, and profile
   interpolation before changing geometry.
8. Revalidate derivatives with respect to pressure amplitude and external-field
   scale; regenerate the free-boundary benchmark and example figures.

Exit: all beta states satisfy the residual and three-grid observable gates, or
the first failing beta becomes the documented supported limit. The target is
50%; a lower limit requires concrete numerical evidence and an explicit plan
amendment.

### M5. Promote the fixed-boundary closed hybrid

1. Establish exact circular-axis parity with ordinary `vmec_jax` and VMEC2000
   using matched boundary, pressure, current, resolution, and normalization.
2. Validate periodic spline position and first/second derivatives, Bishop frame,
   section Jacobian, symmetry maps, and join smoothness.
3. Continue from circular torus to racetrack, then lengthen the straight legs,
   and finally rotate/nonaxisymmetrically shape the return sections.
4. Compare a central long-leg window with the promoted fixed open B-spline
   mirror as curvature approaches zero.
5. Demonstrate nonzero rotational transform and end-to-end field lines without
   attributing vacuum-coil physics to the equilibrium solver.
6. Run fixed-LCFS beta continuation and show interior surface, `|B|`, iota,
   magnetic-well, weak-residual, strong-force, and convergence plots.
7. Revalidate forward/reverse derivatives and produce a periodic MOUT that is
   explicitly distinct from WOUT.
8. Regenerate `mirror_hybrid_fixed_boundary.json` and the root hybrid example.

Exit: the circular parity case, open-leg limit, racetrack continuation,
derivatives, and output round trip all pass.

### M6. Bounded nonaxisymmetric free-boundary attempt

This milestone starts only after M1--M4 pass.

1. Seed the free solve from the promoted weakly rotating fixed equilibrium.
2. Continue one parameter at a time: pressure, ellipticity, then rotation.
3. Run three grids with explicit limits of 1,000 nonlinear iterations, 30
   minutes wall time, and 8 GiB peak memory per state on the reference CPU;
   GPU results are supplementary until CPU parity is shown.
4. Require local nonaxisymmetric coefficients, observables, boundary
   diagnostics, and strong force to satisfy the same promotion contract.

Exit A: promote and regenerate `mirror_free_boundary_nonaxisymmetric.json`.
Exit B: store only a compact negative record with failure mode and resource
measurements, remove public/example scaffolding, and defer the lane.

### M7. Release reduction, documentation, and artifacts

1. Delete superseded benchmark runners, duplicate plotting helpers, temporary
   configurations, and generated outputs. At minimum remove the one-shot
   exterior endpoint runner after canonical data are regenerated and fold the
   standalone performance raster into a retained three-panel showcase.
2. Meet the line, file, API, benchmark, example, and image budgets in section 5.
3. Update the README capability table with `supported`, `research`, and
   `deferred` labels and showcase the three canonical workflows.
4. Document equations, coordinates, boundary conditions, spline spaces,
   normalization, residual definitions, continuation, implicit derivatives,
   MOUT schema, failure modes, and external validation. Remove current
   `supported` wording for free-boundary derivatives and examples until their
   primal spline lane passes the promotion gates.
5. Root examples remain parser-free scripts with parameters at the top and
   produce polished horizontal-mirror 3D geometry, visible equilibrium field
   lines, `|B|`, cross-sections, convergence histories, and relevant profiles.
6. Store no raw run trees. Commit only compact JSON/CSV evidence and at most
   three compressed final figures; CI regenerates smoke-resolution plots.
7. Run all tests, strict Sphinx, examples, packaging/import checks, CPU/GPU
   parity where available, and review the complete PR diff before marking the
   PR ready.

Exit: the draft PR contains only supported code and clearly labeled research
evidence, with no experimental public surface left behind.

## 7. Canonical artifacts and reporting

Canonical benchmark files:

1. `mirror_fixed_boundary.json`;
2. `mirror_free_boundary_axisymmetric.json`;
3. `mirror_free_boundary_nonaxisymmetric.json` only as promoted or explicitly
   negative evidence;
4. `mirror_hybrid_fixed_boundary.json`.

Every record includes commit, clean/dirty state, platform, precision, grid,
basis, represented modes, tolerances, iterations, wall time, peak memory,
variational/weak/strong residuals, geometry checks, observables, comparison
errors, derivative errors when applicable, and promotion status.

Every work report states:

- steps taken;
- results obtained, including failed gates;
- tests and hardware used;
- files changed and why ownership remains clear;
- best next steps;
- completion percentages for all open lanes;
- any concrete input needed from the user.

The plan changes only when evidence invalidates a gate or scope decision. A
plan amendment must cite the benchmark or source that caused the change and
replace, not append to, the affected decision.

## 8. Completion estimate at this audit

Percentages measure promotion evidence, not lines written:

| Lane | Complete | Main remaining evidence |
| --- | ---: | --- |
| Fixed open axisymmetric | 88% | repaired strong force and regenerated benchmark |
| Fixed open nonaxisymmetric | 68% | separate SFLM/rotating-ellipse refinement and force |
| Open fixed B-spline representation | 74% | become sole production state and remove nodal solve |
| Free open axisymmetric | 68% | spline coefficient solve, matrix-free coupling, beta refinement |
| Free open nonaxisymmetric | 28% | conditional three-grid promotion attempt |
| Fixed closed B-spline hybrid | 48% | VMEC parity, open-leg limit, force, derivatives |
| Strong-force diagnostic | 60% | closed-axis first-row limit and three-grid order |
| Coupled preconditioning | 35% | coupled radius--lambda block and A/B gates |
| Implicit differentiation | 70% | unify coefficient maps; rerun promoted primal lanes |
| Code/API simplification | 82% | remove duplicate nodal solver and meet budgets |
| Docs/examples/artifacts | 64% | regenerate evidence and release showcase |
| ESSOS ownership separation | 100% | preserve boundary; no coil code here |

Weighted completion of the four required release models is approximately 65%.
Free closed hybrid and ANIMEC are deferred and excluded from that percentage.

## 9. Primary references

- VMEC2000/ANIMEC source: <https://github.com/PrincetonUniversity/STELLOPT>
- VMEC++ paper: <https://arxiv.org/abs/2502.04374>
- DESC source and experimental branches: <https://github.com/PlasmaControl/DESC>
- DESC mirror branch: <https://github.com/PlasmaControl/DESC/tree/mirror>
- DESC cylindrical/Chebyshev branch:
  <https://github.com/PlasmaControl/DESC/tree/dd/cylindrical>
- DESC free-boundary formulation: <https://arxiv.org/abs/2412.05680>
- SOLVAX: <https://github.com/uwplasma/SOLVAX>
- GVEC G-frame paper: <https://arxiv.org/abs/2410.17595>
- GVEC documentation: <https://gvec.readthedocs.io/>
- Pleiades: <https://github.com/eepeterson/pleiades>
- Ågren and Savenko, *Theory of the straight field line mirror*, 32nd EPS
  Conference on Plasma Physics, ECA 29C, P-4.069 (2005):
  <https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf>
- Cooper, *Three-dimensional magnetohydrodynamic equilibria with anisotropic
  pressure*, Comput. Phys. Commun. 72 (1992):
  <https://doi.org/10.1016/0010-4655(92)90002-G>
- Cooper et al., *Three-dimensional anisotropic pressure free boundary
  equilibria*, Comput. Phys. Commun. 180 (2009):
  <https://doi.org/10.1016/j.cpc.2009.04.006>
- Rodríguez, Helander, and Goodman, *The maximum-J property in
  quasi-isodynamic stellarators*: <https://doi.org/10.1017/S0022377824000345>
- Feng et al., linked mirror concept: <https://arxiv.org/abs/2103.09457>
- Skene and Burns, automated spectral adjoints:
  <https://arxiv.org/abs/2506.14792>
- VEPEC technical report: <https://www.osti.gov/biblio/6351313>
- Goodman, Freidberg, and Lane, analytic long-thin mirror equilibria:
  <https://doi.org/10.1063/1.865851>
- Beklemishev, diamagnetic bubble equilibria:
  <https://arxiv.org/abs/1606.05454>
- JAX custom derivative and implicit-iteration guide:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- JAXopt implicit root differentiation:
  <https://jaxopt.github.io/stable/_autosummary/jaxopt.implicit_diff.custom_root.html>
- Lineax operators and transpose-aware solvers: <https://docs.kidger.site/lineax/>

The reference list supports decisions; only reproduced tests and compact
benchmark records count as release evidence.
