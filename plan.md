# Mirror equilibrium release plan

Status: final authoritative plan for draft PR #22. This file supersedes the
original `/Users/rogeriojorge/Downloads/plan_mirror.md` and every earlier
version of this plan. Do not create parallel roadmaps. Commits, tests, and the
four compact benchmark JSON files are the execution log.

Audit baseline (2026-07-14 CDT / 2026-07-15 UTC, final source review):

- audit started from branch `codex/mirror-geometry` at `43832c0d`, with four
  active evidence, documentation, plan, and solver-policy edits;
- base `origin/main` at `ed4ac7ac`, zero commits behind and 311 ahead;
- PR #22 is open, draft, and mergeable;
- pushed diff is 51 files, 16,795 insertions, and 1,608 deletions;
- `vmec_jax/mirror` contains 13 modules, exactly 8,000 lines, and 20 lazy public
  names;
- mirror tests contain ten substantive owner-aligned files and 4,456 lines;
- the coefficient-API worktree passes 106 tests and skips 10 in 321.53 seconds;
  the later regional-force change passes its focused tests; strict Sphinx,
  changed-file pre-commit, and `git diff --check` pass;
- the final always-matrix-free policy audit passes 105 tests and skips 10 in
  359.28 seconds. Its sole failure is the legacy nodal test that requires zero
  Krylov iterations above the old dense threshold; the new path uses 441
  iterations and reaches true linear residual `8.89e-11`. T2 removes or
  migrates that obsolete algorithm-choice assertion;
- public `ntheta` has been removed. `mpol` denotes the highest retained angular
  Fourier mode and the grid derives `ntheta = 2 * mpol + 1` exactly;
- the repaired strong force reconstructs covariant field and pressure from the
  radial Gauss cells. Its exact cylindrical fixture is second order. On the
  current-free circular spline torus the first-row normalized residual is
  `8.19e-6`, `2.04e-6`, and `5.09e-7` for `ns=5,9,17`, respectively.
- public ``solve_fixed_boundary_cli`` and its public state, boundary, and
  result types are now coefficient-native clamped splines. The CGL fixed solve
  and its custom-VJP wrapper are internal migration references pending M2
  deletion.
- axisymmetric finite-beta spline/CGL parity passes three grids. The rotating
  ellipse and SFLM do not: strong force approaches a nonzero floor and doubles
  when minor radius is halved. Fixed nonaxisymmetric support remains blocked
  until M3.
- direct projection of the analytic SFLM vacuum field into the production
  Clebsch representation reconstructs the Cartesian field to `3.91e-4`
  relative RMS and gives all-volume strong force `4.42e-3`. Starting the
  nonlinear solve from that projected state reduces the previous SFLM force
  from `51.7` to `4.08e-2`, but the final linear residual is `0.842`. This
  isolates initialization and preconditioning as the next blockers rather
  than disproving the field representation or strong-force kernel.
- the environment currently uses SOLVAX `0.7.3`; released `0.8.3` adds pytree
  GMRES, matrix-free Newton--Krylov, cyclic tridiagonal solves, and additive
  preconditioners. These are candidates for measured reuse, not an automatic
  dependency upgrade.

Execution update (2026-07-15): T1 enforced the matrix-free open policy and
compacted its evidence; T2 removed the nodal fixed solver, custom VJP, and
duplicate tests; T3 added and validated the differentiable supplied-field
initializer and made it the default SFLM path. The tree now contains 7,497
mirror source lines, 3,970 mirror-test lines, and 20 lazy public names. T4 is
the active tranche.

The branch contains real equilibrium solvers and useful validation, but it is
not release-ready. The immediate blockers are dense free-boundary Jacobian
storage, a preconditioner that omits radius--stream-function coupling, stale
shaped benchmarks, the artificial open-end collars, and incomplete
closed-hybrid limiting cases. The milestones below remove those blockers in
dependency order and define explicit stop conditions so this PR has a finite
end.

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

Differentiability is part of a supported model, but it follows the primal
physics gates. The fast CLI may use SciPy host optimization and need not be
differentiable. A separate implicit layer differentiates the converged
coefficient residual. Unrolled differentiation through nonlinear iterations,
host callbacks presented as end-to-end JAX solves, and derivatives of an
unconverged state are outside the release contract.

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
  manufactured tests and decreases under physical refinement. For the open
  lanes, the finest all-volume value must be below `5e-2`, the central core
  below `2e-2`, and Richardson extrapolation must be consistent with zero;
- positive Jacobian, nested surfaces, adequate self-clearance, and normalized
  `div(B)` near roundoff;
- physical observables stable on three independently refined grids, assessed
  by observed order or Richardson extrapolation and a predeclared tolerance;
- for open free boundary, separately reported area-weighted `B.n`, total
  pressure jump, and artificial-cap compatibility residuals;
- an analytic, asymptotic, or independent-code comparison;
- forward and reverse implicit derivatives for every advertised control,
  checked against reconverged centered finite differences after the primal
  lane passes all preceding gates. The true primal or transpose linear
  residual must be at most `1e-8`;
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

For a symmetric mirror, both cuts receive the same axisymmetric prescribed
section and compatible through-flux, with opposite outward normals. More
general fixed-boundary fixtures may prescribe different end sections, but the
values and stream data are explicit Dirichlet continuation data, not equations
invented by the optimizer. Moving the cuts outward while retaining the same
central physical field must leave central observables unchanged to the
declared refinement tolerance. The artificial disks used to close the exterior
BIE carry only Neumann compatibility data; total-pressure and tangency
conditions are enforced on the lateral plasma interface, not on those disks.

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
refinement, cut-location independence, and shape derivatives. Do not create a
second spline-specific BIE. Regional force masks diagnose the central 80% and
outer 20% collars separately, but the all-volume norm remains a release gate.

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

Requested beta is defined against the vacuum reference field stated in each
benchmark. Achieved on-axis and volume-averaged beta are reported separately.
The scalar model may be numerically supported at beta 50%, but it must not be
described as a quantitative model of high-beta mirror experiments whose
pressure is strongly anisotropic.

## 3. Current evidence ledger

### 3.1 Accepted evidence

- Axisymmetric fixed boundary reaches `ftol = 1e-12` and passes cylinder and
  flared-tube analytic checks, three-grid spline/CGL parity, an independent
  weak residual, physical observable refinement, and implicit derivative
  tests.
- The coefficient-native open spline representation reproduces CGL geometry,
  energy, and fields with decreasing error while using fewer active axial
  unknowns.
- Spline reverse derivatives agree with reconverged finite differences at
  about `3e-10`, and the forward implicit tangent test passes on small fixed
  systems. This validates the method, not any primal lane that still fails
  force balance.
- A finite-beta radial pressure-balance manufactured case and the VMEC-style
  half-to-full reconstruction show second-order radial convergence for a
  cylindrical polynomial state with nonzero pressure, current, and lambda.
  The current-free circular first-row force decreases by about four on each
  radial refinement from `ns=5` through 17.
- A nonaxisymmetric shaped coordinate map with uniform Cartesian field has
  force below `1e-12`.
- The coefficient-native SFLM field initializer accepts callable or sampled
  Cartesian fields, infers the analytic flux within `6.3e-5` relative, passes
  a field-amplitude JVP, and reaches field/force errors of `3.42e-4`/`5.12e-3`.
- Axis regularity now enforces a single-valued axis `|B|` and a consistent
  derivative pullback.
- The existing axisymmetric free-boundary solve reaches small variational,
  weak, interface, and divergence residuals and shows the expected expansion
  and field depression. Its old strong-force rows predate the accepted M1
  reconstruction and must be regenerated before promotion.
- Periodic cubic racetrack geometry, Bishop transport, circular geometry, and
  field tracing have focused tests.
- Pleiades data are independently generated from commit
  `0161abb3f9c9e0885f5c739d6afec55cb73de733` and provide low-beta external-code
  evidence at 1%, 3%, and 10%.

### 3.2 Evidence that must be regenerated

All shaped benchmark JSON generated before the axis-regularity correction,
exact `mpol` semantics, or accepted M1 strong-force reconstruction is stale.
In particular, old records that declared `mpol = 1` with five or six angular
nodes represented a different coefficient space. The fixed-open compact file
has been rewritten with current axisymmetric evidence and explicit negative 3D
evidence; free-boundary and hybrid records still require regeneration. Keep
the four canonical filenames, but never carry a positive status forward from
an incompatible discretization.

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
- The old continuation basin is demonstrably wrong for the nonaxisymmetric
  vacuum fixtures. On three grids, rotating-ellipse all-volume strong force is
  `15.28, 12.54, 12.09`; SFLM is `59.68` and `51.71`, with a crossed-Jacobian
  medium run. Halving minor radius makes both worse.
- The production spline SFLM projection is the decisive counterexample: field
  reconstruction error is `3.42e-4`, tangency RMS is `1.33e-4`, and strong
  force is `5.12e-3`. The compact nonlinear continuation reaches force
  `3.52e-2`, field-direction cosine above `0.9999996`, and strict variational
  tolerance, but drifts away from the better physical seed. The representation
  and initializer are viable; the coupled preconditioner is not yet viable.
- The periodic preconditioner stalled at 3,000 GMRES iterations with linear
  residual about 0.136. CG/MINRES reached only about 0.016. The scalable closed
  primal path therefore remains disabled.
- Historical circular-hybrid force values predate the accepted M1
  reconstruction and exact `mpol` semantics. They are invalidated rather than
  interpreted; the circular parity study must be regenerated in M5.
- There is no accepted hybrid VMEC parity case, open-leg limit, release MOUT,
  or hybrid implicit-derivative benchmark.

Failed diagnostics must remain visible. Do not tune tolerances, omit boundary
collars, or change normalizations after seeing results without documenting the
reason and rerunning all comparison grids.

## 4. Conclusions from external sources

### 4.1 VMEC2000, VMEC++, and ANIMEC

VMEC2000 source at local commit `728af8bd6c79` was reviewed through
`forces.f`, `residue.f90`, `bcovar.f`, `fbal.f`, `jxbforce.f`, `precon2d.f`,
and the block-tridiagonal solvers. It separates raw force assembly from
preconditioned residuals and uses radial staggering consistently.
`precon2d.f` preserves neighboring radial and field-component coupling instead
of reducing the system to scalar diagonal scaling. These are the primary
references for the mirror strong-force reconstruction and coupled
preconditioner.

The concrete `jxbforce.f` pattern is the acceptance reference: covariant field
is stored on half surfaces, radial differences place current on full interior
surfaces, `sqrt(g) B^u` and `sqrt(g) B^v` are averaged before division by an
averaged Jacobian, and pressure uses a half-to-full difference. Axis and edge
values receive explicit one-sided treatment. VMEC's `fsqr/fsqz/fsql` remain
preconditioned variational residuals and are not interchangeable with this
pointwise diagnostic.

VMEC's fixed-boundary condition `B.n=0` applies to a closed material LCFS and
must not be copied onto open mirror cuts. VMEC2000 and VMEC++ cannot validate
open topology. They are independent references only for the circular
closed-hybrid limit and matched toroidal inputs. VMEC++ adds operational
lessons: restartable continuation, explicit residual history, bounded retries,
and broad VMEC2000 parity.

Cooper's 1992 variational formulation, the 2009 free-boundary ANIMEC paper,
later LHD reports, and the `_ANIMEC` source paths confirm the scope decision in
section 2.3. The source computes `p_parallel(s,B)` and `p_perpendicular`, uses
`sigma = 1 + (p_perpendicular-p_parallel)/(B^2/mu0)` in the effective current
`curl(sigma B)`, and adds profile derivatives evaluated at fixed `B` to radial
force balance. Its free-boundary condition is continuity of
`p_perpendicular+B^2/(2 mu0)`, not scalar pressure. ANIMEC remains a future
physics model, not a pressure-array option inside this scalar solver.

### 4.2 DESC branches

The public DESC `mirror` (`0dba071d`, 2025-09-12) and
`mirror_anisotropy` (`805b77fc`, 2025-11-12) branches provide useful
prototypes for Chebyshev--Zernike coordinates, end-cap constraints, and
continuation. They are not validated production references:

- mirror equilibrium and objective logic is largely duplicated in large
  branch-only modules;
- `FixEndCapR/Z` contains placeholder behavior and explicit mode selection is
  not implemented;
- the nominal mirror boundary-condition test contains imports rather than
  assertions;
- substantial upstream tests are renamed or disabled on the branch;
- notebook and binary artifacts account for branch diffs above one million
  inserted lines and make direct code transfer undesirable.

DESC's newer `dd/cylindrical` branch (`6f85f50a`, 2026-06-26) implements
`DoubleChebyshevFourierBasis`: two nonperiodic Cartesian Chebyshev coordinates
and one periodic Fourier coordinate. It has initial basis tests but no mirror
equilibrium, end-cap, axis-regularity, free-boundary, or force-refinement
evidence. The `finite_element_basis` branches last changed in 2024 and add
experimental scikit-fem paths, debugging output, and incomplete JAX
integration. Use formulas from these branches as review material only. None is
a reason to replace the compact Fourier/radial/longitudinal-spline tensor
product in this PR.

DESC's released free-boundary work is more relevant conceptually: on a closed
plasma--vacuum interface, tangency, total-pressure jump, and any sheet-current
condition are distinct equations. These conditions apply only to the mirror's
lateral LCFS. DESC's smooth-toroidal singular quadrature does not remove the
open cap-rim singularity, which is why the current open exterior formulation
remains separate.

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
test and a first-order ellipticity correction, not an exact finite-beta
equilibrium benchmark. Goodman--Freidberg--Lane expands simultaneously in beta
and inverse aspect ratio and predicts additional quadrupole/diamond distortion;
that supplies section-shape observables at low beta and thin radius. The
rotating ellipse must first match its own vacuum paraxial limit as minor radius
and beta approach zero, then be tested away from that limit.

The linked-mirror configuration of Feng et al. supports the closed topology:
two straight mirror sections joined by two half-tori, with nonparallel
sections producing transform. The 1993 Ilgisonis--Berk--Pastukhov report gives
a separate finite-beta linked-quadrupole asymptotic model and predicts
nonlinear outward displacement through order `(beta/(epsilon E))^2`. These are
geometry and trend references; neither prescribes the boundary representation
or establishes 3D numerical MHD convergence for this code.

Rodríguez, Helander, and Goodman's maximum-`J` paper contains useful paraxial
near-axis mirror equations in its appendix; it is not itself a
straight-field-line-mirror construction.

GVEC validates several architectural choices--coefficient-native B-splines,
independent quadrature, Fourier periodic angles, and a general transported
G-frame--but its published splines are radial rather than longitudinal. Its
G-frame study reduced a strongly shaped case from Fourier `(10,15)` to `(2,10)`
and from 16,000 to 800 iterations. It supports the Bishop-like hybrid frame and
the requirement for valid initial maps, not direct reuse of a GVEC
representation. VEPEC provides historical precedent for vector-potential
variables and divergence-preserving tricubic spline interpolation in high-beta
minimum-`B` mirror studies.

Goodman, Freidberg, and Lane expand simultaneously in beta and the long-thin
parameter and compare section distortions with VEPEC. Their formulas are an
asymptotic low-beta/thin-tube gate, not a beta-50 parity result. Near-unity-beta
diamagnetic-bubble and RealTwin studies generally require anisotropic or
kinetic closures, so they cannot validate the scalar-pressure model
quantitatively.

Pleiades remains the only independent open-mirror numerical comparison in the
current branch. It solves an axisymmetric Grad--Shafranov problem, so compare
LCFS radius, axis field, mirror ratio, and beta response on matched inputs, not
internal coefficients. RealTwin's high-field tandem-mirror study and
diamagnetic-bubble literature provide qualitative high-beta context, including
pressure anisotropy and difficult near-unity-beta behavior; neither is
scalar-pressure numerical parity. Beta 50% is a demanding validation point,
not a claim of a diamagnetic-bubble model.

### 4.4 SOLVAX and differentiation

The branch currently runs with SOLVAX `v0.7.3`. Released `v0.8.3`
(`a904ac20`, 2026-07-14) additionally provides pytree GMRES, matrix-free
Newton--Krylov, cyclic tridiagonal solves, symmetric additive
preconditioners, and elliptic helpers. `vmec_jax` already uses SOLVAX block
Thomas in the toroidal and mirror paths. Generic linear algebra belongs in
SOLVAX, but mirror-specific geometry, residuals, gauges, coefficient maps,
continuation, and physics preconditioners stay here.

Do not depend on unpublished local SOLVAX commits. Evaluate `v0.8.3` in a
temporary environment against the existing mirror operator before changing the
project requirement. A prior direct replacement of the host GMRES path did not
meet runtime/residual gates. Adopt a released primitive only after a
reproducible A/B benchmark shows equal state, lower memory or runtime, and no
loss in true linear residual. Otherwise retain the current narrow calls.

The production derivative strategy remains:

- solve the nonlinear equilibrium to the declared primal tolerance;
- differentiate the converged residual with exact JAX JVP/VJP actions;
- use forward implicit tangents for few controls;
- use reverse adjoints for scalar outputs with many controls;
- solve primal and transpose systems with the same physical preconditioner;
- never reverse-differentiate through the nonlinear iteration history;
- validate against fully reconverged centered finite differences and report
  both nonlinear and linear residuals.

JAX `custom_vjp`, `lax.custom_root`, and `linearize`, JAXopt `custom_root`,
Optimistix, and Lineax all encode variants of the same implicit-function solve.
Lineax offers useful operator abstractions and transpose-aware solver state,
while Optax supplies first-order optimizers and Equinox supplies PyTree
modules; none provides the missing mirror block physics. The Skene--Burns
spectral-adjoint method confirms that sparse operator graphs and transposed
solves are the scalable reverse-mode design, but its symbolic Dedalus graph is
not portable here. The research-grade choice is therefore exact JAX residual
JVP/VJP actions plus an implicit primal/transpose solve, not unrolled AD. Add no
new optimization or linear-solver dependency in this PR.

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

The source audit identifies the exact reductions:

- delete the 244-line private nodal fixed solve and the nodal fixed custom-VJP,
  configuration, adjoint, and duplicate tests in `solver.py` and `implicit.py`;
- keep `_MirrorStateVectorizer` only until free boundary is coefficient-native,
  then replace it with the spline coefficient map and delete it;
- split the 342-line free-boundary workflow into residual assembly, operator
  solve, and result assembly without creating new public modules;
- replace the free solver's materialized Jacobian columns with JVP/VJP actions;
- retain one exterior BIE. `exterior.py` owns closed panels and quadrature;
  `exterior_bie.py` owns the Laplace solve and field coupling;
- retain externally supplied `coil_xyz` only as optional output metadata.
  Remove in-repository coil constructors, ESSOS benchmark runners, and
  Biot--Savart formulas after canonical evidence is regenerated. The root
  integration example may import ESSOS explicitly;
- remove compatibility aliases and private imports from root examples. A root
  example uses the public equilibrium API and submodule diagnostic kernels only
  when no public workflow exists.

Concrete branch budgets at merge:

- no more than 46 changed files and 13 mirror source modules;
- no increase above 8,000 mirror source lines at any accepted milestone, with
  a merge target below 7,200;
- fewer than 4,000 mirror test lines after duplicate nodal tests are removed;
- no mirror source file above 1,000 lines;
- no more than 18 public mirror names, with removals preferred;
- exactly four canonical compact benchmark JSON files;
- at most three root examples and three compressed showcase figures;
- no generated run directories, dense arrays, notebooks, or uncompressed
  raster sequences in git.

No new module is added unless the same commit deletes at least as much obsolete
code and the ownership boundary is clearer. Refactors that do not reduce a
duplicate implementation, public API, or measured complexity are deferred.

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

### M1. Repair the independent strong-force diagnostic -- complete

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

### M2. Finish the coefficient path and physical initialization -- steps 1--5 complete

1. Commit the compact current fixed-boundary evidence only after JSON, focused
   tests, strict docs, and the complete mirror suite pass on the active solver
   policy.
2. Delete the private nodal fixed solve, nodal custom-VJP, nodal fixed adjoint
   configuration, and duplicate live-solve tests. Retain CGL operators and
   evaluated-state parity fixtures only.
3. Add one spline-owned initializer that projects a supplied Cartesian vacuum
   field onto the mirror Clebsch variables. It accepts sampled field values or
   a callable; it does not construct coils or perform Biot--Savart integration.
4. Pin the initializer with the analytic SFLM: reconstructed field relative
   RMS below `5e-4`, all-volume strong force below `6e-3`, correct flux, finite
   lambda, and positive geometry.
5. Use the projected initializer in the root SFLM example and continuation.
   Retain the failed homothetic continuation as compact negative evidence, not
   as a default path.
6. Add an equal-axisymmetric-end fixture and a cut-location study so the
   central solution is demonstrably independent of the artificial collars.

Exit: one public coefficient-native fixed solver remains; the analytic SFLM
initializer is a binary regression test; source and test line counts decrease.

### M3. Build the coupled solver and promote fixed open mirrors

1. Define one coefficient map used by fixed/free primal, tangent, and adjoint
   solves. Radius, stream function, gauge, fixed ends, and scaling are explicit
   blocks rather than unrelated flat slices.
2. Freeze a local approximate Hessian retaining radius--stream `2 x 2` blocks,
   neighboring radial coupling, local axial B-spline support, and Fourier-mode
   coupling required by nonaxisymmetric geometry. Use a block-Jacobi or
   block-tridiagonal factorization as the first implementation; add a more
   elaborate multilevel method only if the measured spectrum requires it.
3. A/B test the current SciPy/SOLVAX `0.7.3` path and released SOLVAX `0.8.3`
   pytree GMRES/Newton--Krylov in an isolated environment. Keep SciPy's host
   optimizer for the nondifferentiable CLI when faster. Upgrade the dependency
   only if the release gates improve.
4. Use the same residual action, scaling, preconditioner, and transpose action
   for Newton, forward tangent, and reverse adjoint solves. Apply gauges and
   fixed-end constraints before factorization.
5. A/B test no preconditioner, the current separable preconditioner, and the
   coupled preconditioner on axisymmetric open, analytic-seeded SFLM, rotating
   ellipse, and circular periodic cases. Record cold/warm runtime, peak memory,
   nonlinear iterations, Krylov iterations, and true linear residual.
6. Rerun SFLM and rotating-ellipse studies with independent refinement of
   `ns`, `mpol`, axial knots, and quadrature, followed by a three-grid combined
   refinement and half-radius paraxial study.
7. Require finite current and lambda, positive Jacobian, nestedness, axis
   regularity, weak residual, repaired strong force, section matrix,
   ellipticity/orientation, axis field, and field-line slope gates.
8. Only after the primal gates pass, rerun forward and reverse implicit
   derivatives and regenerate `mirror_fixed_boundary.json` and the root
   fixed-mirror figures.

Exit gates: both axisymmetric and nonaxisymmetric fixed open mirrors satisfy
section 1.1; true linear residual is at most `1e-8`; Krylov growth is bounded;
the coupled path is at least 2x faster or enables a previously blocked case;
and the milestone has a net source-line reduction. Remove any new solver path
that cannot meet these gates.

### M4. Promote axisymmetric open free boundary through beta 50%

1. Represent boundary, interior geometry, and lambda with the same clamped
   axial spline coefficients used by fixed boundary. Evaluate them on the
   existing CGL/panel nodes and differentiate through that linear map.
2. Replace `_MirrorStateVectorizer` and materialized full free-boundary
   Jacobian columns with the shared coefficient map and JVP/VJP operator
   actions; keep a dense path only for tiny test systems.
3. Preserve the one cap-aware exterior BIE and separately monitor lateral
   tangency, pressure jump, raw/corrected cap compatibility, cap-rim continuity,
   equal symmetric end data, endpoint shape constraints, cut-location
   independence, and shape gauge.
4. Continue one physical equilibrium through beta
   `0, 0.01, 0.03, 0.10, 0.25, 0.50`, warm-starting only between adjacent
   values and recording retries. The external field comes from an ESSOS
   callable or MGRID; no coil representation enters mirror source code.
5. Run three independently refined radial, axial, exterior, and angular grids;
   refine one family at a time before the combined study.
6. Compare 1%, 3%, and 10% with Pleiades. Compare high-beta radius expansion
   and on-axis field depression only to declared paraxial/diamagnetic trends,
   including `B/B_vacuum approximately sqrt(1-beta)` where its ordering is
   valid.
7. Diagnose any beta-insensitive result by checking pressure normalization,
   enclosed volume, boundary work, field-energy balance, and profile
   interpolation before changing geometry.
8. Revalidate derivatives with respect to pressure amplitude and external-field
   scale; regenerate the free-boundary benchmark and example figures.

Exit: all beta states satisfy variational, weak, strong-force, interface,
geometry, and three-grid observable gates, or the first failing beta becomes
the documented supported limit. The target is 50%; a lower limit requires
concrete numerical evidence and an explicit plan amendment. High-beta results
are labeled scalar-pressure equilibria, not ANIMEC or kinetic predictions.

### M5. Promote the fixed-boundary closed hybrid

1. Establish exact circular-axis parity with ordinary `vmec_jax` and VMEC2000
   using matched boundary, pressure, current, resolution, and normalization.
2. Keep the entire longitudinal representation in periodic cubic B-spline
   coefficients: centerline, transported frame controls, section amplitudes,
   and stream function. Fourier is used only in the periodic cross-section
   angle; no longitudinal Fourier projection is introduced.
3. Validate periodic spline position and first/second derivatives, Bishop
   frame and holonomy correction, section Jacobian, clearance, up--down and
   leg-exchange coefficient symmetries, and join smoothness.
4. Continue from circular torus to a racetrack with two curved returns, then
   lengthen both straight legs, and finally impose the rotating-ellipse
   stellarator shaping in the returns. At least half of each straight leg must
   remain below the declared curvature tolerance.
5. Compare a central long-leg window with the promoted fixed open B-spline
   mirror as curvature approaches zero.
6. Demonstrate nonzero rotational transform and circuit-spanning field lines
   from the solved equilibrium, without
   attributing vacuum-coil physics to the equilibrium solver.
7. Run fixed-LCFS beta continuation and show interior surface, `|B|`, iota,
   magnetic-well, weak-residual, strong-force, and convergence plots.
   Compare low-beta displacement/section trends with the linked-mirror report
   only in its asymptotic regime.
8. Revalidate forward/reverse derivatives and produce a periodic MOUT that is
   explicitly distinct from WOUT.
9. Regenerate `mirror_hybrid_fixed_boundary.json` and the root hybrid example.

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

### Finite commit sequence

This is the execution order. A tranche is committed and pushed only after its
listed gate passes; failed experiments are removed in the same tranche or
recorded as compact negative evidence.

| Tranche | Change | Required gate |
| --- | --- | --- |
| T1 (complete) | Validate and commit the active force-region, compact benchmark, docs, and matrix-free policy edits | full mirror suite, strict Sphinx, Ruff, JSON schema |
| T2 (complete) | Delete nodal fixed solve/custom-VJP and redundant tests | public API/import tests; axisymmetric spline parity unchanged; net line reduction |
| T3 (complete) | Add supplied-field-to-Clebsch spline initializer and use it for SFLM | analytic field error `<5e-4`; force `<6e-3`; example smoke |
| T4 | Implement and A/B the coupled radius--stream preconditioner, including SOLVAX `0.8.3` trial | true linear residual `<=1e-8`; bounded Krylov growth; measured benefit |
| T5 | Regenerate rotating-ellipse and SFLM fixed-boundary evidence | all section 1.1 fixed-open gates on three grids and half-radius study |
| T6 | Move free boundary to the shared spline coefficient map and operator Jacobian | dense tiny-case parity; no full Jacobian above threshold; lower peak memory |
| T7 | Regenerate the axisymmetric beta scan through 50% | three-grid physics, interface, force, Pleiades, and trend gates |
| T8 | Establish circular hybrid parity and long-leg/open limit | ordinary VMEC-JAX and VMEC2000 parity; force refinement |
| T9 | Promote the spline racetrack/rotating-return hybrid | positive map, nonzero iota, beta profiles, derivatives, MOUT round trip |
| T10 | Run the bounded nonaxisymmetric free-boundary attempt | promotion or explicit negative record within stated budget |
| T11 | Remove ESSOS runner, compatibility paths, redundant figures/tests, and stale records | section 5 repository budgets |
| T12 | Regenerate README/docs/examples and perform final release audit | all local/CI gates green; PR diff reviewed; draft removed only then |

No additional physics lane enters this PR. ANIMEC, free-boundary hybrid,
arbitrary curved open axes, stability, and mirror Boozer work begin only in a
new plan after T12.

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
| Fixed open axisymmetric | 100% | maintain gates while shared solver code changes |
| Fixed open nonaxisymmetric | 65% | coupled solve and three-grid force |
| Open fixed B-spline representation | 98% | equal-end/cut-location evidence in M2 |
| Free open axisymmetric | 65% | spline coefficient solve, operator coupling, regenerated force/beta study |
| Free open nonaxisymmetric | 25% | conditional three-grid promotion attempt after M4 |
| Fixed closed B-spline hybrid | 45% | current-semantics VMEC parity, open-leg limit, force, derivatives |
| Strong-force diagnostic | 100% | maintain gates in promoted equilibrium lanes |
| Coupled preconditioning | 35% | coupled radius--lambda block and A/B gates |
| Implicit differentiation | 72% | unify coefficient maps; rerun promoted primal lanes |
| Code/API simplification | 82% | reach source-line and public-API budgets in T6/T11 |
| Docs/examples/artifacts | 65% | regenerate remaining release showcase artifacts |
| ESSOS ownership separation | 85% | remove coil benchmark runner/formulas; retain callable integration only |

Weighted completion of the four required release models is approximately 68%.
Free closed hybrid and ANIMEC are deferred and excluded from that percentage.

## 9. Primary references

- VMEC2000/ANIMEC source: <https://github.com/PrincetonUniversity/STELLOPT>
- VMEC++ paper: <https://arxiv.org/abs/2502.04374>
- DESC source and experimental branches: <https://github.com/PlasmaControl/DESC>
- DESC mirror branch: <https://github.com/PlasmaControl/DESC/tree/mirror>
- DESC cylindrical/Chebyshev branch:
  <https://github.com/PlasmaControl/DESC/tree/dd/cylindrical>
- DESC experimental finite-element branch:
  <https://github.com/PlasmaControl/DESC/tree/finite_element_basis>
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
- Asahi et al., *MHD equilibrium analysis with anisotropic pressure in LHD*:
  <https://www.jstage.jst.go.jp/article/pfr/6/0/6_0_2403123/_article>
- Rodríguez, Helander, and Goodman, *The maximum-J property in
  quasi-isodynamic stellarators*: <https://doi.org/10.1017/S0022377824000345>
- Feng et al., linked mirror concept: <https://arxiv.org/abs/2103.09457>
- Ilgisonis, Berk, and Pastukhov, finite-beta toroidally linked mirrors:
  <https://doi.org/10.2172/10179323>
- Skene and Burns, automated spectral adjoints:
  <https://arxiv.org/abs/2506.14792>
- VEPEC technical report: <https://www.osti.gov/biblio/6351313>
- Goodman, Freidberg, and Lane, analytic long-thin mirror equilibria:
  <https://doi.org/10.1063/1.865851>
- Savenko and Ågren, finite-beta SFLM ellipticity:
  <https://doi.org/10.1063/1.2401153>
- Beklemishev, diamagnetic bubble equilibria:
  <https://arxiv.org/abs/1606.05454>
- JAX custom derivative and implicit-iteration guide:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- JAXopt implicit root differentiation:
  <https://jaxopt.github.io/stable/_autosummary/jaxopt.implicit_diff.custom_root.html>
- Lineax operators and transpose-aware solvers: <https://docs.kidger.site/lineax/>

The reference list supports decisions; only reproduced tests and compact
benchmark records count as release evidence.
