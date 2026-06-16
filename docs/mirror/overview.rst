Mirror Geometry Overview
========================

The mirror lane targets fixed-boundary, open-ended ideal-MHD equilibria in
coordinates ``(s, theta, xi)``.  The axial coordinate ``xi`` is nonperiodic and
uses Chebyshev-Gauss-Lobatto nodes in increasing physical order; ``theta`` is
periodic and uses a real Fourier representation.  This is intentionally not a
large-aspect-ratio torus and does not store results as classic ``wout`` files.

The current mirror package provides the fixed-boundary axisymmetric path and
validation surfaces needed to grow the backend without coupling it to toroidal
VMEC assumptions:

- ``vmec_jax.mirror`` as a domain package;
- static mirror resolution/configuration objects;
- Chebyshev-Gauss-Lobatto nodes, differentiation matrices, interpolation, modal
  filtering, and Clenshaw-Curtis quadrature;
- uniform-theta Fourier grids, derivatives, and quadrature;
- axisymmetric fixed side boundaries, state projection, metric and Jacobian
  kernels for straight-axis cylinder/flared tubes;
- theta-dependent radius side boundaries, 3D state projection, and
  nonaxisymmetric cylindrical-radius metric, field, and energy kernels;
- scalar radial profiles, contravariant/covariant/cartesian magnetic-field
  kernels, and magnetic/pressure energy integrals;
- differentiable axisymmetric energy wrappers, projected residuals, and
  manufactured-solution source helpers;
- an experimental fixed-boundary axisymmetric projected-gradient solve path
  with pressure-continuation trace diagnostics;
- mirror-native ``mout_*.nc`` read/write helpers, plot-data extraction, PNG
  writing, ``.npz``/CSV export helpers, and ``vmec --plot mout_*.nc`` dispatch;
- WHAM-inspired circular-loop fixture metadata, deterministic vacuum-field
  reference checks, optional ``magpylib`` comparison hooks, and low-resolution
  runnable examples;
- focused tests for node ordering, polynomial exactness, interpolation, filtering,
  theta orthogonality, analytic axisymmetric geometry, field identities, and
  analytic energy, gradient checks, Hessian symmetry, MMS stationarity, I/O
  roundtrip, plotting numerical content, WHAM fixture parity, and example
  smoke coverage.

Later phases add the nonaxisymmetric fixed-boundary solve path, mirror
straight-field-line diagnostics, and optimization workflows.
