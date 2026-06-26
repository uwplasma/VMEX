# vmec_jax package map

This directory contains the Python package.  Keep new code in a domain folder
when possible; root-level modules are for public facades, compact physics
building blocks, or compatibility entry points.

## Public facades

- `api.py`: curated public imports.
- `driver.py`: high-level VMEC workflows used by Python callers and the CLI.
- `cli.py` and `__main__.py`: command-line entry points.
- `solve.py`, `wout.py`, `free_boundary.py`, `free_boundary_adjoint.py`, and
  `free_boundary_adjoint_controller.py`: compatibility facades for heavily used
  solver and validation APIs.  Prefer adding implementation code under
  `solvers/` or `io/` and re-exporting here only when the API is public.

## Domain folders

- `solvers/`: fixed-boundary and free-boundary solver implementation.
- `optimizers/`: reusable optimization algorithms and residual builders.
- `external_fields/`: coil, mgrid, and ESSOS field providers.
- `io/`: persisted VMEC data formats, especially WOUT netCDF helpers.
- `drivers/`: implementation helpers for `driver.py`; these are not general
  input/output utilities.
- `data/`: package-bundled tiny data used by import-time or CLI helper code.

## Root physics modules

Small standalone physics modules may remain at the root when they are commonly
imported directly, for example `quasisymmetry.py`, `finite_beta.py`,
`profiles.py`, `boundary.py`, and `field.py`.  Large topic families should move
toward domain folders with clear names.  For example, QI-related implementation
should consolidate behind a future `quasi_isodynamic/` package instead of
adding more `qi_*` root modules.

## Naming rules

- Prefer descriptive domain names over generic names like `finish`, `io`, or
  `utils`.
- Avoid creating one-file folders unless they are a stable public domain.
- Keep compatibility facades thin: implementation goes in domain folders,
  user-facing imports are re-exported from the facade.
- Add a short README when creating a package folder that is not obvious from
  its name.
