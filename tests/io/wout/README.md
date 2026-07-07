# WOUT I/O and Physics Tests

Tests in this folder validate VMEC WOUT reading, writing, round-tripping,
VMECPlot2 compatibility, bundled WOUT fixtures, and physics quantities derived
from WOUT files such as geometry, currents, Mercier terms, and magnetic-field
components.

Keep solver-convergence tests under `tests/solvers/`; keep diagnostics-only
artifact renderers under `tests/diagnostics/`. This folder is for WOUT files as
the public equilibrium interchange format.
