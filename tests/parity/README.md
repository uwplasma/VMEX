# Parity and External-VMEC Tests

Tests in this folder compare vmec_jax against VMEC2000, VMEC++, bundled parity
fixtures, or field-by-field reference identities. Optional tests that require
external executables or fetched assets should skip cleanly when those assets are
not available.

Keep pure WOUT file-format tests in `tests/io/wout/`; keep solver unit tests in
`tests/solvers/`. This folder is for reference agreement and physics parity.
