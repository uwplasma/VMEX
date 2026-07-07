# Tools

This directory contains developer-facing tools, not end-user examples.

- `diagnostics/`: reproducibility, profiling, validation, plotting, and release
  artifact generators.
- `benchmarks/`: focused benchmark entry points.
- `fetch_assets.py`: downloads optional large validation/reference assets.

Small inspection/debug utilities live under the relevant diagnostics domain,
for example `diagnostics/assets/inspect_npz.py` and
`diagnostics/parity/vmec2000_driver_probe.py`.

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
