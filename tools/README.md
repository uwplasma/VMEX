# Tools

This directory contains developer-facing tools, not end-user examples.

- `diagnostics/`: reproducibility, profiling, validation, plotting, and release
  artifact generators.
- `benchmarks/`: focused benchmark entry points.
- `fetch_assets.py`: downloads optional large validation/reference assets.

Small inspection/debug utilities live under the relevant diagnostics domain,
for example `diagnostics/assets/inspect_npz.py`.  VMEC2000 parity diagnostics
use the maintained executable-based tools under `diagnostics/parity/`; the old
Python-extension driver probe was removed because it required a nonstandard
`vmec` extension build and was not part of the promoted validation gates.

Tools may write to ignored `outputs/` or a user-selected scratch directory.
They should not write tracked artifacts unless the command is explicitly a
documentation or release-artifact promotion step.
