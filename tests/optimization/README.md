# Optimization Tests

Tests in this folder validate production optimization APIs, workflow builders,
example-script contracts, exact-callback policies, and QI/QS objective assembly.

Diagnostics-only renderers and sweep artifact tests live in
`tests/diagnostics/optimization/`. Keep that split: this folder should prove
that optimization code users call remains correct, while diagnostics folders
cover developer tooling around generated artifacts.
