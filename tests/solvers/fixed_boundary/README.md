# Fixed-Boundary Solver Tests

Tests in this folder cover the production fixed-boundary solve pipeline:
initial guesses, residual iteration, accelerated/VMEC2000-style scans,
implicit differentiation helpers, preconditioners, staged solver steps, and
solver runtime instrumentation.

Keep tests here when they exercise solver behavior that users or CI rely on.
Diagnostics-only renderers and benchmark parsers belong under
`tests/diagnostics/`; WOUT output physics gates belong under the WOUT/parity
domains.
