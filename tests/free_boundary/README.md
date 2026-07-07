# Free-Boundary Tests

Tests in this folder validate production free-boundary functionality:
direct-coil and mgrid providers, JAX-visible NESTOR pieces, branch-local
derivative evidence, coil-optimization examples, finite-beta response, and
bounded VMEC2000/ESSOS parity gates.

Diagnostics-only free-boundary report parsers live in
`tests/diagnostics/free_boundary/` when present. This folder is for behavior
that users or CI rely on as part of the free-boundary solver and optimization
API.
