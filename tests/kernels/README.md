# Kernel and Physics Tests

Tests in this folder validate low-level VMEC physics kernels: real-space
geometry, force and residue helpers, finite-beta terms, field identities,
Fourier/Nyquist helpers, chipf conventions, and small kernel parity checks.

Keep full-solve behavior under `tests/solvers/`, WOUT interchange under
`tests/io/wout/`, and external-code agreement under `tests/parity/`.
