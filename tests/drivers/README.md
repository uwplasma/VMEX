# Driver and CLI Tests

Tests in this folder validate the public command-line interface, the Python
driver API, run policies, finish/output behavior, doctor checks, and
user-facing orchestration around fixed- and free-boundary solves.

Keep low-level solver algebra in `tests/solvers/`; keep output-format tests in
`tests/io/`; keep optional external parity gates in their parity domains. This
folder is for entrypoints and policies that users call directly.
