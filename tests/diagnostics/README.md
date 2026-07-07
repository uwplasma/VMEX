# Diagnostics Tests

This folder contains tests for developer-only tools under
`tools/diagnostics/`. These tests are separate from public API, physics, and
solver tests because they validate reproducibility scripts, profiler parsers,
artifact renderers, and local CI helpers.

Domain folders should mirror `tools/diagnostics/` domains. Keep tests here
small and import-light so they remain safe for fast CI coverage buckets.
