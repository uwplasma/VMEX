# Performance Diagnostics Tests

Tests in this folder cover performance/profiling tools under
`tools/diagnostics/performance/`.

They should validate parsers, command construction, budget classification, and
summary rendering without launching long VMEC solves. Expensive CPU/GPU
profiling belongs in optional local or nightly diagnostics, not required CI.
