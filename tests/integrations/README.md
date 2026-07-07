# Optional Integration Tests

This folder contains tests that require optional third-party packages or
external validation environments, such as SIMSOPT.

Required CI should skip these tests unless the corresponding environment flag is
set. Use this folder for external-code formula checks, not for bundled parity
fixtures that can run in normal CI.
