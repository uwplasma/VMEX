# Optimization Diagnostics Tests

Tests in this folder cover developer diagnostics under
`tools/diagnostics/optimization/`: sweep renderers, README optimization panels,
minimal-seed showcase artifacts, and QS/QI publication-summary utilities.

Keep public optimizer APIs and example-script behavior tests outside this folder
unless they only exercise diagnostics scripts. This keeps production
optimization tests visible while making artifact-rendering tests easier to
maintain.
