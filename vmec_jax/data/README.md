# Bundled Runtime Data

This package contains small files that must be available after `pip install
vmec-jax`.

Keep this directory small. Large WOUTs, MGRID files, generated optimization
outputs, and validation fixtures belong in release assets fetched by
`tools/fetch_assets.py`, not in the Python package.

