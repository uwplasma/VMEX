"""Clean-room core of vmec_jax; will replace the legacy modules — see plan.md §5.

Modules
-------
- :mod:`vmec_jax.core.fourier`    — mode/grid/trig bookkeeping (VMEC2000 ``fixaray.f``).
- :mod:`vmec_jax.core.transforms` — spectral transforms (VMEC2000 ``totzsp_mod.f`` /
  ``tomnsp_mod.f`` / ``symforce.f``).

The numerical conventions here are ported verbatim from the parity-proven
legacy kernels (``vmec_jax/kernels/tomnsp.py``, ``vmec_jax/kernels/realspace.py``)
and are validated A/B against them in ``tests/core_new/``.
"""
