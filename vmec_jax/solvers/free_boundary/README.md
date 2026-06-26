# Free-boundary solver

Free-boundary implementation is organized around the direct-coil/mgrid provider
path and the NESTOR-style boundary response:

- `boundary_fields.py`, `mgrid.py`, `axis_current.py`, and `control.py`: field
  coupling and control helpers.
- `jax_nestor_operator.py`: JAX-visible source and mode-space response pieces.
- `adjoint/`: branch-local replay, controller fingerprints, custom-VJP helpers,
  and validation scaffolding.
- `coil_optimization.py`: single-stage coil optimization utilities.
- `validation.py`: bounded parity and physical-gate helpers.

Claims about arbitrary adaptive branch differentiation should stay conservative
until the adaptive controller itself is JAX-visible and FD-validated.
