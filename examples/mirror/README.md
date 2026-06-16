Mirror Examples
===============

These examples exercise the experimental fixed-boundary mirror backend from a
source checkout.  They intentionally use low resolution and small iteration
budgets so they run quickly.

Run from the repository root:

```bash
python examples/mirror/fixed_cylinder.py --outdir results/mirror/cylinder
python examples/mirror/fixed_flared_tube.py --outdir results/mirror/flared
python examples/mirror/wham_vacuum_boundary.py --outdir results/mirror/wham
```

Each script writes a mirror-native ``mout_*.nc`` file and, unless
``--no-plots`` is passed, a set of PNG diagnostics.  These are research
fixtures for the scalar-pressure fixed-boundary mirror path, not WHAM
predictive modelling tools.
