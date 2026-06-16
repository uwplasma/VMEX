"""Compatibility facade for fixed-boundary solver implementations.

The implementation lives in domain modules under
``vmec_jax.solvers.fixed_boundary`` so this historical public module stays
small while existing imports and internal monkeypatch seams continue to work.
"""

from __future__ import annotations

import sys
import types

from .solvers.fixed_boundary import api as _fixed_boundary_api
from .solvers.fixed_boundary.residual import iteration as _iteration


def _export_symbols(module) -> None:
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        globals()[name] = value


_export_symbols(_iteration)
_export_symbols(_fixed_boundary_api)


class _SolveFacadeModule(types.ModuleType):
    """Forward assignments to the implementation module.

    A number of internal tests and downstream debugging workflows monkeypatch
    private ``vmec_jax.solve`` symbols.  The exported solver functions execute
    in the implementation module's global namespace, so assignments on this
    facade must be mirrored there to preserve legacy behavior.
    """

    def __setattr__(self, name, value):
        if not (name.startswith("__") and name.endswith("__")):
            for module in (_iteration, _fixed_boundary_api):
                if hasattr(module, name):
                    setattr(module, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SolveFacadeModule

__all__ = tuple(name for name in globals() if not (name.startswith("__") and name.endswith("__")))
