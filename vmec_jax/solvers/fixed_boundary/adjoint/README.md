# Fixed-Boundary Adjoint Helpers

This package contains fixed-boundary derivative and adjoint helper code.

Use this layer when a scalar objective needs validated sensitivities through a
VMEC solve. Keep branch- or optimization-policy decisions outside this package;
the files here should focus on differentiating a well-defined solve path.

