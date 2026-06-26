"""Radial profiles for fixed-boundary mirror fields and pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _evaluate_polynomial(coefficients, s):
    coefficients = np.asarray(coefficients)
    s = np.asarray(s)
    powers = s[..., None] ** np.arange(coefficients.size, dtype=s.dtype)
    return powers @ coefficients


@dataclass(frozen=True)
class PsiPrimeProfile:
    """Axial flux derivative profile ``Psi'(s)``."""

    coefficients: np.ndarray

    @classmethod
    def constant(cls, value: float) -> "PsiPrimeProfile":
        """Create a spatially constant axial flux derivative profile."""

        return cls(coefficients=np.asarray([float(value)]))

    @classmethod
    def polynomial(cls, coefficients) -> "PsiPrimeProfile":
        """Create a polynomial ``Psi'(s)`` profile from increasing powers of ``s``."""

        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size < 1:
            raise ValueError("PsiPrimeProfile coefficients must be a nonempty vector")
        return cls(coefficients=coefficients)

    def evaluate(self, s, *, dtype: Any | None = None) -> np.ndarray:
        """Evaluate ``Psi'(s)`` at radial coordinates ``s``."""

        return np.asarray(_evaluate_polynomial(self.coefficients, s), dtype=dtype or float)


@dataclass(frozen=True)
class IPrimeProfile:
    """Twist/current-like radial profile ``I'(s)``."""

    coefficients: np.ndarray

    @classmethod
    def zero(cls) -> "IPrimeProfile":
        """Create a zero twist/current profile."""

        return cls(coefficients=np.asarray([0.0]))

    @classmethod
    def constant(cls, value: float) -> "IPrimeProfile":
        """Create a spatially constant twist/current profile."""

        return cls(coefficients=np.asarray([float(value)]))

    @classmethod
    def polynomial(cls, coefficients) -> "IPrimeProfile":
        """Create a polynomial ``I'(s)`` profile from increasing powers of ``s``."""

        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size < 1:
            raise ValueError("IPrimeProfile coefficients must be a nonempty vector")
        return cls(coefficients=coefficients)

    def evaluate(self, s, *, dtype: Any | None = None) -> np.ndarray:
        """Evaluate ``I'(s)`` at radial coordinates ``s``."""

        return np.asarray(_evaluate_polynomial(self.coefficients, s), dtype=dtype or float)


@dataclass(frozen=True)
class PressureProfile:
    """Scalar pressure profile ``p(s)`` used in the first mirror energy model."""

    coefficients: np.ndarray
    gamma: float = 5.0 / 3.0

    @classmethod
    def zero(cls, *, gamma: float = 5.0 / 3.0) -> "PressureProfile":
        """Create a zero-pressure profile."""

        return cls(coefficients=np.asarray([0.0]), gamma=float(gamma))

    @classmethod
    def constant(cls, value: float, *, gamma: float = 5.0 / 3.0) -> "PressureProfile":
        """Create a spatially constant pressure profile."""

        return cls(coefficients=np.asarray([float(value)]), gamma=float(gamma))

    @classmethod
    def polynomial(cls, coefficients, *, gamma: float = 5.0 / 3.0) -> "PressureProfile":
        """Create a polynomial pressure profile from increasing powers of ``s``."""

        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 1 or coefficients.size < 1:
            raise ValueError("PressureProfile coefficients must be a nonempty vector")
        return cls(coefficients=coefficients, gamma=float(gamma))

    def evaluate(self, s, *, dtype: Any | None = None) -> np.ndarray:
        """Evaluate pressure at radial coordinates ``s``."""

        return np.asarray(_evaluate_polynomial(self.coefficients, s), dtype=dtype or float)

    def derivative(self, s, *, dtype: Any | None = None) -> np.ndarray:
        """Evaluate ``dp/ds`` at radial coordinates ``s``."""

        if self.coefficients.size == 1:
            return np.zeros_like(np.asarray(s, dtype=dtype or float))
        derivative_coefficients = self.coefficients[1:] * np.arange(1, self.coefficients.size)
        return np.asarray(_evaluate_polynomial(derivative_coefficients, s), dtype=dtype or float)
