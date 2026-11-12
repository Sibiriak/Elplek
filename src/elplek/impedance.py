"""Impedance and Y-bus stamping routines ported from Elplek.

Source routines:
- COMPDEF.PAS: Cdiv
- elplex.pas: AdmImp, Twonode

Elplek uses node 0 for ground and numbers physical nodes from 1.
The matrix passed to stamp_two_node is a NumPy array indexed from 0,
so physical node k maps to matrix index k - 1.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

RMIN = 1e-4
XL = 1e35
XS = 1.0 / XL


def _limit_component(value: float) -> float:
    """Clamp one result component to the magnitude limit used by Cdiv."""
    if abs(value) < XL:
        return value
    return math.copysign(XL, value)


def complex_divide(a: complex, b: complex) -> complex:
    """Reproduce the main numerical behavior of Elplek's COMPDEF.Cdiv.

    If both divisor components are <= 1e-35 in magnitude, Elplek returns
    zero. Otherwise it divides using the conjugate and clamps each output
    component to +/-1e35. Python floats are double precision, as are the
    intermediates declared in the original routine.
    """
    if abs(b.real) <= XS and abs(b.imag) <= XS:
        return 0j

    denominator = b.real * b.real + b.imag * b.imag
    numerator = a * b.conjugate()
    real = numerator.real / denominator
    imag = numerator.imag / denominator
    return complex(_limit_component(real), _limit_component(imag))


def impedance_admittance(r: float, x: float) -> complex:
    """Calculate impedance admittance using Elplek's AdmImp rules.

    Negative resistance is clamped to zero. Negative reactance is allowed.
    If abs(x) + r < RMIN, resistance is set to RMIN before inversion.
    """
    r = max(float(r), 0.0)
    x = float(x)

    if abs(x) + r < RMIN:
        r = RMIN

    return complex_divide(1 + 0j, complex(r, x))


def stamp_two_node(
    ybus: NDArray[np.complexfloating],
    upstream: int,
    downstream: int,
    admittance: complex,
    add_diag: complex = 0j,
) -> None:
    """Accumulate a two-node component's contribution to Ybus.

    Mirrors Elplek's Twonode routine. Node 0 means ground; physical nodes
    are numbered 1..n. The matrix uses zero-based NumPy indexing.
    """
    if ybus.ndim != 2 or ybus.shape[0] != ybus.shape[1]:
        raise ValueError("Ybus must be a square matrix")
    n = ybus.shape[0]
    if not (0 <= upstream <= n and 0 <= downstream <= n):
        raise IndexError("Node index outside valid range 0..n")

    u = upstream - 1
    d = downstream - 1

    if upstream > 0:
        ybus[u, u] += admittance + add_diag
        if downstream > 0:
            ybus[u, d] -= admittance

    if downstream > 0:
        ybus[d, d] += admittance + add_diag
        if upstream > 0:
            ybus[d, u] -= admittance
