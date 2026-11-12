"""Symmetrical-component transformations between sequence and phase currents."""
from __future__ import annotations

import cmath
from typing import TypeAlias

PhaseCurrents: TypeAlias = tuple[complex, complex, complex]


def sequence_to_phase_currents(
    zero: complex,
    positive: complex,
    negative: complex,
) -> PhaseCurrents:
    """Convert sequence currents (I0, I1, I2) to phase currents (Ia, Ib, Ic).

    Uses a = exp(j*2*pi/3) and the standard symmetrical-component transform:
    Ia=I0+I1+I2, Ib=I0+a^2*I1+a*I2, Ic=I0+a*I1+a^2*I2.
    """
    i0, i1, i2 = complex(zero), complex(positive), complex(negative)
    a = cmath.exp(2j * cmath.pi / 3)
    ia = i0 + i1 + i2
    ib = i0 + (a**2) * i1 + a * i2
    ic = i0 + a * i1 + (a**2) * i2
    return complex(ia), complex(ib), complex(ic)
