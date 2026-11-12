"""Modified nodal-analysis stamps matching Elplek source equations.

The routines operate on an allocated complex matrix and RHS. Matrix
arguments use Elplek-style 1-based equation indices; NumPy arrays are 0-based.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _put(matrix: NDArray[np.complex128], row: int, col: int, value: complex) -> None:
    """Match cmatmul.pas putCmat: replace a coefficient, do not accumulate."""
    n = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape[1] != n:
        raise ValueError("Coefficient matrix must be square")
    if 1 <= row <= n and 1 <= col <= n:
        matrix[row - 1, col - 1] = value


def _put_rhs(rhs: NDArray[np.complex128], row: int, value: complex) -> None:
    """Match cmatmul.pas putCvec: replace one RHS entry."""
    if rhs.ndim != 1:
        raise ValueError("RHS must be a one-dimensional vector")
    if 1 <= row <= rhs.size:
        rhs[row - 1] = value


def stamp_source(
    matrix: NDArray[np.complex128],
    rhs: NDArray[np.complex128],
    *,
    current_eq: int,
    downstream_eq: int,
    neutral_eq: int = 0,
    impedance: complex,
    phase_voltage: complex,
    diagonal_addition: complex = 0j,
) -> None:
    """Stamp the ordinary, non-parallel PutGenMat / PutFeedmat equations.

    current_eq is the auxiliary source-current unknown. A terminal equation
    of 0 means ground or an unused neutral.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Coefficient matrix must be square")
    if rhs.ndim != 1 or rhs.size != matrix.shape[0]:
        raise ValueError("RHS length must match matrix dimension")
    n = matrix.shape[0]
    if not 1 <= current_eq <= n:
        raise IndexError("current_eq outside matrix")
    for eq in (downstream_eq, neutral_eq):
        if not 0 <= eq <= n:
            raise IndexError("terminal equation outside matrix")

    _put(matrix, current_eq, current_eq, impedance + diagonal_addition)
    _put_rhs(rhs, current_eq, phase_voltage)
    if downstream_eq > 0:
        _put(matrix, current_eq, downstream_eq, 1 + 0j)
        _put(matrix, downstream_eq, current_eq, -1 + 0j)
    if neutral_eq > 0:
        _put(matrix, current_eq, neutral_eq, -1 + 0j)
        _put(matrix, neutral_eq, neutral_eq, 1 + 0j)


def stamp_short_circuit(
    matrix: NDArray[np.complex128],
    rhs: NDArray[np.complex128],
    *,
    fault_bus_eq: int,
    current_eq: int,
    voltage: complex = 0j,
    fault_impedance: complex = 0j,
    iec_mode: bool = False,
) -> None:
    """Stamp a single-fault equation.

    In conventional mode, fault_impedance is the impedance from the faulted
    bus to ground. The auxiliary-current equation is V_fault - Zf*If = voltage.
    IEC mode retains Elplek Shortmat's voltage-source constraint; fault_impedance
    is not applied in that mode.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Coefficient matrix must be square")
    if rhs.ndim != 1 or rhs.size != matrix.shape[0]:
        raise ValueError("RHS length must match matrix dimension")
    n = matrix.shape[0]
    if not (1 <= fault_bus_eq <= n and 1 <= current_eq <= n):
        raise IndexError("fault_bus_eq and current_eq must be valid equation indices")

    if iec_mode:
        _put(matrix, fault_bus_eq, current_eq, 1 + 0j)
        _put(matrix, current_eq, fault_bus_eq, 1 + 0j)
        _put_rhs(rhs, current_eq, voltage)
    else:
        _put(
            matrix, fault_bus_eq, current_eq,
            matrix[fault_bus_eq - 1, current_eq - 1] + 1,
        )
        _put(matrix, current_eq, fault_bus_eq, 1 + 0j)
        _put(matrix, current_eq, current_eq, -complex(fault_impedance))
        _put_rhs(rhs, current_eq, voltage)
