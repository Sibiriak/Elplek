"""Complex linear-system solver for Elplek's assembled equations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix cannot be solved uniquely."""


def solve_linear_system(
    matrix: NDArray[np.complexfloating],
    rhs: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Solve A x = b, reporting malformed or singular systems explicitly.

    This uses NumPy's pivoted LAPACK solver. It does not attempt to reproduce
    the original Elplek LU implementation's pivot thresholds yet.
    """
    a = np.asarray(matrix, dtype=np.complex128)
    b = np.asarray(rhs, dtype=np.complex128)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Coefficient matrix must be square")
    if b.ndim != 1 or b.size != a.shape[0]:
        raise ValueError("RHS length must match matrix dimension")
    if a.shape[0] == 0:
        raise ValueError("System must contain at least one equation")
    if not np.isfinite(a.real).all() or not np.isfinite(a.imag).all():
        raise ValueError("Coefficient matrix must contain only finite values")
    if not np.isfinite(b.real).all() or not np.isfinite(b.imag).all():
        raise ValueError("RHS must contain only finite values")

    try:
        solution = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise SingularSystemError(
            "Coefficient matrix is singular; the network may be disconnected "
            "or its equations may be redundant"
        ) from exc

    return np.asarray(solution, dtype=np.complex128)
