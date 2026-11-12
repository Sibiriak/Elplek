import numpy as np
import pytest

from elplek.mna import stamp_short_circuit, stamp_source
from elplek.solver import SingularSystemError, solve_linear_system


def test_solve_complex_linear_system():
    matrix = np.array([[2 + 1j, 1], [1j, 3 - 1j]], dtype=complex)
    expected = np.array([1 - 2j, 3 + 0.5j])
    rhs = matrix @ expected
    np.testing.assert_allclose(solve_linear_system(matrix, rhs), expected)


def test_singular_matrix_raises_domain_error():
    matrix = np.array([[1, 2], [2, 4]], dtype=complex)
    with pytest.raises(SingularSystemError, match="singular"):
        solve_linear_system(matrix, np.array([1, 2], dtype=complex))


def test_non_square_matrix_rejected():
    with pytest.raises(ValueError, match="square"):
        solve_linear_system(np.zeros((2, 3)), np.zeros(2))


def test_rhs_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="RHS length"):
        solve_linear_system(np.eye(2), np.zeros(3))


def test_non_finite_coefficients_rejected():
    matrix = np.array([[complex(float("nan"), 0)]])
    with pytest.raises(ValueError, match="finite"):
        solve_linear_system(matrix, np.ones(1))


def test_source_and_three_phase_fault_current_matches_v_over_z():
    source_voltage = 100 + 20j
    source_impedance = 0.4 + 1.2j

    # Unknowns: bus voltage V, source current Is, fault current If.
    matrix = np.zeros((3, 3), dtype=complex)
    rhs = np.zeros(3, dtype=complex)

    stamp_source(
        matrix,
        rhs,
        current_eq=2,
        downstream_eq=1,
        impedance=source_impedance,
        phase_voltage=source_voltage,
    )
    stamp_short_circuit(
        matrix, rhs, fault_bus_eq=1, current_eq=3, voltage=0j
    )

    solution = solve_linear_system(matrix, rhs)
    bus_voltage, source_current, fault_current = solution

    assert abs(bus_voltage) < 1e-12
    np.testing.assert_allclose(source_current, source_voltage / source_impedance)
    np.testing.assert_allclose(fault_current, source_voltage / source_impedance)
