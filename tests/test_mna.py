import numpy as np
import pytest

from elplek.mna import stamp_short_circuit, stamp_source


def test_source_stamp_matches_put_gen_mat_coefficients():
    matrix = np.zeros((3, 3), dtype=complex)
    rhs = np.zeros(3, dtype=complex)
    stamp_source(
        matrix, rhs, current_eq=3, downstream_eq=1, neutral_eq=2,
        impedance=0.2 + 0.4j, phase_voltage=100 + 0j,
    )
    np.testing.assert_allclose(
        matrix,
        [[0, 0, -1], [0, 1, 0], [1, -1, 0.2 + 0.4j]],
    )
    assert rhs[2] == 100


def test_source_stamp_grounded_neutral():
    matrix = np.zeros((2, 2), dtype=complex)
    rhs = np.zeros(2, dtype=complex)
    stamp_source(
        matrix, rhs, current_eq=2, downstream_eq=1,
        impedance=1j, phase_voltage=230,
    )
    np.testing.assert_allclose(matrix, [[0, -1], [1, 1j]])
    assert rhs[1] == 230


def test_short_circuit_stamp_zeroes_fault_voltage_equation():
    matrix = np.zeros((2, 2), dtype=complex)
    rhs = np.zeros(2, dtype=complex)
    stamp_short_circuit(matrix, rhs, fault_bus_eq=1, current_eq=2)
    np.testing.assert_allclose(matrix, [[0, 1], [1, 0]])
    np.testing.assert_allclose(rhs, [0, 0])


def test_iec_fault_stamp_sets_voltage_rhs():
    matrix = np.zeros((2, 2), dtype=complex)
    rhs = np.zeros(2, dtype=complex)
    stamp_short_circuit(
        matrix, rhs, fault_bus_eq=1, current_eq=2,
        voltage=-230 + 0j, iec_mode=True,
    )
    np.testing.assert_allclose(matrix, [[0, 1], [1, 0]])
    assert rhs[1] == -230


def test_mna_rejects_rhs_dimension_mismatch():
    with pytest.raises(ValueError, match="RHS length"):
        stamp_source(
            np.zeros((2, 2), dtype=complex), np.zeros(3, dtype=complex),
            current_eq=2, downstream_eq=1, impedance=1j, phase_voltage=1,
        )
