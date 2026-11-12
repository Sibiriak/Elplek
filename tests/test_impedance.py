import numpy as np
import pytest

from elplek.impedance import (
    complex_divide,
    impedance_admittance,
    stamp_two_node,
)


def test_impedance_admittance():
    assert impedance_admittance(3.0, 4.0) == pytest.approx(0.12 - 0.16j)


def test_negative_resistance_is_clamped_to_zero():
    assert impedance_admittance(-2.0, 1.0) == pytest.approx(-1j)


def test_zero_impedance_uses_rmin():
    assert impedance_admittance(0.0, 0.0) == pytest.approx(1e4 + 0j)


def test_negative_reactance_is_allowed():
    assert impedance_admittance(0.0, -2.0) == pytest.approx(0.5j)


def test_two_node_stamp():
    ybus = np.zeros((2, 2), dtype=complex)
    stamp_two_node(ybus, 1, 2, 2 + 3j)

    expected = np.array(
        [[2 + 3j, -2 - 3j], [-2 - 3j, 2 + 3j]],
        dtype=complex,
    )
    np.testing.assert_allclose(ybus, expected)


def test_grounded_branch_only_stamps_connected_node():
    ybus = np.zeros((2, 2), dtype=complex)
    stamp_two_node(ybus, 1, 0, 2 + 3j)

    expected = np.array([[2 + 3j, 0j], [0j, 0j]], dtype=complex)
    np.testing.assert_allclose(ybus, expected)


def test_additional_diagonal_term():
    ybus = np.zeros((1, 1), dtype=complex)
    stamp_two_node(ybus, 1, 0, 2 + 3j, add_diag=0.5 + 0.25j)
    assert ybus[0, 0] == pytest.approx(2.5 + 3.25j)


def test_complex_division_by_zero_returns_zero_like_elplek():
    assert complex_divide(1 + 0j, 0j) == 0j


def test_node_index_out_of_range():
    ybus = np.zeros((2, 2), dtype=complex)
    with pytest.raises(IndexError):
        stamp_two_node(ybus, 3, 1, 1 + 0j)


def test_non_square_matrix_rejected():
    ybus = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError):
        stamp_two_node(ybus, 1, 2, 1 + 0j)
