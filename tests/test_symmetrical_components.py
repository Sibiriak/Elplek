import numpy as np

from elplek.symmetrical_components import sequence_to_phase_currents


def test_balanced_positive_sequence_has_equal_magnitude_phase_currents():
    ia, ib, ic = sequence_to_phase_currents(0j, 10 + 0j, 0j)
    np.testing.assert_allclose(ia, 10)
    np.testing.assert_allclose(ib, 10 * np.exp(-2j * np.pi / 3))
    np.testing.assert_allclose(ic, 10 * np.exp(2j * np.pi / 3))


def test_single_line_to_ground_sequence_currents_reconstruct_only_phase_a():
    ia, ib, ic = sequence_to_phase_currents(5, 5, 5)
    np.testing.assert_allclose(ia, 15)
    np.testing.assert_allclose(ib, 0, atol=1e-12)
    np.testing.assert_allclose(ic, 0, atol=1e-12)


def test_line_to_line_sequence_currents_reconstruct_opposite_b_c_currents():
    ia, ib, ic = sequence_to_phase_currents(0j, 4, -4)
    np.testing.assert_allclose(ia, 0, atol=1e-12)
    np.testing.assert_allclose(ib, -1j * np.sqrt(3) * 4)
    np.testing.assert_allclose(ic, 1j * np.sqrt(3) * 4)
    np.testing.assert_allclose(ib + ic, 0, atol=1e-12)
