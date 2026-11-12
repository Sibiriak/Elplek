import numpy as np
import pytest

from elplek.sequence import solve_sequence_fault


def test_three_phase_fault_uses_positive_sequence_only():
    result = solve_sequence_fault(
        "three_phase", phase_voltage=120, z1=0.2 + 0.4j,
        z2=0.3 + 0.5j, z0=0.6 + 0.8j,
    )
    np.testing.assert_allclose(result.fault_current, 120 / (0.2 + 0.4j))
    assert result.negative_current == 0j
    assert result.zero_current == 0j


def test_single_line_to_ground_fault_connects_all_sequences_in_series():
    result = solve_sequence_fault(
        "single_line_to_ground", phase_voltage=120,
        z1=1 + 0j, z2=2 + 0j, z0=3 + 0j,
    )
    np.testing.assert_allclose(result.positive_current, 20)
    np.testing.assert_allclose(result.negative_current, 20)
    np.testing.assert_allclose(result.zero_current, 20)
    np.testing.assert_allclose(result.fault_current, 60)


def test_line_to_line_fault_has_no_zero_sequence_current():
    result = solve_sequence_fault(
        "line_to_line", phase_voltage=120,
        z1=1 + 0j, z2=2 + 0j, z0=99 + 0j,
    )
    np.testing.assert_allclose(result.positive_current, 40)
    np.testing.assert_allclose(result.negative_current, -40)
    np.testing.assert_allclose(result.zero_current, 0)
    np.testing.assert_allclose(result.fault_current, -1j * np.sqrt(3) * 40)


def test_ground_fault_impedance_appears_tripled_in_sequence_network():
    result = solve_sequence_fault(
        "single_line_to_ground", phase_voltage=120,
        z1=1 + 0j, z2=1 + 0j, z0=1 + 0j,
        fault_impedance=1 + 0j,
    )
    np.testing.assert_allclose(result.fault_current, 3 * 120 / 6)


def test_double_line_to_ground_sequence_currents_satisfy_kcl_relation():
    result = solve_sequence_fault(
        "double_line_to_ground", phase_voltage=120,
        z1=0.2 + 0.3j, z2=0.25 + 0.35j, z0=0.4 + 0.5j,
        fault_impedance=0.1 + 0.05j,
    )
    np.testing.assert_allclose(result.positive_current + result.negative_current + result.zero_current, 0j, atol=1e-12)


@pytest.mark.parametrize(
    ("fault_type", "faulted_phases", "unfaulted_phase"),
    (
        ("double_line_to_ground_ab", (0, 1), 2),
        ("double_line_to_ground_bc", (1, 2), 0),
        ("double_line_to_ground_ca", (2, 0), 1),
    ),
)
def test_double_line_to_ground_orientation_sets_requested_phase_pair(
    fault_type, faulted_phases, unfaulted_phase
):
    result = solve_sequence_fault(
        fault_type,
        phase_voltage=120,
        z1=0.2 + 0.3j,
        z2=0.25 + 0.35j,
        z0=0.4 + 0.5j,
    )

    np.testing.assert_allclose(result.phase_currents[unfaulted_phase], 0j, atol=1e-12)
    for phase_index in faulted_phases:
        np.testing.assert_allclose(result.phase_voltages[phase_index], 0j, atol=1e-10)
    np.testing.assert_allclose(
        result.fault_current,
        3 * result.zero_current,
    )


def test_unknown_fault_type_rejected():
    with pytest.raises(ValueError, match="Unsupported fault type"):
        solve_sequence_fault(
            "invalid", phase_voltage=120, z1=1, z2=1,
        )


def test_fault_result_includes_phase_currents_for_single_line_ground():
    result = solve_sequence_fault(
        "single_line_to_ground", phase_voltage=120,
        z1=1 + 0j, z2=2 + 0j, z0=3 + 0j,
    )
    ia, ib, ic = result.phase_currents
    np.testing.assert_allclose(ia, result.fault_current)
    np.testing.assert_allclose(ib, 0j, atol=1e-12)
    np.testing.assert_allclose(ic, 0j, atol=1e-12)


def test_fault_result_includes_phase_currents_for_line_to_line():
    result = solve_sequence_fault(
        "line_to_line", phase_voltage=120, z1=1, z2=2,
    )
    ia, ib, ic = result.phase_currents
    np.testing.assert_allclose(ia, 0j, atol=1e-12)
    np.testing.assert_allclose(ib, result.fault_current)
    np.testing.assert_allclose(ib + ic, 0j, atol=1e-12)


@pytest.mark.parametrize(
    ("fault_type", "faulted_phase", "unfaulted_phase"),
    (
        ("line_to_line_ab", 0, 2),
        ("line_to_line_bc", 1, 0),
        ("line_to_line_ca", 2, 1),
    ),
)
def test_line_to_line_fault_orientation_sets_requested_phase_pair(
    fault_type, faulted_phase, unfaulted_phase
):
    result = solve_sequence_fault(
        fault_type,
        phase_voltage=120,
        z1=1 + 0.2j,
        z2=2 + 0.3j,
    )
    phases = tuple(result.phase_currents)
    other_faulted_phase = next(
        index for index in range(3) if index not in (faulted_phase, unfaulted_phase)
    )

    np.testing.assert_allclose(phases[unfaulted_phase], 0j, atol=1e-12)
    np.testing.assert_allclose(
        phases[faulted_phase] + phases[other_faulted_phase], 0j, atol=1e-12
    )
    np.testing.assert_allclose(result.fault_current, phases[faulted_phase])


def test_three_phase_fault_result_has_balanced_phase_currents():
    result = solve_sequence_fault(
        "three_phase", phase_voltage=120, z1=1, z2=2,
    )
    ia, ib, ic = result.phase_currents
    np.testing.assert_allclose(ia, result.fault_current)
    np.testing.assert_allclose(abs(ia), abs(ib))
    np.testing.assert_allclose(abs(ia), abs(ic))


def test_single_line_ground_fault_voltage_matches_fault_impedance():
    fault_impedance = 0.2 + 0.1j
    result = solve_sequence_fault(
        "single_line_to_ground", phase_voltage=120,
        z1=0.4 + 0.5j, z2=0.3 + 0.4j, z0=0.6 + 0.7j,
        fault_impedance=fault_impedance,
    )
    va, vb, vc = result.phase_voltages
    ia, _, _ = result.phase_currents
    np.testing.assert_allclose(va, fault_impedance * ia)


def test_bolted_single_line_ground_fault_has_zero_faulted_phase_voltage():
    result = solve_sequence_fault(
        "single_line_to_ground", phase_voltage=120,
        z1=0.4 + 0.5j, z2=0.3 + 0.4j, z0=0.6 + 0.7j,
    )
    np.testing.assert_allclose(result.phase_voltages[0], 0j, atol=1e-10)
