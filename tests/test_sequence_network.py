import numpy as np
import pytest

from elplek.model import Branch, Bus, Network, Shunt, Source
from elplek.sequence_network import calculate_sequence_thevenin


def test_sequence_thevenin_single_bus_source():
    network = Network()
    network.add_bus(Bus(1))
    network.add_source(Source(
        1, 120 + 0j, 1 + 1j, "grid",
        negative_sequence_impedance=2 + 1j,
        zero_sequence_impedance=3 + 1j,
    ))
    result = calculate_sequence_thevenin(network, 1)
    np.testing.assert_allclose(result.prefault_voltage, 120)
    np.testing.assert_allclose(result.z1, 1 + 1j)
    np.testing.assert_allclose(result.z2, 2 + 1j)
    np.testing.assert_allclose(result.z0, 3 + 1j)


def test_sequence_thevenin_includes_branch_impedance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, 0.5, 0.5,
        negative_sequence_impedance=0.7 + 0.4j,
        zero_sequence_impedance=1 + 0.5j,
    ))
    network.add_source(Source(
        1, 120, 0.2 + 0.2j, "grid",
        negative_sequence_impedance=0.3 + 0.3j,
        zero_sequence_impedance=0.4 + 0.4j,
    ))
    result = calculate_sequence_thevenin(network, 2)
    np.testing.assert_allclose(result.prefault_voltage, 120)
    np.testing.assert_allclose(result.z1, 0.7 + 0.7j)
    np.testing.assert_allclose(result.z2, 1.0 + 0.7j)
    np.testing.assert_allclose(result.z0, 1.4 + 0.9j)


def test_sequence_thevenin_requires_zero_sequence_source_data():
    network = Network()
    network.add_bus(Bus(1))
    network.add_source(Source(1, 120, 1j, "grid"))
    with pytest.raises(ValueError, match="zero-sequence impedance"):
        calculate_sequence_thevenin(network, 1)

def test_network_fault_solver_derives_single_line_ground_current():
    from elplek.sequence_network import solve_network_fault

    network = Network()
    network.add_bus(Bus(1))
    network.add_source(Source(
        1, 120, 1 + 0j, "grid",
        negative_sequence_impedance=2 + 0j,
        zero_sequence_impedance=3 + 0j,
    ))

    result = solve_network_fault(network, 1, "single_line_to_ground")

    np.testing.assert_allclose(result.positive_current, 20)
    np.testing.assert_allclose(result.negative_current, 20)
    np.testing.assert_allclose(result.zero_current, 20)
    np.testing.assert_allclose(result.fault_current, 60)


def test_network_fault_solver_includes_branch_sequence_impedances():
    from elplek.sequence_network import solve_network_fault

    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, 0.5, 0.5,
        negative_sequence_impedance=0.7 + 0.4j,
        zero_sequence_impedance=1 + 0.5j,
    ))
    network.add_source(Source(
        1, 120, 0.2 + 0.2j, "grid",
        negative_sequence_impedance=0.3 + 0.3j,
        zero_sequence_impedance=0.4 + 0.4j,
    ))

    result = solve_network_fault(network, 2, "single_line_to_ground")
    expected = 3 * 120 / ((0.7 + 0.7j) + (1.0 + 0.7j) + (1.4 + 0.9j))

    np.testing.assert_allclose(result.fault_current, expected)




def test_sequence_thevenin_includes_shunt_admittance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_source(Source(
        1, 120, 1 + 0j, "grid",
        negative_sequence_impedance=2 + 0j,
        zero_sequence_impedance=3 + 0j,
    ))
    network.add_shunt(Shunt(
        1, 0.5 + 0j,
        negative_sequence_admittance=0.5 + 0j,
        zero_sequence_admittance=0.5 + 0j,
        name="test shunt",
    ))

    result = calculate_sequence_thevenin(network, 1)
    np.testing.assert_allclose(result.prefault_voltage, 80)
    np.testing.assert_allclose(result.z1, 2 / 3)
    np.testing.assert_allclose(result.z2, 1)
    np.testing.assert_allclose(result.z0, 1.2)



def test_sequence_thevenin_includes_nominal_pi_line_charging():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2,
        resistance=1.0,
        reactance=0.0,
        negative_sequence_impedance=1.5 + 0j,
        zero_sequence_impedance=2.0 + 0j,
        shunt_admittance=0.0 + 0.2j,
        negative_sequence_shunt_admittance=0.0 + 0.3j,
        zero_sequence_shunt_admittance=0.0 + 0.1j,
        name="pi line",
    ))
    network.add_source(Source(
        1, 120, 2.0 + 0j, "grid",
        negative_sequence_impedance=3.0 + 0j,
        zero_sequence_impedance=4.0 + 0j,
    ))

    result = calculate_sequence_thevenin(network, 2)

    def expected_zth(z_source, z_line, y_charging):
        y_half = y_charging / 2
        source_side_equivalent = 1 / (1 / z_source + y_half)
        return 1 / (y_half + 1 / (z_line + source_side_equivalent))

    np.testing.assert_allclose(
        result.z1,
        expected_zth(2.0 + 0j, 1.0 + 0j, 0.0 + 0.2j),
    )
    np.testing.assert_allclose(
        result.z2,
        expected_zth(3.0 + 0j, 1.5 + 0j, 0.0 + 0.3j),
    )
    np.testing.assert_allclose(
        result.z0,
        expected_zth(4.0 + 0j, 2.0 + 0j, 0.0 + 0.1j),
    )
