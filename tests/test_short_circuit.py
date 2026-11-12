import numpy as np
import pytest

from elplek.model import Branch, Bus, Network, Shunt, Source, Transformer
from elplek.short_circuit import solve_short_circuit


def test_single_bus_fault_current_equals_source_voltage_over_impedance():
    network = Network()
    network.add_bus(Bus(10, "Fault bus"))
    network.add_source(Source(10, 100 + 20j, 0.4 + 1.2j, "grid"))

    result = solve_short_circuit(network, 10)
    np.testing.assert_allclose(result.fault_current, (100 + 20j) / (0.4 + 1.2j))
    np.testing.assert_allclose(result.bus_voltages[10], 0j)
    np.testing.assert_allclose(result.source_currents["grid"], result.fault_current)


def test_two_bus_fault_includes_line_impedance():
    network = Network()
    network.add_bus(Bus(10, "Source"))
    network.add_bus(Bus(20, "Fault"))
    network.add_branch(Branch(10, 20, 0.3, 0.4))
    network.add_source(Source(10, 120 + 0j, 0.2 + 0.6j, "utility"))

    result = solve_short_circuit(network, 20)
    expected = 120 / ((0.2 + 0.6j) + (0.3 + 0.4j))
    np.testing.assert_allclose(result.fault_current, expected)
    np.testing.assert_allclose(result.bus_voltages[20], 0j, atol=1e-12)
    np.testing.assert_allclose(result.source_currents["utility"], expected)


def test_source_on_unknown_bus_rejected():
    network = Network()
    network.add_bus(Bus(1))
    with pytest.raises(ValueError, match="unknown bus"):
        network.add_source(Source(2, 100, 1j))


def test_short_circuit_without_source_rejected():
    network = Network()
    network.add_bus(Bus(1))
    with pytest.raises(ValueError, match="at least one source"):
        solve_short_circuit(network, 1)

def test_parallel_sources_contribute_to_fault_current():
    network = Network()
    network.add_bus(Bus(1, "Fault bus"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "source_a"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "source_b"))

    result = solve_short_circuit(network, 1)

    np.testing.assert_allclose(result.fault_current, 200 + 0j)
    np.testing.assert_allclose(result.source_currents["source_a"], 100 + 0j)
    np.testing.assert_allclose(result.source_currents["source_b"], 100 + 0j)


def test_parallel_sources_with_different_impedances_match_thevenin_sum():
    network = Network()
    network.add_bus(Bus(1, "Fault bus"))
    network.add_source(Source(1, 100 + 0j, 1 + 1j, "source_a"))
    network.add_source(Source(1, 80 + 0j, 2 + 0j, "source_b"))

    result = solve_short_circuit(network, 1)
    expected_a = (100 + 0j) / (1 + 1j)
    expected_b = (80 + 0j) / 2

    np.testing.assert_allclose(result.source_currents["source_a"], expected_a)
    np.testing.assert_allclose(result.source_currents["source_b"], expected_b)
    np.testing.assert_allclose(result.fault_current, expected_a + expected_b)

def test_fault_impedance_reduces_fault_current():
    network = Network()
    network.add_bus(Bus(1, "Fault bus"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "grid"))

    result = solve_short_circuit(network, 1, fault_impedance=2 + 0j)

    np.testing.assert_allclose(result.fault_current, 100 / (1 + 2))
    np.testing.assert_allclose(result.source_currents["grid"], result.fault_current)
    np.testing.assert_allclose(result.bus_voltages[1], result.fault_current * 2)


def test_complex_fault_impedance_matches_series_equivalent():
    network = Network()
    network.add_bus(Bus(1, "Source"))
    network.add_bus(Bus(2, "Fault"))
    network.add_branch(Branch(1, 2, 0.3, 0.4))
    network.add_source(Source(1, 120 + 0j, 0.2 + 0.6j, "utility"))
    fault_impedance = 0.5 + 0.25j

    result = solve_short_circuit(network, 2, fault_impedance=fault_impedance)
    expected = 120 / ((0.2 + 0.6j) + (0.3 + 0.4j) + fault_impedance)

    np.testing.assert_allclose(result.fault_current, expected)
    np.testing.assert_allclose(result.bus_voltages[2], expected * fault_impedance)


def test_zero_fault_impedance_preserves_bolted_fault_result():
    network = Network()
    network.add_bus(Bus(1, "Fault bus"))
    network.add_source(Source(1, 100 + 0j, 1 + 1j, "grid"))

    default_result = solve_short_circuit(network, 1)
    explicit_zero_result = solve_short_circuit(network, 1, fault_impedance=0j)

    np.testing.assert_allclose(explicit_zero_result.fault_current, default_result.fault_current)




def test_short_circuit_includes_bus_shunt_admittance():
    network = Network()
    network.add_bus(Bus(1, "Fault bus"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "grid"))
    network.add_shunt(Shunt(1, 0.5 + 0j, name="parallel load equivalent"))

    result = solve_short_circuit(network, 1, fault_impedance=1 + 0j)

    # Source current is (100 - 40)/1 = 60 A; the bus voltage and
    # fault-branch current are 40 V and 40 A, respectively.
    np.testing.assert_allclose(result.bus_voltages[1], 40 + 0j)
    np.testing.assert_allclose(result.fault_current, 40 + 0j)
    np.testing.assert_allclose(result.source_currents["grid"], 60 + 0j)



def test_short_circuit_includes_transformer_series_impedance():
    network = Network()
    network.add_bus(Bus(1, "Source side"))
    network.add_bus(Bus(2, "Fault side"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "grid"))
    network.add_transformer(Transformer(
        1, 2,
        positive_sequence_impedance=0.5 + 0.5j,
        negative_sequence_impedance=0.5 + 0.5j,
    ))

    result = solve_short_circuit(network, 2, fault_impedance=0.25 + 0j)
    expected = 100 / (1 + (0.5 + 0.5j) + 0.25)

    np.testing.assert_allclose(result.fault_current, expected)



def test_short_circuit_includes_nominal_pi_line_charging_admittance():
    network = Network()
    network.add_bus(Bus(1, "Source side"))
    network.add_bus(Bus(2, "Fault side"))
    network.add_source(Source(1, 100 + 0j, 1 + 0j, "grid"))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        shunt_admittance=0.0 + 2.0j,
    ))

    result = solve_short_circuit(network, 2, fault_impedance=1 + 0j)

    # Nodal equations with j1 pu on each end of the line charging give
    # V_fault = I_fault*Z_fault = 10 - j20 V.
    np.testing.assert_allclose(result.bus_voltages[2], 10 - 20j)
    np.testing.assert_allclose(result.fault_current, 10 - 20j)
