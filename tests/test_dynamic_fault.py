import numpy as np
import pytest

from elplek.dynamic_fault import (
    initialize_dynamic_fault_rotor_state,
    make_dynamic_fault_current_solver,
    make_dynamic_fault_phase_current_solver,
    make_dynamic_fault_positive_sequence_current_solver,
    solve_dynamic_network_prefault_source_current,
    solve_dynamic_network_fault,
)
from elplek.model import Bus, Network, Shunt, Source
from elplek.synchronous_machine import (
    AxisReactanceData,
    SynchronousMachineParameters,
    solve_synchronous_machine_parameters,
)


def _single_bus_network():
    network = Network()
    network.add_bus(Bus(1, "G1"))
    network.add_source(Source(
        bus_id=1,
        voltage=120 + 0j,
        impedance=1 + 0j,
        negative_sequence_impedance=2 + 0j,
        zero_sequence_impedance=3 + 0j,
        name="G1",
    ))
    return network


def test_dynamic_fault_returns_source_currents_for_all_sequences():
    result = solve_dynamic_network_fault(
        _single_bus_network(),
        fault_bus_id=1,
        fault_type="single_line_to_ground",
        source_voltage=120 + 0j,
    )

    np.testing.assert_allclose(result.fault.fault_current, 60)
    np.testing.assert_allclose(result.source_positive_current, 20)
    np.testing.assert_allclose(result.source_negative_current, 20)
    np.testing.assert_allclose(result.source_zero_current, 20)


@pytest.mark.parametrize(
    "fault_type",
    (
        "three_phase",
        "line_to_line",
        "line_to_line_ab",
        "single_line_to_ground",
        "double_line_to_ground",
        "double_line_to_ground_ab",
    ),
)
def test_dynamic_network_adapter_matches_sequence_solver_for_each_fault_type(
    fault_type,
):
    result = solve_dynamic_network_fault(
        _single_bus_network(),
        fault_bus_id=1,
        fault_type=fault_type,
        source_voltage=120 + 0j,
    )

    np.testing.assert_allclose(
        result.source_positive_current,
        result.fault.positive_current,
    )
    np.testing.assert_allclose(
        result.source_negative_current,
        result.fault.negative_current,
    )
    np.testing.assert_allclose(
        result.source_zero_current,
        result.fault.zero_current,
    )


def test_dynamic_fault_callback_projects_positive_current_to_reference_dq():
    reference_voltage = 120 * np.exp(1j * 0.4)
    callback = make_dynamic_fault_current_solver(
        _single_bus_network(),
        fault_bus_id=1,
        fault_type="single_line_to_ground",
        reference_voltage=reference_voltage,
    )

    d_current, q_current = callback(reference_voltage)

    np.testing.assert_allclose(d_current, 0, atol=1e-12)
    np.testing.assert_allclose(q_current, 20, atol=1e-12)


def test_phase_current_callback_reconstructs_sequence_currents():
    callback = make_dynamic_fault_phase_current_solver(
        _single_bus_network(),
        fault_bus_id=1,
        fault_type="single_line_to_ground",
    )

    np.testing.assert_allclose(callback(120 + 0j), 60)


def test_positive_sequence_callback_returns_machine_current_for_rotor_equations():
    callback = make_dynamic_fault_positive_sequence_current_solver(
        _single_bus_network(),
        fault_bus_id=1,
        fault_type="single_line_to_ground",
    )

    np.testing.assert_allclose(callback(120 + 0j), 20)


def test_positive_sequence_callback_reuses_exact_linear_current_sensitivity():
    network = _single_bus_network()
    callback = make_dynamic_fault_positive_sequence_current_solver(
        network,
        fault_bus_id=1,
        fault_type="single_line_to_ground",
    )
    trial_voltage = 97 + 31j

    direct = solve_dynamic_network_fault(
        network,
        fault_bus_id=1,
        fault_type="single_line_to_ground",
        source_voltage=trial_voltage,
    )

    np.testing.assert_allclose(callback(trial_voltage), direct.source_positive_current)


def test_prefault_source_current_includes_explicit_load_admittance():
    network = _single_bus_network()
    network.add_shunt(Shunt(bus_id=1, admittance=0.01 + 0j))

    current = solve_dynamic_network_prefault_source_current(network)

    np.testing.assert_allclose(current, 120 * 0.01 / 1.01)


def test_integrated_dynamic_fault_initialization_uses_network_prefault_current():
    network = _single_bus_network()
    machine = SynchronousMachineParameters(
        stator_resistance=0.0044,
        d_axis=AxisReactanceData(2.39, 0.3445, 0.26, 0.8433, 0.0251),
        q_axis=AxisReactanceData(2.39, 0.534, 0.26, 0.123, 0.02),
    )
    physical = solve_synchronous_machine_parameters(machine)

    initial = initialize_dynamic_fault_rotor_state(
        network,
        fault_bus_id=1,
        fault_type="single_line_to_ground",
        machine_parameters=machine,
        physical_parameters=physical,
    )

    np.testing.assert_allclose(
        initial.voltage_corrections.prefault_current,
        0,
        atol=1e-12,
    )
    assert np.isfinite(initial.rotor_initial_conditions.internal_voltage.real)
    assert np.isfinite(initial.rotor_initial_conditions.internal_voltage.imag)
