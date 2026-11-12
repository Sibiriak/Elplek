import numpy as np

from elplek.synchronous_machine import (
    AxisReactanceData,
    advance_rotor_state_elplek,
    advance_rotor_state_trapezoidal,
    field_winding_voltage_for_prefault_state,
    SynchronousMachineParameters,
    initialize_rotor_state,
    internal_voltage_from_rotor_state,
    solve_initial_rotor_voltage_corrections,
    solve_synchronous_machine_parameters,
    simulate_coupled_rotor_steps,
)


def _machine():
    parameters = SynchronousMachineParameters(
        stator_resistance=0.0044,
        d_axis=AxisReactanceData(2.39, 0.3445, 0.26, 0.8433, 0.0251),
        q_axis=AxisReactanceData(2.39, 0.534, 0.26, 0.123, 0.02),
        frequency_hz=50,
    )
    return parameters, solve_synchronous_machine_parameters(parameters)


def test_rotor_initialization_at_zero_voltage_change_is_steady_state():
    parameters, physical = _machine()
    prefault_voltage = 12_000 * np.exp(1j * 0.3)

    initial = initialize_rotor_state(
        parameters,
        physical,
        prefault_voltage,
        d_axis_voltage_change_v=0,
        q_axis_voltage_change_v=0,
    )

    assert initial.rotor_state.d_damper == 0.0
    assert initial.rotor_state.q_damper == 0.0
    assert initial.rotor_state.p_damper == 0.0
    np.testing.assert_allclose(initial.internal_voltage, prefault_voltage)
    np.testing.assert_allclose(
        internal_voltage_from_rotor_state(
            physical, initial.rotor_state, prefault_voltage
        ),
        prefault_voltage,
    )


def test_rotor_initialization_applies_dq_voltage_corrections():
    parameters, physical = _machine()

    initial = initialize_rotor_state(
        parameters,
        physical,
        12_000 + 0j,
        d_axis_voltage_change_v=300,
        q_axis_voltage_change_v=-150,
    )

    assert initial.rotor_state.d_damper != 0
    assert initial.rotor_state.q_damper != 0
    np.testing.assert_allclose(initial.internal_voltage, 12_300 - 150j)
    np.testing.assert_allclose(
        internal_voltage_from_rotor_state(
            physical, initial.rotor_state, 12_000 + 0j
        ),
        initial.internal_voltage,
        atol=1e-9,
    )


def test_initial_dq_correction_solves_single_machine_fault_equations():
    parameters, _ = _machine()
    reference_voltage = 12_000 * np.exp(1j * 0.3)
    faulted_current = 60 * np.exp(1j * 0.3)
    current_gain = 0.5

    result = solve_initial_rotor_voltage_corrections(
        parameters,
        reference_voltage,
        0j,
        lambda voltage: faulted_current + current_gain * (
            voltage - reference_voltage
        ),
    )

    np.testing.assert_allclose(result.faulted_current, faulted_current)
    np.testing.assert_allclose(result.current_derivative, current_gain)
    d_x = parameters.d_axis.synchronous_reactance - parameters.d_axis.subtransient_reactance
    d_xq = parameters.d_axis.synchronous_reactance - parameters.q_axis.subtransient_reactance
    expected = np.linalg.solve(
        [[-1, -d_x * current_gain], [d_xq * current_gain, -1]],
        [0, -d_xq * 60],
    )
    np.testing.assert_allclose(
        [result.d_axis_voltage_change_v, result.q_axis_voltage_change_v],
        expected,
    )


def test_elplek_rotor_update_uses_pascal_quatre_orthogonal_column():
    parameters, physical = _machine()
    previous = initialize_rotor_state(
        parameters,
        physical,
        12_000 + 0j,
        d_axis_voltage_change_v=0,
        q_axis_voltage_change_v=0,
    ).rotor_state
    sensitivity = 0.2 + 0.3j
    dt = 0.001

    d, q = physical.d_axis, physical.q_axis
    a, b = sensitivity.real, sensitivity.imag
    # Pascal Calc_dIdV writes the orthogonal column via
    # PutdIdV(dIrdV, -dIidV, 2): (ri, ii) = (-b, a).
    rr, ir, ri, ii = a, b, -b, a
    omega = physical.angular_frequency
    network = 1.5 * omega * np.array(
        [
            [d.main_to_stator_mutual * ir * d.main_to_stator_mutual,
             d.main_to_stator_mutual * ir * d.damping_to_stator_mutual,
             d.main_to_stator_mutual * ii * q.main_to_stator_mutual,
             d.main_to_stator_mutual * ii * q.damping_to_stator_mutual],
            [d.damping_to_stator_mutual * ir * d.main_to_stator_mutual,
             d.damping_to_stator_mutual * ir * d.damping_to_stator_mutual,
             d.damping_to_stator_mutual * ii * q.main_to_stator_mutual,
             d.damping_to_stator_mutual * ii * q.damping_to_stator_mutual],
            [-q.main_to_stator_mutual * rr * d.main_to_stator_mutual,
             -q.main_to_stator_mutual * rr * d.damping_to_stator_mutual,
             -q.main_to_stator_mutual * ri * q.main_to_stator_mutual,
             -q.main_to_stator_mutual * ri * q.damping_to_stator_mutual],
            [-q.damping_to_stator_mutual * rr * d.main_to_stator_mutual,
             -q.damping_to_stator_mutual * rr * d.damping_to_stator_mutual,
             -q.damping_to_stator_mutual * ri * q.main_to_stator_mutual,
             -q.damping_to_stator_mutual * ri * q.damping_to_stator_mutual],
        ]
    )
    inductance = np.array(
        [
            [d.main_inductance, d.mutual_inductance, 0, 0],
            [d.mutual_inductance, d.damping_inductance, 0, 0],
            [0, 0, q.main_inductance, q.mutual_inductance],
            [0, 0, q.mutual_inductance, q.damping_inductance],
        ]
    )
    resistance = np.diag(
        [d.main_resistance, d.damping_resistance, q.main_resistance, q.damping_resistance]
    )
    old = previous.as_array()
    expected = np.linalg.solve(
        inductance + network + dt / 2 * resistance,
        (inductance + network - dt / 2 * resistance) @ old
        + np.array([dt * -100.0, 0, 0, 0]),
    )
    result = advance_rotor_state_elplek(
        physical,
        previous,
        time_step_s=dt,
        field_voltage_v=-100.0,
        stator_current_sensitivity=sensitivity,
    )
    np.testing.assert_allclose(result.as_array(), expected)


def test_multistep_driver_preserves_unperturbed_machine_steady_state():
    parameters, physical = _machine()
    reference_voltage = 12_000 * np.exp(1j * 0.3)
    initial = initialize_rotor_state(
        parameters,
        physical,
        reference_voltage,
        d_axis_voltage_change_v=0,
        q_axis_voltage_change_v=0,
    )
    field_voltage = field_winding_voltage_for_prefault_state(
        physical, reference_voltage
    )

    trajectory = simulate_coupled_rotor_steps(
        physical,
        initial.rotor_state,
        time_steps_s=(0.01, 0.02),
        field_voltage_v=field_voltage,
        reference_voltage=reference_voltage,
        initial_stator_d_current_a=0,
        initial_stator_q_current_a=0,
        solve_stator_currents=lambda voltage: (0.0, 0.0),
    )

    assert [sample.time_s for sample in trajectory] == [0.01, 0.03]
    np.testing.assert_allclose(
        [
            sample.step.rotor_state.field
            for sample in trajectory
        ],
        initial.rotor_state.field,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [sample.step.internal_voltage for sample in trajectory],
        reference_voltage,
        atol=1e-9,
    )


def test_rotor_trapezoidal_update_matches_initial_rotor_current_conversion():
    parameters, physical = _machine()
    prefault_voltage = 12_000 * np.exp(1j * 0.3)
    prefault = initialize_rotor_state(
        parameters,
        physical,
        prefault_voltage,
        d_axis_voltage_change_v=0,
        q_axis_voltage_change_v=0,
    )
    d_voltage_change = 300.0
    q_voltage_change = -150.0
    faulted = initialize_rotor_state(
        parameters,
        physical,
        prefault_voltage,
        d_axis_voltage_change_v=d_voltage_change,
        q_axis_voltage_change_v=q_voltage_change,
    )
    d_current_change = d_voltage_change / (
        parameters.d_axis.synchronous_reactance
        - parameters.d_axis.subtransient_reactance
    )
    q_current_change = q_voltage_change / (
        parameters.d_axis.synchronous_reactance
        - parameters.q_axis.subtransient_reactance
    )

    stepped = advance_rotor_state_trapezoidal(
        physical,
        prefault.rotor_state,
        time_step_s=1e-9,
        field_voltage_v=field_winding_voltage_for_prefault_state(
            physical, prefault_voltage
        ),
        stator_d_current_new_a=d_current_change,
        stator_q_current_new_a=q_current_change,
    )

    np.testing.assert_allclose(
        stepped.as_array(),
        faulted.rotor_state.as_array(),
        rtol=5e-5,
        atol=1e-4,
    )
