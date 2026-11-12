import numpy as np
import pytest

from elplek.load_flow import LoadFlowConvergenceError, solve_load_flow
from elplek.model import Branch, Bus, Network


def _two_bus_network() -> Network:
    network = Network()
    network.add_bus(Bus(10, "Slack"))
    network.add_bus(Bus(20, "Load"))
    network.add_branch(Branch(10, 20, 0.01, 0.1))
    return network


def test_no_load_load_flow_preserves_slack_voltage_at_all_buses():
    result = solve_load_flow(_two_bus_network(), slack_bus_id=10)

    np.testing.assert_allclose(result.bus_voltages[10], 1 + 0j)
    np.testing.assert_allclose(result.bus_voltages[20], 1 + 0j)
    np.testing.assert_allclose(result.bus_power_injections[10], 0j, atol=1e-12)
    np.testing.assert_allclose(result.bus_power_injections[20], 0j, atol=1e-12)
    assert result.iterations == 0


def test_two_bus_load_flow_solves_pq_voltage_and_slack_injection():
    network = _two_bus_network()
    load = -0.5 - 0.2j

    result = solve_load_flow(
        network,
        slack_bus_id=10,
        slack_voltage=1.02 + 0j,
        specified_power_injections={20: load},
    )

    assert result.iterations > 0
    assert result.maximum_mismatch <= 1e-10
    assert abs(result.bus_voltages[20]) < 1.02
    np.testing.assert_allclose(result.bus_power_injections[20], load, atol=1e-10)
    assert result.bus_power_injections[10].real > 0.5
    assert result.bus_power_injections[10].imag > 0.2


def test_load_flow_accepts_nonconsecutive_bus_ids_and_initial_guess():
    network = _two_bus_network()
    result = solve_load_flow(
        network,
        slack_bus_id=10,
        specified_power_injections={20: -0.2 - 0.05j},
        initial_voltages={20: 0.98 - 0.01j},
    )

    assert set(result.bus_voltages) == {10, 20}
    assert result.maximum_mismatch <= 1e-10


def test_load_flow_holds_pv_voltage_and_solves_reactive_injection():
    result = solve_load_flow(
        _two_bus_network(),
        slack_bus_id=10,
        specified_power_injections={20: 0.5 + 0j},
        pv_bus_voltage_magnitudes={20: 1.02},
    )

    assert result.maximum_mismatch <= 1e-10
    np.testing.assert_allclose(abs(result.bus_voltages[20]), 1.02, atol=1e-12)
    np.testing.assert_allclose(result.bus_power_injections[20].real, 0.5, atol=1e-10)
    assert result.bus_power_injections[20].imag > 0


def test_load_flow_supports_pv_and_pq_buses_in_same_network():
    network = _two_bus_network()
    network.add_bus(Bus(35, "PQ"))
    network.add_branch(Branch(20, 35, 0.02, 0.15))

    result = solve_load_flow(
        network,
        slack_bus_id=10,
        specified_power_injections={20: 0.3 + 0j, 35: -0.4 - 0.15j},
        pv_bus_voltage_magnitudes={20: 1.02},
    )

    assert result.maximum_mismatch <= 1e-10
    np.testing.assert_allclose(abs(result.bus_voltages[20]), 1.02, atol=1e-12)
    np.testing.assert_allclose(result.bus_power_injections[20].real, 0.3, atol=1e-10)
    np.testing.assert_allclose(result.bus_power_injections[35], -0.4 - 0.15j, atol=1e-10)


def test_pv_bus_changes_to_pq_when_reactive_limit_is_reached():
    result = solve_load_flow(
        _two_bus_network(),
        slack_bus_id=10,
        specified_power_injections={20: 0.5 + 0j},
        pv_bus_voltage_magnitudes={20: 1.02},
        pv_bus_reactive_limits={20: (-0.01, 0.01)},
    )

    assert result.maximum_mismatch <= 1e-10
    np.testing.assert_allclose(result.bus_power_injections[20].real, 0.5, atol=1e-10)
    np.testing.assert_allclose(result.bus_power_injections[20].imag, 0.01, atol=1e-10)
    assert abs(result.bus_voltages[20]) != pytest.approx(1.02, abs=1e-8)


def test_load_flow_rejects_unknown_slack_or_injection_buses():
    network = _two_bus_network()
    with pytest.raises(ValueError, match="Unknown slack"):
        solve_load_flow(network, slack_bus_id=99)
    with pytest.raises(ValueError, match="unknown bus"):
        solve_load_flow(network, slack_bus_id=10, specified_power_injections={99: -1j})


def test_load_flow_rejects_slack_power_specification():
    with pytest.raises(ValueError, match="Slack-bus power"):
        solve_load_flow(_two_bus_network(), slack_bus_id=10, specified_power_injections={10: 1 + 0j})


def test_load_flow_validates_pv_bus_data():
    network = _two_bus_network()
    with pytest.raises(ValueError, match="unknown bus"):
        solve_load_flow(network, slack_bus_id=10, pv_bus_voltage_magnitudes={99: 1.0})
    with pytest.raises(ValueError, match="requires a specified active-power"):
        solve_load_flow(network, slack_bus_id=10, pv_bus_voltage_magnitudes={20: 1.0})
    with pytest.raises(ValueError, match="specify only active power"):
        solve_load_flow(
            network,
            slack_bus_id=10,
            specified_power_injections={20: 0.1 + 0.1j},
            pv_bus_voltage_magnitudes={20: 1.0},
        )
    with pytest.raises(ValueError, match="Reactive limits require PV"):
        solve_load_flow(network, slack_bus_id=10, pv_bus_reactive_limits={20: (-1.0, 1.0)})
    with pytest.raises(ValueError, match=r"finite \(Q_min, Q_max\)"):
        solve_load_flow(
            network,
            slack_bus_id=10,
            specified_power_injections={20: 0.1 + 0j},
            pv_bus_voltage_magnitudes={20: 1.0},
            pv_bus_reactive_limits={20: (1.0, -1.0)},
        )


def test_load_flow_reports_nonconvergence():
    with pytest.raises(LoadFlowConvergenceError, match="did not converge"):
        solve_load_flow(
            _two_bus_network(),
            slack_bus_id=10,
            specified_power_injections={20: -0.5 - 0.2j},
            max_iterations=1,
        )
