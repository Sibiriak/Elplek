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


def test_load_flow_rejects_unknown_slack_or_injection_buses():
    network = _two_bus_network()
    with pytest.raises(ValueError, match="Unknown slack"):
        solve_load_flow(network, slack_bus_id=99)
    with pytest.raises(ValueError, match="unknown bus"):
        solve_load_flow(network, slack_bus_id=10, specified_power_injections={99: -1j})


def test_load_flow_rejects_slack_power_specification():
    with pytest.raises(ValueError, match="Slack-bus power"):
        solve_load_flow(_two_bus_network(), slack_bus_id=10, specified_power_injections={10: 1 + 0j})


def test_load_flow_reports_nonconvergence():
    with pytest.raises(LoadFlowConvergenceError, match="did not converge"):
        solve_load_flow(
            _two_bus_network(),
            slack_bus_id=10,
            specified_power_injections={20: -0.5 - 0.2j},
            max_iterations=1,
        )
