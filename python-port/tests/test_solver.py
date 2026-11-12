import numpy as np
import pytest
from elplek import Branch, Bus, Network, SolveError, solve_network

def test_two_bus_voltage_with_series_impedance_and_shunt():
    # V1=100 V, Zline=1 ohm, Zload=1 ohm => V2=50 V.
    network = Network(
        buses=[Bus("source", slack_voltage=100+0j), Bus("load")],
        branches=[Branch("line", "source", "load", 1+0j)],
        shunts_s={"load": 1+0j},
    )
    result = solve_network(network)
    assert result.bus_voltages_v["source"] == pytest.approx(100+0j)
    assert result.bus_voltages_v["load"] == pytest.approx(50+0j)

def test_ybus_is_symmetric_for_reciprocal_branches():
    network = Network(
        buses=[Bus("source", slack_voltage=230+0j), Bus("b2"), Bus("b3")],
        branches=[Branch("l1", "source", "b2", 1+1j), Branch("l2", "b2", "b3", 2+3j)],
    )
    result = solve_network(network)
    assert np.allclose(result.ybus_s, result.ybus_s.T)
    assert set(result.bus_voltages_v) == {"source", "b2", "b3"}

def test_requires_exactly_one_slack_bus():
    with pytest.raises(ValueError, match="Exactly one slack bus"):
        Network(buses=[Bus("a"), Bus("b")])

def test_disconnected_unknown_bus_raises_solve_error():
    network = Network(buses=[Bus("source", slack_voltage=1+0j), Bus("isolated")])
    with pytest.raises(SolveError, match="singular"):
        solve_network(network)
