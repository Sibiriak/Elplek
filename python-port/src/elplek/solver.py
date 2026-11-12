"""Linear nodal-admittance solver; not Elplek's full load-flow algorithm."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .model import Network

class SolveError(RuntimeError):
    """Raised when network equations cannot be solved."""

@dataclass(frozen=True, slots=True)
class SolveResult:
    bus_voltages_v: dict[str, complex]
    ybus_s: NDArray[np.complex128]

def build_ybus(network: Network) -> NDArray[np.complex128]:
    """Build the bus-admittance matrix in siemens."""
    network.validate()
    index = {bus.name: i for i, bus in enumerate(network.buses)}
    ybus = np.zeros((len(network.buses), len(network.buses)), dtype=np.complex128)
    for branch in network.branches:
        i, j = index[branch.from_bus], index[branch.to_bus]
        y = 1 / branch.impedance_ohm
        ybus[i, i] += y + branch.shunt_from_s
        ybus[j, j] += y + branch.shunt_to_s
        ybus[i, j] -= y
        ybus[j, i] -= y
    for bus_name, admittance in network.shunts_s.items():
        ybus[index[bus_name], index[bus_name]] += admittance
    return ybus

def solve_network(network: Network) -> SolveResult:
    """Solve a linear AC network with one fixed-voltage slack bus."""
    ybus = build_ybus(network)
    slack = next(i for i, bus in enumerate(network.buses) if bus.is_slack)
    unknown = [i for i in range(len(network.buses)) if i != slack]
    voltage = np.zeros(len(network.buses), dtype=np.complex128)
    voltage[slack] = network.buses[slack].slack_voltage
    if unknown:
        yuu = ybus[np.ix_(unknown, unknown)]
        yus = ybus[np.ix_(unknown, [slack])].ravel()
        try:
            voltage[unknown] = np.linalg.solve(yuu, -yus * voltage[slack])
        except np.linalg.LinAlgError as exc:
            raise SolveError("Network matrix is singular; check connectivity and reference bus") from exc
    return SolveResult({bus.name: complex(voltage[i]) for i, bus in enumerate(network.buses)}, ybus)
