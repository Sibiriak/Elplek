"""Single-phase-equivalent short-circuit analysis for small networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mna import stamp_short_circuit, stamp_source
from .model import Network
from .solver import solve_linear_system
from .ybus import build_ybus


@dataclass(frozen=True, slots=True)
class ShortCircuitResult:
    """Solved phasors for a single fault, including finite fault impedance."""

    bus_voltages: dict[int, complex]
    source_currents: dict[str, complex]
    fault_current: complex


def solve_short_circuit(
    network: Network,
    fault_bus_id: int,
    *,
    fault_impedance: complex = 0j,
) -> ShortCircuitResult:
    """Solve one fault using MNA and the positive-sequence-style equivalent.

    The network must contain at least one source. The conventional Y-bus
    supports series R+jX branches, two-winding transformer series equivalents
    (including magnitude taps and clock phase shift), and constant shunt
    admittances. Sources are represented by voltage behind series impedance;
    fault_impedance connects the faulted bus to ground.

    This is not an asymmetrical-fault sequence solver or a full IEC 60909
    calculation. It does not run load flow, apply IEC voltage factors, or model
    detailed transformer winding/zero-sequence behavior in this single-phase
    equivalent.
    """
    if not network.buses:
        raise ValueError("Network must contain at least one bus")
    if not network.sources:
        raise ValueError("Network must contain at least one source")
    index = network.bus_index
    if fault_bus_id not in index:
        raise ValueError(f"Unknown fault bus id: {fault_bus_id}")

    bus_count = len(network.buses)
    source_count = len(network.sources)
    fault_current_eq = bus_count + source_count + 1
    dimension = fault_current_eq
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    rhs = np.zeros(dimension, dtype=np.complex128)
    matrix[:bus_count, :bus_count] = build_ybus(network)

    for i, source in enumerate(network.sources):
        current_eq = bus_count + i + 1
        bus_eq = index[source.bus_id] + 1
        stamp_source(
            matrix,
            rhs,
            current_eq=current_eq,
            downstream_eq=bus_eq,
            impedance=source.impedance,
            phase_voltage=source.voltage,
        )

    fault_bus_eq = index[fault_bus_id] + 1
    stamp_short_circuit(
        matrix,
        rhs,
        fault_bus_eq=fault_bus_eq,
        current_eq=fault_current_eq,
        fault_impedance=complex(fault_impedance),
    )
    solution = solve_linear_system(matrix, rhs)

    voltages = {
        bus.id: complex(solution[index[bus.id]])
        for bus in network.buses
    }
    currents = {
        source.name or f"source_{i + 1}": complex(solution[bus_count + i])
        for i, source in enumerate(network.sources)
    }
    return ShortCircuitResult(
        bus_voltages=voltages,
        source_currents=currents,
        fault_current=complex(solution[fault_current_eq - 1]),
    )
