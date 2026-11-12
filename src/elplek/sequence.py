"""Sequence-network fault calculations for a single equivalent source.

These analytical helpers are independent of the bus-network solver. Voltages
and impedances are phase-domain sequence equivalents in consistent units.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

from .symmetrical_components import PhaseCurrents, sequence_to_phase_currents

FaultType = Literal[
    "three_phase",
    "line_to_line",
    "line_to_line_ab",
    "line_to_line_bc",
    "line_to_line_ca",
    "single_line_to_ground",
    "double_line_to_ground",
    "double_line_to_ground_ab",
    "double_line_to_ground_bc",
    "double_line_to_ground_ca",
]


@dataclass(frozen=True, slots=True)
class SequenceFaultResult:
    """Sequence currents and a representative fault current.

    For line-to-line faults, ``fault_current`` is the current in one of the
    faulted phases. For ground faults, it is total ground current. For
    three-phase faults, it is phase-a current.
    """

    positive_current: complex
    negative_current: complex
    zero_current: complex
    fault_current: complex
    phase_currents: PhaseCurrents
    phase_voltages: PhaseCurrents


def solve_sequence_fault(
    fault_type: FaultType,
    *,
    phase_voltage: complex,
    z1: complex,
    z2: complex,
    z0: complex = 0j,
    fault_impedance: complex = 0j,
) -> SequenceFaultResult:
    """Calculate standard sequence-network currents for common fault types.

    phase_voltage is the prefault phase-to-neutral Thevenin voltage. z1, z2
    and z0 are positive-, negative- and zero-sequence Thevenin impedances at
    the fault location. ``line_to_line`` retains the legacy B-C orientation;
    ``line_to_line_ab``, ``line_to_line_bc`` and ``line_to_line_ca`` select an
    explicit phase pair. For ground faults, fault impedance is phase-to-ground
    and appears as 3*Zf in the sequence network. This does not assemble
    sequence networks from bus-level network data.
    """
    z1, z2, z0, zf = map(complex, (z1, z2, z0, fault_impedance))
    voltage = complex(phase_voltage)

    if fault_type == "three_phase":
        i1 = voltage / (z1 + zf)
        i2 = i0 = 0j
        fault_current = i1
    elif fault_type == "single_line_to_ground":
        i0 = voltage / (z1 + z2 + z0 + 3 * zf)
        i1 = i2 = i0
        fault_current = 3 * i0
    elif fault_type in (
        "line_to_line",
        "line_to_line_ab",
        "line_to_line_bc",
        "line_to_line_ca",
    ):
        i1 = voltage / (z1 + z2 + zf)
        if fault_type == "line_to_line_ab":
            i2 = -(complex(-0.5, sqrt(3) / 2) ** 2) * i1
        elif fault_type == "line_to_line_ca":
            i2 = -complex(-0.5, sqrt(3) / 2) * i1
        else:  # legacy "line_to_line" means B-C
            i2 = -i1
        i0 = 0j
        fault_current = 0j  # assigned from the phase currents below
    elif fault_type in (
        "double_line_to_ground",
        "double_line_to_ground_ab",
        "double_line_to_ground_bc",
        "double_line_to_ground_ca",
    ):
        z0f = z0 + 3 * zf
        parallel_denominator = z2 + z0f
        if parallel_denominator == 0:
            raise ZeroDivisionError("Negative- and zero-sequence parallel impedance is zero")
        parallel = z2 * z0f / parallel_denominator
        i1 = voltage / (z1 + parallel)
        i2 = -i1 * z0f / parallel_denominator
        i0 = -i1 * z2 / parallel_denominator
        a = complex(-0.5, sqrt(3) / 2)
        if fault_type == "double_line_to_ground_ab":
            i0 *= a
            i2 *= a**2
        elif fault_type == "double_line_to_ground_ca":
            i0 *= a**2
            i2 *= a
        fault_current = 3 * i0
    else:
        raise ValueError(f"Unsupported fault type: {fault_type!r}")

    currents = sequence_to_phase_currents(i0, i1, i2)
    if fault_type in ("line_to_line", "line_to_line_bc"):
        fault_current = currents[1]
    elif fault_type == "line_to_line_ab":
        fault_current = currents[0]
    elif fault_type == "line_to_line_ca":
        fault_current = currents[2]
    v1 = voltage - z1 * i1
    v2 = -z2 * i2
    v0 = -z0 * i0
    phase_voltages = sequence_to_phase_currents(v0, v1, v2)
    return SequenceFaultResult(i1, i2, i0, fault_current, currents, phase_voltages)
