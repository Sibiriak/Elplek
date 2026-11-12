"""Sequence-network adapter for a time-stepped synchronous generator.

The adapter keeps the network linear at each rotor-state iteration. It uses
fixed sequence impedances and caches the sequence-current sensitivities for
the selected fault, then returns generator stator current in the rotor frame.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable

import numpy as np

from .model import Network
from .sequence import FaultType, SequenceFaultResult, solve_sequence_fault
from .sequence_network import calculate_sequence_thevenin
from .synchronous_machine import (
    RotorInitialConditions,
    RotorVoltageCorrections,
    SynchronousMachineParameters,
    SynchronousMachinePhysicalParameters,
    initialize_rotor_state,
    field_winding_voltage_for_prefault_state,
    solve_initial_rotor_voltage_corrections,
)
from .ybus import build_sequence_ybus


@dataclass(frozen=True, slots=True)
class DynamicFaultNetworkResult:
    """Fault and generator-terminal sequence currents for one voltage trial."""

    fault: SequenceFaultResult
    source_positive_current: complex
    source_negative_current: complex
    source_zero_current: complex


@dataclass(frozen=True, slots=True)
class DynamicRotorInitialState:
    """One-machine fault-on rotor state, current, and fixed field excitation."""

    voltage_corrections: RotorVoltageCorrections
    rotor_initial_conditions: RotorInitialConditions
    stator_d_current_a: float
    stator_q_current_a: float
    field_voltage_v: float


@dataclass(frozen=True, slots=True)
class _SequenceCurrentAffineSolver:
    reference_voltage: complex
    base_currents: tuple[complex, complex, complex]
    sensitivities: tuple[complex, complex, complex]

    def __call__(self, source_voltage: complex) -> tuple[complex, complex, complex]:
        voltage = complex(source_voltage)
        if not isfinite(voltage.real) or not isfinite(voltage.imag):
            raise ValueError("source_voltage must be finite")
        offset = voltage - self.reference_voltage
        return tuple(
            base + sensitivity * offset
            for base, sensitivity in zip(self.base_currents, self.sensitivities)
        )


def _make_sequence_current_affine_solver(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    reference_voltage: complex,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
    sensitivity_step_v: float = 1.0,
) -> _SequenceCurrentAffineSolver:
    """Build an affine sequence-current response using Elplek-style dI/dV.

    With fixed impedances, the sequence network is linear. One voltage
    perturbation along the reference-voltage axis therefore determines the
    exact complex current sensitivity. Caching the base currents and dI/dV
    avoids solving all three sequence networks on every rotor iteration.
    """
    reference = complex(reference_voltage)
    if (
        not isfinite(reference.real)
        or not isfinite(reference.imag)
        or abs(reference) == 0
    ):
        raise ValueError("reference_voltage must be finite and nonzero")
    if not isfinite(sensitivity_step_v) or sensitivity_step_v <= 0:
        raise ValueError("sensitivity_step_v must be finite and positive")

    def solve_network(voltage: complex) -> tuple[complex, complex, complex]:
        result = solve_dynamic_network_fault(
            network,
            fault_bus_id,
            fault_type,
            voltage,
            fault_impedance=fault_impedance,
            source_sequence_impedances=source_sequence_impedances,
        )
        return (
            result.source_positive_current,
            result.source_negative_current,
            result.source_zero_current,
        )

    base_currents = solve_network(reference)
    perturbation = reference / abs(reference) * sensitivity_step_v
    perturbed_currents = solve_network(reference + perturbation)
    sensitivities = tuple(
        (perturbed - base) / perturbation
        for base, perturbed in zip(base_currents, perturbed_currents)
    )

    return _SequenceCurrentAffineSolver(reference, base_currents, sensitivities)


def solve_dynamic_network_prefault_source_current(
    network: Network,
    source_voltage: complex | None = None,
) -> complex:
    """Calculate source positive-sequence current before a fault from Y-bus.

    This solves the linear network represented by explicit branch and bus
    admittances. It is not a power-flow solution for constant-power loads.
    The source impedance in ``network`` is used; a later fault-stage impedance
    override does not change this prefault calculation.
    """
    if len(network.sources) != 1:
        raise ValueError("Dynamic fault adapter currently supports one source")
    source = network.sources[0]
    impedance = complex(source.impedance)
    if impedance == 0:
        raise ValueError("Zero positive-sequence source impedance is unsupported")
    voltage = complex(source.voltage if source_voltage is None else source_voltage)
    if not (isfinite(voltage.real) and isfinite(voltage.imag)):
        raise ValueError("source_voltage must be finite")

    ybus = build_sequence_ybus(network, "positive").copy()
    source_index = network.bus_index[source.bus_id]
    source_admittance = 1 / impedance
    ybus[source_index, source_index] += source_admittance
    rhs = np.zeros(len(network.buses), dtype=np.complex128)
    rhs[source_index] += source_admittance * voltage
    try:
        bus_voltages = np.linalg.solve(ybus, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular positive-sequence prefault network") from exc
    return complex((voltage - bus_voltages[source_index]) / impedance)


def _source_impedance(source, sequence: str) -> complex:
    if sequence == "positive":
        impedance = complex(source.impedance)
    elif sequence == "negative":
        impedance = source.negative_sequence_impedance
    else:
        impedance = source.zero_sequence_impedance
    if impedance is None:
        raise ValueError(f"Source {source.name or source.bus_id} has no {sequence} impedance")
    impedance = complex(impedance)
    if impedance == 0:
        raise ValueError(f"Source {source.name or source.bus_id} has zero {sequence} impedance")
    return impedance


def solve_dynamic_network_fault(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    source_voltage: complex,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
) -> DynamicFaultNetworkResult:
    """Solve sequence fault networks for one trial generator voltage.

    This first adapter supports one voltage source, which is the Elplek
    reference case. ``source_sequence_impedances`` may set fixed positive,
    negative and zero impedances for a dynamic step, usually subtransient
    values. The returned source currents include the connected shunt loads.
    """
    if len(network.sources) != 1:
        raise ValueError("Dynamic fault adapter currently supports one source")
    source = network.sources[0]
    voltage = complex(source_voltage)
    if not (isfinite(voltage.real) and isfinite(voltage.imag)):
        raise ValueError("source_voltage must be finite")
    if fault_bus_id not in network.bus_index:
        raise ValueError(f"Unknown bus id: {fault_bus_id}")

    if source_sequence_impedances is not None:
        if len(source_sequence_impedances) != 3:
            raise ValueError("source_sequence_impedances must contain Z1, Z2, and Z0")
        z1, z2, z0 = map(complex, source_sequence_impedances)
        updated_source = replace(
            source,
            voltage=voltage,
            impedance=z1,
            negative_sequence_impedance=z2,
            zero_sequence_impedance=z0,
        )
    else:
        z1 = _source_impedance(source, "positive")
        z2 = _source_impedance(source, "negative")
        z0 = _source_impedance(source, "zero")
        updated_source = replace(source, voltage=voltage)
    if z1 == 0 or z2 == 0 or z0 == 0:
        raise ValueError("All source sequence impedances must be nonzero")

    working = Network(
        buses=list(network.buses),
        branches=list(network.branches),
        sources=[updated_source],
        transformers=list(network.transformers),
        shunts=list(network.shunts),
    )
    equivalent = calculate_sequence_thevenin(working, fault_bus_id)
    fault = solve_sequence_fault(
        fault_type,
        phase_voltage=equivalent.prefault_voltage,
        z1=equivalent.z1,
        z2=equivalent.z2,
        z0=equivalent.z0,
        fault_impedance=fault_impedance,
    )

    source_index = working.bus_index[source.bus_id]
    fault_index = working.bus_index[fault_bus_id]
    sequence_currents = {
        "positive": fault.positive_current,
        "negative": fault.negative_current,
        "zero": fault.zero_current,
    }
    source_currents: dict[str, complex] = {}
    for sequence, impedance, emf in (
        ("positive", z1, voltage),
        ("negative", z2, 0j),
        ("zero", z0, 0j),
    ):
        ybus = build_sequence_ybus(working, sequence).copy()
        source_admittance = 1 / impedance
        ybus[source_index, source_index] += source_admittance
        rhs = np.zeros(len(working.buses), dtype=np.complex128)
        rhs[source_index] += source_admittance * emf
        rhs[fault_index] -= sequence_currents[sequence]
        try:
            voltages = np.linalg.solve(ybus, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Singular {sequence}-sequence fault network") from exc
        source_currents[sequence] = (emf - voltages[source_index]) / impedance

    return DynamicFaultNetworkResult(
        fault=fault,
        source_positive_current=source_currents["positive"],
        source_negative_current=source_currents["negative"],
        source_zero_current=source_currents["zero"],
    )


def make_dynamic_fault_current_solver(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    reference_voltage: complex,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
) -> Callable[[complex], tuple[float, float]]:
    """Build the callback expected by ``solve_coupled_rotor_step``.

    The callback maps positive-sequence source current into Elplek D/Q
    coordinates aligned with ``reference_voltage``: D is the orthogonal
    (imaginary) component and Q is the parallel (real) component. Negative- and
    zero-sequence currents are retained in the fault solution but are not
    mapped into the rotor state yet.
    """
    reference = complex(reference_voltage)
    magnitude = abs(reference)
    if not (isfinite(reference.real) and isfinite(reference.imag)) or magnitude == 0:
        raise ValueError("reference_voltage must be finite and nonzero")
    unit = reference / magnitude
    sequence_currents = _make_sequence_current_affine_solver(
        network,
        fault_bus_id,
        fault_type,
        reference,
        fault_impedance=fault_impedance,
        source_sequence_impedances=source_sequence_impedances,
    )

    def solve(source_voltage: complex) -> tuple[float, float]:
        positive_current, _, _ = sequence_currents(source_voltage)
        aligned_current = positive_current * unit.conjugate()
        return float(aligned_current.imag), float(aligned_current.real)

    # The rotor equations use the same fixed-network current sensitivity that
    # the affine sequence solver cached for these time steps.
    solve.current_sensitivity = sequence_currents.sensitivities[0]
    return solve


def make_dynamic_fault_phase_current_solver(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
) -> Callable[[complex], complex]:
    """Build a callback returning phase-A current at generator terminals.

    The current is reconstructed as the sum of the source positive-, negative-
    and zero-sequence currents. It is suitable for comparing CT phase-A
    measurements. This does not include multiple generators.
    """

    if not network.sources:
        raise ValueError("Dynamic fault adapter currently supports one source")
    sequence_currents = _make_sequence_current_affine_solver(
        network,
        fault_bus_id,
        fault_type,
        network.sources[0].voltage,
        fault_impedance=fault_impedance,
        source_sequence_impedances=source_sequence_impedances,
    )

    def solve(source_voltage: complex) -> complex:
        return sum(sequence_currents(source_voltage))

    return solve


def make_dynamic_fault_positive_sequence_current_solver(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
) -> Callable[[complex], complex]:
    """Build a callback returning positive-sequence source current.

    Elplek ``ConstrVeq`` initializes the rotor from positive-sequence machine
    current. Use ``make_dynamic_fault_phase_current_solver`` to compare phase-A
    current measurements at a CT.
    """

    if not network.sources:
        raise ValueError("Dynamic fault adapter currently supports one source")
    sequence_currents = _make_sequence_current_affine_solver(
        network,
        fault_bus_id,
        fault_type,
        network.sources[0].voltage,
        fault_impedance=fault_impedance,
        source_sequence_impedances=source_sequence_impedances,
    )

    def solve(source_voltage: complex) -> complex:
        return sequence_currents(source_voltage)[0]

    return solve


def initialize_dynamic_fault_rotor_state(
    network: Network,
    fault_bus_id: int,
    fault_type: FaultType,
    machine_parameters: SynchronousMachineParameters,
    physical_parameters: SynchronousMachinePhysicalParameters,
    *,
    fault_impedance: complex = 0j,
    source_sequence_impedances: tuple[complex, complex, complex] | None = None,
    prefault_internal_voltage: complex | None = None,
    prefault_stator_current: complex | None = None,
) -> DynamicRotorInitialState:
    """Calculate one-machine rotor initial conditions from a network fault.

    If ``prefault_stator_current`` is omitted, it is computed as the
    positive-sequence current from explicit network admittances and the source impedance
    currently stored in ``network``. This is appropriate for constant-
    admittance loads; pass the actual load-flow current for a constant-power
    prefault model. ``source_sequence_impedances`` applies only to the faulted
    network solve, allowing subtransient machine impedances there.
    """
    if len(network.sources) != 1:
        raise ValueError("Dynamic fault adapter currently supports one source")
    source = network.sources[0]
    voltage = complex(
        source.voltage
        if prefault_internal_voltage is None
        else prefault_internal_voltage
    )
    prefault_current = (
        solve_dynamic_network_prefault_source_current(network, voltage)
        if prefault_stator_current is None
        else complex(prefault_stator_current)
    )
    fault_current_solver = make_dynamic_fault_positive_sequence_current_solver(
        network,
        fault_bus_id,
        fault_type,
        fault_impedance=fault_impedance,
        source_sequence_impedances=source_sequence_impedances,
    )
    corrections = solve_initial_rotor_voltage_corrections(
        machine_parameters,
        voltage,
        prefault_current,
        fault_current_solver,
    )
    initial_conditions = initialize_rotor_state(
        machine_parameters,
        physical_parameters,
        voltage,
        d_axis_voltage_change_v=corrections.d_axis_voltage_change_v,
        q_axis_voltage_change_v=corrections.q_axis_voltage_change_v,
    )
    stator_current_solver = make_dynamic_fault_current_solver(
        network,
        fault_bus_id,
        fault_type,
        voltage,
        fault_impedance=fault_impedance,
        source_sequence_impedances=source_sequence_impedances,
    )
    stator_d_current, stator_q_current = stator_current_solver(
        initial_conditions.internal_voltage
    )
    field_voltage = field_winding_voltage_for_prefault_state(
        physical_parameters, voltage
    )
    return DynamicRotorInitialState(
        corrections,
        initial_conditions,
        stator_d_current,
        stator_q_current,
        field_voltage,
    )
