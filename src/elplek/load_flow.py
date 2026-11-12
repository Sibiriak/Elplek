"""Newton-Raphson load flow for fixed-voltage slack and PQ buses.

Power values use the same per-unit base as the network admittances. Positive
specified complex power means injection into the network; loads therefore use
negative real and reactive power.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from .model import Network
from .ybus import build_ybus


@dataclass(frozen=True, slots=True)
class LoadFlowResult:
    """Solved bus voltages and calculated complex bus power injections."""

    bus_voltages: Mapping[int, complex]
    bus_power_injections: Mapping[int, complex]
    iterations: int
    maximum_mismatch: float


class LoadFlowConvergenceError(RuntimeError):
    """Raised when the Newton-Raphson iterations do not converge."""


def solve_load_flow(
    network: Network,
    *,
    slack_bus_id: int,
    slack_voltage: complex = 1 + 0j,
    specified_power_injections: Mapping[int, complex] | None = None,
    pv_bus_voltage_magnitudes: Mapping[int, float] | None = None,
    initial_voltages: Mapping[int, complex] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> LoadFlowResult:
    """Solve a balanced positive-sequence load flow with slack, PQ, and PV buses.

    ``specified_power_injections`` maps bus IDs to net ``P + jQ`` injections in
    per unit. Omitted PQ buses have zero specified injection. A PV bus uses the
    real part of its specified injection and its voltage magnitude from
    ``pv_bus_voltage_magnitudes``; its reactive injection is solved. The
    slack-bus power is solved from the network equations. Generator reactive
    limits, controls, and constant-impedance load conversion are not part of
    this API.
    """
    if not network.buses:
        raise ValueError("Load flow requires at least one bus")
    if slack_bus_id not in network.bus_index:
        raise ValueError(f"Unknown slack bus id: {slack_bus_id}")
    if not (isinstance(max_iterations, int) and not isinstance(max_iterations, bool) and max_iterations > 0):
        raise ValueError("max_iterations must be a positive integer")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    slack_voltage = complex(slack_voltage)
    if not (isfinite(slack_voltage.real) and isfinite(slack_voltage.imag)) or abs(slack_voltage) == 0:
        raise ValueError("slack_voltage must be a finite, nonzero complex value")

    index = network.bus_index
    slack = index[slack_bus_id]
    specified_power_injections = specified_power_injections or {}
    pv_bus_voltage_magnitudes = pv_bus_voltage_magnitudes or {}
    for bus_id, power in specified_power_injections.items():
        if bus_id not in index:
            raise ValueError(f"Power injection references unknown bus id: {bus_id}")
        value = complex(power)
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise ValueError("Specified power injections must be finite")
    if slack_bus_id in specified_power_injections:
        raise ValueError("Slack-bus power is solved by the load flow")

    for bus_id, magnitude in pv_bus_voltage_magnitudes.items():
        if bus_id not in index:
            raise ValueError(f"PV voltage target references unknown bus id: {bus_id}")
        if bus_id not in specified_power_injections:
            raise ValueError(f"PV bus {bus_id} requires a specified active-power injection")
        if not isfinite(magnitude) or magnitude <= 0:
            raise ValueError("PV bus voltage magnitudes must be finite and positive")
        if abs(complex(specified_power_injections[bus_id]).imag) > tolerance:
            raise ValueError("PV-bus reactive power is solved; specify only active power")
    pv = np.array([index[bus_id] for bus_id in pv_bus_voltage_magnitudes], dtype=int)
    if slack in pv:
        raise ValueError("Slack bus cannot also be a PV bus")

    for bus_id, voltage in (initial_voltages or {}).items():
        if bus_id not in index:
            raise ValueError(f"Initial voltage references unknown bus id: {bus_id}")
        value = complex(voltage)
        if not (isfinite(value.real) and isfinite(value.imag)) or abs(value) == 0:
            raise ValueError("Initial bus voltages must be finite and nonzero")

    ybus = build_ybus(network)
    n_bus = len(network.buses)
    non_slack = np.array([i for i in range(n_bus) if i != slack], dtype=int)
    pq = np.array([i for i in non_slack if i not in set(pv)], dtype=int)
    specified = np.zeros(n_bus, dtype=np.complex128)
    for bus_id, power in specified_power_injections.items():
        specified[index[bus_id]] = complex(power)

    voltage = np.ones(n_bus, dtype=np.complex128)
    for bus_id, value in (initial_voltages or {}).items():
        voltage[index[bus_id]] = complex(value)
    for bus_id, magnitude in pv_bus_voltage_magnitudes.items():
        pv_index = index[bus_id]
        angle = np.angle(voltage[pv_index])
        voltage[pv_index] = magnitude * np.exp(1j * angle)
    voltage[slack] = slack_voltage

    if non_slack.size == 0:
        injection = voltage * np.conjugate(ybus @ voltage)
        return _result(network, voltage, injection, 0, 0.0)

    g, b = ybus.real, ybus.imag
    for iteration in range(max_iterations + 1):
        injection = voltage * np.conjugate(ybus @ voltage)
        mismatch_p = specified.real[non_slack] - injection.real[non_slack]
        mismatch_q = specified.imag[pq] - injection.imag[pq]
        mismatch = np.concatenate((mismatch_p, mismatch_q))
        maximum_mismatch = float(np.max(np.abs(mismatch)))
        if maximum_mismatch <= tolerance:
            return _result(network, voltage, injection, iteration, maximum_mismatch)
        if iteration == max_iterations:
            break

        angle = np.angle(voltage)
        magnitude = np.abs(voltage)
        h = np.empty((non_slack.size, non_slack.size), dtype=float)
        n = np.empty((non_slack.size, pq.size), dtype=float)
        m = np.empty((pq.size, non_slack.size), dtype=float)
        ell = np.empty((pq.size, pq.size), dtype=float)
        p, q = injection.real, injection.imag

        for row, i in enumerate(non_slack):
            for col, j in enumerate(non_slack):
                if i == j:
                    h[row, col] = -q[i] - b[i, i] * magnitude[i] ** 2
                else:
                    delta = angle[i] - angle[j]
                    h[row, col] = magnitude[i] * magnitude[j] * (
                        g[i, j] * np.sin(delta) - b[i, j] * np.cos(delta)
                    )
        for row, i in enumerate(non_slack):
            for col, j in enumerate(pq):
                if i == j:
                    n[row, col] = p[i] / magnitude[i] + g[i, i] * magnitude[i]
                else:
                    delta = angle[i] - angle[j]
                    n[row, col] = magnitude[i] * (
                        g[i, j] * np.cos(delta) + b[i, j] * np.sin(delta)
                    )
        for row, i in enumerate(pq):
            for col, j in enumerate(non_slack):
                if i == j:
                    m[row, col] = p[i] - g[i, i] * magnitude[i] ** 2
                else:
                    delta = angle[i] - angle[j]
                    m[row, col] = -magnitude[i] * magnitude[j] * (
                        g[i, j] * np.cos(delta) + b[i, j] * np.sin(delta)
                    )
        for row, i in enumerate(pq):
            for col, j in enumerate(pq):
                if i == j:
                    ell[row, col] = q[i] / magnitude[i] - b[i, i] * magnitude[i]
                else:
                    delta = angle[i] - angle[j]
                    ell[row, col] = magnitude[i] * (
                        g[i, j] * np.sin(delta) - b[i, j] * np.cos(delta)
                    )

        jacobian = np.block([[h, n], [m, ell]])
        try:
            correction = np.linalg.solve(jacobian, mismatch)
        except np.linalg.LinAlgError as exc:
            raise LoadFlowConvergenceError(
                "Load-flow Jacobian is singular; check network connectivity and bus data"
            ) from exc
        if not np.isfinite(correction).all():
            raise LoadFlowConvergenceError("Load-flow iteration produced a non-finite correction")
        new_angle = angle[non_slack] + correction[: non_slack.size]
        voltage[non_slack] = magnitude[non_slack] * np.exp(1j * new_angle)
        if pq.size:
            voltage[pq] = (magnitude[pq] + correction[non_slack.size :]) * np.exp(
                1j * new_angle[np.searchsorted(non_slack, pq)]
            )
        for bus_id, target_magnitude in pv_bus_voltage_magnitudes.items():
            pv_index = index[bus_id]
            voltage[pv_index] = target_magnitude * np.exp(1j * np.angle(voltage[pv_index]))

    raise LoadFlowConvergenceError(
        f"Load flow did not converge in {max_iterations} iterations; "
        f"maximum power mismatch is {maximum_mismatch:.6g} pu"
    )


def _result(
    network: Network,
    voltage: np.ndarray,
    injection: np.ndarray,
    iterations: int,
    maximum_mismatch: float,
) -> LoadFlowResult:
    return LoadFlowResult(
        bus_voltages={bus.id: complex(voltage[i]) for i, bus in enumerate(network.buses)},
        bus_power_injections={bus.id: complex(injection[i]) for i, bus in enumerate(network.buses)},
        iterations=iterations,
        maximum_mismatch=maximum_mismatch,
    )
