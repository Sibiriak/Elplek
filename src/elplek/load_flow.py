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
    initial_voltages: Mapping[int, complex] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> LoadFlowResult:
    """Solve a balanced positive-sequence load flow with one slack and PQ buses.

    ``specified_power_injections`` maps bus IDs to net ``P + jQ`` injections in
    per unit. Omitted buses have zero specified injection. The slack-bus power
    is solved from the network equations. PV buses, generator reactive limits,
    controls, and constant-impedance load conversion are not part of this API.
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
    specified_power_injections = specified_power_injections or {}
    for bus_id, power in specified_power_injections.items():
        if bus_id not in index:
            raise ValueError(f"Power injection references unknown bus id: {bus_id}")
        value = complex(power)
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise ValueError("Specified power injections must be finite")
    if slack_bus_id in specified_power_injections:
        raise ValueError("Slack-bus power is solved by the load flow")

    for bus_id, voltage in (initial_voltages or {}).items():
        if bus_id not in index:
            raise ValueError(f"Initial voltage references unknown bus id: {bus_id}")
        value = complex(voltage)
        if not (isfinite(value.real) and isfinite(value.imag)) or abs(value) == 0:
            raise ValueError("Initial bus voltages must be finite and nonzero")

    ybus = build_ybus(network)
    n_bus = len(network.buses)
    slack = index[slack_bus_id]
    pq = np.array([i for i in range(n_bus) if i != slack], dtype=int)
    specified = np.zeros(n_bus, dtype=np.complex128)
    for bus_id, power in specified_power_injections.items():
        specified[index[bus_id]] = complex(power)

    voltage = np.ones(n_bus, dtype=np.complex128)
    for bus_id, value in (initial_voltages or {}).items():
        voltage[index[bus_id]] = complex(value)
    voltage[slack] = slack_voltage

    if pq.size == 0:
        injection = voltage * np.conjugate(ybus @ voltage)
        return _result(network, voltage, injection, 0, 0.0)

    g, b = ybus.real, ybus.imag
    for iteration in range(max_iterations + 1):
        injection = voltage * np.conjugate(ybus @ voltage)
        mismatch_p = specified.real[pq] - injection.real[pq]
        mismatch_q = specified.imag[pq] - injection.imag[pq]
        mismatch = np.concatenate((mismatch_p, mismatch_q))
        maximum_mismatch = float(np.max(np.abs(mismatch)))
        if maximum_mismatch <= tolerance:
            return _result(network, voltage, injection, iteration, maximum_mismatch)
        if iteration == max_iterations:
            break

        angle = np.angle(voltage)
        magnitude = np.abs(voltage)
        h = np.empty((pq.size, pq.size), dtype=float)
        n = np.empty_like(h)
        m = np.empty_like(h)
        ell = np.empty_like(h)
        p, q = injection.real, injection.imag

        for row, i in enumerate(pq):
            for col, j in enumerate(pq):
                if i == j:
                    h[row, col] = -q[i] - b[i, i] * magnitude[i] ** 2
                    n[row, col] = p[i] / magnitude[i] + g[i, i] * magnitude[i]
                    m[row, col] = p[i] - g[i, i] * magnitude[i] ** 2
                    ell[row, col] = q[i] / magnitude[i] - b[i, i] * magnitude[i]
                else:
                    delta = angle[i] - angle[j]
                    h[row, col] = magnitude[i] * magnitude[j] * (
                        g[i, j] * np.sin(delta) - b[i, j] * np.cos(delta)
                    )
                    n[row, col] = magnitude[i] * (
                        g[i, j] * np.cos(delta) + b[i, j] * np.sin(delta)
                    )
                    m[row, col] = -magnitude[i] * magnitude[j] * (
                        g[i, j] * np.cos(delta) + b[i, j] * np.sin(delta)
                    )
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
        voltage[pq] = (magnitude[pq] + correction[pq.size :]) * np.exp(
            1j * (angle[pq] + correction[: pq.size])
        )

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
