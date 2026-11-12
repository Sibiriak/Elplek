"""Physical rotor-winding parameters for Elplek-style synchronous machines.

This module ports the parameter conversion in ``GenProc.SolveGenPar``. It
converts steady-state/transient/subtransient reactances and short-circuit time
constants into the winding inductances used by the dynamic rotor equations.
The module also provides an isolated trapezoidal state update and a generic
fixed-point coupling hook for network solvers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class AxisReactanceData:
    """Reactances and short-circuit time constants for one rotor axis."""

    synchronous_reactance: float
    transient_reactance: float
    subtransient_reactance: float
    transient_time_constant: float
    subtransient_time_constant: float


@dataclass(frozen=True, slots=True)
class AxisWindingParameters:
    """Physical two-winding rotor equivalent returned by Elplek conversion."""

    main_inductance: float
    damping_inductance: float
    mutual_inductance: float
    main_to_stator_mutual: float
    damping_to_stator_mutual: float
    main_resistance: float
    damping_resistance: float
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class SynchronousMachineParameters:
    """SLD-style synchronous-generator data needed by the rotor model."""

    stator_resistance: float
    d_axis: AxisReactanceData
    q_axis: AxisReactanceData
    frequency_hz: float = 50.0
    parallel_units: int = 1


@dataclass(frozen=True, slots=True)
class SynchronousMachinePhysicalParameters:
    """Four rotor winding circuits and stator parameters for dynamic solving."""

    stator_resistance: float
    stator_inductance: float
    angular_frequency: float
    d_axis: AxisWindingParameters
    q_axis: AxisWindingParameters


@dataclass(frozen=True, slots=True)
class RotorCurrentState:
    """Currents in field, d-damper, q-damper and second q-damper windings."""

    field: float
    d_damper: float
    q_damper: float
    p_damper: float

    def as_array(self) -> np.ndarray:
        """Return winding currents in the Elplek state ordering."""
        return np.asarray(
            (self.field, self.d_damper, self.q_damper, self.p_damper),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class RotorStepResult:
    """Converged one-step rotor and network state."""

    rotor_state: RotorCurrentState
    internal_voltage: complex
    stator_d_current: float
    stator_q_current: float
    iterations: int


@dataclass(frozen=True, slots=True)
class RotorTrajectorySample:
    """One converged sample from a multi-step rotor/network integration."""

    time_s: float
    step: RotorStepResult


@dataclass(frozen=True, slots=True)
class RotorInitialConditions:
    """Fault-on rotor currents and internal voltage at the initial instant."""

    rotor_state: RotorCurrentState
    internal_voltage: complex


@dataclass(frozen=True, slots=True)
class RotorVoltageCorrections:
    """D/Q internal-voltage changes from the single-machine fault equations."""

    d_axis_voltage_change_v: float
    q_axis_voltage_change_v: float
    prefault_current: complex
    faulted_current: complex
    current_derivative: complex


def solve_initial_rotor_voltage_corrections(
    parameters: SynchronousMachineParameters,
    prefault_internal_voltage: complex,
    prefault_stator_current: complex,
    solve_faulted_stator_current: Callable[[complex], complex],
    *,
    derivative_step_v: float = 1.0,
) -> RotorVoltageCorrections:
    """Solve the one-machine D/Q voltage correction from Elplek ``ConstrVeq``.

    The callback returns the machine positive-sequence stator current during
    the fault for a trial internal positive-sequence voltage.
    ``prefault_stator_current`` must use the same positive-sequence convention.
    Elplek uses this generator current in ``ConstrVeq``; phase-A current is the
    sum of all three sequences and is the right quantity for CT outputs.
    The derivative is evaluated with a centered perturbation along the
    prefault-voltage axis; the network model is linear, so this is equivalent
    to Elplek's unit-voltage sensitivity solve.

    This is the single synchronous-machine equation only. Networks with
    multiple machines or asynchronous-motor couplings require the full
    multi-machine matrix from Elplek ``ConstrVeq``.
    """
    voltage = complex(prefault_internal_voltage)
    prefault_current = complex(prefault_stator_current)
    if not (isfinite(voltage.real) and isfinite(voltage.imag)) or abs(voltage) == 0:
        raise ValueError("prefault_internal_voltage must be finite and nonzero")
    if not (
        isfinite(prefault_current.real) and isfinite(prefault_current.imag)
    ):
        raise ValueError("prefault_stator_current must be finite")
    if not isfinite(derivative_step_v) or derivative_step_v <= 0:
        raise ValueError("derivative_step_v must be finite and positive")

    unit = voltage / abs(voltage)
    current = complex(solve_faulted_stator_current(voltage))
    current_plus = complex(
        solve_faulted_stator_current(voltage + derivative_step_v * unit)
    )
    current_minus = complex(
        solve_faulted_stator_current(voltage - derivative_step_v * unit)
    )
    currents = (current, current_plus, current_minus)
    if any(not (isfinite(item.real) and isfinite(item.imag)) for item in currents):
        raise ValueError("Fault-current callback returned a non-finite value")
    derivative = (current_plus - current_minus) * unit.conjugate() / (
        2 * derivative_step_v
    )

    d_reactance = (
        parameters.d_axis.synchronous_reactance
        - parameters.d_axis.subtransient_reactance
    )
    q_reactance = (
        parameters.d_axis.synchronous_reactance
        - parameters.q_axis.subtransient_reactance
    )
    if d_reactance <= 0 or q_reactance <= 0:
        raise ValueError("Initial voltage solve requires positive Xd-Xd'' and Xd-Xq''")

    prefault_aligned = prefault_current * unit.conjugate()
    faulted_aligned = current * unit.conjugate()
    d_i, d_r = derivative.imag, derivative.real
    # Elplek stores Quatre as (rr, ir, ri, ii). Calc_dIdV saves the
    # parallel column (a, b), then calls PutdIdV(dIrdV, -dIidV, 2),
    # yielding the orthogonal column (ri, ii)=(-b, a).
    matrix = np.asarray(
        [
        [-d_reactance * d_i - 1.0, -d_reactance * d_r],
        [q_reactance * d_r, -q_reactance * d_i - 1.0],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray(
        [
            d_reactance * (faulted_aligned.imag - prefault_aligned.imag),
            -q_reactance * (faulted_aligned.real - prefault_aligned.real),
        ],
        dtype=np.float64,
    )
    try:
        d_voltage, q_voltage = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular initial D/Q voltage correction equations") from exc
    if not np.isfinite((d_voltage, q_voltage)).all():
        raise ValueError("Initial D/Q voltage correction produced non-finite values")
    return RotorVoltageCorrections(
        d_axis_voltage_change_v=float(d_voltage),
        q_axis_voltage_change_v=float(q_voltage),
        prefault_current=prefault_current,
        faulted_current=current,
        current_derivative=derivative,
    )


def initialize_rotor_state(
    parameters: SynchronousMachineParameters,
    physical: SynchronousMachinePhysicalParameters,
    prefault_internal_voltage: complex,
    *,
    d_axis_voltage_change_v: float,
    q_axis_voltage_change_v: float,
) -> RotorInitialConditions:
    """Port Elplek ``InitialRotorCurrents`` for one synchronous machine.

    The voltage changes are the machine D/Q corrections obtained by solving
    Elplek-style ``SolveDQVoltages`` for the initial faulted network state.
    This routine does not derive those corrections itself.
    """
    voltage = complex(prefault_internal_voltage)
    if not (isfinite(voltage.real) and isfinite(voltage.imag)):
        raise ValueError("prefault_internal_voltage must be finite")
    dva, dvq = d_axis_voltage_change_v, q_axis_voltage_change_v
    if not (isfinite(dva) and isfinite(dvq)):
        raise ValueError("D/Q voltage changes must be finite")

    xd = parameters.d_axis.synchronous_reactance
    xdpp = parameters.d_axis.subtransient_reactance
    xqpp = parameters.q_axis.subtransient_reactance
    d_reactance = xd - xdpp
    q_reactance = xd - xqpp
    if d_reactance <= 0 or q_reactance <= 0:
        raise ValueError("Initial rotor state requires positive Xd-Xd'' and Xd-Xq''")

    d, q = physical.d_axis, physical.q_axis
    d_det = d.main_inductance * d.damping_inductance - d.mutual_inductance**2
    q_det = q.main_inductance * q.damping_inductance - q.mutual_inductance**2
    root_two = sqrt(2.0)
    d_i_sin = dva / d_reactance
    d_i_cos = dvq / q_reactance

    if d_det > 0:
        d_damper = (
            root_two * 1.5
            * (d.main_to_stator_mutual * d.mutual_inductance
               - d.damping_to_stator_mutual * d.main_inductance)
            * d_i_sin / d_det
        )
        field_delta = (
            root_two * 1.5
            * (d.mutual_inductance * d.damping_to_stator_mutual
               - d.damping_inductance * d.main_to_stator_mutual)
            * d_i_sin / d_det
        )
    else:
        d_damper = field_delta = 0.0
    if d.main_to_stator_mutual > 0 and physical.angular_frequency > 0:
        field = field_delta - (
            root_two * abs(voltage)
            / (d.main_to_stator_mutual * physical.angular_frequency)
        )
    else:
        field = field_delta

    if q_det > 0:
        q_damper = (
            -root_two * 1.5
            * (q.main_to_stator_mutual * q.damping_inductance
               - q.damping_to_stator_mutual * q.mutual_inductance)
            * d_i_cos / q_det
        )
        p_damper = (
            -root_two * 1.5
            * (q.damping_to_stator_mutual * q.main_inductance
               - q.main_to_stator_mutual * q.mutual_inductance)
            * d_i_cos / q_det
        )
    else:
        q_damper = p_damper = 0.0

    magnitude = abs(voltage)
    if magnitude > 0:
        unit = voltage / magnitude
        initial_voltage = voltage + dva * unit + 1j * dvq * unit
    else:
        initial_voltage = complex(dva, dvq)
    return RotorInitialConditions(
        rotor_state=RotorCurrentState(field, d_damper, q_damper, p_damper),
        internal_voltage=initial_voltage,
    )


def field_winding_voltage_for_prefault_state(
    physical: SynchronousMachinePhysicalParameters,
    prefault_internal_voltage: complex,
) -> float:
    """Return the fixed field voltage that sustains Elplek's prefault state."""
    voltage = complex(prefault_internal_voltage)
    if not (isfinite(voltage.real) and isfinite(voltage.imag)):
        raise ValueError("prefault_internal_voltage must be finite")
    mutual = physical.d_axis.main_to_stator_mutual
    omega = physical.angular_frequency
    if mutual <= 0 or omega <= 0:
        raise ValueError("Physical machine parameters require positive Mf and omega")
    prefault_field_current = -sqrt(2.0) * abs(voltage) / (mutual * omega)
    return physical.d_axis.main_resistance * prefault_field_current


def _validate_axis(name: str, axis: AxisReactanceData) -> None:
    values = (
        axis.synchronous_reactance,
        axis.transient_reactance,
        axis.subtransient_reactance,
        axis.transient_time_constant,
        axis.subtransient_time_constant,
    )
    if any(not isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name}-axis reactances and time constants must be positive")
    if not (
        axis.synchronous_reactance
        > axis.transient_reactance
        > axis.subtransient_reactance
    ):
        raise ValueError(f"{name}-axis reactances must satisfy X > X' > X''")
    if axis.transient_time_constant <= axis.subtransient_time_constant:
        raise ValueError(f"{name}-axis time constants must satisfy T' > T''")


def _inductances_for_mm(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
    omega: float,
    a1: float,
    a2: float,
    b1: float,
    b2: float,
    discriminant: float,
    mm: float,
) -> tuple[float, float, float, float, float, float]:
    if mm <= 0 or discriminant < 0:
        raise ValueError("Invalid rotor-inductance solution branch")

    bm = 0.5 * (
        -(a1 * b2 + b1 * a2) - sqrt(discriminant)
    ) / (a1 * a2 - 1)
    lfd = bm + mm
    lff = a1 * bm + b1 + mm
    ldd = a2 * bm + b2 + mm
    mf = sqrt(data.synchronous_reactance * mm / (1.5 * omega))
    md = mf

    aa = lff - 1.5 * omega * mf * mf / data.synchronous_reactance
    bb = lfd - 1.5 * omega * mf * md / data.synchronous_reactance
    cc = ldd - 1.5 * omega * md * md / data.synchronous_reactance
    apu = aa * damping_resistance - cc * main_resistance
    root = sqrt(
        apu * apu + 4 * bb * bb * main_resistance * damping_resistance
    )
    denominator = 2 * bb * damping_resistance
    if denominator == 0:
        raise ValueError("Singular rotor-inductance coefficient")
    k1 = (-apu + root) / denominator
    k2 = (-apu - root) / denominator

    determinant = lff * ldd - lfd * lfd
    if determinant == 0 or data.subtransient_reactance == 0:
        raise ValueError("Singular rotor-inductance determinant")
    ifo = 1.5 * (-ldd * mf + lfd * md) / (
        data.subtransient_reactance * determinant
    )
    ido = 1.5 * (-lff * md + lfd * mf) / (
        data.subtransient_reactance * determinant
    )
    y1o = main_resistance * ifo + k2 * damping_resistance * ido
    if k2 == k1:
        raise ValueError("Singular diagonalized rotor coefficient")
    c1 = (
        -omega
        / (k2 - k1)
        * y1o
        * (k2 * mf / main_resistance - md / damping_resistance)
        / data.synchronous_reactance
        - (1 / data.transient_reactance - 1 / data.synchronous_reactance)
    )
    return lff, ldd, lfd, mf, md, c1


def _zero_discriminant(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
) -> tuple[float, float, float, float, float, float, float, float] | None:
    rf, rd = main_resistance, damping_resistance
    t1, t2 = data.transient_time_constant, data.subtransient_time_constant
    if rf == rd or rf == 0 or rd == 0:
        return None

    a1 = 2 * rf / (rf - rd)
    a2 = 2 * rd / (rd - rf)
    beta = rf * rd * (t1 + t2) / (rf - rd)
    gamma = 4 * (rf + rd) ** 2 * t1 * t2 * rf * rd / (rf - rd) ** 2
    aa = a1 * a2
    bb = 2 * beta * (a1 - a2)
    cc = beta * beta * ((a1 + a2) ** 2 - 4) - gamma
    disc = bb * bb - 4 * aa * cc
    if disc < 0 or aa == 0:
        return None

    root = sqrt(disc)
    k1 = (-bb + root) / (2 * aa)
    k2 = (-bb - root) / (2 * aa)
    delta = (data.synchronous_reactance - data.subtransient_reactance)
    if k1 == 0 or k2 == 0:
        return None
    mm1 = delta * t1 * t2 * rf * rd / (k1 * data.subtransient_reactance)
    mm2 = delta * t1 * t2 * rf * rd / (k2 * data.subtransient_reactance)
    b11 = 0.5 * k1 * a1 - beta
    b12 = 0.5 * k1 * a2 + beta
    b21 = 0.5 * k2 * a1 - beta
    b22 = 0.5 * k2 * a2 + beta
    return mm1, mm2, a1, a2, b11, b12, b21, b22


def _discriminant_coefficients(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
    mm: float,
) -> tuple[float, float, float, float, float]:
    rf, rd = main_resistance, damping_resistance
    t1, t2 = data.transient_time_constant, data.subtransient_time_constant
    a1 = 2 * rf / (rf - rd)
    a2 = 2 * rd / (rd - rf)
    apu = (
        (data.synchronous_reactance - data.subtransient_reactance)
        * t1
        * t2
        * rf
        * rd
        / (data.subtransient_reactance * mm)
    )
    b1 = (apu * rf - rf * rd * (t1 + t2)) / (rf - rd)
    b2 = (apu * rd - rf * rd * (t1 + t2)) / (rd - rf)
    u = a1 * b2 + b1 * a2
    v = (b1 * b2 - t1 * t2 * rf * rd) * (a1 * a2 - 1)
    return a1, a2, b1, b2, u * u - 4 * v


def _solve_inductances(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
    frequency_hz: float,
) -> AxisWindingParameters | None:
    roots = _zero_discriminant(data, main_resistance, damping_resistance)
    if roots is None:
        return None
    mm1, mm2, a1, a2, b11, b12, b21, b22 = roots
    if mm1 <= 0 or mm2 <= 0:
        return None

    omega = 2 * pi * frequency_hz
    t1, t2 = data.transient_time_constant, data.subtransient_time_constant
    ld = data.synchronous_reactance / omega
    ldpp = data.subtransient_reactance / omega
    dd = (
        (t1 - t2) * data.synchronous_reactance / data.transient_reactance
        + t2 * data.synchronous_reactance / data.subtransient_reactance
    )
    apu = 1.5 * (1 / main_resistance + 1 / damping_resistance) / ld
    apv = dd + t1 * t2 * ld / (dd * ldpp) - (t1 + t2)
    mm_guess = apv / apu if apu > 0 and apv > 0 else 0.001
    trial = 1.5 * omega * mm_guess * mm_guess / data.synchronous_reactance

    def evaluate(mm: float, b1: float, b2: float, forced_disc: float | None = None):
        a1c, a2c, b1c, b2c, disc = _discriminant_coefficients(
            data, main_resistance, damping_resistance, mm
        )
        if forced_disc is not None:
            disc = forced_disc
        if disc < 0:
            return None
        return _inductances_for_mm(
            data,
            main_resistance,
            damping_resistance,
            omega,
            a1c,
            a2c,
            b1,
            b2,
            disc,
            mm,
        )

    edge1 = evaluate(mm1, b11, b12, 0.0)
    edge2 = evaluate(mm2, b21, b22, 0.0)
    if edge1 is None or edge2 is None or edge1[-1] * edge2[-1] >= 0:
        return None

    lo, hi = (mm1, mm2) if mm1 < mm2 else (mm2, mm1)
    f_lo, f_hi = (edge1[-1], edge2[-1]) if mm1 < mm2 else (edge2[-1], edge1[-1])
    best = None
    for _ in range(10):
        denom = f_hi - f_lo
        mm = (f_hi * lo - f_lo * hi) / denom if denom else 0.5 * (lo + hi)
        a1c, a2c, b1c, b2c, disc = _discriminant_coefficients(
            data, main_resistance, damping_resistance, mm
        )
        if disc < 0:
            disc = 0.0
        values = _inductances_for_mm(
            data,
            main_resistance,
            damping_resistance,
            omega,
            a1c,
            a2c,
            b1c,
            b2c,
            disc,
            mm,
        )
        best = values
        f_mid = values[-1]
        if f_mid == 0 or abs(hi - lo) < 1e-5 * mm:
            break
        if f_mid * f_lo > 0:
            lo, f_lo = mm, f_mid
        else:
            hi, f_hi = mm, f_mid

    if best is None:
        return None
    return AxisWindingParameters(
        main_inductance=best[0],
        damping_inductance=best[1],
        mutual_inductance=best[2],
        main_to_stator_mutual=best[3],
        damping_to_stator_mutual=best[4],
        main_resistance=main_resistance,
        damping_resistance=damping_resistance,
        used_fallback=False,
    )


def _fallback_inductances(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
    frequency_hz: float,
    mutual_ratio: float,
) -> AxisWindingParameters:
    omega = 2 * pi * frequency_hz
    xd, xdp, xdpp = (
        data.synchronous_reactance,
        data.transient_reactance,
        data.subtransient_reactance,
    )
    tdp, tdpp = data.transient_time_constant, data.subtransient_time_constant
    ld, ldpp = xd / omega, xdpp / omega
    dd = (tdp - tdpp) * xd / xdp + tdpp * xd / xdpp
    apu = 1.5 * (1 / main_resistance + mutual_ratio**2 / damping_resistance) / ld
    apv = dd + tdp * tdpp * ld / (dd * ldpp) - (tdp + tdpp)
    mf = sqrt(apv / apu) if apu > 0 and apv > 0 else 0.001
    md = mf * mutual_ratio

    cc = 2 * (ld - ldpp) * main_resistance * damping_resistance * tdp * tdpp * ld / (3 * ldpp)
    determinant = md * md / damping_resistance - mf * mf / main_resistance
    if determinant != 0:
        # SolvePhysPar recalculates this term after solving Mf/Md. Reusing
        # the coefficient from the mutual-inductance estimate changes b1/b2
        # and produces nearly singular rotor-winding inductances.
        apu = dd + tdp * tdpp * ld / (dd * ldpp)
        a1 = 2 * mf * md / (damping_resistance * determinant)
        b1 = cc / (damping_resistance * determinant) - apu * mf * mf / determinant
        a2 = -2 * mf * md / (main_resistance * determinant)
        b2 = -cc / (main_resistance * determinant) + apu * md * md / determinant
        apv = a1 * b2 + a2 * b1
        apw = 4 * (
            b1 * b2 - main_resistance * damping_resistance * tdp * tdpp * xd / xdpp
        ) * (a1 * a2 - 1)
        divisor = a1 * a2 - 1
        disc = apv * apv - apw
    else:
        divisor = 0.0
        disc = -1.0

    if divisor != 0:
        lfd = 0.5 * (-apv - sqrt(disc)) / divisor if disc >= 0 else 0.5 * (-apv) / divisor
        lff = a1 * lfd + b1
        ldd = a2 * lfd + b2
    else:
        lfd, lff, ldd = 1.8 * mf, 2 * mf, 2 * mf

    return AxisWindingParameters(
        main_inductance=lff,
        damping_inductance=ldd,
        mutual_inductance=lfd,
        main_to_stator_mutual=mf,
        damping_to_stator_mutual=md,
        main_resistance=main_resistance,
        damping_resistance=damping_resistance,
        used_fallback=True,
    )


def _solve_axis(
    data: AxisReactanceData,
    main_resistance: float,
    damping_resistance: float,
    frequency_hz: float,
    *,
    mutual_ratio: float,
) -> AxisWindingParameters:
    solution = _solve_inductances(
        data, main_resistance, damping_resistance, frequency_hz
    )
    if solution is not None:
        return solution
    return _fallback_inductances(
        data, main_resistance, damping_resistance, frequency_hz, mutual_ratio
    )


def solve_synchronous_machine_parameters(
    parameters: SynchronousMachineParameters,
) -> SynchronousMachinePhysicalParameters:
    """Port Elplek's ``SolveGenPar`` defaults and axis parameter conversion.

    Elplek guesses rotor winding resistances rather than reading them from the
    SLD: field/d-axis damper are 0.3/0.1 ohm, and q/p dampers are 0.2/0.1 ohm.
    Its fallback conversion uses a mutual-inductance ratio of 0.5.
    """
    if not isfinite(parameters.stator_resistance) or parameters.stator_resistance < 0:
        raise ValueError("stator_resistance must be finite and nonnegative")
    if not isfinite(parameters.frequency_hz) or parameters.frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    if (
        not isinstance(parameters.parallel_units, int)
        or isinstance(parameters.parallel_units, bool)
        or parameters.parallel_units < 1
    ):
        raise ValueError("parallel_units must be a positive integer")
    _validate_axis("d", parameters.d_axis)
    _validate_axis("q", parameters.q_axis)
    if parameters.q_axis.synchronous_reactance != parameters.d_axis.synchronous_reactance:
        raise ValueError(
            "Elplek assumes the q-axis synchronous reactance equals the d-axis value"
        )

    units = parameters.parallel_units
    d_data = AxisReactanceData(
        parameters.d_axis.synchronous_reactance / units,
        parameters.d_axis.transient_reactance / units,
        parameters.d_axis.subtransient_reactance / units,
        parameters.d_axis.transient_time_constant,
        parameters.d_axis.subtransient_time_constant,
    )
    q_data = AxisReactanceData(
        parameters.q_axis.synchronous_reactance / units,
        parameters.q_axis.transient_reactance / units,
        parameters.q_axis.subtransient_reactance / units,
        parameters.q_axis.transient_time_constant,
        parameters.q_axis.subtransient_time_constant,
    )
    d_axis = _solve_axis(
        d_data, 0.3, 0.1, parameters.frequency_hz, mutual_ratio=0.5
    )
    q_axis = _solve_axis(
        q_data, 0.2, 0.1, parameters.frequency_hz, mutual_ratio=0.5
    )
    return SynchronousMachinePhysicalParameters(
        stator_resistance=parameters.stator_resistance / units,
        stator_inductance=d_data.synchronous_reactance
        / (2 * pi * parameters.frequency_hz),
        angular_frequency=2 * pi * parameters.frequency_hz,
        d_axis=d_axis,
        q_axis=q_axis,
    )


def advance_rotor_state_trapezoidal(
    parameters: SynchronousMachinePhysicalParameters,
    previous: RotorCurrentState,
    *,
    time_step_s: float,
    field_voltage_v: float,
    stator_d_current_old_a: float = 0.0,
    stator_d_current_new_a: float = 0.0,
    stator_q_current_old_a: float = 0.0,
    stator_q_current_new_a: float = 0.0,
) -> RotorCurrentState:
    """Advance the four rotor winding currents by one trapezoidal step.

    The supplied endpoint stator currents provide an isolated flux-linkage
    update using the ``1.5*sqrt(2)`` stator-current scaling. The network
    callback couples this update in ``solve_coupled_rotor_step``.
    """
    scalar_inputs = (
        time_step_s,
        field_voltage_v,
        stator_d_current_old_a,
        stator_d_current_new_a,
        stator_q_current_old_a,
        stator_q_current_new_a,
    )
    if any(not isfinite(value) for value in scalar_inputs):
        raise ValueError("Rotor update inputs must be finite")
    if time_step_s <= 0:
        raise ValueError("time_step_s must be positive")

    d, q = parameters.d_axis, parameters.q_axis
    inductance = np.asarray(
        [
            [d.main_inductance, d.mutual_inductance, 0.0, 0.0],
            [d.mutual_inductance, d.damping_inductance, 0.0, 0.0],
            [0.0, 0.0, q.main_inductance, q.mutual_inductance],
            [0.0, 0.0, q.mutual_inductance, q.damping_inductance],
        ],
        dtype=np.float64,
    )
    resistance = np.diag(
        (
            d.main_resistance,
            d.damping_resistance,
            q.main_resistance,
            q.damping_resistance,
        )
    )
    previous_currents = previous.as_array()
    if not np.isfinite(previous_currents).all():
        raise ValueError("Previous rotor currents must be finite")

    winding_voltage = np.asarray((field_voltage_v, 0.0, 0.0, 0.0))
    half_step = time_step_s / 2
    stator_flux_scale = 1.5 * sqrt(2.0)
    old_stator_flux = stator_flux_scale * np.asarray(
        (
            d.main_to_stator_mutual * stator_d_current_old_a,
            d.damping_to_stator_mutual * stator_d_current_old_a,
            q.main_to_stator_mutual * stator_q_current_old_a,
            q.damping_to_stator_mutual * stator_q_current_old_a,
        ),
        dtype=np.float64,
    )
    new_stator_flux = stator_flux_scale * np.asarray(
        (
            d.main_to_stator_mutual * stator_d_current_new_a,
            d.damping_to_stator_mutual * stator_d_current_new_a,
            q.main_to_stator_mutual * stator_q_current_new_a,
            q.damping_to_stator_mutual * stator_q_current_new_a,
        ),
        dtype=np.float64,
    )
    lhs = inductance + half_step * resistance
    rhs = (
        (inductance - half_step * resistance) @ previous_currents
        + time_step_s * winding_voltage
        - (new_stator_flux - old_stator_flux)
    )
    try:
        next_currents = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular rotor winding inductance matrix") from exc

    if not np.isfinite(next_currents).all():
        raise ValueError("Rotor update produced non-finite winding currents")
    return RotorCurrentState(*map(float, next_currents))


def advance_rotor_state_elplek(
    parameters: SynchronousMachinePhysicalParameters,
    previous: RotorCurrentState,
    *,
    time_step_s: float,
    field_voltage_v: float,
    stator_current_sensitivity: complex,
) -> RotorCurrentState:
    """Advance rotor currents with Elplek's ``RotorDQEquations`` coupling.

    ``stator_current_sensitivity`` is the complex response of the synchronous
    machine current to a unit change in its internal voltage. The sensitivity
    is converted to Elplek's four-real ``Quatre`` layout before assembling the
    rotor/network terms in its trapezoidal winding equations.
    """
    if not isfinite(time_step_s) or time_step_s <= 0:
        raise ValueError("time_step_s must be finite and positive")
    if not isfinite(field_voltage_v):
        raise ValueError("field_voltage_v must be finite")
    sensitivity = complex(stator_current_sensitivity)
    if not isfinite(sensitivity.real) or not isfinite(sensitivity.imag):
        raise ValueError("stator_current_sensitivity must be finite")

    d, q = parameters.d_axis, parameters.q_axis
    omega = parameters.angular_frequency
    d_current_per_parallel_v = sensitivity.real
    q_current_per_parallel_v = sensitivity.imag
    # Elplek PutdIdV stores (rr, ir, ri, ii) = (real, imag, -imag, real).
    rr = d_current_per_parallel_v
    ir = q_current_per_parallel_v
    ri = -q_current_per_parallel_v
    ii = d_current_per_parallel_v
    rotor_network = 1.5 * omega * np.asarray(
        [
            [
                d.main_to_stator_mutual * ir * d.main_to_stator_mutual,
                d.main_to_stator_mutual * ir * d.damping_to_stator_mutual,
                d.main_to_stator_mutual * ii * q.main_to_stator_mutual,
                d.main_to_stator_mutual * ii * q.damping_to_stator_mutual,
            ],
            [
                d.damping_to_stator_mutual * ir * d.main_to_stator_mutual,
                d.damping_to_stator_mutual * ir * d.damping_to_stator_mutual,
                d.damping_to_stator_mutual * ii * q.main_to_stator_mutual,
                d.damping_to_stator_mutual * ii * q.damping_to_stator_mutual,
            ],
            [
                -q.main_to_stator_mutual * rr * d.main_to_stator_mutual,
                -q.main_to_stator_mutual * rr * d.damping_to_stator_mutual,
                -q.main_to_stator_mutual * ri * q.main_to_stator_mutual,
                -q.main_to_stator_mutual * ri * q.damping_to_stator_mutual,
            ],
            [
                -q.damping_to_stator_mutual * rr * d.main_to_stator_mutual,
                -q.damping_to_stator_mutual * rr * d.damping_to_stator_mutual,
                -q.damping_to_stator_mutual * ri * q.main_to_stator_mutual,
                -q.damping_to_stator_mutual * ri * q.damping_to_stator_mutual,
            ],
        ],
        dtype=np.float64,
    )
    inductance = np.asarray(
        [
            [d.main_inductance, d.mutual_inductance, 0.0, 0.0],
            [d.mutual_inductance, d.damping_inductance, 0.0, 0.0],
            [0.0, 0.0, q.main_inductance, q.mutual_inductance],
            [0.0, 0.0, q.mutual_inductance, q.damping_inductance],
        ],
        dtype=np.float64,
    )
    resistance = np.diag(
        (d.main_resistance, d.damping_resistance, q.main_resistance, q.damping_resistance)
    )
    previous_currents = previous.as_array()
    lhs = inductance + rotor_network + time_step_s / 2 * resistance
    rhs = (
        (inductance + rotor_network - time_step_s / 2 * resistance)
        @ previous_currents
    )
    rhs[0] += time_step_s * field_voltage_v
    try:
        next_currents = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular Elplek rotor/network inductance matrix") from exc
    if not np.isfinite(next_currents).all():
        raise ValueError("Elplek rotor/network update produced non-finite currents")
    return RotorCurrentState(*map(float, next_currents))


def internal_voltage_from_rotor_state(
    parameters: SynchronousMachinePhysicalParameters,
    state: RotorCurrentState,
    reference_voltage: complex,
) -> complex:
    """Build the machine internal phasor using Elplek's D/Q projection."""
    reference = complex(reference_voltage)
    magnitude = abs(reference)
    if not isfinite(reference.real) or not isfinite(reference.imag) or magnitude == 0:
        raise ValueError("reference_voltage must be finite and nonzero")
    currents = state.as_array()
    if not np.isfinite(currents).all():
        raise ValueError("Rotor currents must be finite")

    d, q = parameters.d_axis, parameters.q_axis
    voltage_d = -parameters.angular_frequency * (
        d.main_to_stator_mutual * state.field
        + d.damping_to_stator_mutual * state.d_damper
    ) / sqrt(2)
    voltage_q = -parameters.angular_frequency * (
        q.main_to_stator_mutual * state.q_damper
        + q.damping_to_stator_mutual * state.p_damper
    ) / sqrt(2)
    unit = reference / magnitude
    return complex(voltage_d) * unit + 1j * voltage_q * unit


def solve_coupled_rotor_step(
    parameters: SynchronousMachinePhysicalParameters,
    previous: RotorCurrentState,
    *,
    time_step_s: float,
    field_voltage_v: float,
    reference_voltage: complex,
    stator_d_current_old_a: float,
    stator_q_current_old_a: float,
    solve_stator_currents: Callable[[complex], tuple[float, float]],
    voltage_tolerance_v: float = 1e-6,
    current_tolerance_a: float = 1e-8,
    max_iterations: int = 50,
) -> RotorStepResult:
    """Solve one rotor/network step with a two-variable Newton iteration.

    ``solve_stator_currents`` solves the network/fault equations for a trial
    machine internal voltage and returns its stator D/Q currents. Each residual
    evaluation recomputes the rotor state from the same previous time step.
    A numerical 2x2 Jacobian updates the trial voltage until the rotor voltage
    and network current agree.
    """
    tolerances = (voltage_tolerance_v, current_tolerance_a)
    if any(not isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("Coupling tolerances must be finite and positive")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not isfinite(stator_d_current_old_a) or not isfinite(stator_q_current_old_a):
        raise ValueError("Previous stator currents must be finite")

    current_sensitivity = getattr(
        solve_stator_currents, "current_sensitivity", None
    )
    if current_sensitivity is not None:
        state = advance_rotor_state_elplek(
            parameters,
            previous,
            time_step_s=time_step_s,
            field_voltage_v=field_voltage_v,
            stator_current_sensitivity=current_sensitivity,
        )
        voltage = internal_voltage_from_rotor_state(
            parameters, state, reference_voltage
        )
        currents = solve_stator_currents(voltage)
        if len(currents) != 2 or any(not isfinite(value) for value in currents):
            raise ValueError("Network callback must return two finite D/Q currents")
        return RotorStepResult(
            rotor_state=state,
            internal_voltage=voltage,
            stator_d_current=currents[0],
            stator_q_current=currents[1],
            iterations=1,
        )

    trial_voltage = internal_voltage_from_rotor_state(parameters, previous, reference_voltage)
    previous_network_currents: tuple[float, float] | None = None

    def evaluate(voltage: complex):
        currents = solve_stator_currents(voltage)
        if len(currents) != 2 or any(not isfinite(value) for value in currents):
            raise ValueError("Network callback must return two finite D/Q currents")
        d_current, q_current = currents
        state = advance_rotor_state_trapezoidal(
            parameters,
            previous,
            time_step_s=time_step_s,
            field_voltage_v=field_voltage_v,
            stator_d_current_old_a=stator_d_current_old_a,
            stator_d_current_new_a=d_current,
            stator_q_current_old_a=stator_q_current_old_a,
            stator_q_current_new_a=q_current,
        )
        updated = internal_voltage_from_rotor_state(
            parameters, state, reference_voltage
        )
        return currents, state, updated, updated - voltage

    for iteration in range(1, max_iterations + 1):
        network_currents, state, updated_voltage, residual = evaluate(trial_voltage)
        voltage_error = abs(residual)
        current_error = (
            float("inf")
            if previous_network_currents is None
            else max(
                abs(network_currents[0] - previous_network_currents[0]),
                abs(network_currents[1] - previous_network_currents[1]),
            )
        )
        if (
            voltage_error <= voltage_tolerance_v
            and current_error <= current_tolerance_a
        ):
            return RotorStepResult(
                rotor_state=state,
                internal_voltage=updated_voltage,
                stator_d_current=network_currents[0],
                stator_q_current=network_currents[1],
                iterations=iteration,
            )
        if previous_network_currents is None and voltage_error <= voltage_tolerance_v:
            previous_network_currents = network_currents
            continue

        difference_step = max(1e-3, abs(trial_voltage) * 1e-6)
        _, _, _, residual_real = evaluate(trial_voltage + difference_step)
        _, _, _, residual_imag = evaluate(trial_voltage + 1j * difference_step)
        jacobian = np.asarray(
            [
                [
                    (residual_real.real - residual.real) / difference_step,
                    (residual_imag.real - residual.real) / difference_step,
                ],
                [
                    (residual_real.imag - residual.imag) / difference_step,
                    (residual_imag.imag - residual.imag) / difference_step,
                ],
            ],
            dtype=np.float64,
        )
        try:
            correction = np.linalg.solve(
                jacobian,
                np.asarray((-residual.real, -residual.imag), dtype=np.float64),
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Singular rotor/network Newton Jacobian") from exc
        voltage_correction = complex(float(correction[0]), float(correction[1]))
        accepted_voltage: complex | None = None
        scale = 1.0
        for _ in range(12):
            candidate = trial_voltage + scale * voltage_correction
            _, _, _, candidate_residual = evaluate(candidate)
            if (
                abs(candidate_residual) < voltage_error
                or abs(candidate_residual) <= voltage_tolerance_v
            ):
                accepted_voltage = candidate
                break
            scale *= 0.5
        if accepted_voltage is None:
            raise RuntimeError("Rotor/network Newton step failed to reduce residual")
        trial_voltage = accepted_voltage
        previous_network_currents = network_currents

    raise RuntimeError(
        f"Rotor/network coupling did not converge in {max_iterations} iterations"
    )


def simulate_coupled_rotor_steps(
    parameters: SynchronousMachinePhysicalParameters,
    initial_state: RotorCurrentState,
    *,
    time_steps_s: Iterable[float],
    field_voltage_v: float,
    reference_voltage: complex,
    initial_stator_d_current_a: float,
    initial_stator_q_current_a: float,
    solve_stator_currents: Callable[[complex], tuple[float, float]],
    voltage_tolerance_v: float = 1e-6,
    current_tolerance_a: float = 1e-8,
    max_iterations: int = 50,
) -> tuple[RotorTrajectorySample, ...]:
    """Integrate a sequence of fault-on time steps with network coupling.

    ``time_steps_s`` contains the duration of each interval, so it can be
    nonuniform. The initial rotor state and D/Q stator currents must describe
    the same instant immediately before the first interval.
    """
    state = initial_state
    current_d = initial_stator_d_current_a
    current_q = initial_stator_q_current_a
    elapsed = 0.0
    samples: list[RotorTrajectorySample] = []
    for time_step in time_steps_s:
        result = solve_coupled_rotor_step(
            parameters,
            state,
            time_step_s=time_step,
            field_voltage_v=field_voltage_v,
            reference_voltage=reference_voltage,
            stator_d_current_old_a=current_d,
            stator_q_current_old_a=current_q,
            solve_stator_currents=solve_stator_currents,
            voltage_tolerance_v=voltage_tolerance_v,
            current_tolerance_a=current_tolerance_a,
            max_iterations=max_iterations,
        )
        elapsed += time_step
        samples.append(RotorTrajectorySample(elapsed, result))
        state = result.rotor_state
        current_d = result.stator_d_current
        current_q = result.stator_q_current
    return tuple(samples)
