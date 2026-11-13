"""Steady-state equivalent impedance for Elplek asynchronous motors.

This ports ``AsMotPar``, ``SolveAll``, ``IstartAndParam``, ``AsMotLoad``, and
``AsMotPower`` from the preserved Elplek Delphi source.
The returned impedance is the per-phase input impedance Elplek stamps during
normal steady-state calculations and load flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


class AsynchronousMotorError(ValueError):
    """Raised when Elplek-style motor data cannot produce a physical equivalent."""


@dataclass(frozen=True, slots=True)
class AsynchronousMotorEquivalent:
    """Solved per-phase steady-state equivalent of an induction motor."""

    impedance: complex
    slip: float
    stator_resistance: float
    stator_reactance: float
    rotor_resistance: float
    rotor_reactance: float
    magnetizing_reactance: float
    load_resistance: float
    locked_rotor_current: float
    used_locked_rotor_fallback: bool


def solve_asynchronous_motor_equivalent(
    *,
    nominal_voltage_ll: float,
    nominal_power: float,
    nominal_slip: float,
    nominal_power_factor: float,
    efficiency: float,
    locked_rotor_current_ratio: float,
    output_power: float,
) -> AsynchronousMotorEquivalent:
    """Solve the Elplek steady-state induction-motor impedance in ohms/phase.

    Parameters are the Elplek SLD values: line-to-line nominal voltage, total
    nominal motor output power, nominal slip, nominal PF, efficiency, locked
    rotor/nominal current ratio, and actual useful output power. This routine
    Negative output power selects Elplek's induction-generator calculation.
    """
    values = (
        nominal_voltage_ll,
        nominal_power,
        nominal_slip,
        nominal_power_factor,
        efficiency,
        locked_rotor_current_ratio,
        output_power,
    )
    if not all(isfinite(value) for value in values):
        raise AsynchronousMotorError("Motor parameters must be finite")
    if nominal_voltage_ll <= 0.5 or nominal_power <= 1 or abs(output_power) <= 1:
        raise AsynchronousMotorError("Motor voltage and power magnitudes must be positive")
    if not 0.00099 < nominal_slip < 0.125:
        raise AsynchronousMotorError("nominal_slip must be between 0.00099 and 0.125")
    if not 0.4 <= nominal_power_factor < 0.995:
        raise AsynchronousMotorError("nominal_power_factor must be in [0.4, 0.995)")
    if not 0.699 <= efficiency < 1 - nominal_slip:
        raise AsynchronousMotorError("efficiency must be in [0.699, 1 - nominal_slip)")
    if locked_rotor_current_ratio <= 1.2:
        raise AsynchronousMotorError("locked_rotor_current_ratio must exceed 1.2")

    phase_voltage = nominal_voltage_ll / sqrt(3)
    nominal_power_per_phase = nominal_power / 3
    nominal_current = nominal_power_per_phase / (
        phase_voltage * nominal_power_factor * efficiency
    )
    locked_rotor_current = nominal_current * locked_rotor_current_ratio

    # Elplek SolveAll: fit the per-phase steady-state equivalent at the nominal
    # point, then tune X2+X3 to reproduce the locked-rotor current.
    stator_r = (
        phase_voltage
        * nominal_power_factor
        * (1 - efficiency / (1 - nominal_slip))
        / nominal_current
    )
    nominal_current_phasor = complex(
        nominal_current * nominal_power_factor,
        -nominal_current * sqrt(1 - nominal_power_factor**2),
    )
    apparent_z = phase_voltage / nominal_current_phasor
    apparent_resistance_without_stator = apparent_z.real - stator_r
    apparent_reactance = apparent_z.imag
    if apparent_resistance_without_stator <= 0 or apparent_reactance <= 0:
        raise AsynchronousMotorError("Motor nameplate values give an invalid equivalent")

    x_sum_min = (
        apparent_resistance_without_stator**2 / apparent_reactance
        + apparent_reactance
    )
    x_sum_max = phase_voltage / (nominal_current * 0.1)
    if x_sum_max < 10 * x_sum_min:
        x_sum_max = 10 * x_sum_min

    def parameters_for_x_sum(x_sum: float) -> tuple[float, float, float, float, float]:
        delta = x_sum - apparent_reactance
        if delta <= 0:
            raise AsynchronousMotorError("Could not bracket the locked-rotor reactance")
        x3_squared = (
            x_sum * delta
            + x_sum * apparent_resistance_without_stator**2 / delta
        )
        if x3_squared <= 0:
            raise AsynchronousMotorError("Invalid magnetizing reactance in motor equivalent")
        x3 = sqrt(x3_squared)
        x2 = x_sum - x3
        x1 = x2
        r2 = (
            apparent_resistance_without_stator
            * x_sum
            * nominal_slip
            / delta
        )
        return stator_r, x1, r2, x2, x3

    def locked_current_for_x_sum(x_sum: float) -> float:
        r1, x1, r2, x2, x3 = parameters_for_x_sum(x_sum)
        rotor_at_standstill = complex(r2, x2)
        magnetizing = complex(0, x3)
        parallel = rotor_at_standstill * magnetizing / (rotor_at_standstill + magnetizing)
        return abs(phase_voltage / (complex(r1, x1) + parallel))

    x_sum = x_sum_max
    mismatch = locked_current_for_x_sum(x_sum) - locked_rotor_current
    bracket_low: tuple[float, float] | None = None
    bracket_high: tuple[float, float] | None = None
    fallback = False
    if mismatch < 0:
        bracket_high = (x_sum, mismatch)
        for _ in range(20):
            x_sum = x_sum_min + 0.5 * (x_sum - x_sum_min)
            mismatch = locked_current_for_x_sum(x_sum) - locked_rotor_current
            if mismatch >= 0:
                bracket_low = (x_sum, mismatch)
                break
            bracket_high = (x_sum, mismatch)
    else:
        bracket_low = (x_sum, mismatch)
        for _ in range(20):
            x_sum = x_sum_min + 1.5 * (x_sum - x_sum_min)
            mismatch = locked_current_for_x_sum(x_sum) - locked_rotor_current
            if mismatch <= 0:
                bracket_high = (x_sum, mismatch)
                break
            bracket_low = (x_sum, mismatch)

    if bracket_low is None or bracket_high is None:
        x_sum = x_sum_min
        fallback = True
    else:
        low_x, low_f = bracket_low
        high_x, high_f = bracket_high
        for _ in range(10):
            denominator = high_f - low_f
            if abs(denominator) > 1e-6:
                x_sum = (high_f * low_x - low_f * high_x) / denominator
            else:
                x_sum = 0.5 * (low_x + high_x)
            mismatch = locked_current_for_x_sum(x_sum) - locked_rotor_current
            if mismatch > 0:
                low_x, low_f = x_sum, mismatch
            else:
                high_x, high_f = x_sum, mismatch

    stator_r, x1, rotor_r, x2, x3 = parameters_for_x_sum(x_sum)
    z1 = complex(stator_r, x1)
    z2 = complex(rotor_r, x2)
    z3 = complex(0, x3)

    if output_power > 0:
        # Elplek AsMotLoad chooses the solution nearest nominal slip.
        z13 = z1 + z3
        z12_23 = z1 * z2 + z1 * z3 + z2 * z3
        a = abs(z13) ** 2
        b = 2 * (z12_23 * z13.conjugate()).real - (
            phase_voltage**2 * abs(z3) ** 2 * 3 / output_power
        )
        c = abs(z12_23) ** 2
        discriminant = b * b - 4 * a * c
        if discriminant < 0 or b > 0 or a == 0:
            raise AsynchronousMotorError(
                "No physical load resistance for the requested motor power"
            )
        root = sqrt(discriminant)
        load_resistance_roots = ((-b + root) / (2 * a), (-b - root) / (2 * a))
        slip_roots = tuple(
            rotor_r / (rotor_r + load_r) if rotor_r + load_r != 0 else 0.001
            for load_r in load_resistance_roots
        )
        selected = min(
            range(2), key=lambda index: abs(nominal_slip - slip_roots[index])
        )
        load_resistance = load_resistance_roots[selected]
        slip = slip_roots[selected]
        if rotor_r + load_resistance <= 0 or slip <= 0:
            raise AsynchronousMotorError(
                "Solved motor slip or load resistance is nonphysical"
            )
    else:
        # Elplek AsMotPower uses the negative-power quadratic and its lower
        # root to obtain the generator's negative slip and rotor load resistance.
        power_per_voltage_squared = output_power / (3 * phase_voltage**2)
        aa = z2.imag + z3.imag
        bb = (z1.imag * z2.imag + z1.imag * z3.imag + z2.imag * z3.imag)
        cc = (z2.imag + z3.imag) * z1.real
        dd = z1.imag + z3.imag
        a = z1.real - power_per_voltage_squared * (z1.real**2 + dd**2)
        b = aa * dd - bb + 2 * power_per_voltage_squared * bb * z1.real - (
            2 * power_per_voltage_squared * cc * dd
        )
        c = aa * cc - power_per_voltage_squared * (bb**2 + cc**2)
        discriminant = b * b - 4 * a * c
        if discriminant < 0 or a == 0:
            raise AsynchronousMotorError(
                "No physical generator slip for the requested output power"
            )
        rotor_load_ratio = (-b - sqrt(discriminant)) / (2 * a)
        if rotor_load_ratio == 0:
            raise AsynchronousMotorError("Generator slip is singular")
        slip = rotor_r / rotor_load_ratio
        if slip == 0:
            raise AsynchronousMotorError("Generator slip is zero")
        load_resistance = rotor_r * (1 - slip) / slip
        if slip >= 0 or rotor_r + load_resistance >= 0:
            raise AsynchronousMotorError(
                "Solved generator slip or load resistance is nonphysical"
            )

    loaded_rotor = complex(rotor_r + load_resistance, x2)
    parallel = loaded_rotor * z3 / (loaded_rotor + z3)
    impedance = z1 + parallel
    if not (isfinite(impedance.real) and isfinite(impedance.imag)):
        raise AsynchronousMotorError("Solved machine impedance is nonphysical")
    if output_power > 0 and impedance.real <= 0:
        raise AsynchronousMotorError("Solved motor impedance is nonphysical")
    if output_power < 0 and impedance.real >= 0:
        raise AsynchronousMotorError("Solved generator impedance is nonphysical")

    return AsynchronousMotorEquivalent(
        impedance=impedance,
        slip=slip,
        stator_resistance=stator_r,
        stator_reactance=x1,
        rotor_resistance=rotor_r,
        rotor_reactance=x2,
        magnetizing_reactance=x3,
        load_resistance=load_resistance,
        locked_rotor_current=locked_rotor_current,
        used_locked_rotor_fallback=fallback,
    )
