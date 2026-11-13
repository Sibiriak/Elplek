from __future__ import annotations

from math import isclose, isfinite

import pytest

from elplek.asynchronous_motor import (
    AsynchronousMotorError,
    solve_asynchronous_motor_equivalent,
)


def test_elplek_motor_equivalent_matches_nominal_power_factor_and_slip() -> None:
    equivalent = solve_asynchronous_motor_equivalent(
        nominal_voltage_ll=5_200.0,
        nominal_power=5e6,
        nominal_slip=0.018,
        nominal_power_factor=0.8,
        efficiency=0.94,
        locked_rotor_current_ratio=5.0,
        output_power=5e6,
    )

    apparent_power = 5_200.0**2 / equivalent.impedance.conjugate()
    assert isclose(apparent_power.real, 5e6 / 0.94, rel_tol=1e-10)
    assert isclose(apparent_power.real / abs(apparent_power), 0.8, rel_tol=1e-10)
    assert isclose(equivalent.slip, 0.018, abs_tol=1e-10)
    assert isclose(equivalent.locked_rotor_current, 3_691.1202766, rel_tol=1e-8)
    assert not equivalent.used_locked_rotor_fallback


def test_elplek_motor_equivalent_supports_part_load_and_is_finite() -> None:
    equivalent = solve_asynchronous_motor_equivalent(
        nominal_voltage_ll=5_200.0,
        nominal_power=5e6,
        nominal_slip=0.02,
        nominal_power_factor=0.8,
        efficiency=0.94,
        locked_rotor_current_ratio=5.0,
        output_power=2e6,
    )

    assert 0 < equivalent.slip < 0.02
    assert equivalent.impedance.real > 0
    assert all(
        isfinite(value)
        for value in (
            equivalent.impedance.real,
            equivalent.impedance.imag,
            equivalent.load_resistance,
            equivalent.locked_rotor_current,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"nominal_voltage_ll": float("nan")},
        {"nominal_slip": 0.2},
        {"efficiency": 0.999},
        {"locked_rotor_current_ratio": 1.0},
        {"output_power": 0.0},
    ],
)
def test_motor_equivalent_rejects_invalid_nameplate_data(overrides) -> None:
    parameters = {
        "nominal_voltage_ll": 5_200.0,
        "nominal_power": 5e6,
        "nominal_slip": 0.018,
        "nominal_power_factor": 0.8,
        "efficiency": 0.94,
        "locked_rotor_current_ratio": 5.0,
        "output_power": 5e6,
    }
    parameters.update(overrides)
    with pytest.raises(AsynchronousMotorError):
        solve_asynchronous_motor_equivalent(**parameters)
