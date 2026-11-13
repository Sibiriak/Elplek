from __future__ import annotations

import sys
from cmath import exp
from math import isclose, pi
from pathlib import Path

from elplek.load_flow import LoadFlowResult


REFERENCE_CASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "python-port"
    / "reference-cases"
)
sys.path.insert(0, str(REFERENCE_CASE_DIR))

from calpos_beispiel_load_flow import (  # noqa: E402
    S_BASE_VA,
    V_BASE_LL,
    build_calpos_example,
    solve_calpos_example,
)


def _shown_angle(voltage: complex) -> float:
    from cmath import phase

    return phase(voltage) * 180 / 3.141592653589793 + 30


def test_calpos_beispiel_reconstructed_load_flow_matches_terminal_voltages() -> None:
    result: LoadFlowResult = solve_calpos_example()

    assert result.iterations == 6
    assert result.maximum_mismatch < 1e-10
    expected = {
        7: (65_687.5, 32.5),
        10: (5_102.2, -3.5),
        11: (5_033.6, -0.2),
        12: (5_518.2, -2.5),
        13: (8_747.0, 7.1),
    }
    for bus_id, (expected_voltage, expected_angle) in expected.items():
        voltage = result.bus_voltages[bus_id]
        calculated_voltage = abs(voltage) * V_BASE_LL[bus_id]
        assert isclose(calculated_voltage, expected_voltage, rel_tol=0.003)
        assert abs(_shown_angle(voltage) - expected_angle) < 1.5


def test_calpos_beispiel_reconstructed_source_power_is_close_to_elplek() -> None:
    result = solve_calpos_example()
    f1 = result.bus_power_injections[1] * S_BASE_VA / 1e6
    g1 = result.bus_power_injections[13] * S_BASE_VA / 1e6

    assert abs(f1.real - (-24.7858)) < 0.1
    assert abs(f1.imag - 8.5344) < 3.0
    assert abs(g1.real - 39.9994) < 0.01
    assert abs(g1.imag - 10.4937) < 0.8


def test_calpos_beispiel_motor_equivalents_match_elplek_terminal_powers() -> None:
    network = build_calpos_example()
    result = solve_calpos_example()
    expected = {
        "Asm1": (6.4012e6, 36.9),
        "Asm2": (3.6741e6, 57.8),
        "Asm3": (6.6303e6, 32.5),
    }

    for motor in network.shunts:
        magnitude_va, angle_degrees = expected[motor.name]
        expected_power = magnitude_va * exp(1j * angle_degrees * pi / 180)
        voltage = result.bus_voltages[motor.bus_id]
        calculated_power = (
            abs(voltage) ** 2 * motor.admittance.conjugate() * S_BASE_VA
        )
        assert abs(calculated_power - expected_power) / magnitude_va < 0.005
