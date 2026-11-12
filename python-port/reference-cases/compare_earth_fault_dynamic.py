"""Compare dynamic phase and residual currents with Elplek CT1 and CT7 exports.

Run from the repository root after installing the Python package:
    python python-port/reference-cases/compare_earth_fault_dynamic.py

The Elplek initial-state option was left disabled. The prefault current is
calculated from the explicit constant-admittance network, matching the
non-precalculated state path. CT1 currents are transformed from the generator
side through T1's 11:1 ratio; CT7 currents are checked on the generator side.
The exported ``I1``/``3I2`` columns do not equal the symmetrical components
reconstructed from the exported phase currents, so this comparison uses the
direct phase and residual measurements.
"""
from __future__ import annotations

import csv
from cmath import exp, phase
from math import pi, sqrt
from pathlib import Path

from compare_earth_fault_thevenin import network_for_stage
from elplek.dynamic_fault import (
    initialize_dynamic_fault_rotor_state,
    make_dynamic_fault_current_solver,
    solve_dynamic_network_fault,
)
from elplek.synchronous_machine import (
    AxisReactanceData,
    SynchronousMachineParameters,
    simulate_coupled_rotor_steps,
    solve_synchronous_machine_parameters,
)


ROOT = Path(__file__).parent
REFERENCE_CSV = ROOT / "exports" / "earth_fault_base.csv"
V_PHASE = 12_000 / sqrt(3)
TRANSFORMER_CURRENT_RATIO = 165_000 / 15_000
PREFAULT_CURRENT_A = None
# EarthShort is the non-IEC, time-domain path. Genmat uses the generator's
# normal impedance (Xd) for both its initial fault solution and later steps.
# The separately printed sequence-impedance report can still show Xd''.
DYNAMIC_SEQUENCE_IMPEDANCES = (
    0.0044 + 2.39j,
    0.0044 + 0.26j,
    0.0044 + 0j,
)
INITIAL_SEQUENCE_IMPEDANCES = DYNAMIC_SEQUENCE_IMPEDANCES
MAX_COMPARISON_TIME_S = 11.032
MAGNITUDE_TOLERANCE_A = 2.5
ANGLE_TOLERANCE_DEG = 0.25


def phase_angle(value: complex) -> float:
    return phase(value) * 180 / pi


def read_reference_samples(
    reference_csv: Path = REFERENCE_CSV,
    ct_label: str = "CT1",
) -> dict[float, dict[str, complex]]:
    with reference_csv.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {
            float(row["time_s"]): {
                **{
                    f"I{phase_name}": float(row[f"I{phase_name}_A"])
                    * exp(1j * float(row[f"I{phase_name}_deg"]) * pi / 180)
                    for phase_name in ("a", "b", "c")
                },
                "Isum": float(row["Isum_A"])
                * exp(1j * float(row["Isum_deg"]) * pi / 180),
            }
            for row in rows
            if row["ct_label"] == ct_label
            and float(row["time_s"]) <= MAX_COMPARISON_TIME_S
        }


def compare_case(
    case_name: str,
    reference_csv: Path,
    transformer_multiplier: float,
    z1_branch: complex = 2.85 + 8.15j,
    fault_type: str = "single_line_to_ground",
) -> tuple[float, float]:
    network = network_for_stage(2.39, transformer_multiplier, z1_branch)
    machine = SynchronousMachineParameters(
        stator_resistance=0.0044,
        d_axis=AxisReactanceData(2.39, 0.3445, 0.26, 0.8433, 0.0251),
        q_axis=AxisReactanceData(2.39, 0.534, 0.26, 0.123, 0.02),
        frequency_hz=50,
    )
    physical = solve_synchronous_machine_parameters(machine)
    initial = initialize_dynamic_fault_rotor_state(
        network,
        fault_bus_id=2,
        fault_type=fault_type,
        machine_parameters=machine,
        physical_parameters=physical,
        source_sequence_impedances=INITIAL_SEQUENCE_IMPEDANCES,
        prefault_stator_current=PREFAULT_CURRENT_A,
    )
    current_solver = make_dynamic_fault_current_solver(
        network,
        fault_bus_id=2,
        fault_type=fault_type,
        reference_voltage=V_PHASE,
        source_sequence_impedances=DYNAMIC_SEQUENCE_IMPEDANCES,
    )
    samples_by_ct = {
        label: read_reference_samples(reference_csv, label)
        for label in ("CT1", "CT7")
    }
    samples = samples_by_ct["CT1"]
    positive_times = sorted(time_s for time_s in samples if time_s > 0)
    time_steps = [
        current - previous
        for previous, current in zip([0.0, *positive_times[:-1]], positive_times)
    ]
    trajectory = simulate_coupled_rotor_steps(
        physical,
        initial.rotor_initial_conditions.rotor_state,
        time_steps_s=time_steps,
        field_voltage_v=initial.field_voltage_v,
        reference_voltage=V_PHASE,
        initial_stator_d_current_a=initial.stator_d_current_a,
        initial_stator_q_current_a=initial.stator_q_current_a,
        solve_stator_currents=current_solver,
    )

    predicted: dict[float, dict[str, complex]] = {}
    voltages = {
        0.0: initial.rotor_initial_conditions.internal_voltage,
        **{sample.time_s: sample.step.internal_voltage for sample in trajectory},
    }
    for time_s, voltage in voltages.items():
        result = solve_dynamic_network_fault(
            network,
            fault_bus_id=2,
            fault_type=fault_type,
            source_voltage=voltage,
            source_sequence_impedances=(
                INITIAL_SEQUENCE_IMPEDANCES
                if time_s == 0
                else DYNAMIC_SEQUENCE_IMPEDANCES
            ),
        )
        scale = TRANSFORMER_CURRENT_RATIO
        i1 = result.source_positive_current / scale
        i2 = result.source_negative_current / scale
        i0 = result.source_zero_current / scale
        phase_operator = exp(2j * pi / 3)
        predicted[time_s] = {
            "Ia": i0 + i1 + i2,
            "Ib": i0 + phase_operator**2 * i1 + phase_operator * i2,
            "Ic": i0 + phase_operator * i1 + phase_operator**2 * i2,
            "Isum": 3 * i0,
        }

    channels = (
        ("Ia", "Ib", "Ic")
        if fault_type == "three_phase" or fault_type.startswith("line_to_line")
        else ("Ia", "Ib", "Ic", "Isum")
    )
    magnitude_tolerance = (
        4.0
        if fault_type.startswith("double_line_to_ground")
        else MAGNITUDE_TOLERANCE_A
    )
    angle_tolerance = (
        0.35
        if fault_type.startswith("double_line_to_ground") or fault_type == "three_phase"
        else 0.30
        if fault_type.startswith("line_to_line")
        else ANGLE_TOLERANCE_DEG
    )
    overall_max_magnitude_error = 0.0
    overall_max_angle_error = 0.0
    for ct_label, samples in samples_by_ct.items():
        current_scale = 1.0 if ct_label == "CT1" else TRANSFORMER_CURRENT_RATIO
        ct_predicted = {
            time_s: {
                channel: value * current_scale
                for channel, value in currents.items()
            }
            for time_s, currents in predicted.items()
        }
        if samples.keys() != ct_predicted.keys():
            raise ValueError(f"Sample times differ for {case_name} at {ct_label}")
        maximum_magnitude_error = 0.0
        maximum_angle_error = 0.0
        print(f"{ct_label} phase/residual validation: {case_name}")
        print(" time   channel  Python A/deg    Elplek A/deg    dA      ddeg")
        for time_s in sorted(samples):
            if time_s not in (0.0, 0.001, 0.002, MAX_COMPARISON_TIME_S):
                continue
            for channel in channels:
                model_value = ct_predicted[time_s][channel]
                reference = samples[time_s][channel]
                angle_error = (phase_angle(model_value / reference) + 180) % 360 - 180
                print(
                    f"{time_s:6.3f}  {channel:5s} "
                    f"{abs(model_value):8.1f}/{phase_angle(model_value):7.1f} "
                    f"{abs(reference):8.1f}/{phase_angle(reference):7.1f} "
                    f"{abs(model_value) - abs(reference):+7.1f} {angle_error:+7.1f}"
                )
        for channel in channels:
            magnitude_errors = [
                abs(ct_predicted[time_s][channel]) - abs(samples[time_s][channel])
                for time_s in samples
            ]
            angle_errors = [
                (phase_angle(ct_predicted[time_s][channel] / samples[time_s][channel]) + 180)
                % 360
                - 180
                for time_s in samples
            ]
            max_mag = max(abs(error) for error in magnitude_errors)
            max_ang = max(abs(error) for error in angle_errors)
            maximum_magnitude_error = max(maximum_magnitude_error, max_mag)
            maximum_angle_error = max(maximum_angle_error, max_ang)
            print(
                f"{channel}: max magnitude error {max_mag:.2f} A; "
                f"max angle error {max_ang:.2f}° over {len(magnitude_errors)} samples"
            )
        passed = (
            maximum_magnitude_error <= magnitude_tolerance * current_scale
            and maximum_angle_error <= angle_tolerance
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"{status}: max errors {maximum_magnitude_error:.2f} A, "
            f"{maximum_angle_error:.2f}° "
            f"(limits {magnitude_tolerance * current_scale:.1f} A, "
            f"{angle_tolerance:.2f}°)"
        )
        overall_max_magnitude_error = max(
            overall_max_magnitude_error, maximum_magnitude_error
        )
        overall_max_angle_error = max(overall_max_angle_error, maximum_angle_error)
        if not passed:
            raise ValueError(f"Dynamic reference comparison failed for {case_name} at {ct_label}")
    return overall_max_magnitude_error, overall_max_angle_error


def main() -> None:
    cases = (
        ("baseline", REFERENCE_CSV, 1.0, 2.85 + 8.15j),
        (
            "T1 R/X doubled",
            ROOT / "exports" / "earth_fault_t1_z2.csv",
            2.0,
            2.85 + 8.15j,
        ),
        (
            "Z1 R/X doubled",
            ROOT / "exports" / "earth_fault_z2.csv",
            1.0,
            5.70 + 16.30j,
        ),
        (
            "Z1 resistance changed",
            ROOT / "exports" / "earth_fault_r1.csv",
            1.0,
            1.00 + 8.15j,
        ),
        (
            "line-to-line fault",
            ROOT / "exports" / "fault_line_to_line.csv",
            1.0,
            2.85 + 8.15j,
            "line_to_line_ab",
        ),
        (
            "line-to-line-to-earth fault",
            ROOT / "exports" / "fault_line_to_line_to_earth.csv",
            1.0,
            2.85 + 8.15j,
            "double_line_to_ground_ab",
        ),
        (
            "three-phase fault",
            ROOT / "exports" / "fault_three_phase.csv",
            1.0,
            2.85 + 8.15j,
            "three_phase",
        ),
    )
    for case in cases:
        compare_case(*case)


if __name__ == "__main__":
    main()
