"""Rebuild the Elplek 0.995 ``Calpos Beispiel`` load-flow case.

Run from the repository root with:
    python python-port/reference-cases/calpos_beispiel_load_flow.py

Line, transformer, and asynchronous-motor data/topology are transcribed from
the bundled ``beispiel.sld``. The motor equivalent calculation ports the
corresponding Elplek Delphi routines and is compared against the saved Elplek
terminal S and V results.
"""
from __future__ import annotations

from cmath import phase
from math import pi

from elplek.asynchronous_motor import solve_asynchronous_motor_equivalent
from elplek.load_flow import LoadFlowResult, solve_load_flow
from elplek.model import Branch, Bus, Network, Shunt, Transformer


S_BASE_VA = 100e6
FREQUENCY_HZ = 50.0
V_BASE_LL = {
    1: 220e3,
    2: 65e3,
    3: 65e3,
    4: 65e3,
    5: 65e3,
    6: 65e3,
    7: 65e3,
    8: 65e3,
    9: 65e3,
    10: 5_500.0,
    11: 5_200.0,
    12: 5_800.0,
    13: 8_500.0,
}

# from bus, to bus, R1/km, B/km, R0/km, length/km, X1/km, X0/km
LINE_DATA = (
    ("TL1", 2, 3, 0.103, 2.827e-6, 0.403, 0.590, 0.140, 1.599),
    ("TL2", 2, 4, 0.103, 2.827e-6, 0.403, 1.160, 0.140, 1.499),
    ("TL3", 2, 4, 0.103, 2.827e-6, 0.403, 1.160, 0.150, 1.400),
    ("TL4", 2, 5, 0.091, 2.827e-6, 0.415, 7.797, 0.130, 1.659),
    ("TL5", 3, 9, 0.113, 2.827e-6, 0.413, 0.156, 0.154, 1.639),
    ("TL6", 4, 8, 0.103, 2.827e-6, 0.403, 1.160, 0.150, 1.400),
    ("TL7", 3, 8, 0.113, 2.827e-6, 0.413, 0.370, 0.153, 1.619),
    ("TL8", 5, 6, 0.141, 2.827e-6, 0.413, 11.900, 0.160, 1.649),
    ("TL9", 8, 7, 0.112, 2.827e-6, 0.400, 19.100, 0.144, 1.587),
    ("TL10", 6, 7, 0.066, 3.142e-6, 0.382, 1.612, 0.085, 1.459),
)

# name, primary bus, secondary bus, R (ohm), X (ohm), encoded +30 flag
TRANSFORMER_DATA = (
    ("T1", 1, 2, 1.210, 21.750, False),
    ("T2", 6, 13, 0.000, 8.450, True),
    ("T3", 9, 10, 5.630, 59.300, True),
    ("T4", 8, 11, 2.800, 35.100, True),
    ("T5", 8, 12, 4.204, 42.040, True),
)

# label, bus, nominal V LL, nominal output W, nominal slip, pf, efficiency,
# locked-rotor current ratio, actual output W
MOTOR_DATA = (
    ("Asm1", 10, 5_200.0, 5e6, 0.018, 0.8, 0.94, 5.0, 5e6),
    ("Asm2", 11, 5_200.0, 5e6, 0.020, 0.8, 0.94, 5.0, 2e6),
    ("Asm3", 12, 5_500.0, 3e6, 0.020, 0.8, 0.94, 5.0, 5e6),
)
LOAD_POWER_PU = complex(
    2.829e6 * 0.707,
    2.829e6 * (1 - 0.707**2) ** 0.5,
) / S_BASE_VA

# Elplek stores the YD phase flag as 1. With the primary/secondary ordering
# below, this clock position reproduces the reported terminal angles. Treat
# this as a case-specific mapping until other YD orientations are compared.
ELPLEK_YD_CLOCK = 1


def build_calpos_example() -> Network:
    """Build the balanced positive-sequence pu model from ``beispiel.sld``."""
    network = Network()
    for bus_id in V_BASE_LL:
        network.add_bus(Bus(bus_id, f"B{bus_id}"))

    for name, from_bus, to_bus, r1, b1, r0, length, x1, x0 in LINE_DATA:
        z_base = V_BASE_LL[from_bus] ** 2 / S_BASE_VA
        z1_pu = complex(r1 * length, x1 * length) / z_base
        z0_pu = complex(r0 * length, x0 * length) / z_base
        # The SLD value is susceptance B/km (not capacitance); the manual says
        # zero-sequence susceptance equals B. Split total charging equally.
        y_charge_pu = 1j * b1 * length * z_base
        network.add_branch(
            Branch(
                from_bus,
                to_bus,
                z1_pu.real,
                z1_pu.imag,
                name=name,
                zero_sequence_impedance=z0_pu,
                shunt_admittance=y_charge_pu,
                zero_sequence_shunt_admittance=y_charge_pu,
            )
        )

    for name, primary, secondary, resistance, reactance, yd_plus30 in TRANSFORMER_DATA:
        z_base = V_BASE_LL[primary] ** 2 / S_BASE_VA
        z_pu = complex(resistance, reactance) / z_base
        clock = ELPLEK_YD_CLOCK if yd_plus30 else 0
        network.add_transformer(
            Transformer(
                primary,
                secondary,
                z_pu,
                z_pu,
                name=name,
                vector_group_clock=clock,
                from_connection="grounded_wye" if yd_plus30 else None,
                to_connection="delta" if yd_plus30 else None,
            )
        )

    # Ld1 is a proper Elplek PQ load. Its SLD parameters are 2.829 MVA,
    # pf=0.707, so the load-flow power remains constant.
    # Elplek models induction motors as fixed impedances during load flow. Port
    # the equivalent from SLD motor nameplate values, independently of results.
    for (
        name,
        bus_id,
        nominal_voltage_ll,
        nominal_power,
        nominal_slip,
        power_factor,
        efficiency,
        locked_rotor_ratio,
        output_power,
    ) in MOTOR_DATA:
        motor = solve_asynchronous_motor_equivalent(
            nominal_voltage_ll=nominal_voltage_ll,
            nominal_power=nominal_power,
            nominal_slip=nominal_slip,
            nominal_power_factor=power_factor,
            efficiency=efficiency,
            locked_rotor_current_ratio=locked_rotor_ratio,
            output_power=output_power,
        )
        y_physical = 1 / motor.impedance
        y_pu = y_physical * V_BASE_LL[bus_id] ** 2 / S_BASE_VA
        network.add_shunt(Shunt(bus_id, y_pu, name=name))

    return network


def solve_calpos_example() -> LoadFlowResult:
    """Solve the example with F1 as slack and G1 as the PV bus."""
    network = build_calpos_example()
    return solve_load_flow(
        network,
        slack_bus_id=1,
        slack_voltage=1 + 0j,
        specified_power_injections={
            7: -LOAD_POWER_PU,
            13: 40e6 / S_BASE_VA,
        },
        pv_bus_voltage_magnitudes={13: 8_747.0 / V_BASE_LL[13]},
    )


def _line_angle(voltage: complex) -> float:
    """Convert the phase-neutral solver angle to Elplek's Vab display angle."""
    return phase(voltage) * 180 / pi + 30


def main() -> None:
    result = solve_calpos_example()
    print(f"Newton-Raphson: {result.iterations} iterations; "
          f"mismatch={result.maximum_mismatch:.3g} pu")
    print("Bus   Elplek Vab [kV]   Python [kV]   Elplek angle   Python angle")
    for label, bus_id, measured_kv, angle_deg in (
        ("B7/Ld1", 7, 65.6875, 32.5),
        ("B10/Asm1", 10, 5.1022, -3.5),
        ("B11/Asm2", 11, 5.0336, -0.2),
        ("B12/Asm3", 12, 5.5182, -2.5),
        ("B13/G1", 13, 8.7470, 7.1),
    ):
        voltage = result.bus_voltages[bus_id]
        value_kv = abs(voltage) * V_BASE_LL[bus_id] / 1e3
        print(f"{label:9} {measured_kv:14.4f} {value_kv:13.4f} "
              f"{angle_deg:13.2f} {_line_angle(voltage):13.2f}")

    for label, bus_id, measured_mva, angle_deg in (
        ("F1", 1, 26.214, 161.0),
        ("G1", 13, 41.353, 14.7),
    ):
        power = result.bus_power_injections[bus_id] * S_BASE_VA / 1e6
        print(f"{label}: P={power.real:.4f} MW, Q={power.imag:.4f} MVAr "
              f"(|S|={abs(power):.4f} MVA, angle={phase(power) * 180 / pi:.2f} deg); "
              f"Elplek |S|={measured_mva:.3f} MVA, angle={angle_deg:.1f} deg")


if __name__ == "__main__":
    main()
