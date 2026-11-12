"""Compare the Elplek sequence Thevenin report with the Python network model.

Run from the repository root with:
    python python-port/reference-cases/compare_earth_fault_thevenin.py

The saved values are the user's Elplek 0.995 results for the baseline SLD and
the variant where T1 R/X and R0/X0 are doubled. This is a steady-state
sequence-impedance comparison; it does not validate Elplek dynamic currents.
"""
from __future__ import annotations

from math import sqrt

from elplek.model import Bus, Network, Shunt, Source, Transformer
from elplek.sequence_network import calculate_sequence_thevenin


TOLERANCE_OHM = 0.02
TRANSFORMER_REFLECTION = (165_000 / 15_000) ** 2
TRANSFORMER_TAP = 15_000 / 165_000
GENERATOR_RESISTANCE = 0.0044
GENERATOR_XD = 2.39
GENERATOR_XD_TRANSIENT = 0.3445
GENERATOR_XD_SUBTRANSIENT = 0.26
GENERATOR_X2 = 0.26
GENERATOR_X0 = 0.0
TRANSFORMER_Z = (0.015 + 0.0735j) * TRANSFORMER_REFLECTION
Z1_BRANCH = 2.85 + 8.15j
LOAD_V_LL = 121_000.0
LOAD_VA = 100_000_000.0
LOAD_PF = 0.85

# (case, stage, sequence): Elplek R+jX in ohms.
REFERENCE = {
    ("baseline", "normal", "Z1"): 69.394 + 88.997j,
    ("baseline", "transient", "Z1"): 10.816 + 39.870j,
    ("baseline", "subtransient", "Z1"): 8.0725 + 33.343j,
    ("baseline", "subtransient", "Z2"): 8.0725 + 33.343j,
    ("baseline", "subtransient", "Z0"): 2.3474 + 8.8935j,
    ("T1 doubled", "normal", "Z1"): 70.665 + 89.070j,
    ("T1 doubled", "transient", "Z1"): 14.389 + 44.444j,
    ("T1 doubled", "subtransient", "Z1"): 11.55 + 38.567j,
    ("T1 doubled", "subtransient", "Z2"): 11.55 + 38.567j,
    ("T1 doubled", "subtransient", "Z0"): 4.1623 + 17.787j,
}


def network_for_stage(
    generator_x: float,
    transformer_multiplier: float,
    z1_branch: complex = Z1_BRANCH,
) -> Network:
    network = Network()
    network.add_bus(Bus(1, "G1"))
    network.add_bus(Bus(2, "CT1/Z1 fault node"))

    # T1's tap reflects G1's sequence impedances by 11^2 into the 165 kV
    # network. Keep the generator impedance on its original side here.
    network.add_source(
        Source(
            bus_id=1,
            voltage=12_000 / sqrt(3),
            impedance=GENERATOR_RESISTANCE + 1j * generator_x,
            name="G1",
            negative_sequence_impedance=GENERATOR_RESISTANCE + 1j * GENERATOR_X2,
            zero_sequence_impedance=GENERATOR_RESISTANCE + 1j * GENERATOR_X0,
        )
    )

    transformer_z = TRANSFORMER_Z * transformer_multiplier
    network.add_transformer(
        Transformer(
            from_bus=1,
            to_bus=2,
            positive_sequence_impedance=transformer_z,
            negative_sequence_impedance=transformer_z,
            zero_sequence_impedance=transformer_z,
            zero_sequence_mode="through",
            name="T1",
            from_connection="grounded_wye",
            to_connection="grounded_wye",
            tap_ratio=TRANSFORMER_TAP,
        )
    )

    # In the sequence-impedance calculation Elplek includes the load. Z1 and
    # Ld1 form the shunt branch at the fault bus; its neutral is ungrounded,
    # so it has no zero-sequence admittance.
    load_r = LOAD_V_LL**2 / LOAD_VA * LOAD_PF
    load_x = LOAD_V_LL**2 / LOAD_VA * sqrt(1 - LOAD_PF**2)
    branch_impedance = z1_branch + complex(load_r, load_x)
    branch_admittance = 1 / branch_impedance
    network.add_shunt(
        Shunt(
            bus_id=2,
            admittance=branch_admittance,
            name="Z1 + Ld1",
            negative_sequence_admittance=branch_admittance,
            zero_sequence_admittance=0j,
        )
    )
    return network


def main() -> None:
    cases = (("baseline", 1.0), ("T1 doubled", 2.0))
    stages = (
        ("normal", GENERATOR_XD),
        ("transient", GENERATOR_XD_TRANSIENT),
        ("subtransient", GENERATOR_XD_SUBTRANSIENT),
    )
    failures: list[str] = []

    for case, multiplier in cases:
        for stage, generator_x in stages:
            equivalent = calculate_sequence_thevenin(
                network_for_stage(generator_x, multiplier), 2
            )
            sequences = {"Z1": equivalent.z1, "Z2": equivalent.z2, "Z0": equivalent.z0}
            selected = ("Z1",) if stage != "subtransient" else ("Z1", "Z2", "Z0")
            for sequence in selected:
                key = (case, stage, sequence)
                calculated = sequences[sequence]
                expected = REFERENCE[key]
                error = abs(calculated - expected)
                status = "OK" if error <= TOLERANCE_OHM else "MISMATCH"
                print(
                    f"{case:12s} {stage:13s} {sequence}: "
                    f"Python {calculated.real:.4f} {calculated.imag:+.4f}j; "
                    f"Elplek {expected.real:.4f} {expected.imag:+.4f}j; "
                    f"error {error:.4f} ohm [{status}]"
                )
                if error > TOLERANCE_OHM:
                    failures.append(f"{key}: {error:.4f} ohm")

    if failures:
        raise SystemExit("Sequence Thevenin comparison failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
