"""Compare the settled earth-fault residual current with Elplek exports.

Run from the repository root with:
    python python-port/reference-cases/compare_earth_fault.py

The Python model uses G1's normal positive-sequence reactance and includes the
Z1 + Ld1 shunt load branch. Its fault current is compared with CT1's residual
current, which is the corresponding Elplek measurement for this grounded-fault
case. This does not reproduce Elplek's full time-domain waveform or every CT
sequence-current output.
"""
from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path

from elplek.model import Bus, Network, Shunt, Source, Transformer
from elplek.sequence_network import solve_network_fault


ROOT = Path(__file__).parent
EXPORTS = ROOT / "exports"
V_GENERATOR_LL = 12_000.0
V_TRAFO_PRIMARY = 15_000.0
V_TRAFO_SECONDARY = 165_000.0
TRAFO_TAP_FROM_SIDE = V_TRAFO_PRIMARY / V_TRAFO_SECONDARY
TRAFO_REFLECTION = (V_TRAFO_SECONDARY / V_TRAFO_PRIMARY) ** 2
LOAD_V_LL = 121_000.0
LOAD_VA = 100_000_000.0
LOAD_PF = 0.85


def read_last_ct_sample(path: Path, label: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["ct_label"] == label]
    if not rows:
        raise ValueError(f"No samples for {label} in {path.name}")
    return max(rows, key=lambda row: float(row["time_s"]))


def solve_fault_current(
    transformer_multiplier: float,
    z1_branch: complex = 2.85 + 8.15j,
) -> float:
    network = Network()
    network.add_bus(Bus(1, "G1"))
    network.add_bus(Bus(2, "CT1/Z1 fault node"))

    network.add_source(
        Source(
            bus_id=1,
            voltage=V_GENERATOR_LL / sqrt(3),
            impedance=0.0044 + 2.39j,
            name="G1",
            negative_sequence_impedance=0.0044 + 0.26j,
            zero_sequence_impedance=0.0044 + 0j,
        )
    )

    transformer_z = (0.015 + 0.0735j) * TRAFO_REFLECTION * transformer_multiplier
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
            tap_ratio=TRAFO_TAP_FROM_SIDE,
        )
    )

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

    result = solve_network_fault(network, 2, "single_line_to_ground")
    return abs(result.fault_current)


def main() -> None:
    cases = (
        ("baseline", "earth_fault_base.csv", 1.0, 2.85 + 8.15j),
        ("T1 doubled", "earth_fault_t1_z2.csv", 2.0, 2.85 + 8.15j),
        ("Z1 R/X doubled", "earth_fault_z2.csv", 1.0, 5.70 + 16.30j),
        ("Z1 resistance changed", "earth_fault_r1.csv", 1.0, 1.00 + 8.15j),
    )
    results: dict[str, tuple[float, float]] = {}

    for name, filename, multiplier, z1_branch in cases:
        python_current = solve_fault_current(multiplier, z1_branch)
        ct1 = read_last_ct_sample(EXPORTS / filename, "CT1")
        time_s = float(ct1["time_s"])
        elplek_residual = float(ct1["Isum_A"])
        error_pct = (python_current / elplek_residual - 1) * 100
        results[name] = (python_current, elplek_residual)
        print(
            f"{name} at t={time_s:g} s (CT1 residual): "
            f"Python {python_current:.1f} A; Elplek {elplek_residual:.1f} A; "
            f"difference {error_pct:+.2f}%"
        )

    for case_name in ("T1 doubled", "Z1 R/X doubled", "Z1 resistance changed"):
        python_delta = (results[case_name][0] / results["baseline"][0] - 1) * 100
        elplek_delta = (results[case_name][1] / results["baseline"][1] - 1) * 100
        print(
            f"Change for {case_name}: "
            f"Python {python_delta:+.2f}%; Elplek {elplek_delta:+.2f}%"
        )


if __name__ == "__main__":
    main()
