"""Compare one steady-state sequence sensitivity with Elplek reference data."""

from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path

from elplek.model import Bus, Network, Source, Transformer
from elplek.sequence_network import solve_network_fault


REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "python-port"
    / "reference-cases"
    / "exports"
)
ELPLEK_I1_DELTA_TOLERANCE_PERCENT_POINTS = 0.5


def _elplek_last_i1(filename: str) -> float:
    with (REFERENCE_DIR / filename).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["ct_label"] == "CT1"]
    last = max(rows, key=lambda row: float(row["time_s"]))
    assert float(last["time_s"]) == 11.032
    return float(last["I1_A"])


def _python_i1(transformer_multiplier: float) -> float:
    # Generator parameters from G1 in faultest.sld.
    network = Network()
    network.add_bus(Bus(1, "G1"))
    network.add_bus(Bus(2, "CT1/Z1 fault node"))
    network.add_source(
        Source(
            bus_id=1,
            voltage=12_000 / sqrt(3),
            impedance=0.0044 + 2.39j,
            name="G1",
            negative_sequence_impedance=0.0044 + 0.26j,
            zero_sequence_impedance=0.0044 + 0j,
        )
    )

    # T1's impedances are given on the 15 kV primary side in the SLD.
    # Refer them to the 165 kV to-side used by this network model.
    ratio = 165_000 / 15_000
    z = (0.015 + 0.0735j) * ratio**2 * transformer_multiplier
    network.add_transformer(
        Transformer(
            from_bus=1,
            to_bus=2,
            positive_sequence_impedance=z,
            negative_sequence_impedance=z,
            zero_sequence_impedance=z,
            zero_sequence_mode="through",
            name="T1",
            from_connection="grounded_wye",
            to_connection="grounded_wye",
            tap_ratio=1 / ratio,
        )
    )
    result = solve_network_fault(network, 2, "single_line_to_ground")
    return abs(result.positive_current)


def test_t1_doubling_positive_sequence_sensitivity_matches_elplek() -> None:
    python_base = _python_i1(1)
    python_t1_doubled = _python_i1(2)
    elplek_base = _elplek_last_i1("earth_fault_base.csv")
    elplek_t1_doubled = _elplek_last_i1("earth_fault_t1_z2.csv")

    python_delta = (python_t1_doubled / python_base - 1) * 100
    elplek_delta = (elplek_t1_doubled / elplek_base - 1) * 100

    assert abs(python_delta - elplek_delta) < ELPLEK_I1_DELTA_TOLERANCE_PERCENT_POINTS
