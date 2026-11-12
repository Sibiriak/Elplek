"""Y-bus assembly from the basic network model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .impedance import impedance_admittance, stamp_two_node
from .model import Network


def _clock_displacement_rad(clock: int) -> float:
    """Return positive-sequence to-side displacement from the IEC clock number.

    The from side is the reference. Clock 11 means to-side voltage leads by
    30 degrees; clock 1 means it lags by 30 degrees. Values 0 and 12 both
    represent zero displacement (0 is retained as the legacy default).
    """
    if clock in (0, 12):
        return 0.0
    return float(np.deg2rad((12 - clock) * 30.0))


def _stamp_with_phase_displacement(
    ybus: NDArray[np.complex128],
    from_eq: int,
    to_eq: int,
    admittance: complex,
    phase_displacement: float,
    tap_ratio: float = 1.0,
) -> None:
    """Stamp an ideal transformer with magnitude tap and phase displacement.

    phase_displacement is the angle of the to-side positive-sequence voltage
    relative to the from side. tap_ratio is the ideal magnitude ratio on the
    from side. Sequence-network callers reverse phase sign for negative
    sequence. With a = tap_ratio*exp(-j*phase_displacement):
    Yff = y/|a|^2, Yft = -y/conj(a), Ytf = -y/a, Ytt = y.
    """
    f = from_eq - 1
    t = to_eq - 1
    tap = tap_ratio * np.exp(-1j * phase_displacement)
    ybus[f, f] += admittance / (abs(tap) ** 2)
    ybus[f, t] -= admittance / np.conjugate(tap)
    ybus[t, f] -= admittance / tap
    ybus[t, t] += admittance


def build_ybus(network: Network) -> NDArray[np.complex128]:
    """Build conventional nodal admittance matrix using nominal-pi branches.

    Ground is represented by terminal ID 0 and has no matrix row/column.
    Branch shunt_admittance is the total line charging admittance and is split
    equally between both terminal buses. Matrix ordering follows bus insertion
    order. Transformer positive-sequence impedances, magnitude taps, vector
    group phase displacement and independent bus shunts are included. Mutual
    coupling and IEC peak corrections are not included.
    """
    n = len(network.buses)
    ybus = np.zeros((n, n), dtype=np.complex128)
    index = network.bus_index

    for branch in network.branches:
        y = impedance_admittance(branch.resistance, branch.reactance)
        upstream = 0 if branch.from_bus == 0 else index[branch.from_bus] + 1
        downstream = 0 if branch.to_bus == 0 else index[branch.to_bus] + 1
        stamp_two_node(ybus, upstream, downstream, y)

        # Nominal-pi line model: the declared total charging admittance is
        # divided equally between the two ends. A grounded end has no row.
        half_shunt = complex(branch.shunt_admittance) / 2
        if branch.from_bus != 0:
            k = index[branch.from_bus]
            ybus[k, k] += half_shunt
        if branch.to_bus != 0:
            k = index[branch.to_bus]
            ybus[k, k] += half_shunt

    for transformer in network.transformers:
        y = impedance_admittance(
            complex(transformer.positive_sequence_impedance).real,
            complex(transformer.positive_sequence_impedance).imag,
        )
        _stamp_with_phase_displacement(
            ybus,
            index[transformer.from_bus] + 1,
            index[transformer.to_bus] + 1,
            y,
            _clock_displacement_rad(transformer.vector_group_clock),
            transformer.tap_ratio,
        )

    for shunt in network.shunts:
        ybus[index[shunt.bus_id], index[shunt.bus_id]] += complex(shunt.admittance)

    return ybus


def build_sequence_ybus(
    network: Network,
    sequence: str,
) -> NDArray[np.complex128]:
    """Build a sequence-specific Y-bus matrix.

    Positive and negative transformer networks use the vector-group phase
    displacement with opposite signs; zero-sequence phase displacement is
    zero and connection-dependent grounding rules are applied.
    Positive branches use R+jX. Negative branch impedances and charging
    admittances fall back to positive sequence when omitted. A zero-sequence
    branch impedance is required for every branch; zero-sequence charging is
    required when the positive-sequence charging admittance is nonzero.
    Independent bus shunts require explicit zero-sequence admittances.
    """
    if sequence not in {"positive", "negative", "zero"}:
        raise ValueError("sequence must be 'positive', 'negative', or 'zero'")

    n = len(network.buses)
    ybus = np.zeros((n, n), dtype=np.complex128)
    index = network.bus_index

    for branch in network.branches:
        if sequence == "positive":
            impedance = complex(branch.resistance, branch.reactance)
        elif sequence == "negative":
            impedance = branch.negative_sequence_impedance
            if impedance is None:
                impedance = complex(branch.resistance, branch.reactance)
        else:
            impedance = branch.zero_sequence_impedance
            if impedance is None:
                label = branch.name or f"{branch.from_bus}-{branch.to_bus}"
                raise ValueError(f"Branch {label} has no zero-sequence impedance")

        y = impedance_admittance(impedance.real, impedance.imag)
        upstream = 0 if branch.from_bus == 0 else index[branch.from_bus] + 1
        downstream = 0 if branch.to_bus == 0 else index[branch.to_bus] + 1
        stamp_two_node(ybus, upstream, downstream, y)

        if sequence == "positive":
            line_shunt = branch.shunt_admittance
        elif sequence == "negative":
            line_shunt = branch.negative_sequence_shunt_admittance
            if line_shunt is None:
                line_shunt = branch.shunt_admittance
        else:
            line_shunt = branch.zero_sequence_shunt_admittance
            if line_shunt is None:
                if branch.shunt_admittance != 0j:
                    label = branch.name or f"{branch.from_bus}-{branch.to_bus}"
                    raise ValueError(
                        f"Branch {label} has no zero-sequence shunt admittance"
                    )
                line_shunt = 0j

        half_shunt = complex(line_shunt) / 2
        if branch.from_bus != 0:
            k = index[branch.from_bus]
            ybus[k, k] += half_shunt
        if branch.to_bus != 0:
            k = index[branch.to_bus]
            ybus[k, k] += half_shunt

    for transformer in network.transformers:
        if sequence == "positive":
            impedance = transformer.positive_sequence_impedance
            mode = "through"
        elif sequence == "negative":
            impedance = transformer.negative_sequence_impedance
            mode = "through"
        else:
            mode = transformer.zero_sequence_mode
            impedance = transformer.zero_sequence_impedance
            from_eq = index[transformer.from_bus] + 1
            to_eq = index[transformer.to_bus] + 1

            if transformer.from_connection is not None:
                # Explicit winding connections take precedence over the legacy
                # zero_sequence_mode. The zero-sequence connection model remains
                # simplified; clock displacement is applied only to positive
                # and negative sequence networks.
                from_conn = transformer.from_connection
                to_conn = transformer.to_connection
                if from_conn == "ungrounded_wye" or to_conn == "ungrounded_wye":
                    continue
                if from_conn == "delta" and to_conn == "delta":
                    continue
                if impedance is None:
                    raise ValueError(
                        f"Transformer {transformer.name or transformer.from_bus} "
                        "needs zero-sequence impedance for its winding connections"
                    )

                y = impedance_admittance(complex(impedance).real, complex(impedance).imag)
                if from_conn == "grounded_wye" and to_conn == "grounded_wye":
                    z_total = (
                        complex(impedance)
                        + 3 * complex(transformer.from_neutral_impedance)
                        + 3 * complex(transformer.to_neutral_impedance)
                    )
                    y_total = impedance_admittance(z_total.real, z_total.imag)
                    _stamp_with_phase_displacement(
                        ybus, from_eq, to_eq, y_total, 0.0, transformer.tap_ratio
                    )
                elif from_conn == "grounded_wye" and to_conn == "delta":
                    z_total = complex(impedance) + 3 * complex(transformer.from_neutral_impedance)
                    y_total = impedance_admittance(z_total.real, z_total.imag)
                    ybus[from_eq - 1, from_eq - 1] += y_total / (transformer.tap_ratio ** 2)
                elif from_conn == "delta" and to_conn == "grounded_wye":
                    z_total = complex(impedance) + 3 * complex(transformer.to_neutral_impedance)
                    y_total = impedance_admittance(z_total.real, z_total.imag)
                    ybus[to_eq - 1, to_eq - 1] += y_total
                continue

            if mode == "blocked":
                continue
            if impedance is None:
                raise ValueError(
                    f"Transformer {transformer.name or transformer.from_bus} "
                    "has no zero-sequence impedance"
                )

        y = impedance_admittance(complex(impedance).real, complex(impedance).imag)
        from_eq = index[transformer.from_bus] + 1
        to_eq = index[transformer.to_bus] + 1
        if sequence == "zero" and mode == "grounded_wye_delta":
            # Legacy simplified mode: zero-sequence current returns through
            # the grounded neutral on the from side; no transfer to to_bus.
            ybus[from_eq - 1, from_eq - 1] += y / (transformer.tap_ratio ** 2)
        elif sequence == "zero":
            # Zero sequence is not phase-shifted; through-mode still observes
            # the magnitude tap on the from side.
            _stamp_with_phase_displacement(
                ybus, from_eq, to_eq, y, 0.0, transformer.tap_ratio
            )
        else:
            phase = _clock_displacement_rad(transformer.vector_group_clock)
            if sequence == "negative":
                phase = -phase
            _stamp_with_phase_displacement(
                ybus, from_eq, to_eq, y, phase, transformer.tap_ratio
            )

    for shunt in network.shunts:
        if sequence == "positive":
            admittance = shunt.admittance
        elif sequence == "negative":
            admittance = shunt.negative_sequence_admittance
            if admittance is None:
                admittance = shunt.admittance
        else:
            admittance = shunt.zero_sequence_admittance
            if admittance is None:
                label = shunt.name or f"bus {shunt.bus_id}"
                raise ValueError(
                    f"Shunt {label} has no zero-sequence admittance"
                )
        bus_index = index[shunt.bus_id]
        ybus[bus_index, bus_index] += complex(admittance)

    return ybus

