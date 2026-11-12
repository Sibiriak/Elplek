import numpy as np
import pytest

from elplek.model import Branch, Bus, Network, Shunt, Transformer
from elplek.ybus import build_sequence_ybus, build_ybus


def test_two_bus_network_builds_expected_ybus():
    network = Network()
    network.add_bus(Bus(10, "A"))
    network.add_bus(Bus(20, "B"))
    network.add_branch(Branch(10, 20, resistance=3.0, reactance=4.0))

    expected = np.array(
        [[0.12 - 0.16j, -0.12 + 0.16j],
         [-0.12 + 0.16j, 0.12 - 0.16j]],
        dtype=complex,
    )
    np.testing.assert_allclose(build_ybus(network), expected)


def test_branch_to_ground_stamps_only_bus_diagonal():
    network = Network()
    network.add_bus(Bus(7))
    network.add_branch(Branch(7, 0, resistance=2.0, reactance=0.0))

    np.testing.assert_allclose(build_ybus(network), np.array([[0.5 + 0j]]))


def test_bus_ids_map_by_insertion_order_not_numeric_value():
    network = Network()
    network.add_bus(Bus(20))
    network.add_bus(Bus(5))
    network.add_branch(Branch(20, 5, 1.0, 0.0))

    np.testing.assert_allclose(
        build_ybus(network),
        np.array([[1 + 0j, -1 + 0j], [-1 + 0j, 1 + 0j]]),
    )


def test_duplicate_bus_rejected():
    network = Network()
    network.add_bus(Bus(1))
    with pytest.raises(ValueError, match="Duplicate"):
        network.add_bus(Bus(1))


def test_unknown_branch_terminal_rejected():
    network = Network()
    network.add_bus(Bus(1))
    with pytest.raises(ValueError, match="unknown bus"):
        network.add_branch(Branch(1, 2, 1.0, 0.0))


def test_ground_to_ground_branch_rejected():
    with pytest.raises(ValueError, match="ground to ground"):
        Branch(0, 0, 1.0, 0.0)


def test_sequence_ybus_uses_branch_specific_negative_impedance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(
        Branch(1, 2, 1.0, 0.0, negative_sequence_impedance=2 + 0j)
    )

    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        np.array([[0.5 + 0j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0j]]),
    )


def test_sequence_ybus_requires_zero_sequence_data():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(1, 2, 1.0, 0.0))

    with pytest.raises(ValueError, match="zero-sequence impedance"):
        build_sequence_ybus(network, "zero")


def test_sequence_ybus_assembles_zero_sequence_and_grounded_branch():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(
        Branch(1, 2, 1.0, 0.0, zero_sequence_impedance=4 + 0j)
    )
    network.add_branch(
        Branch(2, 0, 1.0, 0.0, zero_sequence_impedance=2 + 0j)
    )

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        np.array([[0.25 + 0j, -0.25 + 0j], [-0.25 + 0j, 0.75 + 0j]]),
    )


def test_sequence_ybus_rejects_unknown_sequence():
    with pytest.raises(ValueError, match="sequence must"):
        build_sequence_ybus(Network(), "negative-sequence")


def test_transformer_stamps_positive_and_negative_sequence():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1 + 0.2j, 0.15 + 0.25j,
        zero_sequence_impedance=0.3 + 0.4j,
        zero_sequence_mode="through",
    ))

    y1 = 1 / (0.1 + 0.2j)
    y2 = 1 / (0.15 + 0.25j)
    np.testing.assert_allclose(
        build_sequence_ybus(network, "positive"),
        [[y1, -y1], [-y1, y1]],
    )
    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        [[y2, -y2], [-y2, y2]],
    )


def test_transformer_blocked_zero_sequence_does_not_connect_buses():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1 + 0.2j, 0.1 + 0.2j,
        zero_sequence_mode="blocked",
    ))

    np.testing.assert_allclose(build_sequence_ybus(network, "zero"), np.zeros((2, 2)))


def test_transformer_grounded_wye_delta_stamps_zero_sequence_shunt_on_from_side():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z0 = 0.3 + 0.4j
    network.add_transformer(Transformer(
        1, 2, 0.1 + 0.2j, 0.1 + 0.2j,
        zero_sequence_impedance=z0,
        zero_sequence_mode="grounded_wye_delta",
    ))

    expected = np.array([[1 / z0, 0j], [0j, 0j]])
    np.testing.assert_allclose(build_sequence_ybus(network, "zero"), expected)


def test_transformer_rejects_missing_zero_sequence_impedance_for_through_mode():
    with pytest.raises(ValueError, match="zero-sequence impedance"):
        Transformer(1, 2, 0.1j, 0.1j, zero_sequence_mode="through")


def test_network_rejects_transformer_with_unknown_bus():
    network = Network()
    network.add_bus(Bus(1))
    with pytest.raises(ValueError, match="unknown bus"):
        network.add_transformer(Transformer(1, 2, 0.1j, 0.1j))


def test_positive_sequence_ybus_includes_transformer_impedance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.2 + 0.4j, 0.3 + 0.5j,
    ))
    y = 1 / (0.2 + 0.4j)
    np.testing.assert_allclose(
        build_ybus(network),
        [[y, -y], [-y, y]],
    )



def test_explicit_grounded_wye_delta_zero_sequence_shunt_on_grounded_wye_side():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=0.2j,
        from_connection="grounded_wye",
        to_connection="delta",
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[1 / (0.2j), 0j], [0j, 0j]],
    )


def test_explicit_delta_grounded_wye_places_zero_sequence_shunt_on_to_side():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=0.2j,
        from_connection="delta",
        to_connection="grounded_wye",
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[0j, 0j], [0j, 1 / (0.2j)]],
    )


def test_explicit_grounded_wye_wye_includes_three_times_neutral_impedance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=0.2j,
        from_connection="grounded_wye",
        to_connection="grounded_wye",
        from_neutral_impedance=0.01j,
        to_neutral_impedance=0.02j,
    ))
    y = 1 / (0.2j + 3 * 0.01j + 3 * 0.02j)

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[y, -y], [-y, y]],
    )


@pytest.mark.parametrize(
    ("from_connection", "to_connection"),
    [
        ("ungrounded_wye", "grounded_wye"),
        ("grounded_wye", "ungrounded_wye"),
        ("delta", "delta"),
    ],
)
def test_explicit_blocking_winding_connections_block_zero_sequence(
    from_connection, to_connection
):
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=0.2j,
        from_connection=from_connection,
        to_connection=to_connection,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        np.zeros((2, 2)),
    )


def test_transformer_requires_both_explicit_winding_connections():
    with pytest.raises(ValueError, match="specified together"):
        Transformer(
            1, 2, 0.1j, 0.1j,
            from_connection="grounded_wye",
        )


def test_transformer_rejects_neutral_impedance_on_delta_winding():
    with pytest.raises(ValueError, match="requires grounded_wye"):
        Transformer(
            1, 2, 0.1j, 0.1j,
            from_connection="grounded_wye",
            to_connection="delta",
            to_neutral_impedance=0.01j,
        )



def test_vector_group_clock_11_applies_positive_sequence_phase_displacement():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z = 0.2 + 0.4j
    y = 1 / z
    delta = np.deg2rad(30.0)
    network.add_transformer(Transformer(
        1, 2, z, z,
        zero_sequence_impedance=0.5 + 0.2j,
        from_connection="grounded_wye",
        to_connection="delta",
        vector_group_clock=11,
    ))

    expected = np.array([
        [y, -y * np.exp(-1j * delta)],
        [-y * np.exp(1j * delta), y],
    ])
    np.testing.assert_allclose(build_ybus(network), expected)
    np.testing.assert_allclose(
        build_sequence_ybus(network, "positive"),
        expected,
    )


def test_vector_group_phase_displacement_reverses_for_negative_sequence():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z2 = 0.3 + 0.6j
    y2 = 1 / z2
    delta = np.deg2rad(30.0)
    network.add_transformer(Transformer(
        1, 2, 0.2 + 0.4j, z2,
        zero_sequence_impedance=0.5 + 0.2j,
        from_connection="grounded_wye",
        to_connection="delta",
        vector_group_clock=11,
    ))

    expected = np.array([
        [y2, -y2 * np.exp(1j * delta)],
        [-y2 * np.exp(-1j * delta), y2],
    ])
    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        expected,
    )


def test_vector_group_clock_does_not_rotate_zero_sequence_network():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z0 = 0.2 + 0.3j
    network.add_transformer(Transformer(
        1, 2, 0.1 + 0.2j, 0.1 + 0.2j,
        zero_sequence_impedance=z0,
        from_connection="grounded_wye",
        to_connection="delta",
        vector_group_clock=11,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[1 / z0, 0j], [0j, 0j]],
    )


@pytest.mark.parametrize("clock", [-1, 13, 1.5, True])
def test_transformer_rejects_invalid_vector_group_clock(clock):
    with pytest.raises(ValueError, match="vector_group_clock"):
        Transformer(1, 2, 0.1j, 0.1j, vector_group_clock=clock)


def test_nonzero_vector_group_clock_requires_explicit_connections():
    with pytest.raises(ValueError, match="requires both winding connections"):
        Transformer(1, 2, 0.1j, 0.1j, vector_group_clock=11)


def test_vector_group_clock_12_represents_zero_phase_displacement():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z = 0.2 + 0.4j
    network.add_transformer(Transformer(
        1, 2, z, z,
        from_connection="grounded_wye",
        to_connection="grounded_wye",
        vector_group_clock=12,
    ))

    y = 1 / z
    np.testing.assert_allclose(
        build_ybus(network),
        [[y, -y], [-y, y]],
    )



@pytest.mark.parametrize(
    ("from_connection", "to_connection", "clock"),
    [
        ("grounded_wye", "grounded_wye", 11),
        ("grounded_wye", "delta", 12),
        ("delta", "delta", 1),
    ],
)
def test_transformer_rejects_clock_parity_inconsistent_with_windings(
    from_connection, to_connection, clock
):
    with pytest.raises(ValueError, match="parity is inconsistent"):
        Transformer(
            1, 2, 0.1j, 0.1j,
            from_connection=from_connection,
            to_connection=to_connection,
            vector_group_clock=clock,
        )



def test_transformer_tap_ratio_stamps_correct_positive_sequence_ybus():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z = 0.2 + 0.4j
    tap = 1.1
    y = 1 / z
    network.add_transformer(Transformer(
        1, 2, z, z,
        tap_ratio=tap,
    ))

    expected = np.array([
        [y / tap**2, -y / tap],
        [-y / tap, y],
    ])
    np.testing.assert_allclose(build_ybus(network), expected)
    np.testing.assert_allclose(
        build_sequence_ybus(network, "positive"),
        expected,
    )


def test_transformer_tap_ratio_applies_to_negative_sequence_ybus():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z2 = 0.3 + 0.6j
    tap = 0.9
    y2 = 1 / z2
    network.add_transformer(Transformer(
        1, 2, 0.2 + 0.4j, z2,
        tap_ratio=tap,
    ))

    expected = np.array([
        [y2 / tap**2, -y2 / tap],
        [-y2 / tap, y2],
    ])
    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        expected,
    )


def test_transformer_tap_ratio_combines_with_vector_group_phase_shift():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z = 0.2 + 0.4j
    tap = 1.1
    y = 1 / z
    delta = np.deg2rad(30.0)
    network.add_transformer(Transformer(
        1, 2, z, z,
        from_connection="grounded_wye",
        to_connection="delta",
        vector_group_clock=11,
        tap_ratio=tap,
    ))

    expected = np.array([
        [y / tap**2, -y * np.exp(-1j * delta) / tap],
        [-y * np.exp(1j * delta) / tap, y],
    ])
    np.testing.assert_allclose(build_ybus(network), expected)


def test_transformer_tap_ratio_applies_to_zero_sequence_through_mode():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z0 = 0.2 + 0.3j
    tap = 1.2
    y0 = 1 / z0
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=z0,
        zero_sequence_mode="through",
        tap_ratio=tap,
    ))

    expected = np.array([
        [y0 / tap**2, -y0 / tap],
        [-y0 / tap, y0],
    ])
    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        expected,
    )


def test_transformer_tap_ratio_scales_grounded_wye_delta_zero_sequence_shunt():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    z0 = 0.2 + 0.3j
    tap = 1.2
    network.add_transformer(Transformer(
        1, 2, 0.1j, 0.1j,
        zero_sequence_impedance=z0,
        from_connection="grounded_wye",
        to_connection="delta",
        tap_ratio=tap,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[1 / z0 / tap**2, 0j], [0j, 0j]],
    )


@pytest.mark.parametrize(
    "tap_ratio",
    [0, -1, float("inf"), float("-inf"), float("nan"), True, "1.1"],
)
def test_transformer_rejects_invalid_tap_ratio(tap_ratio):
    with pytest.raises(ValueError, match="tap_ratio must be a finite positive number"):
        Transformer(1, 2, 0.1j, 0.1j, tap_ratio=tap_ratio)



def test_ybus_stamps_shunt_admittance_on_correct_bus_diagonal():
    network = Network()
    network.add_bus(Bus(20))
    network.add_bus(Bus(5))
    network.add_shunt(Shunt(5, 0.02 + 0.1j, name="bus capacitor"))

    np.testing.assert_allclose(
        build_ybus(network),
        [[0j, 0j], [0j, 0.02 + 0.1j]],
    )


def test_ybus_accumulates_multiple_shunts_at_same_bus():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(1, 0.01 + 0.02j))
    network.add_shunt(Shunt(1, 0.03 + 0.04j))

    np.testing.assert_allclose(
        build_ybus(network),
        [[0.04 + 0.06j]],
    )


def test_sequence_ybus_uses_shunt_admittance_for_positive_sequence():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(1, 0.02 + 0.1j))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "positive"),
        [[0.02 + 0.1j]],
    )


def test_sequence_ybus_uses_explicit_negative_sequence_shunt_admittance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(
        1, 0.02 + 0.1j,
        negative_sequence_admittance=0.03 + 0.2j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        [[0.03 + 0.2j]],
    )


def test_sequence_ybus_negative_shunt_falls_back_to_positive_admittance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(1, 0.02 + 0.1j))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        [[0.02 + 0.1j]],
    )


def test_sequence_ybus_requires_zero_sequence_shunt_admittance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(1, 0.02 + 0.1j, name="compensation reactor"))

    with pytest.raises(ValueError, match="no zero-sequence admittance"):
        build_sequence_ybus(network, "zero")


def test_sequence_ybus_stamps_zero_sequence_shunt_admittance():
    network = Network()
    network.add_bus(Bus(1))
    network.add_shunt(Shunt(
        1, 0.02 + 0.1j,
        negative_sequence_admittance=0.03 + 0.2j,
        zero_sequence_admittance=0.04 + 0.3j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[0.04 + 0.3j]],
    )


def test_network_rejects_shunt_on_unknown_bus():
    network = Network()
    network.add_bus(Bus(1))

    with pytest.raises(ValueError, match="Shunt references unknown bus"):
        network.add_shunt(Shunt(2, 0.1j))


def test_shunt_rejects_nonpositive_bus_id():
    with pytest.raises(ValueError, match="Shunt bus_id must be positive"):
        Shunt(0, 0.1j)


@pytest.mark.parametrize(
    "admittance",
    [complex(float("nan"), 0), complex(0, float("inf"))],
)
def test_shunt_rejects_nonfinite_admittance(admittance):
    with pytest.raises(ValueError, match="must be finite"):
        Shunt(1, admittance)



def test_branch_nominal_pi_model_splits_total_shunt_between_both_buses():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        shunt_admittance=0.0 + 2.0j,
    ))

    np.testing.assert_allclose(
        build_ybus(network),
        [[1.0 + 1.0j, -1.0 + 0j], [-1.0 + 0j, 1.0 + 1.0j]],
    )


def test_branch_nominal_pi_model_stamps_only_connected_end_when_other_is_ground():
    network = Network()
    network.add_bus(Bus(1))
    network.add_branch(Branch(
        1, 0, resistance=1.0, reactance=0.0,
        shunt_admittance=0.0 + 2.0j,
    ))

    np.testing.assert_allclose(build_ybus(network), [[1.0 + 1.0j]])


def test_sequence_pi_model_uses_positive_and_negative_shunt_data():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        negative_sequence_impedance=2.0 + 0j,
        shunt_admittance=0.0 + 0.2j,
        negative_sequence_shunt_admittance=0.0 + 0.6j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "positive"),
        [[1.0 + 0.1j, -1.0 + 0j], [-1.0 + 0j, 1.0 + 0.1j]],
    )
    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        [[0.5 + 0.3j, -0.5 + 0j], [-0.5 + 0j, 0.5 + 0.3j]],
    )


def test_sequence_pi_model_negative_shunt_defaults_to_positive_shunt():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        shunt_admittance=0.0 + 0.2j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "negative"),
        [[1.0 + 0.1j, -1.0 + 0j], [-1.0 + 0j, 1.0 + 0.1j]],
    )


def test_sequence_pi_model_requires_zero_sequence_shunt_if_positive_is_nonzero():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        zero_sequence_impedance=4.0 + 0j,
        shunt_admittance=0.0 + 0.2j,
        name="line with charging",
    ))

    with pytest.raises(ValueError, match="no zero-sequence shunt admittance"):
        build_sequence_ybus(network, "zero")


def test_sequence_pi_model_stamps_explicit_zero_sequence_shunt():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        zero_sequence_impedance=4.0 + 0j,
        shunt_admittance=0.0 + 0.2j,
        zero_sequence_shunt_admittance=0.0 + 0.4j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[0.25 + 0.2j, -0.25 + 0j], [-0.25 + 0j, 0.25 + 0.2j]],
    )


def test_sequence_pi_model_with_zero_positive_shunt_defaults_zero_sequence_shunt_to_zero():
    network = Network()
    network.add_bus(Bus(1))
    network.add_bus(Bus(2))
    network.add_branch(Branch(
        1, 2, resistance=1.0, reactance=0.0,
        zero_sequence_impedance=4.0 + 0j,
    ))

    np.testing.assert_allclose(
        build_sequence_ybus(network, "zero"),
        [[0.25 + 0j, -0.25 + 0j], [-0.25 + 0j, 0.25 + 0j]],
    )


@pytest.mark.parametrize(
    "admittance_field",
    [
        {"shunt_admittance": complex(float("nan"), 0)},
        {"negative_sequence_shunt_admittance": complex(0, float("inf"))},
        {"zero_sequence_shunt_admittance": complex(float("-inf"), 0)},
    ],
)
def test_branch_rejects_nonfinite_sequence_shunt_admittance(admittance_field):
    with pytest.raises(ValueError, match="Branch shunt admittances must be finite"):
        Branch(1, 2, 1.0, 0.0, **admittance_field)
