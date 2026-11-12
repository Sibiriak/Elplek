"""Basic network data structures for the incremental Elplek port."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True)
class Bus:
    """Electrical bus. IDs are positive integers; ground is represented by 0."""

    id: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError("Bus id must be positive; use 0 only for ground terminals")


@dataclass(frozen=True, slots=True)
class Branch:
    """Branch with positive-sequence R/X and optional negative/zero impedances.

    If negative_sequence_impedance is omitted, the positive-sequence
    impedance is used as an explicit engineering approximation. Zero-sequence
    impedance must be supplied before a zero-sequence matrix can be assembled.

    shunt_admittance is the total positive-sequence line charging admittance,
    divided equally between the two terminals (nominal-pi model). Negative
    shunt admittance falls back to positive when omitted. If positive shunt
    admittance is nonzero, zero_sequence_shunt_admittance must be supplied
    before a zero-sequence matrix is assembled.
    """

    from_bus: int
    to_bus: int
    resistance: float
    reactance: float
    name: str = ""
    negative_sequence_impedance: complex | None = None
    zero_sequence_impedance: complex | None = None
    shunt_admittance: complex = 0j
    negative_sequence_shunt_admittance: complex | None = None
    zero_sequence_shunt_admittance: complex | None = None

    def __post_init__(self) -> None:
        shunt_values = [self.shunt_admittance]
        if self.negative_sequence_shunt_admittance is not None:
            shunt_values.append(self.negative_sequence_shunt_admittance)
        if self.zero_sequence_shunt_admittance is not None:
            shunt_values.append(self.zero_sequence_shunt_admittance)
        if any(
            not (isfinite(complex(value).real) and isfinite(complex(value).imag))
            for value in shunt_values
        ):
            raise ValueError("Branch shunt admittances must be finite complex numbers")
        if self.from_bus < 0 or self.to_bus < 0:
            raise ValueError("Branch terminal IDs cannot be negative")
        if self.from_bus == 0 and self.to_bus == 0:
            raise ValueError("A branch cannot connect ground to ground")


@dataclass(frozen=True, slots=True)
class Source:
    """Voltage source with positive-sequence impedance and optional Z2/Z0.

    voltage and impedance represent the positive-sequence source. Negative- and
    zero-sequence impedances must be supplied explicitly for asymmetrical
    fault equivalents; omitted values are not assumed.
    """

    bus_id: int
    voltage: complex
    impedance: complex
    name: str = ""
    negative_sequence_impedance: complex | None = None
    zero_sequence_impedance: complex | None = None

    def __post_init__(self) -> None:
        if self.bus_id <= 0:
            raise ValueError("Source bus_id must be positive")


@dataclass(frozen=True, slots=True)
class Shunt:
    """Admittance connected from one bus to ground.

    Positive-sequence admittance is required. Negative-sequence admittance
    falls back to positive sequence when omitted. Zero-sequence admittance
    must be specified before assembling a zero-sequence Y-bus.
    """

    bus_id: int
    admittance: complex
    name: str = ""
    negative_sequence_admittance: complex | None = None
    zero_sequence_admittance: complex | None = None

    def __post_init__(self) -> None:
        if self.bus_id <= 0:
            raise ValueError("Shunt bus_id must be positive")
        values = [self.admittance]
        if self.negative_sequence_admittance is not None:
            values.append(self.negative_sequence_admittance)
        if self.zero_sequence_admittance is not None:
            values.append(self.zero_sequence_admittance)
        if any(
            not (isfinite(complex(value).real) and isfinite(complex(value).imag))
            for value in values
        ):
            raise ValueError("Shunt admittances must be finite complex numbers")


@dataclass(frozen=True, slots=True)
class Transformer:
    """Simplified two-winding transformer sequence equivalent.

    zero_sequence_mode:
      blocked: no zero-sequence connection between terminals.
      through: zero-sequence impedance connects both terminals.
      grounded_wye_delta: zero-sequence impedance is a grounded shunt at
      from_bus only; zero sequence does not transfer to to_bus.

    Explicit winding connections, neutral impedances, and a vector-group clock
    number are supported in a simplified sequence model. The clock number
    defines the positive-sequence phase displacement of to_bus relative to
    from_bus; 0 is the legacy no-shift default and 12 is the standard zero-shift
    clock position. tap_ratio is the positive magnitude of the ideal complex tap
    on the from_bus side; it must be finite and greater than zero. Core effects
    and the magnetizing branch are not represented yet.
    """
    from_bus: int
    to_bus: int
    positive_sequence_impedance: complex
    negative_sequence_impedance: complex
    zero_sequence_impedance: complex | None = None
    zero_sequence_mode: str = "blocked"
    name: str = ""
    from_connection: str | None = None
    to_connection: str | None = None
    from_neutral_impedance: complex = 0j
    to_neutral_impedance: complex = 0j
    vector_group_clock: int = 0
    tap_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.from_bus <= 0 or self.to_bus <= 0:
            raise ValueError("Transformer terminal IDs must be positive bus IDs")
        if self.from_bus == self.to_bus:
            raise ValueError("Transformer cannot connect a bus to itself")
        if (
            isinstance(self.tap_ratio, bool)
            or not isinstance(self.tap_ratio, (int, float))
            or not isfinite(self.tap_ratio)
            or self.tap_ratio <= 0
        ):
            raise ValueError("tap_ratio must be a finite positive number")
        if (
            not isinstance(self.vector_group_clock, int)
            or isinstance(self.vector_group_clock, bool)
            or not 0 <= self.vector_group_clock <= 12
        ):
            raise ValueError("vector_group_clock must be an integer from 0 to 12")
        if self.vector_group_clock != 0 and (
            self.from_connection is None or self.to_connection is None
        ):
            raise ValueError(
                "Nonzero vector_group_clock requires both winding connections"
            )
        if self.zero_sequence_mode not in {"blocked", "through", "grounded_wye_delta"}:
            raise ValueError(
                "zero_sequence_mode must be 'blocked', 'through', or 'grounded_wye_delta'"
            )
        if self.zero_sequence_mode != "blocked" and self.zero_sequence_impedance is None:
            raise ValueError("Transformer zero-sequence impedance is required for this mode")
        valid_connections = {"grounded_wye", "ungrounded_wye", "delta"}
        for label, connection in (
            ("from_connection", self.from_connection),
            ("to_connection", self.to_connection),
        ):
            if connection is not None and connection not in valid_connections:
                raise ValueError(
                    f"{label} must be 'grounded_wye', 'ungrounded_wye', or 'delta'"
                )
        if (self.from_connection is None) != (self.to_connection is None):
            raise ValueError("Both transformer winding connections must be specified together")
        if self.vector_group_clock != 0 and self.from_connection is not None:
            from_is_delta = self.from_connection == "delta"
            to_is_delta = self.to_connection == "delta"
            mixed_connections = from_is_delta != to_is_delta
            if (self.vector_group_clock % 2 == 1) != mixed_connections:
                raise ValueError(
                    "vector_group_clock parity is inconsistent with winding connections"
                )
        if self.from_neutral_impedance.real < 0 or self.to_neutral_impedance.real < 0:
            raise ValueError("Neutral impedance resistance cannot be negative")
        if self.from_neutral_impedance != 0j and self.from_connection != "grounded_wye":
            raise ValueError("from_neutral_impedance requires grounded_wye connection")
        if self.to_neutral_impedance != 0j and self.to_connection != "grounded_wye":
            raise ValueError("to_neutral_impedance requires grounded_wye connection")


@dataclass(slots=True)
class Network:
    """Network with bus IDs independent of matrix row ordering."""

    buses: list[Bus] = field(default_factory=list)
    branches: list[Branch] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    transformers: list[Transformer] = field(default_factory=list)
    shunts: list[Shunt] = field(default_factory=list)

    def add_bus(self, bus: Bus) -> None:
        if any(existing.id == bus.id for existing in self.buses):
            raise ValueError(f"Duplicate bus id: {bus.id}")
        self.buses.append(bus)

    def add_branch(self, branch: Branch) -> None:
        known_ids = {bus.id for bus in self.buses}
        for terminal in (branch.from_bus, branch.to_bus):
            if terminal != 0 and terminal not in known_ids:
                raise ValueError(f"Branch references unknown bus id: {terminal}")
        self.branches.append(branch)

    def add_source(self, source: Source) -> None:
        if source.bus_id not in {bus.id for bus in self.buses}:
            raise ValueError(f"Source references unknown bus id: {source.bus_id}")
        self.sources.append(source)

    def add_transformer(self, transformer: Transformer) -> None:
        known_ids = {bus.id for bus in self.buses}
        for terminal in (transformer.from_bus, transformer.to_bus):
            if terminal not in known_ids:
                raise ValueError(f"Transformer references unknown bus id: {terminal}")
        self.transformers.append(transformer)

    def add_shunt(self, shunt: Shunt) -> None:
        if shunt.bus_id not in {bus.id for bus in self.buses}:
            raise ValueError(f"Shunt references unknown bus id: {shunt.bus_id}")
        self.shunts.append(shunt)

    @property
    def bus_index(self) -> dict[int, int]:
        """Map bus IDs to zero-based Y-bus matrix indices."""
        return {bus.id: index for index, bus in enumerate(self.buses)}
