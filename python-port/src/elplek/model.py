"""Minimal AC network model for the Python port scaffold."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

@dataclass(frozen=True, slots=True)
class Bus:
    """Electrical bus; slack voltage is phase-to-neutral RMS as a complex phasor."""
    name: str
    slack_voltage: complex | None = None

    @property
    def is_slack(self) -> bool:
        return self.slack_voltage is not None

@dataclass(frozen=True, slots=True)
class Branch:
    """Series branch; impedance in ohms and terminal shunts in siemens."""
    name: str
    from_bus: str
    to_bus: str
    impedance_ohm: complex
    shunt_from_s: complex = 0j
    shunt_to_s: complex = 0j

    def __post_init__(self) -> None:
        if self.from_bus == self.to_bus:
            raise ValueError("A branch must connect two different buses")
        if self.impedance_ohm == 0:
            raise ValueError("Branch impedance must be non-zero")

@dataclass(slots=True)
class Network:
    """Network data for nodal-admittance analysis."""
    buses: list[Bus] = field(default_factory=list)
    branches: list[Branch] = field(default_factory=list)
    shunts_s: dict[str, complex] = field(default_factory=dict)

    def __init__(self, buses: Iterable[Bus] = (), branches: Iterable[Branch] = (), shunts_s: dict[str, complex] | None = None) -> None:
        self.buses = list(buses)
        self.branches = list(branches)
        self.shunts_s = dict(shunts_s or {})
        self.validate()

    def validate(self) -> None:
        names = [bus.name for bus in self.buses]
        if len(names) != len(set(names)):
            raise ValueError("Bus names must be unique")
        if not names:
            raise ValueError("Network must contain at least one bus")
        if sum(bus.is_slack for bus in self.buses) != 1:
            raise ValueError("Exactly one slack bus is required")
        known = set(names)
        for branch in self.branches:
            if branch.from_bus not in known or branch.to_bus not in known:
                raise ValueError(f"Branch {branch.name!r} references an unknown bus")
        for name in self.shunts_s:
            if name not in known:
                raise ValueError(f"Shunt references unknown bus {name!r}")
