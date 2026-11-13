# Elplek
Saved copy of Elplek v. 0.995  (short circuit and load flow, freeware).

This is not an official repository of the program and this account is not affiliated with the original author of Elplek software. The goal of this repository is to preserve a unique freeware program to download and for possible further development of the program as the source code was available to download on the official site after the author had finished its development. If you use any parts of Elplek source code please provide links to the original author:  Ilkka Leikkonen, Kokkola, Finland.

This program is written in Delphi 2 and with minor editions can be compiled in Delphi 3. If you have any ideas how to upgrade the source code to Lazarus or Delphi of recent versions, feel free to share your suggestions here.

Read the preserved original description of the program in the Wiki: https://github.com/Sibiriak/Elplek/wiki



## Incremental Python port (`python-port`)

The Python implementation is an incremental port and is not yet a drop-in
replacement for the original Delphi program. Current transformer models support
positive-, negative- and simplified zero-sequence equivalents, winding
connection labels, neutral impedances, and an optional vector-group clock
number.

Example:

```python
from elplek.model import Network, Bus, Shunt, Transformer
from elplek.ybus import build_ybus, build_sequence_ybus

network = Network()
network.add_bus(Bus(1, "HV"))
network.add_bus(Bus(2, "LV"))
network.add_transformer(Transformer(
    from_bus=1,
    to_bus=2,
    positive_sequence_impedance=0.01 + 0.08j,
    negative_sequence_impedance=0.01 + 0.08j,
    zero_sequence_impedance=0.02 + 0.10j,
    from_connection="grounded_wye",
    to_connection="delta",
    vector_group_clock=11,
    tap_ratio=1.05,  # ideal magnitude ratio on the from_bus side
))
network.add_shunt(Shunt(
    bus_id=2,
    admittance=0.0 + 0.04j,
    negative_sequence_admittance=0.0 + 0.04j,
    zero_sequence_admittance=0.0 + 0.0j,
    name="LV capacitor bank",
))
y_positive = build_ybus(network)
y_negative = build_sequence_ybus(network, "negative")
y_zero = build_sequence_ybus(network, "zero")
```

The clock number describes the phase displacement of the `to_bus` winding
relative to `from_bus`: clock 11 means the to-side positive-sequence voltage
leads by 30 degrees; clock 1 means it lags by 30 degrees. Clock 0 is the legacy
no-shift default, while clock 12 is the explicit zero-displacement clock
position. Negative-sequence phase displacement is reversed. Zero-sequence
network behavior depends on the configured winding connections and neutral
impedances; its simplified model does not encode all transformer core designs.

Tap magnitude ratios are supported as ideal complex taps on the from_bus side.
Their positive- and negative-sequence Y-bus stamps are included; zero-sequence
tap treatment follows the simplified connection model. Magnetizing branches,
complete vector-group/terminal conventions, full transformer grounding details,
transformer magnetizing-current effects in load flow, and complete IEC 60909
corrections remain outside the current implementation.
The port is still under validation and should not yet be used as a certified
replacement for the original program or an independently validated
short-circuit study package.


### Bus shunts

The `Shunt` model stamps a complex admittance directly onto the selected
bus diagonal. Use admittance values in the same per-unit or physical-unit
system as the rest of the network. Positive-sequence admittance is required;
negative-sequence admittance defaults to positive sequence if omitted. For
zero-sequence assembly, provide `zero_sequence_admittance` explicitly rather
than assuming that the positive-sequence value is valid. Shunts are included
in both the conventional `build_ybus` and `build_sequence_ybus` matrices.


### Conventional short-circuit solver

`solve_short_circuit(network, fault_bus_id, fault_impedance=...)` uses the
conventional Y-bus with source equivalents and a single fault-current unknown.
Consequently, transformer positive-sequence impedance, ideal magnitude taps,
configured clock displacement, and bus shunts affect its result. This API
remains a single-phase-equivalent solver; use `solve_network_fault` for the
separate, simplified symmetrical-component fault path. Neither API is yet a
certified implementation of IEC 60909.


### Newton-Raphson load flow

`solve_load_flow(network, slack_bus_id=..., specified_power_injections=...)`
solves a balanced positive-sequence AC load flow with one fixed-voltage slack
bus, PQ buses, and PV buses with specified active power and voltage magnitude.
Specify net complex power injections in per unit; loads use negative real and
reactive power. The slack-bus power and PV reactive injections are calculated
from the solved network. Optional per-bus reactive limits convert a PV bus to a
PQ bus when its limit is reached. Individual generator capability curves and
voltage controls are not implemented. This solver does not convert
constant-power loads into impedances for short-circuit calculations.

### Asynchronous-motor steady-state equivalent

`solve_asynchronous_motor_equivalent(...)` ports Elplek's positive-power
induction-motor equivalent calculation from its nameplate data. It returns the
per-phase steady-state impedance in ohms, which can be converted to a shunt
admittance for load-flow modeling. The `Calpos Beispiel` reference case checks
this calculation against saved Elplek terminal voltages and powers. Induction
generators and dynamic induction-machine coupling are not implemented.


### Nominal-pi line model

A `Branch` can include its total positive-sequence line charging admittance
using `shunt_admittance`. The nominal-pi model stamps half of that admittance
at each terminal bus. When the branch connects to ground, only the half at the
non-ground terminal contributes to the reduced Y-bus.

```python
from elplek.model import Branch

network.add_branch(Branch(
    from_bus=1,
    to_bus=2,
    resistance=0.02,
    reactance=0.15,
    shunt_admittance=0.0 + 0.04j,  # total Yc for the line
    negative_sequence_shunt_admittance=0.0 + 0.04j,
    zero_sequence_impedance=0.06 + 0.30j,
    zero_sequence_shunt_admittance=0.0 + 0.01j,
    name="line 1-2",
))
```

Negative-sequence line charging falls back to positive sequence when omitted.
If positive-sequence line charging is nonzero, explicitly provide
`zero_sequence_shunt_admittance` before constructing the zero-sequence
matrix. Branch impedance and admittance values must use consistent units or
the same per-unit base as the rest of the network.

### Synchronous-machine parameter conversion

`solve_synchronous_machine_parameters` converts d- and q-axis reactances and
short-circuit time constants into the four rotor-winding inductance sets used
by the Elplek dynamic model. It follows Elplek’s `SolveGenPar` resistance and
mutual-inductance assumptions. This module provides a four-winding rotor-state
trapezoidal step, a one-machine Newton-coupled network step, and a generic
multi-step driver. `make_dynamic_fault_current_solver` adapts a
single-source sequence network to that hook: it solves the fault using all
three sequences and returns the source positive-sequence current in D/Q
coordinates. `initialize_rotor_state` ports Elplek's initial rotor-current
calculation when the initial D/Q voltage corrections are known. Solving those
corrections for a single machine is available with
`solve_initial_rotor_voltage_corrections`, together with
`make_dynamic_fault_positive_sequence_current_solver`. The prefault machine
current must use the same positive-sequence convention, matching Elplek
`ConstrVeq`; phase-A current is used separately for CT comparisons.
`initialize_dynamic_fault_rotor_state` combines these steps and derives the
prefault current from explicit network admittances when it is not provided;
this does not replace a power-flow solution for constant-power loads.
`simulate_coupled_rotor_steps` advances a supplied initial state over
nonuniform time intervals. The single-machine dynamic fault path is compared
with all 28 exported CT1 samples for `Ia`, `Ib`, `Ic`, and `Isum` in the
baseline, T1 R/X doubled, Z1 R/X doubled, and Z1 resistance-only earth faults,
the three phase-current channels for an A-B line-to-line fault, and all phase
and residual channels for an A-B-to-earth fault, plus three phase-current
channels for a three-phase fault, by
`python-port/reference-cases/compare_earth_fault_dynamic.py`. EarthShort
errors stay below 2.0 A and 0.24 degrees at CT1; the line-to-line case stays
below 1.30 A and 0.27 degrees, and the line-to-line-to-earth case below 3.32 A and
0.33 degrees; the three-phase case stays below 1.69 A and 0.33 degrees. The
same cases are compared at CT7 after applying T1's 11:1 current ratio; the
largest CT7 error is below 34.5 A and 0.33 degrees. The
exported `I1` and
`3I2` columns are not used because they do not equal the symmetrical components
reconstructed from the exported phase currents. Multiple-machine coupling
remains in development.
