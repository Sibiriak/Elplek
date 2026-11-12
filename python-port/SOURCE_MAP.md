# Original Elplek source map

This map was prepared by inspecting the uploaded `elpleksource.zip` and the original Delphi units. It is a planning aid, not a claim that every dependency has been exhaustively traced.

## Application entry point and UI

- `Imagopro.dpr`: Delphi application entry point; registers forms and units.
- `Imagopas.pas`: main application form and orchestration. It imports the load-flow, short-circuit, IEC, node, component and diagram units.
- `Imagopas.dfm` and the other `.dfm` files: Delphi VCL form resources.
- `Drawcomp.pas`, `Skemadef.pas`, `Skemproc.pas`, `Nodes.pas`: diagram/component model, interactions and connectivity logic. These are tightly coupled to UI objects and global component state.

## Numerical calculation areas

- `LoadFlow.pas`: load-flow entry point `LoadFlowCalculation` and class `TLoadFlow`; the source imports `Nodes`, `Elplex`, `Compdef`, `Skemproc`, `GenVolt`, `Cmatmul` and `Matmul`.
- `IEC.pas`: IEC 60909 short-circuit entry points `IECShort`, `IECTwoPhase`, `IECEarthFault`, and `IECLineLineEarth`; depends on node topology, component data and multiple fault-analysis units.
- `Shortcir.pas`: dynamic/superposition short-circuit calculations, generator/motor models and relay/current-transformer calculations; very large and strongly coupled.
- `TWOSHORT.PAS`: phase-to-phase faults and symmetrical-component conversions. Exposes `PhaseToPhaseShort`, `PhaseToSeq`/related sequence transformations and fault-current routines.
- `earthflt.pas`: earth-fault calculations.
- `AllShort.pas`: orchestration/UI for calculations across buses and measurement points; not a good first numerical module to port.
- `GenProc.pas`: synchronous-machine parameter and equation helpers; imports `Shortcir`, so it is not yet a clean standalone numerical library.
- `Elplex.pas`, `Matmul.pas`, `Cmatmul.pas`, `Utildef.pas`, `Imagodef.pas`, `Compdef.pas`: shared types, complex/matrix utilities, component definitions and global application data. These dependencies need tracing before translating algorithms.

## Dynamic synchronous-generator path

The single-machine time-domain EarthShort path is now matched against every
exported baseline CT1 phase and residual-current sample. The relevant source
path is:

- `Shortcir.pas`: `ShortCircuit` creates the generator/motor state model;
  `Calc_dIdV` couples machine-current changes to network-voltage changes;
  `TGenMot.RotorDQEquations` integrates rotor winding currents with the
  trapezoidal method; and `TGenMot.TraFDQPVolt` advances the machine state and
  updates the network source terms at each time step.
- `Shortcir.pas`: `GetGenPar` reads stator resistance, d-axis reactances and
  time constants, plus transient/subtransient q-axis reactances and time
  constants. It assigns the internal `Rot_res=0.3` estimate to the rotor
  resistance field in its parameter record.
- `GenProc.pas`: `SolveGenPar`, `SolveInductances`, and `SolvePhysPar` convert
  the reactances and short-circuit time constants into the physical winding
  parameters used by the dynamic equations. `SolveGenPar` uses source defaults
  `Mdp/Mf=0.5`, `Rf=0.3`, `Rd=0.1`, `Rq=0.2`, and `Rp=0.1`; these winding
  resistances are not SLD inputs.
- `earthflt.pas`: the fault is solved through the positive-, negative-, and
  zero-sequence networks, then the resulting sequence currents are converted
  and combined for phase and residual-current outputs.

The `faultest.sld` generator supplies `Rs=0.0044`, `Xd=2.39`,
`Xd'=0.3445`, `Xd''=0.26`, `Td'=0.8433 s`, `Td''=0.0251 s`, `X2=0.26`,
`X0=0`, `Xq'=0.534`, `Xq''=0.26`, `Tq'=0.123 s`, and `Tq''=0.02 s`.
`src/elplek/synchronous_machine.py` now ports the conversion from these
reactances and time constants to rotor inductances and resistance assumptions,
provides a trapezoidal update for the four rotor winding currents, and includes
a Newton-coupled one-machine step with a network-current solver.
`src/elplek/dynamic_fault.py` adapts a one-source `Network` to that hook by
solving the fault in all sequence networks, calculating their linear `dI/dV`
sensitivity once, and reusing it during rotor iterations. The source positive-
sequence current is projected into Elplek D/Q coordinates. When the callback
provides that linear sensitivity, the dynamic step uses Elplek's direct coupled
rotor/network matrix; arbitrary callbacks can use the numerical 2x2 Newton
fallback. This remains a one-machine port, not Elplek's full multi-machine
`dIdV` matrix.
`initialize_rotor_state`
now ports Elplek `InitialRotorCurrents` when
the initial D/Q voltage corrections are provided. The one-machine linear
equations from `ConstrVeq` are now solved by
`solve_initial_rotor_voltage_corrections`; it estimates the network current
sensitivity by a centered voltage perturbation and uses the generator's
positive-sequence stator current, matching `ConstrVeq`. The phase-A sum of all
three sequences is retained for CT comparisons. `initialize_dynamic_fault_rotor_state`
can derive that prefault current from the explicit linear admittances in a
one-source `Network`, or accept a load-flow current for constant-power loads.
`simulate_coupled_rotor_steps` now advances a supplied initial state over
nonuniform intervals and holds the prefault field excitation. The script
`python-port/reference-cases/compare_earth_fault_dynamic.py` compares all 28
CT1 samples for phase currents `Ia`, `Ib`, `Ic` and residual current `Isum` in
the baseline, T1 R/X doubled, Z1 R/X doubled, and Z1 resistance-only cases;
it also compares CT1/CT7 phase currents for A-B line-to-line and three-phase
faults, and phase and residual currents for an A-B-to-earth fault. At CT1,
EarthShort errors stay below 2.0 A and 0.24 degrees; line-to-line errors stay
below 1.30 A and 0.27 degrees; line-to-line-to-earth errors stay below 3.32 A
and 0.33 degrees; three-phase errors stay below 1.69 A and 0.33 degrees. CT7
is compared using T1's 11:1 current ratio, with maximum error below 34.5 A and
0.33 degrees. A separate steady-state residual comparison covers all four
EarthShort cases and matches within 0.08%. For this
EarthShort path, Elplek uses the normal
generator impedance Xd for both the initial fault network and later steps;
the separately printed subtransient Thevenin values are not the impedance
selected by that path. The prefault current is calculated from the explicit
linear-admittance network when no precomputed load-flow state is supplied.
Multiple-machine coupling, asynchronous motors, and constant-power prefault
loads remain out of scope. The
exported `I1`/`3I2` columns are not treated as symmetrical components because
they do not reconstruct from Elplek's exported phase currents.

## Recommended migration order

1. **Keep this scaffold separate from the Delphi program.** Do not overwrite or rename original files.
2. **Inventory core types and units:** identify complex-number representation, matrix indexing, node numbering, component kinds, sign conventions, units and global state in `Utildef`, `Imagodef`, `Compdef`, `Elplex`, `Matmul` and `Cmatmul`.
3. **Build reference fixtures:** small networks with known values and matching outputs from the original executable. Record voltages, currents, fault type, source parameters and calculation mode.
4. **Port a narrow calculation path:** start with the smallest independent matrix/impedance routine, then expand toward load flow and IEC 60909.
5. **Implement SLD import only after documenting the binary/text format** and making sample files available as tests.
6. **Rebuild the GUI last** with a separate diagram model and PySide6/Qt Graphics View or another graph-editor framework.

## Important engineering distinction

The current `solver.py` is only a linear Y-bus voltage solver with one fixed slack voltage and constant-admittance shunts. It is **not** a Newton-Raphson load-flow solver and does not model constant-power loads, generator limits, transformer tap control, sequence networks, IEC 60909 corrections or protection relays. Its tests have been added but have not been executed in this environment. Numerical equivalence must be established against reference cases before use.

## License and attribution

The repository README names Ilkka Leikkonen as the original author and asks derivative users to credit the original source. Keep that attribution. Check the repository's LICENSE and third-party component licenses before publishing a derivative.
