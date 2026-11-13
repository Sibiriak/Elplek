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

## Current Python load-flow path

`src/elplek/load_flow.py` implements a balanced positive-sequence Newton-Raphson
calculation over `build_ybus()`. It accepts net per-unit complex power
injections for PQ buses, accepts active power and a fixed voltage magnitude for
PV buses, fixes one slack bus voltage, and calculates slack power and PV
reactive injections from the resulting voltages. Optional per-bus reactive
limits switch a constrained PV bus to PQ at its limit. Its Jacobian is the
conventional polar-form P/Q Jacobian. Individual generator capability curves,
voltage controls, and load-flow conversion of constant-power loads for
short-circuit calculations are not implemented. Focused unit tests are in
`tests/test_load_flow.py`.

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
Multiple-machine coupling, asynchronous motors in the dynamic fault model, and
constant-power prefault loads remain out of scope. The
exported `I1`/`3I2` columns are not treated as symmetrical components because
they do not reconstruct from Elplek's exported phase currents.

## Next development stages

1. Extend asynchronous-machine support to induction-generator operation and
   couple induction-machine states to the dynamic fault path.
2. Extend load flow with individual generator capability curves and bus voltage
   controls, first using analytically solved fixtures and then Elplek reference
   cases.
3. Couple a converged prefault load-flow state to the dynamic fault model,
   especially for constant-power loads and multiple generators.
4. Continue the IEC 60909 path against matching Elplek calculations, keeping its
   correction factors and initial-condition conventions explicit.
5. Add SLD import only after documenting its format and testing it against the
   saved sample files.
6. Rebuild the GUI after the numerical models and file format are stable.

## Important engineering distinction

The Newton-Raphson load-flow implementation supports a fixed slack bus, PQ
buses, and PV buses with fixed voltage magnitude and specified active power.
Optional bus-level reactive limits convert PV buses to PQ. It does not support
generator capability curves or bus controls. The wider package remains an
incremental port and is not a certified replacement for Elplek or a validated
engineering study package. The test suite is run locally and in GitHub Actions;
feature-specific comparisons against the original program are documented with
their reference cases.

## License and attribution

The repository README names Ilkka Leikkonen as the original author and asks derivative users to credit the original source. Keep that attribution. Check the repository's LICENSE and third-party component licenses before publishing a derivative.
