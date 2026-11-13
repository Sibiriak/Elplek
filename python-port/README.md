# Elplek Python Port

The Python implementation is an incremental numerical port alongside the
original Delphi application. It is not a drop-in replacement and has not been
validated for certified engineering use.

## Current scope

- Positive-sequence Y-bus construction for branches, nominal-pi line charging,
  shunts, and simplified transformer taps and phase displacement.
- Linear network and conventional short-circuit solvers.
- Symmetrical-component sequence networks and several asymmetrical fault
  types.
- A balanced Newton-Raphson load flow with one fixed-voltage slack bus, PQ
  buses, and PV buses with specified active power and voltage magnitude. Net
  bus power injections use per-unit values; loads are negative injections.
  Optional per-bus reactive limits switch a PV bus to PQ when reached.
  Individual generator capability curves and voltage controls are not
  implemented.
- A single-machine dynamic fault path and synchronous-machine parameter
  conversion, checked against saved Elplek reference exports.
- Elplek's steady-state asynchronous-machine equivalent impedance from motor
  nameplate data, including induction-generator operation at negative output
  power. The reconstructed `Calpos Beispiel` load-flow case compares motor
  equivalents with the saved terminal results.

## Development and validation

Install the package and test dependencies, then run the suite:

```powershell
python -m pip install -e ".[test]"
python -m pytest -v
```

The repository workflow runs the tests on Python 3.10 and 3.14. Reference-case
comparators and saved Elplek `.sld`/text exports are under
`reference-cases/`.

## Limitations

This port does not yet implement individual generator capability curves,
constant-power load coupling to short-circuit calculations, complete IEC
60909 corrections, dynamic induction-machine coupling, multiple-machine
dynamic coupling, protection relays, or the original graphical interface. The
transformer zero-sequence and magnetizing-branch models remain simplified.
Check the root README and `SOURCE_MAP.md` for feature-specific assumptions and
validation coverage.

## Attribution

The parent repository identifies Ilkka Leikkonen as the original author.
Preserve that attribution and verify the applicable license and third-party
component terms before redistribution.
