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
- A balanced Newton-Raphson load flow with one fixed-voltage slack bus and PQ
  buses. Net bus power injections use per-unit values; loads are negative
  injections. PV buses and generator voltage/reactive-power controls are not
  implemented.
- A single-machine dynamic fault path and synchronous-machine parameter
  conversion, checked against saved Elplek reference exports.

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

This port does not yet implement PV-bus controls, generator reactive limits,
constant-power load coupling to short-circuit calculations, complete IEC
60909 corrections, multiple-machine dynamic coupling, asynchronous motors,
protection relays, or the original graphical interface. The transformer
zero-sequence and magnetizing-branch models remain simplified. Check the root
README and `SOURCE_MAP.md` for feature-specific assumptions and validation
coverage.

## Attribution

The parent repository identifies Ilkka Leikkonen as the original author.
Preserve that attribution and verify the applicable license and third-party
component terms before redistribution.
