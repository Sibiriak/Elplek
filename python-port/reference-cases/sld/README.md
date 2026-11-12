# Elplek SLD Reference Cases

Four variants based on `faultest.sld`, included with Elplek 0.995. Each file
keeps the original diagram and changes the network title and the electrical
parameter(s) named below.

| File | Changed electrical values | Purpose |
|---|---|---|
| `earth_fault_base.sld` | Z1: R = 2.85 ohm, X = 8.15 ohm | Baseline case |
| `earth_fault_z2.sld` | Z1: R = 5.70 ohm, X = 16.30 ohm | Double Z1 R and X |
| `earth_fault_r1.sld` | Z1: R = 1.00 ohm, X = 8.15 ohm | Isolate the effect of Z1 resistance |
| `earth_fault_t1_z2.sld` | T1 R/X and R0/X0 doubled | Change source-to-fault impedance |

## Baseline and Z1-doubled exports

Elplek exported `exports/earth_fault_base.txt` and `exports/earth_fault_z2.txt`.
Both identify the calculation as **Line to earth fault** at **common point of
Z1 CT1**. Each TXT contains 28 samples for `CT1` and 28 for `CT7`, from 0 to
11.032 s. The corresponding CSV files are normalized versions; the original
TXT exports are preserved.

The columns include phase-current magnitudes and angles, residual current,
positive-sequence current (`I1`), and negative-sequence current (`3I2`). Dynamic
superposition mode was selected (confirmed during the runs); IEC 60909 was not
selected. These exports are time-domain results.

The lower-resistance case is exported to `exports/earth_fault_r1.txt` and
normalized to `exports/earth_fault_r1.csv`. Its header confirms **Line to
earth fault** at the same point, with 28 samples each for CT1 and CT7. An
earlier three-phase export was replaced after its header and balanced currents
revealed the wrong fault type. The separate line-to-line and line-to-line-to-
earth exports use the baseline network and have the same CT/sample counts.
The three-phase case is saved as `exports/fault_three_phase.txt` and
`exports/fault_three_phase.csv`.

| Measurement at t = 11.032 s | Baseline | Z1 doubled | Change |
|---|---:|---:|---:|
| CT1 `Ia` | 572.2 A | 574.3 A | +0.37% |
| CT1 `I1` | 300.4 A | 299.5 A | -0.30% |
| CT7 `Ia` | 6295 A | 6318 A | +0.37% |
| CT7 `I1` | 3305 A | 3294 A | -0.33% |

The fault is at the node shared by `CT1` and `Z1`; `Z1` is on a branch leaving
the fault node, rather than in series between the source and the fault. The
small change is consistent with this topology. These cases show the influence
of the `Z1` branch, but are not a strong test of the source Thevenin impedance.

## Source-path impedance case

`earth_fault_t1_z2.sld` doubles T1 series resistance and reactance and its
zero-sequence resistance and reactance (R/X from 0.015/0.0735 ohm to
0.030/0.147 ohm). T1 is upstream of the fault. The Elplek export is saved as
`exports/earth_fault_t1_z2.txt`; its normalized CSV is
`exports/earth_fault_t1_z2.csv`. The export has the same fault type, location,
and 28-sample time grid as the baseline.

| Measurement at t = 11.032 s | Baseline | T1 R/X doubled | Change |
|---|---:|---:|---:|
| CT1 `Ia` | 572.2 A | 515.7 A | -9.9% |
| CT1 `I1` | 300.4 A | 279.2 A | -7.1% |
| CT1 `3I2` | 739.8 A | 634.5 A | -14.2% |
| CT7 `Ia` | 6295 A | 5673 A | -9.9% |
| CT7 `I1` | 3305 A | 3071 A | -7.1% |
| CT7 `3I2` | 8137 A | 6979 A | -14.2% |

The measured decrease confirms that changing T1 affects the source-to-fault
path. Elplek exported dynamic time series, while the current Python sequence
solver calculates steady-state phasors. The comparison below uses the final
sample after matching the generator and transformer sequence models; the full
transient curve remains outside the current Python solver scope.

## Sequence Thevenin impedance comparison

The Elplek `Calculate sequence impedances` report was compared with the
Python sequence network. The match requires the complete path represented in
the SLD: G1's sequence impedances are reflected through T1's 11:1 tap
(impedance factor 121), and the shunt branch at the fault node includes the
series combination `Z1 + Ld1`. The load has no zero-sequence return path.

Run `python-port/reference-cases/compare_earth_fault_thevenin.py` from the
repository root to reproduce the comparison. It checks the normal and
transient positive-sequence equivalents and subtransient `Z1`, `Z2`, and `Z0`
for both cases, with a 0.02 ohm tolerance for Elplek report rounding.

| Case | Elplek quantity | Elplek (ohm) | Python model (ohm) |
|---|---|---:|---:|
| Baseline | Normal `ZT` | 69.394 + j88.997 | 69.4057 + j88.9994 |
| Baseline | Transient `ZT` | 10.816 + j39.870 | 10.8171 + j39.8712 |
| Baseline | Subtransient `Z1` = `Z2` | 8.0725 + j33.343 | 8.0730 + j33.3440 |
| Baseline | Subtransient `Z0` | 2.3474 + j8.8935 | 2.3474 + j8.8935 |
| T1 doubled | Normal `ZT` | 70.665 + j89.070 | 70.6764 + j89.0730 |
| T1 doubled | Transient `ZT` | 14.389 + j44.444 | 14.3894 + j44.4448 |
| T1 doubled | Subtransient `Z1` = `Z2` | 11.550 + j38.567 | 11.5511 + j38.5676 |
| T1 doubled | Subtransient `Z0` | 4.1623 + j17.787 | 4.1624 + j17.7870 |

This validates the sequence-network topology and impedance conversion for this
reference case. The baseline dynamic current waveform is validated separately
against the raw CT1 phase and residual-current measurements.

## Settled earth-fault residual current

`python-port/reference-cases/compare_earth_fault.py` calculates the fault with
G1's normal positive-sequence reactance and the complete `Z1 + Ld1` shunt
branch, then compares the fault residual current with CT1's final exported
sample. Run it from the repository root after installing the root Python
package with `python -m pip install -e .`.

| Case | Python CT1 residual | Elplek CT1 residual | Difference |
|---|---:|---:|---:|
| Baseline | 563.5 A | 563.7 A | -0.03% |
| T1 R/X doubled | 500.6 A | 501.0 A | -0.08% |
| Z1 R/X doubled | 565.7 A | 565.9 A | -0.03% |
| Z1 resistance changed | 562.4 A | 562.6 A | -0.03% |

All four residuals agree within 0.08%. The changes from baseline also match:
doubling T1 R/X changes the residual by -11.17% in Python and -11.12% in
Elplek; doubling Z1 R/X changes it by +0.39% in both; reducing Z1 resistance
changes it by -0.20% in both. The R-only dynamic waveform comparison includes
all 28 samples and passes with maximum errors of 1.67 A and 0.24 degrees.

`python-port/reference-cases/compare_earth_fault_dynamic.py` validates all 28
samples at CT1 and CT7 for `Ia`, `Ib`, `Ic`, and (where applicable) `Isum` in
baseline, T1 R/X doubled,
Z1 R/X doubled, and Z1 resistance-only earth faults. It also validates the
three phase-current channels for all 28 samples of the A-B line-to-line fault,
all four phase/residual channels for the A-B-to-earth fault, and the three
phase-current channels for all 28 samples of the three-phase fault.
The EarthShort cases stay within 2.0 A and 0.24 degrees; the line-to-line case
stays within 1.30 A and 0.27 degrees (2.5 A and 0.30-degree limits); the
line-to-line-to-earth case stays within 3.32 A and 0.33 degrees (4 A and
0.35-degree limits); the three-phase case stays within 1.69 A and 0.33 degrees
(2.5 A and 0.35-degree limits). CT7 is checked at T1's 11:1 current ratio;
the worst CT7 error is 34.49 A and 0.33 degrees for line-to-line-to-earth
(44 A and 0.35-degree limits). The Z1 variants check that a fault-node shunt
change propagates through prefault rotor initialization and the fault network.
The two-phase cases exposed Elplek's A-B phase-pair convention, represented by
`line_to_line_ab` and `double_line_to_ground_ab` in the sequence solver. It
uses the
non-precalculated prefault current derived from the explicit linear-admittance
network and the normal generator impedance used by Elplek for the time-domain
EarthShort path. The exported `I1`/`3I2` values are not used as
symmetrical-component references because they do not reconstruct from the
exported phase currents. Multi-machine coupling remains outstanding.
