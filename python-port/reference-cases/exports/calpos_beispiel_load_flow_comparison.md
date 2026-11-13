# Calpos Beispiel load-flow comparison

The case was reconstructed from `beispiel.sld` and the Elplek 0.995 manual.
The model uses a 100 MVA base, 220 kV at B1, 65 kV at B2-B9, and the motor / generator
nameplate voltages at B10-B13. F1 is the 220 kV slack bus, G1 is a PV bus at
40 MW and 8.747 kV, and Ld1 is a 2.829 MVA, 0.707 power-factor PQ load.

The line connections are TL1: B2-B3, TL2/TL3: B2-B4, TL4: B2-B5,
TL5: B3-B9, TL6: B4-B8, TL7: B3-B8, TL8: B5-B6, TL9: B8-B7, and
TL10: B6-B7. T1 connects B1-B2; T2 B6-B13; T3 B9-B10; T4 B8-B11;
and T5 B8-B12. Line charging uses the SLD susceptance value in S/km and
the nominal-pi split. Transformer R/X values are referred to the primary side.

## Comparison

Voltage magnitudes and angles below are Elplek Vab results and Python
positive-sequence bus voltages converted to Elplek's displayed angle convention.

| Point | Elplek (kV) | Python (kV) | Elplek angle | Python angle |
|---|---:|---:|---:|---:|
| B7 / Ld1 | 65.6875 | 65.6249 | 32.50° | 31.27° |
| B10 / Asm1 | 5.1022 | 5.1127 | -3.50° | -3.49° |
| B11 / Asm2 | 5.0336 | 5.0426 | -0.20° | -0.18° |
| B12 / Asm3 | 5.5182 | 5.5285 | -2.50° | -2.49° |
| B13 / G1 | 8.7470 | 8.7470 | 7.10° | 5.71° |

The solved Python case converges in six Newton-Raphson iterations with a
maximum mismatch of approximately `6.9e-13 pu`. The motor impedances are
calculated directly from the SLD nameplate parameters using the ported Elplek
Delphi routines; no measured motor terminal output is used as an input.

| Motor | Elplek P (MW) | Python P (MW) | Elplek Q (MVAr) | Python Q (MVAr) |
|---|---:|---:|---:|---:|
| Asm1 | 5.1189 | 5.1421 | 3.8434 | 3.8565 |
| Asm2 | 1.9578 | 1.9658 | 3.1090 | 3.1194 |
| Asm3 | 5.5919 | 5.6113 | 3.5625 | 3.5781 |

The slack and generator powers are:

| Component | Elplek P (MW) | Python P (MW) | Elplek Q (MVAr) | Python Q (MVAr) |
|---|---:|---:|---:|---:|
| F1 | -24.7858 | -24.7401 | 8.5344 | 6.1544 |
| G1 | 39.9994 | 40.0000 | 10.4937 | 11.1887 |

Motor powers agree within 0.42%, and terminal voltage magnitudes agree within
0.21%. The largest listed angle difference is 1.39°. F1 and G1 reactive power
differ by about 2.38 MVAr and 0.70 MVAr, respectively.

## Scope and known differences

The steady-state motor equivalent ports Elplek `AsMotPar`, `SolveAll`,
`IstartAndParam`, and `AsMotLoad` from `elpleksource.zip`'s `elplex.pas`. It
currently handles positive-power asynchronous motors; induction-generator
operation and dynamic motor behavior are not included in this routine.

Elplek reported that T1-T5 had very small magnetizing current and substituted
default values. The Python transformer model currently omits magnetizing
branches, so the remaining source reactive-power differences are recorded
rather than hidden by fitting the results. The YD phase flag is represented by
clock position 1 for this particular terminal ordering because that reproduces
the observed motor and generator angles. Test this sign mapping against other
transformer orientations before generalizing it into an SLD importer.

The executable comparison is
`python-port/reference-cases/calpos_beispiel_load_flow.py`. The relevant manual
sections are Load Flow (pp. 35-37), transmission lines (pp. 49-50), feeders and
motors (pp. 55-57), and transformers (pp. 59-60) in `elplek.pdf`.
