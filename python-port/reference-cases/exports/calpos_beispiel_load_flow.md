# Elplek load-flow reference: Calpos Beispiel

Transcribed from Elplek 0.995 result boxes after running Load Flow on
`beispiel.sld`. The application reported that T1-T5 had small magnetization
current and used its default values. This was a warning; the calculation
produced the results below.

## Reported component results

All values below are the values shown by Elplek. Voltage and current angles are
in degrees; `S` is apparent power in VA and its angle is in degrees.

| Component | Voltage (V) | V angle | Current (A) | I angle | S (VA) | S angle |
|---|---:|---:|---:|---:|---:|---:|
| F1 | V1 220,090 | 29.9 | 68.79 | -161.0 | 26,214,000 | 161.0 |
| F1 | V2 220,000 | 30.0 |  |  |  |  |
| G1 | V1 16,674.9 | 51.9 | 2,729.5 | -37.6 | 41,353,000 | 14.7 |
| G1 | V2 8,747.0 | 7.1 |  |  |  |  |
| Ld1 | 65,687.5 | 32.5 | 24.87 | -42.5 | 2,829,000 | 45.0 |
| Asm1 | 5,102.2 | -3.5 | 724.3 | -70.3 | 6,401,200 | 36.9 |
| Asm2 | 5,033.6 | -0.2 | 421.4 | -87.9 | 3,674,100 | 57.8 |
| Asm3 | 5,518.2 | -2.5 | 693.7 | -65.0 | 6,630,300 | 32.5 |

F1 also reports `Vn = 0` and `fiin = 0.0`.

## Derived complex powers

Using `P = |S| cos(angle)` and `Q = |S| sin(angle)`:

| Component | P (MW) | Q (MVAr) |
|---|---:|---:|
| F1 | -24.7858 | 8.5344 |
| G1 | 39.9994 | 10.4937 |
| Ld1 | 2.0004 | 2.0004 |
| Asm1 | 5.1189 | 3.8434 |
| Asm2 | 1.9578 | 3.1090 |
| Asm3 | 5.5919 | 3.5625 |
| Total PQ demand (Ld1, Asm1-Asm3) | 14.6691 | 12.5153 |

With generation positive and F1's displayed active power negative, the active
power balance leaves approximately 0.5445 MW for network losses and rounding:
`39.9994 - 24.7858 - 14.6691 = 0.5445 MW`.

This is a recorded Elplek operating point. A complete Python comparison still
requires reconstructing the example's branch topology, line parameters,
transformers, and asynchronous-machine equivalents.
