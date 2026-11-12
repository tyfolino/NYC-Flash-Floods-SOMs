# D2 stratification (preview on the published SOMs)

Events: 118 warm-season FF events; Stage IV coverage n = 101; ASOS n = 117. Window (t0-6h, t0+6h], hour-ending. Area = land/coastal cells of 38.5-44N, 78-70W (open ocean masked). D3 anomalies: month-day climatology (as D3); Z500 in the trough_T0 box (43.0, 36.0, -86.5, -79.0).

Stage IV NYC peak vs D4 extract: max |diff| = 0 mm/h.

## Threshold dependence of the rain-area classification

- Median area (km^2): {10: 31383.629893612488, 25: 2441.0038790337467, 50: 35.22033890723016}; events with zero area: {10: 0, 25: 2, 50: 39}
- Spearman between thresholds: {'10_vs_25': 0.7448420059495936, '10_vs_50': 0.292452646951426, '25_vs_50': 0.6432441150243368}
- Median-split agreement (%): {'10_vs_25': 76.23762376237624, '10_vs_50': 52.475247524752476, '25_vs_50': 72.27722772277228}; Cohen's kappa: {'10_vs_25': 0.52, '10_vs_50': 0.05, '25_vs_50': 0.45}
- Spearman(area, Stage IV NYC peak): {10: 0.13073916827509582, 25: 0.36627109538894187, 50: 0.2831113199093608}

## Node mix and D3 metrics by group

| Split | n (low/high) | evSOM p | static p | all-days p | A4+A5 % (high vs low) | θe T0 med (high/low), p | TCWV T0 p | Z500 T0 med (high/low), p |
|---|---|---|---|---|---|---|---|---|
| area10_median | 51/50 | 0.01 | 0.19 | 0.61 | 42 vs 25 | 1.48/0.87, 0.00 | 0.00 | -0.60/-0.41, 0.78 |
| area10_tercile | 34/34 | 0.02 | 0.08 | 0.54 | 50 vs 21 | 1.59/0.55, 0.00 | 0.00 | -0.85/-0.40, 0.19 |
| area25_median | 51/50 | 0.03 | 0.56 | 0.78 | 40 vs 27 | 1.50/0.96, 0.00 | 0.00 | -0.53/-0.53, 0.85 |
| area25_tercile | 34/34 | 0.05 | 0.33 | 0.75 | 41 vs 26 | 1.65/0.98, 0.00 | 0.00 | -0.44/-0.47, 0.87 |
| area50_median | 51/50 | 0.88 | 0.85 | 0.12 | 32 vs 35 | 1.27/1.06, 0.49 | 0.24 | -0.50/-0.56, 0.66 |
| area50_tercile | 39/34 | 0.70 | 0.93 | 0.20 | 26 vs 38 | 1.39/1.02, 0.16 | 0.13 | -0.18/-0.41, 0.42 |
| s4_rate_median | 53/48 | 0.58 | 0.48 | 0.76 | 27 vs 40 | 1.09/1.17, 0.23 | 0.42 | -0.47/-0.54, 0.57 |
| s4_rate_1in | 22/79 | 0.85 | 0.64 | 0.18 | 30 vs 45 | 1.17/1.02, 0.05 | 0.02 | -0.53/-0.85, 0.19 |
| asos_rate_median | 59/58 | 0.13 | 0.35 | 0.88 | 34 vs 29 | 1.35/0.97, 0.00 | 0.00 | -0.35/-0.69, 0.13 |
| asos_rate_1in | 66/51 | 0.12 | 0.45 | 1.00 | 35 vs 29 | 1.36/0.98, 0.00 | 0.00 | -0.36/-0.65, 0.29 |

Rules: area10_median: area(>= 10 mm/h) > median (31384 km^2); area10_tercile: area(>= 10 mm/h): bottom tercile (<= 19833) vs top (> 39369 km^2); area25_median: area(>= 25 mm/h) > median (2441 km^2); area25_tercile: area(>= 25 mm/h): bottom tercile (<= 1356) vs top (> 3819 km^2); area50_median: area(>= 50 mm/h) > median (35 km^2); area50_tercile: area(>= 50 mm/h): bottom tercile (<= 0) vs top (> 73 km^2); s4_rate_median: S4 NYC peak > median (34.0 mm/h); s4_rate_1in: S4 NYC peak >= 25.4 mm/h; asos_rate_median: ASOS NYC peak > median (23.1 mm/h); asos_rate_1in: ASOS NYC peak >= 25.4 mm/h

## Moisture nearly always, troughing sometimes? (T0, % of group)

| Split | group | θe>0 | TCWV>0 | Z500<0 | moist+trough | moist, no trough | dry+trough | dry, no trough |
|---|---|---|---|---|---|---|---|---|
| area10_median | widespread | 98 | 98 | 68 | 66 | 32 | 2 | 0 |
| area10_median | localized | 84 | 88 | 69 | 55 | 29 | 14 | 2 |
| area10_tercile | widespread | 100 | 100 | 76 | 76 | 24 | 0 | 0 |
| area10_tercile | localized | 79 | 85 | 74 | 56 | 24 | 18 | 3 |
| area25_median | widespread | 98 | 98 | 70 | 68 | 30 | 2 | 0 |
| area25_median | localized | 84 | 88 | 67 | 53 | 31 | 14 | 2 |
| area25_tercile | widespread | 97 | 97 | 68 | 65 | 32 | 3 | 0 |
| area25_tercile | localized | 85 | 88 | 68 | 53 | 32 | 15 | 0 |
| area50_median | widespread | 94 | 96 | 66 | 62 | 32 | 4 | 2 |
| area50_median | localized | 88 | 90 | 71 | 59 | 29 | 12 | 0 |
| area50_tercile | widespread | 91 | 94 | 59 | 53 | 38 | 6 | 3 |
| area50_tercile | localized | 90 | 87 | 67 | 56 | 33 | 10 | 0 |
| s4_rate_median | heavy | 96 | 96 | 67 | 62 | 33 | 4 | 0 |
| s4_rate_median | modest | 87 | 91 | 70 | 58 | 28 | 11 | 2 |
| s4_rate_1in | heavy | 96 | 97 | 66 | 62 | 34 | 4 | 0 |
| s4_rate_1in | modest | 73 | 77 | 77 | 55 | 18 | 23 | 5 |
| asos_rate_median | heavy | 97 | 97 | 66 | 62 | 34 | 3 | 0 |
| asos_rate_median | modest | 85 | 90 | 73 | 59 | 25 | 14 | 2 |
| asos_rate_1in | heavy | 96 | 96 | 69 | 65 | 31 | 4 | 0 |
| asos_rate_1in | modest | 86 | 91 | 70 | 58 | 29 | 12 | 2 |

## evSOM node mix (counts)

- area25_median: {'localized': {'A1': 20, 'A2': 9, 'B1': 6, 'B2': 16}, 'widespread': {'A1': 15, 'A2': 2, 'B1': 14, 'B2': 19}}
- s4_rate_median: {'heavy': {'A1': 15, 'A2': 7, 'B1': 8, 'B2': 18}, 'modest': {'A1': 20, 'A2': 4, 'B1': 12, 'B2': 17}}
- asos_rate_median: {'heavy': {'A1': 16, 'A2': 6, 'B1': 15, 'B2': 21}, 'modest': {'A1': 26, 'A2': 8, 'B1': 7, 'B2': 18}}
