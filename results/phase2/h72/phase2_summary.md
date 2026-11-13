# Phase 2 (D1): 72-h evSOM

## Seed baseline (ARI median [min–max])

- static_static: ARI 0.96 [0.88–1.00], agreement 0.98 [0.95–1.00]
- evsom_evsom: ARI 0.59 [0.39–1.00], agreement 0.80 [0.64–1.00]
- static_evsom: ARI 0.30 [0.22–0.38], agreement 0.60 [0.47–0.68]
- share of cross-method pairs above the within-evSOM 5th percentile: 0.00

Representative members: static seed 8 (agreement with published 0.86); evSOM seed 14 (agreement with published 24 h 0.57). evSOM node counts: {'A1': 35, 'B1': 30, 'A2': 24, 'B2': 29}.

## Reclassified events: 59

| static → evSOM | A1 | B1 | A2 | B2 |
|---|---|---|---|---|
| A1 | 22 | 16 | 0 | 1 |
| B1 | 2 | 7 | 2 | 6 |
| A2 | 9 | 2 | 8 | 0 |
| B2 | 2 | 5 | 14 | 22 |

Mean standardized anomalies (box means) by evSOM node:

- **A1**: Z500_Midwest T-71 -0.31, T-48 -0.43, T-24 -0.95, T-12 -1.03, T+0 -0.90; Z500_Plains T-71 -0.10, T-48 -0.31, T-24 -0.43, T-12 -0.48, T+0 -0.47; Z500_Canada T-71 +0.01, T-48 -0.15, T-24 -0.17, T-12 -0.14, T+0 -0.08; thetae_NYC T-71 -0.47, T-48 -0.46, T-24 -0.19, T-12 +0.04, T+0 +0.56
- **B1**: Z500_Midwest T-71 -0.32, T-48 -0.62, T-24 -1.04, T-12 -1.28, T+0 -1.07; Z500_Plains T-71 -0.69, T-48 -0.92, T-24 -1.03, T-12 -0.89, T+0 -0.62; Z500_Canada T-71 -0.21, T-48 -0.61, T-24 -0.69, T-12 -0.68, T+0 -0.61; thetae_NYC T-71 +0.43, T-48 +0.57, T-24 +0.81, T-12 +0.84, T+0 +1.18
- **A2**: Z500_Midwest T-71 +0.72, T-48 +0.62, T-24 +0.30, T-12 +0.18, T+0 +0.27; Z500_Plains T-71 +0.55, T-48 +0.45, T-24 +0.39, T-12 +0.42, T+0 +0.47; Z500_Canada T-71 +0.68, T-48 +0.61, T-24 +0.36, T-12 +0.27, T+0 +0.39; thetae_NYC T-71 +0.13, T-48 +0.10, T-24 +0.34, T-12 +0.65, T+0 +0.99
- **B2**: Z500_Midwest T-71 +0.77, T-48 +0.70, T-24 +0.34, T-12 +0.12, T+0 +0.18; Z500_Plains T-71 +0.09, T-48 -0.05, T-24 +0.07, T-12 +0.10, T+0 +0.32; Z500_Canada T-71 +0.55, T-48 +0.43, T-24 -0.05, T-12 -0.21, T+0 -0.10; thetae_NYC T-71 +0.78, T-48 +0.77, T-24 +1.09, T-12 +1.20, T+0 +1.45

Reclassifications:

- **A1->B1** (n=16): Z500_Midwest T-71 -0.43, T-48 -0.66, T-24 -1.23, T-12 -1.59, T+0 -1.43; Z500_Plains T-71 -0.73, T-48 -0.92, T-24 -1.15, T-12 -1.00, T+0 -0.80; Z500_Canada T-71 -0.25, T-48 -0.62, T-24 -0.87, T-12 -0.89, T+0 -0.91; thetae_NYC T-71 +0.50, T-48 +0.24, T-24 +0.81, T-12 +0.78, T+0 +1.08
- **B2->A2** (n=14): Z500_Midwest T-71 +0.74, T-48 +0.69, T-24 +0.54, T-12 +0.45, T+0 +0.55; Z500_Plains T-71 +0.48, T-48 +0.42, T-24 +0.46, T-12 +0.48, T+0 +0.47; Z500_Canada T-71 +0.75, T-48 +0.79, T-24 +0.44, T-12 +0.19, T+0 +0.21; thetae_NYC T-71 +0.02, T-48 -0.10, T-24 +0.32, T-12 +0.74, T+0 +1.22
- **A2->A1** (n=9): Z500_Midwest T-71 -0.11, T-48 -0.37, T-24 -0.50, T-12 -0.36, T+0 -0.12; Z500_Plains T-71 +0.21, T-48 +0.06, T-24 -0.07, T-12 -0.11, T+0 -0.08; Z500_Canada T-71 -0.32, T-48 -0.25, T-24 +0.12, T-12 +0.34, T+0 +0.56; thetae_NYC T-71 -0.34, T-48 -0.44, T-24 -0.36, T-12 -0.28, T+0 +0.33
- **B1->B2** (n=6): Z500_Midwest T-71 +0.68, T-48 +0.32, T-24 -0.30, T-12 -0.45, T+0 -0.39; Z500_Plains T-71 -0.78, T-48 -1.01, T-24 -0.33, T-12 +0.05, T+0 +0.41; Z500_Canada T-71 +0.41, T-48 +0.45, T-24 -0.67, T-12 -0.96, T+0 -0.88; thetae_NYC T-71 +1.19, T-48 +0.87, T-24 +1.11, T-12 +1.50, T+0 +1.27
- **B2->B1** (n=5): Z500_Midwest T-71 -0.28, T-48 -0.47, T-24 -0.38, T-12 -0.39, T+0 -0.06; Z500_Plains T-71 -0.44, T-48 -0.58, T-24 -0.26, T-12 -0.11, T+0 +0.16; Z500_Canada T-71 +0.17, T-48 -0.24, T-24 +0.21, T-12 +0.27, T+0 +0.42; thetae_NYC T-71 +0.59, T-48 +0.94, T-24 +0.88, T-12 +0.84, T+0 +1.04
- **B1->A1** (n=2): Z500_Midwest T-71 -0.23, T-48 -0.28, T-24 -0.85, T-12 -1.23, T+0 -1.10; Z500_Plains T-71 +0.21, T-48 -0.09, T-24 -0.39, T-12 -0.29, T+0 +0.09; Z500_Canada T-71 +0.04, T-48 -0.80, T-24 -1.19, T-12 -1.35, T+0 -1.36; thetae_NYC T-71 +0.55, T-48 -0.00, T-24 +0.20, T-12 +0.50, T+0 +1.07
- **B1->A2** (n=2): Z500_Midwest T-71 +0.37, T-48 +0.04, T-24 -0.79, T-12 -1.03, T+0 -0.82; Z500_Plains T-71 +0.48, T-48 -0.63, T-24 -0.82, T-12 -0.13, T+0 +0.37; Z500_Canada T-71 +0.34, T-48 -0.08, T-24 -0.41, T-12 -0.41, T+0 -0.14; thetae_NYC T-71 -0.50, T-48 +0.23, T-24 +0.60, T-12 +1.18, T+0 +1.17
- **A2->B1** (n=2): Z500_Midwest T-71 +0.07, T-48 -0.31, T-24 -0.46, T-12 -0.32, T+0 -0.00; Z500_Plains T-71 -1.39, T-48 -0.72, T-24 -0.41, T-12 -0.17, T+0 +0.10; Z500_Canada T-71 +0.00, T-48 -0.50, T-24 -0.10, T-12 +0.25, T+0 +0.59; thetae_NYC T-71 -1.02, T-48 +0.19, T-24 +1.13, T-12 +1.18, T+0 +1.08
- **B2->A1** (n=2): Z500_Midwest T-71 -0.64, T-48 -0.04, T-24 +0.10, T-12 +0.12, T+0 +0.46; Z500_Plains T-71 -0.32, T-48 +0.04, T-24 +0.01, T-12 +0.08, T+0 +0.24; Z500_Canada T-71 +0.13, T-48 +0.06, T-24 +0.14, T-12 +0.38, T+0 +0.24; thetae_NYC T-71 -1.17, T-48 -0.72, T-24 -0.31, T-12 -0.20, T+0 +0.40
- **A1->B2** (n=1): Z500_Midwest T-71 +1.28, T-48 +0.84, T-24 +0.08, T-12 -0.79, T+0 -1.10; Z500_Plains T-71 +0.57, T-48 +0.34, T-24 -0.22, T-12 -0.56, T+0 -0.04; Z500_Canada T-71 -0.51, T-48 -0.83, T-24 -0.83, T-12 -0.79, T+0 -0.52; thetae_NYC T-71 +2.20, T-48 +2.29, T-24 +2.13, T-12 +1.16, T+0 +1.92

## All-days pathways (5×3 seed 25, 118 FF days)

High-risk nodes (≥2×): ['A4', 'A5']

- **A1**: d-3 9% high-risk (top {'C5': 13, 'C1': 4, 'C3': 4}); d-2 9% high-risk (top {'C5': 9, 'C4': 6, 'B5': 4}); d-1 20% high-risk (top {'B5': 7, 'C4': 7, 'B4': 6}); d+0 37% high-risk (top {'A5': 11, 'C5': 9, 'C4': 3})
- **B1**: d-3 37% high-risk (top {'A1': 7, 'A5': 7, 'A4': 4}); d-2 50% high-risk (top {'A5': 14, 'A3': 7, 'A1': 4}); d-1 80% high-risk (top {'A5': 16, 'A4': 8, 'A3': 5}); d+0 87% high-risk (top {'A5': 19, 'A4': 7, 'A1': 1})
- **A2**: d-3 0% high-risk (top {'C1': 7, 'A1': 4, 'C3': 4}); d-2 0% high-risk (top {'C2': 9, 'C1': 8, 'B2': 3}); d-1 0% high-risk (top {'B1': 6, 'C1': 5, 'B2': 4}); d+0 12% high-risk (top {'B1': 5, 'C2': 4, 'A1': 3})
- **B2**: d-3 3% high-risk (top {'A1': 17, 'B1': 2, 'C1': 2}); d-2 3% high-risk (top {'A1': 15, 'B1': 4, 'A3': 3}); d-1 3% high-risk (top {'A1': 17, 'A2': 5, 'A3': 4}); d+0 14% high-risk (top {'A1': 8, 'A3': 8, 'A2': 6})

## Independent variables: static vs evSOM grouping

Effect size = rank ε² (Kruskal–Wallis) or Cramér's V (TC). p = permutation p of each grouping; Δ CI = paired bootstrap 95% CI of evSOM − static.

| variable | n | static | evSOM | Δ 95% CI | P(evSOM > static) |
|---|---|---|---|---|---|
| Stage IV area >= 10 mm/h | 101 | 0.117 (p=0.004) | 0.087 (p=0.025) | [-0.179, +0.106] | 0.36 |
| Stage IV area >= 25 mm/h | 101 | 0.030 (p=0.410) | 0.042 (p=0.245) | [-0.082, +0.114] | 0.61 |
| Stage IV regional peak rate | 101 | 0.021 (p=0.570) | 0.059 (p=0.113) | [-0.070, +0.154] | 0.76 |
| Stage IV NYC peak rate | 101 | 0.070 (p=0.071) | 0.012 (p=0.741) | [-0.166, +0.052] | 0.16 |
| ASOS peak hourly rate | 117 | 0.052 (p=0.116) | 0.041 (p=0.203) | [-0.119, +0.089] | 0.42 |
| TC within 48 h (D7 rule) | 118 | 0.115 (p=0.691) | 0.170 (p=0.339) | [-0.168, +0.206] | 0.64 |
| TCWV at T0 (related to θe) | 118 | 0.169 (p=0.000) | 0.220 (p=0.000) | [-0.090, +0.191] | 0.75 |
| TCWV daily mean (related to θe) | 118 | 0.117 (p=0.003) | 0.250 (p=0.000) | [+0.001, +0.256] | 0.98 |

## Lead-time test (in-sample; chance = 0.25)

- T-0 (72 h used): 1.00 match; by node A1 1.00, B1 1.00, A2 1.00, B2 1.00
- T-6 (66 h used): 1.00 match; by node A1 1.00, B1 1.00, A2 1.00, B2 1.00
- T-12 (60 h used): 0.99 match; by node A1 1.00, B1 0.97, A2 1.00, B2 1.00
- T-18 (54 h used): 0.95 match; by node A1 0.94, B1 0.93, A2 0.96, B2 0.97
- T-24 (48 h used): 0.91 match; by node A1 0.91, B1 0.87, A2 0.96, B2 0.90
- T-30 (42 h used): 0.81 match; by node A1 0.77, B1 0.80, A2 0.96, B2 0.76
- T-36 (36 h used): 0.80 match; by node A1 0.74, B1 0.80, A2 0.92, B2 0.76
- T-42 (30 h used): 0.75 match; by node A1 0.74, B1 0.77, A2 0.79, B2 0.72
- T-48 (24 h used): 0.74 match; by node A1 0.74, B1 0.70, A2 0.79, B2 0.72
- T-54 (18 h used): 0.70 match; by node A1 0.74, B1 0.67, A2 0.71, B2 0.69
- T-60 (12 h used): 0.70 match; by node A1 0.77, B1 0.63, A2 0.67, B2 0.72
- T-66 (6 h used): 0.65 match; by node A1 0.71, B1 0.53, A2 0.62, B2 0.72
