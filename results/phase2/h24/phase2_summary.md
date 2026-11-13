# Phase 2 (D1): 24-h evSOM

## Seed baseline (ARI median [min–max])

- static_static: ARI 0.96 [0.88–1.00], agreement 0.98 [0.95–1.00]
- evsom_evsom: ARI 0.92 [0.79–1.00], agreement 0.97 [0.91–1.00]
- static_evsom: ARI 0.65 [0.55–0.72], agreement 0.84 [0.78–0.87]
- share of cross-method pairs above the within-evSOM 5th percentile: 0.00

Representative members: static seed 8 (agreement with published 0.86); evSOM seed 17 (agreement with published 24 h 0.92). evSOM node counts: {'A1': 38, 'B1': 21, 'A2': 19, 'B2': 40}.

## Reclassified events: 17

| static → evSOM | A1 | B1 | A2 | B2 |
|---|---|---|---|---|
| A1 | 34 | 5 | 0 | 0 |
| B1 | 1 | 14 | 1 | 1 |
| A2 | 3 | 0 | 15 | 1 |
| B2 | 0 | 2 | 3 | 38 |

Mean standardized anomalies (box means) by evSOM node:

- **A1**: Z500_Midwest T-23 -1.22, T-12 -1.46, T+0 -1.35; Z500_Plains T-23 -0.81, T-12 -0.84, T+0 -0.76; Z500_Canada T-23 -0.39, T-12 -0.41, T+0 -0.37; thetae_NYC T-23 +0.21, T-12 +0.36, T+0 +0.77
- **B1**: Z500_Midwest T-23 -0.92, T-12 -1.10, T+0 -0.87; Z500_Plains T-23 -0.85, T-12 -0.60, T+0 -0.22; Z500_Canada T-23 -0.79, T-12 -0.92, T+0 -0.82; thetae_NYC T-23 +0.88, T-12 +1.01, T+0 +1.30
- **A2**: Z500_Midwest T-23 -0.14, T-12 -0.12, T+0 +0.03; Z500_Plains T-23 +0.19, T-12 +0.17, T+0 +0.23; Z500_Canada T-23 +0.20, T-12 +0.38, T+0 +0.54; thetae_NYC T-23 +0.20, T-12 +0.17, T+0 +0.50
- **B2**: Z500_Midwest T-23 +0.50, T-12 +0.35, T+0 +0.43; Z500_Plains T-23 +0.29, T-12 +0.26, T+0 +0.36; Z500_Canada T-23 +0.22, T-12 +0.07, T+0 +0.16; thetae_NYC T-23 +0.80, T-12 +0.98, T+0 +1.37

Reclassifications:

- **A1->B1** (n=5): Z500_Midwest T-23 -0.94, T-12 -1.07, T+0 -0.82; Z500_Plains T-23 -0.63, T-12 -0.34, T+0 -0.13; Z500_Canada T-23 -0.83, T-12 -0.98, T+0 -1.05; thetae_NYC T-23 +1.07, T-12 +0.87, T+0 +1.09
- **A2->A1** (n=3): Z500_Midwest T-23 -0.87, T-12 -0.92, T+0 -0.50; Z500_Plains T-23 -0.38, T-12 -0.40, T+0 -0.02; Z500_Canada T-23 +0.40, T-12 +0.47, T+0 +0.75; thetae_NYC T-23 -0.00, T-12 +0.01, T+0 +0.26
- **B2->A2** (n=3): Z500_Midwest T-23 -0.03, T-12 -0.21, T+0 +0.12; Z500_Plains T-23 -0.16, T-12 -0.09, T+0 +0.18; Z500_Canada T-23 +0.08, T-12 +0.22, T+0 +0.15; thetae_NYC T-23 +0.36, T-12 +0.35, T+0 +0.65
- **B2->B1** (n=2): Z500_Midwest T-23 -0.91, T-12 -0.90, T+0 -0.47; Z500_Plains T-23 -0.20, T-12 +0.01, T+0 +0.35; Z500_Canada T-23 +0.37, T-12 +0.53, T+0 +0.77; thetae_NYC T-23 +0.85, T-12 +0.94, T+0 +0.91
- **B1->A1** (n=1): Z500_Midwest T-23 -1.01, T-12 -1.24, T+0 -1.05; Z500_Plains T-23 -0.38, T-12 -0.45, T+0 -0.28; Z500_Canada T-23 -0.58, T-12 -0.75, T+0 -1.18; thetae_NYC T-23 +0.28, T-12 +0.84, T+0 +1.27
- **B1->A2** (n=1): Z500_Midwest T-23 -0.63, T-12 -0.63, T+0 -0.59; Z500_Plains T-23 -0.02, T-12 +0.27, T+0 +0.61; Z500_Canada T-23 -0.25, T-12 -0.24, T+0 -0.18; thetae_NYC T-23 +1.14, T-12 +1.19, T+0 +0.81
- **B1->B2** (n=1): Z500_Midwest T-23 +0.30, T-12 +0.08, T+0 -0.23; Z500_Plains T-23 +0.47, T-12 +0.68, T+0 +0.72; Z500_Canada T-23 -0.81, T-12 -0.81, T+0 -0.57; thetae_NYC T-23 +1.37, T-12 +1.40, T+0 +0.78
- **A2->B2** (n=1): Z500_Midwest T-23 +0.28, T-12 -0.18, T+0 -0.21; Z500_Plains T-23 +0.11, T-12 +0.19, T+0 +0.27; Z500_Canada T-23 -0.44, T-12 +0.04, T+0 +0.58; thetae_NYC T-23 +0.31, T-12 +1.72, T+0 +1.99

## All-days pathways (5×3 seed 25, 118 FF days)

High-risk nodes (≥2×): ['A4', 'A5']

- **A1**: d-3 18% high-risk (top {'C5': 10, 'A5': 6, 'A1': 4}); d-2 24% high-risk (top {'A5': 8, 'C5': 8, 'A3': 3}); d-1 45% high-risk (top {'A5': 13, 'B5': 7, 'B4': 4}); d+0 66% high-risk (top {'A5': 21, 'C5': 9, 'B5': 4})
- **B1**: d-3 24% high-risk (top {'A1': 7, 'A4': 3, 'C1': 2}); d-2 33% high-risk (top {'A1': 5, 'A5': 5, 'A3': 4}); d-1 62% high-risk (top {'A5': 8, 'A4': 5, 'A3': 4}); d+0 81% high-risk (top {'A5': 9, 'A4': 8, 'A3': 3})
- **A2**: d-3 5% high-risk (top {'C1': 3, 'C5': 3, 'A2': 2}); d-2 0% high-risk (top {'C1': 4, 'C4': 4, 'A2': 2}); d-1 5% high-risk (top {'C1': 4, 'C3': 3, 'C4': 3}); d+0 16% high-risk (top {'C3': 4, 'C2': 3, 'A4': 3})
- **B2**: d-3 5% high-risk (top {'A1': 16, 'C1': 8, 'C3': 6}); d-2 8% high-risk (top {'A1': 11, 'C2': 9, 'C1': 5}); d-1 2% high-risk (top {'A1': 16, 'B1': 6, 'B2': 5}); d+0 2% high-risk (top {'A1': 12, 'A2': 9, 'A3': 7})

## Independent variables: static vs evSOM grouping

Effect size = rank ε² (Kruskal–Wallis) or Cramér's V (TC). p = permutation p of each grouping; Δ CI = paired bootstrap 95% CI of evSOM − static.

| variable | n | static | evSOM | Δ 95% CI | P(evSOM > static) |
|---|---|---|---|---|---|
| Stage IV area >= 10 mm/h | 101 | 0.117 (p=0.009) | 0.089 (p=0.028) | [-0.143, +0.074] | 0.31 |
| Stage IV area >= 25 mm/h | 101 | 0.030 (p=0.386) | 0.048 (p=0.173) | [-0.055, +0.095] | 0.68 |
| Stage IV regional peak rate | 101 | 0.021 (p=0.533) | 0.017 (p=0.634) | [-0.082, +0.070] | 0.48 |
| Stage IV NYC peak rate | 101 | 0.070 (p=0.064) | 0.022 (p=0.537) | [-0.128, +0.021] | 0.09 |
| ASOS peak hourly rate | 117 | 0.052 (p=0.114) | 0.060 (p=0.071) | [-0.070, +0.090] | 0.59 |
| TC within 48 h (D7 rule) | 118 | 0.115 (p=0.698) | 0.111 (p=0.706) | [-0.147, +0.131] | 0.47 |
| TCWV at T0 (related to θe) | 118 | 0.169 (p=0.000) | 0.242 (p=0.000) | [-0.018, +0.162] | 0.94 |
| TCWV daily mean (related to θe) | 118 | 0.117 (p=0.003) | 0.185 (p=0.000) | [-0.016, +0.154] | 0.94 |

## Lead-time test (in-sample; chance = 0.28)

- T-0 (24 h used): 1.00 match; by node A1 1.00, B1 1.00, A2 1.00, B2 1.00
- T-6 (18 h used): 0.98 match; by node A1 0.97, B1 1.00, A2 0.95, B2 1.00
- T-12 (12 h used): 0.97 match; by node A1 0.97, B1 1.00, A2 0.89, B2 1.00
- T-18 (6 h used): 0.95 match; by node A1 0.95, B1 1.00, A2 0.79, B2 1.00
