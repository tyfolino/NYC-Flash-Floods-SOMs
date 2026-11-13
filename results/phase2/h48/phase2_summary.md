# Phase 2 (D1): 48-h evSOM

## Seed baseline (ARI median [min–max])

- static_static: ARI 0.96 [0.88–1.00], agreement 0.98 [0.95–1.00]
- evsom_evsom: ARI 0.96 [0.77–1.00], agreement 0.98 [0.89–1.00]
- static_evsom: ARI 0.49 [0.35–0.54], agreement 0.74 [0.64–0.75]
- share of cross-method pairs above the within-evSOM 5th percentile: 0.00

Representative members: static seed 8 (agreement with published 0.86); evSOM seed 16 (agreement with published 24 h 0.85). evSOM node counts: {'A1': 38, 'B1': 23, 'A2': 15, 'B2': 42}.

## Reclassified events: 30

| static → evSOM | A1 | B1 | A2 | B2 |
|---|---|---|---|---|
| A1 | 30 | 9 | 0 | 0 |
| B1 | 2 | 10 | 1 | 4 |
| A2 | 5 | 2 | 11 | 1 |
| B2 | 1 | 2 | 3 | 37 |

Mean standardized anomalies (box means) by evSOM node:

- **A1**: Z500_Midwest T-47 -0.56, T-24 -1.13, T-12 -1.32, T+0 -1.21; Z500_Plains T-47 -0.47, T-24 -0.71, T-12 -0.73, T+0 -0.68; Z500_Canada T-47 -0.45, T-24 -0.50, T-12 -0.46, T+0 -0.39; thetae_NYC T-47 -0.26, T-24 -0.04, T-12 +0.20, T+0 +0.63
- **B1**: Z500_Midwest T-47 -0.49, T-24 -0.98, T-12 -1.16, T+0 -0.90; Z500_Plains T-47 -1.06, T-24 -0.88, T-12 -0.66, T+0 -0.32; Z500_Canada T-47 -0.31, T-24 -0.62, T-12 -0.71, T+0 -0.66; thetae_NYC T-47 +0.81, T-24 +1.00, T-12 +1.02, T+0 +1.32
- **A2**: Z500_Midwest T-47 +0.33, T-24 +0.02, T-12 -0.05, T+0 +0.04; Z500_Plains T-47 +0.48, T-24 +0.27, T-12 +0.18, T+0 +0.26; Z500_Canada T-47 +0.32, T-24 +0.43, T-12 +0.57, T+0 +0.73; thetae_NYC T-47 +0.14, T-24 +0.33, T-12 +0.32, T+0 +0.60
- **B2**: Z500_Midwest T-47 +0.65, T-24 +0.43, T-12 +0.26, T+0 +0.33; Z500_Plains T-47 +0.14, T-24 +0.20, T-12 +0.23, T+0 +0.34; Z500_Canada T-47 +0.52, T-24 +0.18, T-12 +0.01, T+0 +0.11; thetae_NYC T-47 +0.39, T-24 +0.74, T-12 +0.99, T+0 +1.36

Reclassifications:

- **A1->B1** (n=9): Z500_Midwest T-47 -0.34, T-24 -1.04, T-12 -1.36, T+0 -1.10; Z500_Plains T-47 -0.92, T-24 -0.78, T-12 -0.58, T+0 -0.31; Z500_Canada T-47 -0.09, T-24 -0.57, T-12 -0.80, T+0 -0.86; thetae_NYC T-47 +0.53, T-24 +1.21, T-12 +0.92, T+0 +1.29
- **A2->A1** (n=5): Z500_Midwest T-47 -0.40, T-24 -0.76, T-12 -0.59, T+0 -0.24; Z500_Plains T-47 +0.07, T-24 -0.15, T-12 -0.09, T+0 +0.04; Z500_Canada T-47 -0.59, T-24 -0.27, T-12 -0.07, T+0 +0.18; thetae_NYC T-47 -0.09, T-24 -0.47, T-12 -0.36, T+0 +0.17
- **B1->B2** (n=4): Z500_Midwest T-47 +0.54, T-24 -0.27, T-12 -0.62, T+0 -0.64; Z500_Plains T-47 -0.80, T-24 -0.52, T-12 -0.10, T+0 +0.25; Z500_Canada T-47 +0.37, T-24 -0.68, T-12 -0.93, T+0 -0.63; thetae_NYC T-47 +0.51, T-24 +0.82, T-12 +1.35, T+0 +1.30
- **B2->A2** (n=3): Z500_Midwest T-47 +0.68, T-24 +0.27, T-12 +0.02, T+0 +0.22; Z500_Plains T-47 +0.27, T-24 -0.14, T-12 -0.11, T+0 +0.20; Z500_Canada T-47 +0.33, T-24 +0.17, T-12 +0.23, T+0 +0.33; thetae_NYC T-47 +0.64, T-24 +0.84, T-12 +1.03, T+0 +1.25
- **B1->A1** (n=2): Z500_Midwest T-47 -0.31, T-24 -0.85, T-12 -1.23, T+0 -1.10; Z500_Plains T-47 -0.14, T-24 -0.39, T-12 -0.29, T+0 +0.09; Z500_Canada T-47 -0.87, T-24 -1.19, T-12 -1.35, T+0 -1.36; thetae_NYC T-47 -0.02, T-24 +0.20, T-12 +0.50, T+0 +1.07
- **A2->B1** (n=2): Z500_Midwest T-47 -0.28, T-24 -0.46, T-12 -0.32, T+0 -0.00; Z500_Plains T-47 -0.67, T-24 -0.41, T-12 -0.17, T+0 +0.10; Z500_Canada T-47 -0.55, T-24 -0.10, T-12 +0.25, T+0 +0.59; thetae_NYC T-47 +0.23, T-24 +1.13, T-12 +1.18, T+0 +1.08
- **B2->B1** (n=2): Z500_Midwest T-47 -0.90, T-24 -0.95, T-12 -0.90, T+0 -0.47; Z500_Plains T-47 -0.84, T-24 -0.26, T-12 +0.01, T+0 +0.35; Z500_Canada T-47 +0.04, T-24 +0.35, T-12 +0.53, T+0 +0.77; thetae_NYC T-47 +1.13, T-24 +0.82, T-12 +0.94, T+0 +0.91
- **B1->A2** (n=1): Z500_Midwest T-47 +0.05, T-24 -0.61, T-12 -0.63, T+0 -0.59; Z500_Plains T-47 +0.42, T-24 -0.05, T-12 +0.27, T+0 +0.61; Z500_Canada T-47 -0.26, T-24 -0.26, T-12 -0.24, T+0 -0.18; thetae_NYC T-47 +1.21, T-24 +1.16, T-12 +1.19, T+0 +0.81
- **A2->B2** (n=1): Z500_Midwest T-47 +0.70, T-24 +0.33, T-12 -0.18, T+0 -0.21; Z500_Plains T-47 +0.51, T-24 +0.18, T-12 +0.19, T+0 +0.27; Z500_Canada T-47 +0.43, T-24 -0.40, T-12 +0.04, T+0 +0.58; thetae_NYC T-47 +0.26, T-24 +0.29, T-12 +1.72, T+0 +1.99
- **B2->A1** (n=1): Z500_Midwest T-47 -0.23, T-24 -0.10, T-12 -0.15, T+0 +0.56; Z500_Plains T-47 -0.14, T-24 -0.05, T-12 +0.06, T+0 +0.39; Z500_Canada T-47 -0.93, T-24 -0.93, T-12 -0.50, T+0 -0.40; thetae_NYC T-47 -0.96, T-24 -0.59, T-12 -0.54, T+0 -0.02

## All-days pathways (5×3 seed 25, 118 FF days)

High-risk nodes (≥2×): ['A4', 'A5']

- **A1**: d-3 18% high-risk (top {'C5': 11, 'A5': 7, 'B1': 3}); d-2 21% high-risk (top {'C5': 9, 'A5': 6, 'C4': 4}); d-1 39% high-risk (top {'A5': 10, 'B5': 7, 'C4': 6}); d+0 53% high-risk (top {'A5': 18, 'C5': 9, 'B5': 4})
- **B1**: d-3 26% high-risk (top {'A1': 10, 'A4': 4, 'A5': 2}); d-2 35% high-risk (top {'A5': 7, 'A1': 6, 'A3': 4}); d-1 70% high-risk (top {'A5': 11, 'A3': 6, 'A4': 5}); d+0 91% high-risk (top {'A5': 12, 'A4': 9, 'A3': 1})
- **A2**: d-3 0% high-risk (top {'C1': 3, 'A1': 2, 'A2': 2}); d-2 0% high-risk (top {'C1': 4, 'A1': 2, 'C2': 2}); d-1 0% high-risk (top {'C1': 4, 'C3': 3, 'A3': 2}); d+0 20% high-risk (top {'C2': 3, 'A4': 3, 'C1': 2})
- **B2**: d-3 5% high-risk (top {'A1': 15, 'C1': 8, 'C3': 6}); d-2 7% high-risk (top {'A1': 12, 'C2': 9, 'C1': 5}); d-1 2% high-risk (top {'A1': 17, 'B1': 6, 'A2': 6}); d+0 5% high-risk (top {'A1': 12, 'A2': 9, 'A3': 9})

## Independent variables: static vs evSOM grouping

Effect size = rank ε² (Kruskal–Wallis) or Cramér's V (TC). p = permutation p of each grouping; Δ CI = paired bootstrap 95% CI of evSOM − static.

| variable | n | static | evSOM | Δ 95% CI | P(evSOM > static) |
|---|---|---|---|---|---|
| Stage IV area >= 10 mm/h | 101 | 0.117 (p=0.007) | 0.116 (p=0.007) | [-0.150, +0.136] | 0.51 |
| Stage IV area >= 25 mm/h | 101 | 0.030 (p=0.383) | 0.098 (p=0.018) | [-0.033, +0.172] | 0.91 |
| Stage IV regional peak rate | 101 | 0.021 (p=0.530) | 0.032 (p=0.362) | [-0.090, +0.108] | 0.59 |
| Stage IV NYC peak rate | 101 | 0.070 (p=0.072) | 0.015 (p=0.677) | [-0.154, +0.045] | 0.13 |
| ASOS peak hourly rate | 117 | 0.052 (p=0.104) | 0.044 (p=0.171) | [-0.107, +0.086] | 0.42 |
| TC within 48 h (D7 rule) | 118 | 0.115 (p=0.684) | 0.122 (p=0.638) | [-0.155, +0.158] | 0.50 |
| TCWV at T0 (related to θe) | 118 | 0.169 (p=0.000) | 0.269 (p=0.000) | [-0.021, +0.217] | 0.94 |
| TCWV daily mean (related to θe) | 118 | 0.117 (p=0.003) | 0.246 (p=0.000) | [+0.006, +0.258] | 0.98 |

## Lead-time test (in-sample; chance = 0.28)

- T-0 (48 h used): 1.00 match; by node A1 1.00, B1 1.00, A2 1.00, B2 1.00
- T-6 (42 h used): 0.99 match; by node A1 1.00, B1 1.00, A2 1.00, B2 0.98
- T-12 (36 h used): 0.96 match; by node A1 1.00, B1 1.00, A2 0.93, B2 0.90
- T-18 (30 h used): 0.94 match; by node A1 1.00, B1 1.00, A2 0.87, B2 0.88
- T-24 (24 h used): 0.92 match; by node A1 1.00, B1 0.91, A2 0.87, B2 0.86
- T-30 (18 h used): 0.86 match; by node A1 0.89, B1 0.91, A2 0.73, B2 0.83
- T-36 (12 h used): 0.81 match; by node A1 0.82, B1 0.96, A2 0.60, B2 0.81
- T-42 (6 h used): 0.74 match; by node A1 0.79, B1 0.87, A2 0.40, B2 0.74
