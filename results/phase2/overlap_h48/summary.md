# Overlapping 48-h evSOM windows: sensitivity

5 of 118 events are < 48 h after the previous event, so their windows share hours with it:

| first event | second event | gap (h) | evSOM nodes |
|---|---|---|---|
| 2008-08-14 21Z | 2008-08-15 22Z | 25 | A1, A1 |
| 2013-05-08 13Z | 2013-05-09 15Z | 26 | A2, B2 |
| 2014-07-03 00Z | 2014-07-04 00Z | 24 | B1, B1 |
| 2017-08-02 21Z | 2017-08-04 12Z | 39 | A1, A1 |
| 2019-07-22 23Z | 2019-07-23 12Z | 13 | B1, A1 |

Retrained without the second event of each pair (113 events, 10 seeds, final-ensemble settings) and compared with the representative full-data evSOM (seed 16) on the 113 events:

| comparison | matched agreement, median [min–max] | ARI, median [min–max] |
|---|---|---|
| without overlaps vs reference | 0.973 [0.850–0.991] | 0.94 [0.68–0.98] |
| full-data ensemble (29 other seeds) vs reference | 0.982 [0.912–1.000] | 0.96 [0.81–1.00] |

Dropped events projected onto the retrained SOMs land in their reference node in a median 100% of seeds' assignments.
