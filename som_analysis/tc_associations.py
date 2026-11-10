"""
TC association CSVs for the flash-flood SOMs.

Applies the association rule in node_statistics.compute_tc_associations
(T-48 h to T0; 32-54N, 100-68W; storms that reached TD strength or stronger at
any point, all of their in-domain track points counted, remnants included) to
the evSOM and static-SOM BMU files and writes:

    data/som_2x2_evsom_24h_tc_associations.csv   (Fig. S9, manuscript TC numbers)
    data/som_2x2_tc_associations.csv             (static SOM)

Usage:
    python -m som_analysis.tc_associations
"""

import os

import pandas as pd

from .config import DATA_DIR
from .helpers import node_label
from .node_statistics import compute_tc_associations

TARGETS = {
    "evSOM": (
        "som_2x2_evsom_24h_bmus_thetae.csv",
        "som_2x2_evsom_24h_tc_associations.csv",
    ),
    "static SOM": ("som_2x2_bmus_thetae.csv", "som_2x2_tc_associations.csv"),
}


def main():
    for name, (bmu_name, out_name) in TARGETS.items():
        print(f"\n=== {name} ===")
        bmu_df = pd.read_csv(
            os.path.join(DATA_DIR, bmu_name), parse_dates=["timestamp"]
        )
        tc_df, *_ = compute_tc_associations(bmu_df, 2, 2)
        n_tc = int(tc_df["tc_present"].sum())
        print(f"TC-associated: {n_tc}/{len(tc_df)} ({100 * n_tc / len(tc_df):.1f}%)")
        for (i, j), g in tc_df.groupby(["node_i", "node_j"]):
            k = int(g["tc_present"].sum())
            print(f"  {node_label(i, j)}: {k}/{len(g)} ({100 * k / len(g):.1f}%)")
        tc_df.to_csv(os.path.join(DATA_DIR, out_name), index=False)
        print(f"Wrote data/{out_name}")


if __name__ == "__main__":
    main()
