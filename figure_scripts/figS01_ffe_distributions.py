"""
Supplementary Figure S1 — Flash Flood Episode Distributions

Three-panel figure:
  (a) Annual distribution (1996–2025) with linear trend line
  (b) Seasonal distribution (by month)
  (c) Diurnal distribution (by hour of day, EST)

Usage:
    python -m figure_scripts.figS01_ffe_distributions
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from som_analysis.config import DATA_DIR, setup_plotting

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS01")
os.makedirs(OUT_DIR, exist_ok=True)

STORM_CSV = os.path.join(DATA_DIR, "storm_data_search_results.csv")

# ── Figure parameters ─────────────────────────────────────────────────────────
FIG_WIDTH = 7.0  # GRL double column
FIG_HEIGHT = 4.2
DPI_RASTER = 300
BAR_COLOR = "#3a7ebf"  # clean steel blue, no gradient


def load_data():
    df = pd.read_csv(STORM_CSV)
    df = df[df["EVENT_ID"].astype(str).str.isdigit()]
    df["BEGIN_TIME"] = df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
    begin_str = df["BEGIN_DATE"] + " " + df["BEGIN_TIME"]
    df["BEGIN_DATETIME"] = pd.to_datetime(
        begin_str, format="%m/%d/%Y %H%M", errors="coerce"
    )
    df = df.drop_duplicates(subset=["EPISODE_ID"], keep="first").copy()
    df["EVENT_COUNT"] = 1
    return df


def main():
    setup_plotting()

    df = load_data()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df_yearly = df.groupby(df["BEGIN_DATETIME"].dt.year)["EVENT_COUNT"].sum()
    df_monthly = df.groupby(df["BEGIN_DATETIME"].dt.month)["EVENT_COUNT"].sum()
    df_hourly = df.groupby(df["BEGIN_DATETIME"].dt.hour)["EVENT_COUNT"].sum()

    # Ensure all months (1–12) and hours (0–23) are represented
    df_monthly = df_monthly.reindex(range(1, 13), fill_value=0)
    df_hourly = df_hourly.reindex(range(0, 24), fill_value=0)

    # Linear trend for annual panel
    years = df_yearly.index.values
    counts = df_yearly.values
    slope, intercept, r_value, p_value, _ = stats.linregress(years, counts)
    trend_y = slope * years + intercept

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_RASTER, constrained_layout=True
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.08, wspace=0.04)

    ax1 = fig.add_subplot(gs[0, :])  # (a) annual — full width
    ax2 = fig.add_subplot(gs[1, 0])  # (b) seasonal
    ax3 = fig.add_subplot(gs[1, 1])  # (c) diurnal

    # ── (a) Annual ────────────────────────────────────────────────────────────
    ax1.bar(years, counts, width=0.8, color=BAR_COLOR, linewidth=0)
    ax1.plot(
        years,
        trend_y,
        color="firebrick",
        linewidth=0.9,
        linestyle="--",
        label=f"Trend: {slope:+.2f} events yr$^{{-1}}$ ($p={p_value:.2f}$)",
    )
    ax1.set_xlim(1995.5, 2025.5)
    ax1.set_xticks(years[::2])
    ax1.tick_params(axis="x", rotation=45, labelsize=5.5)
    ax1.tick_params(axis="y", labelsize=5.5)
    ax1.set_xlabel("Year", fontsize=6)
    ax1.set_ylabel("Number of Episodes", fontsize=6)
    ax1.legend(fontsize=5.5, loc="upper left", framealpha=0.7)
    ax1.text(
        -0.07,
        1.02,
        "(a)",
        transform=ax1.transAxes,
        fontsize=6.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ── (b) Seasonal ──────────────────────────────────────────────────────────
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax2.bar(
        df_monthly.index, df_monthly.values, width=0.8, color=BAR_COLOR, linewidth=0
    )
    ax2.set_xlim(0.5, 12.5)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_labels, fontsize=5.5)
    ax2.tick_params(axis="y", labelsize=5.5)
    ax2.set_xlabel("Month", fontsize=6)
    ax2.set_ylabel("Number of Episodes", fontsize=6)
    ax2.text(
        -0.14,
        1.02,
        "(b)",
        transform=ax2.transAxes,
        fontsize=6.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ── (c) Diurnal ───────────────────────────────────────────────────────────
    ax3.bar(
        df_hourly.index + 0.5, df_hourly.values, width=0.8, color=BAR_COLOR, linewidth=0
    )
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 24, 3))
    ax3.tick_params(axis="x", labelsize=5.5)
    ax3.tick_params(axis="y", labelsize=5.5)
    ax3.set_xlabel("Hour of Day (EST)", fontsize=6)
    ax3.set_ylabel("Number of Episodes", fontsize=6)
    ax3.text(
        -0.14,
        1.02,
        "(c)",
        transform=ax3.transAxes,
        fontsize=6.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS01_ffe_distributions")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
