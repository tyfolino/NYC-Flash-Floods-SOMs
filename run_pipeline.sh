#!/usr/bin/env bash
# Full NYC flash-flood SOM pipeline: preprocessing (optional) -> training of
# all three SOMs -> statistics -> figures -> results/manuscript_numbers.json.
#
# Run from the project root. Examples:
#
#   # Re-make statistics, figures and numbers from the existing (published) SOMs:
#   bash run_pipeline.sh --stages "stats figures numbers"
#
#   # Final (revision) configuration: v2 intermediates, 5x length, lr2 0.02,
#   # all-days 5x3 (seed 25, highest-risk node -> A5), 2x2 SOMs at lr1 0.5 with
#   # the representative ensemble seeds, node names matched to the published
#   # SOMs, 48-h evSOM window. Reproduces the robustness-ensemble members exactly.
#   bash run_pipeline.sh --intermediate v2 --output-root /mnt/drive2/SOM_runs/final \
#       --evsom-hours 48 --train-args "--length 5 --lr2 0.02" \
#       --alldays-args "--xdim 5 --ydim 3 --seed 25 --orient-risk" \
#       --static-args "--seed 8 --orient-to figs/Z500-and-thetae-SOM/.cache/som_results.npz" \
#       --evsom-args "--seed 16 --orient-to figs/Z500-and-thetae-evSOM-24h/.cache/som_results.npz"
#
#   # Build the v2 intermediates first (slow; ~1 h, ~100 GB RAM peak for theta_e):
#   bash run_pipeline.sh --preprocess --intermediate v2 --stages none
#
# Options:
#   --stages "LIST"        any of: train stats figures numbers (default: all four),
#                          or "none" (only --preprocess / --d3 / --d4 / --movies).
#   --preprocess           regenerate intermediates (refuses to write into v1).
#   --climatology MODE     monthday (default) | doy (as published), for --preprocess.
#   --intermediate WHICH   v1 (published; default) | v2 | DIR  (NYCFF_INTERMEDIATE).
#   --output-root DIR      where SOM caches, CSVs, figures and results go
#                          (NYCFF_OUTPUT_ROOT; default: the project tree).
#   --overwrite-published  allow the train stage to write into the project tree.
#                          The published SOM caches are gitignored, so they are NOT
#                          recoverable from git once overwritten.
#   --train-args "ARGS"    hyperparameters for all three trainers (see
#                          som_analysis/training.py: --length --lr1 --lr2 --sigma1
#                          --sigma2 --neighborhood --mode --topology --init --seed).
#   --alldays-args/--static-args/--evsom-args "ARGS"   per-SOM extras (appended).
#                          --static-args defaults to "--rotate" (as published);
#                          passing it replaces that default. --orient-to NPZ /
#                          --orient-risk fix node names (see som_analysis/training.py).
#   --evsom-hours N        evSOM window length (default 24); also selects the
#                          evSOM read by the stats/figure stages (NYCFF_EVSOM_HOURS).
#   --d3 / --d4            also run the SOM-independent D3 / D4 analyses.
#   --movies               also render Movie S1 (slow).
#   --workers N            cap for dask threads / process pools (default 4).

set -euo pipefail
cd "$(dirname "$0")"

STAGES="train stats figures numbers"
PREPROCESS=0
CLIMATOLOGY=monthday
INTERMEDIATE="${NYCFF_INTERMEDIATE:-v1}"
OUTPUT_ROOT="${NYCFF_OUTPUT_ROOT:-}"
OVERWRITE=0
TRAIN_ARGS=""
ALLDAYS_ARGS=""
STATIC_ARGS="--rotate"
EVSOM_ARGS=""
EVSOM_HOURS=24
RUN_D3=0
RUN_D4=0
MOVIES=0
WORKERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --preprocess) PREPROCESS=1; shift ;;
        --climatology) CLIMATOLOGY="$2"; shift 2 ;;
        --intermediate) INTERMEDIATE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --overwrite-published) OVERWRITE=1; shift ;;
        --train-args) TRAIN_ARGS="$2"; shift 2 ;;
        --alldays-args) ALLDAYS_ARGS="$2"; shift 2 ;;
        --static-args) STATIC_ARGS="$2"; shift 2 ;;
        --evsom-args) EVSOM_ARGS="$2"; shift 2 ;;
        --evsom-hours) EVSOM_HOURS="$2"; shift 2 ;;
        --d3) RUN_D3=1; shift ;;
        --d4) RUN_D4=1; shift ;;
        --movies) MOVIES=1; shift ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# ── Environment: thread caps (shared 64-core server), paths, determinism ─────
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NYCFF_DASK_WORKERS="${WORKERS}"
export NYCFF_INTERMEDIATE="${INTERMEDIATE}"
export NYCFF_EVSOM_HOURS="${EVSOM_HOURS}"
if [[ -n "${OUTPUT_ROOT}" ]]; then
    mkdir -p "${OUTPUT_ROOT}"
    export NYCFF_OUTPUT_ROOT
    NYCFF_OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
fi
# Fixed PDF/PS metadata dates so unchanged figures are byte-identical.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1767225600}"

ROOT_ABS="$(pwd)"
OUT_ABS="${NYCFF_OUTPUT_ROOT:-${ROOT_ABS}}"
echo "Intermediates : ${NYCFF_INTERMEDIATE}"
echo "evSOM window  : ${EVSOM_HOURS} h"
echo "Output root   : ${OUT_ABS}"
echo "Stages        : ${STAGES}$( ((PREPROCESS)) && echo ' +preprocess')"

if has_stage train && [[ "${OUT_ABS}" == "${ROOT_ABS}" ]] && ((OVERWRITE == 0)); then
    echo "Refusing to train into the project tree: the published SOM caches" >&2
    echo "(figs/*/.cache, gitignored) would be overwritten. Pass --output-root DIR," >&2
    echo "or --overwrite-published if that is really intended." >&2
    exit 1
fi

# The output tree needs the input event list for scripts that read data/.
if [[ "${OUT_ABS}" != "${ROOT_ABS}" ]]; then
    mkdir -p "${OUT_ABS}/data"
fi

run() { echo "+ $*"; "$@"; }

# ── 1. Preprocessing (optional) ───────────────────────────────────────────────
if ((PREPROCESS)); then
    echo; echo "=== Preprocessing (${CLIMATOLOGY} climatology) ==="
    for v in Z500 theta_e ivt tcwv mslp tp; do
        run python -m preprocessing.som_preprocessing "$v" --no-plots --no-snapshots \
            --climatology "${CLIMATOLOGY}"
    done
    for v in Z500 theta_e; do
        run python -m preprocessing.som_preprocessing_evolution "$v" \
            --n-hours "${EVSOM_HOURS}" --climatology "${CLIMATOLOGY}"
    done
fi

# ── 2. Training ───────────────────────────────────────────────────────────────
if has_stage train; then
    echo; echo "=== Training ==="
    # shellcheck disable=SC2086
    run python -m som_analysis.train_alldays_som --moisture-var thetae ${TRAIN_ARGS} ${ALLDAYS_ARGS}
    # shellcheck disable=SC2086
    run python -m som_analysis.train_som --moisture-var thetae ${TRAIN_ARGS} ${STATIC_ARGS}
    # shellcheck disable=SC2086
    run python -m som_analysis.train_evsom --moisture-var thetae --n-hours "${EVSOM_HOURS}" \
        ${TRAIN_ARGS} ${EVSOM_ARGS}
fi

# ── 3. Statistics ─────────────────────────────────────────────────────────────
if has_stage stats; then
    echo; echo "=== Statistics ==="
    run python -m som_analysis.tc_associations
fi
if ((RUN_D3)); then
    run python -m som_analysis.moisture_forcing_metrics --workers "${WORKERS}"
fi
if ((RUN_D4)); then
    run python -m som_analysis.extreme_day_comparison
fi

# ── 4. Figures ────────────────────────────────────────────────────────────────
if has_stage figures; then
    echo; echo "=== Figures ==="
    for f in fig01_som_node_weights fig02_evsom_key_hours fig03_evsom_composite_key_hours \
             figS01_ffe_distributions figS02_sammon_map figS03_alldays_raw_composites \
             figS04_evsom_sammon_map figS05_alldays_pathways \
             figS09_evsom_tc_tracks_and_histogram; do
        run python -m "figure_scripts.${f}"
    done
    for f in fig04_stageiv_pmm_composites figS06_ffe_som_node_weights_and_composites \
             figS07_evsom_ivt_tcwv_composites figS08_precip_histograms; do
        run python -m "figure_scripts.${f}"
        run python -m "figure_scripts.${f}" --wide
    done
fi
if ((MOVIES)); then
    run python -m figure_scripts.movieS01_evsom_node_weights_animation
    run python -m figure_scripts.movieS01_evsom_combined_animation
    run python -m figure_scripts.movieS01_evsom_combined_animation --wide
    run python -m figure_scripts.movieS01_evsom_combined_animation --wide --split
fi

# ── 5. Manuscript numbers ─────────────────────────────────────────────────────
if has_stage numbers; then
    echo; echo "=== Manuscript numbers ==="
    run python -m som_analysis.manuscript_numbers
fi

echo; echo "Done."
