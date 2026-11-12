#!/bin/bash
# Subset ERA5 hourly files to the SOM domain (30-54N, 100-60W).
#
# Copied from /mnt/drive2/ERA5/NC_files/slice.sh (which hard-coded tcwv); the
# exact command for every variable is recorded in each sliced file's
# `history` attribute:
#   cdo -O -f nc4 -z zip sellonlatbox,260,300,30,54 <var>/<file> hourly_sliced/<file>
# (theta-e is subset in create_thetae.py instead.)
#
# Run from /mnt/drive2/ERA5/NC_files:
#   bash slice.sh z500 mslp tcwv viwv          # default: tcwv
#   (tp files sat at the top level: era5_tp_NWHem_*.nc)
# Uses GNU parallel; keep -j modest on the shared server.

mkdir -p hourly_sliced

slice_file() {
    file="$1"
    base=$(basename "$file")
    out="hourly_sliced/${base}"

    if [[ -f "$out" ]]; then
        echo "Skipping $file (already exists)"
        return
    fi

    echo "Slicing $file -> $out"
    cdo -O -f nc4 -z zip sellonlatbox,260,300,30,54 "$file" "$out"
}

export -f slice_file

vars=("${@:-tcwv}")
for v in "${vars[@]}"; do
    parallel -j "${NJOBS:-8}" --line-buffer slice_file ::: "${v}"/era5_"${v}"_*.nc
    echo "All ${v} files have been sliced and saved to hourly_sliced/"
done
