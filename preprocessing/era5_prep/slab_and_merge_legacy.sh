#!/bin/bash
# LEGACY (copied from /mnt/drive2/ERA5/NC_files/slab_and_merge.sh). Not used by the
# SOM pipeline: it merges six z500 files into a misnamed era5_pw_nwhem.nc for an
# older, larger (25-60N, 105-55W) exploratory domain. Kept for provenance only.
set -euo pipefail

files=(
  era5_z500_00061218.nc
  era5_z500_01071319.nc
  era5_z500_02081420.nc
  era5_z500_03091521.nc
  era5_z500_04101622.nc
  era5_z500_05111723.nc
)

# Build per-file sellonlatbox inputs then merge
cdo -O -f nc4 -z zip mergetime \
  $(for f in "${files[@]}"; do echo -n "-sellonlatbox,250,305,25,60 $f "; done) \
  era5_pw_nwhem.nc

echo "Wrote era5_pw_nwhem.nc (lat 25–50N, lon 260–295E)."
