# ERA5 preparation (before `preprocessing/`)

Order used to build `/mnt/drive2/ERA5/NC_files/hourly_sliced/` (inputs to the SOM preprocessing):

1. **Download** (May–Oct 1996–2025, hourly, 0–90°N, 180°W–0°): `ERA5_download_scripts/ERA5_pres_levels_1996to2025_warm_season.py`
   (z500; thetae = q + t at 850 hPa) and `ERA5_download_scripts/ERA5_single_levels_1996to2025_warm_season.py`
   (mslp, tcwv, viwv = vertical integral of water vapour flux, tp) → `/mnt/drive2/ERA5/NC_files/<var>/`.
2. **Subset to the SOM domain** (30–54°N, 100–60°W): `slice.sh` (CDO `sellonlatbox,260,300,30,54`) for z500, mslp,
   tcwv, viwv, tp → `hourly_sliced/`. The exact command is in each output file's `history` attribute.
3. **θe at 850 hPa**: `create_thetae.py` (MetPy; subsets the domain itself) → `hourly_sliced/era5_thetae_NWHem_<year>.nc`.
4. **Standardized anomalies and event files**: from the project root,
   `NYCFF_INTERMEDIATE=v2 python -m preprocessing.som_preprocessing <VAR> --no-plots --no-snapshots`
   for Z500, theta_e, ivt, tcwv, mslp, tp; then
   `NYCFF_INTERMEDIATE=v2 python -m preprocessing.som_preprocessing_evolution <VAR> [--n-hours N]` for Z500 and theta_e.
   (Or run `bash run_pipeline.sh --preprocess`.)

`slab_and_merge_legacy.sh` is kept for provenance only; the SOM pipeline does not use it.
