import os

import cdsapi

years = list(range(1996, 2026))
dataset = "reanalysis-era5-single-levels"
base_outdir = "/mnt/drive2/ERA5/NC_files"

# Variable lookup: short name -> CDS API variable name(s)
VARIABLES = {
    "mslp": ["mean_sea_level_pressure"],
    "tp": ["total_precipitation"],
    "tcwv": ["total_column_water_vapour"],
    "viwv": [
        "vertical_integral_of_eastward_water_vapour_flux",
        "vertical_integral_of_northward_water_vapour_flux",
    ],
}

# ---- Edit this to select the variable to download ----
var_name = "tcwv"
# -------------------------------------------------------

cds_names = VARIABLES[var_name]
outdir = os.path.join(base_outdir, var_name)
os.makedirs(outdir, exist_ok=True)

client = cdsapi.Client()

for year in years:
    outfile = os.path.join(outdir, f"era5_{var_name}_NWHem_{year}.nc")

    if os.path.exists(outfile):
        print(f"Skipping {var_name} {year} — file already exists.")
        continue

    print(f"Downloading {var_name} for {year}...")
    request = {
        "product_type": ["reanalysis"],
        "variable": cds_names,
        "year": str(year),
        "month": ["05", "06", "07", "08", "09", "10"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [90, -180, 0, 0],
    }

    client.retrieve(dataset, request).download(outfile)
