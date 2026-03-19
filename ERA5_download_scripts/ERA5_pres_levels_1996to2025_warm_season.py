import os

import cdsapi

years = list(range(1996, 2026))
dataset = "reanalysis-era5-pressure-levels"
base_outdir = "/mnt/drive2/ERA5/NC_files"

# Variable lookup: short name -> CDS API variable name(s) and pressure level
VARIABLES = {
    "thetae": {"cds_names": ["specific humidity", "temperature"], "level": "850"},
    "z500":   {"cds_names": ["geopotential"], "level": "500"},
}

# ---- Edit this to select the variable to download ----
var_name = "thetae"
# -------------------------------------------------------

cds_names = VARIABLES[var_name]["cds_names"]
level = VARIABLES[var_name]["level"]
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
        "pressure_level": [level],
    }

    client.retrieve(dataset, request).download(outfile)
