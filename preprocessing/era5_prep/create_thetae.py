"""
Theta-e Production Script
By: Ty Janoski
Updated: 2/20/26

Copied into the repository from ~/nyc_flash_flood/create_thetae.py (A23).
Computes 850-hPa equivalent potential temperature (MetPy, Bolton 1980) from
ERA5 specific humidity and temperature (downloaded by
ERA5_download_scripts/ERA5_pres_levels_1996to2025_warm_season.py with
var_name = "thetae"), subset to the SOM domain (30-54N, 100-60W), and writes
one file per year to hourly_sliced/ for preprocessing/som_preprocessing.py.
"""

import os

import metpy.calc as mpcalc
import xarray as xr
from metpy.units import units

# Read in data
path = "/mnt/drive2/ERA5/NC_files/thetae"
out_dir = "/mnt/drive2/ERA5/NC_files/hourly_sliced"

for filename in os.listdir(path):
    if filename.endswith(".nc"):
        out_path = os.path.join(out_dir, filename)

        # Skip if already exists
        if os.path.exists(out_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Processing {filename}...")

        # Open the dataset
        ds = xr.open_dataset(os.path.join(path, filename)).sel(
            latitude=slice(54, 30), longitude=slice(-100, -60)
        )

        # Extract necessary variables
        t = ds["t"].metpy.quantify()
        q = ds["q"].metpy.quantify()
        p = ds["pressure_level"].metpy.quantify() * units.hPa

        # Calculate dewpoint from specific humidity
        dewpoint = mpcalc.dewpoint_from_specific_humidity(p, t, q)

        # Calculate theta-e
        theta_e = mpcalc.equivalent_potential_temperature(p, t, dewpoint)

        # Create new dataset with theta-e
        theta_e_ds = xr.Dataset(
            {
                "theta_e": (
                    ("valid_time", "latitude", "longitude"),
                    theta_e.squeeze().metpy.magnitude,
                )
            },
            coords={
                "valid_time": ds["valid_time"],
                "latitude": ds["latitude"],
                "longitude": ds["longitude"],
            },
        )
        theta_e_ds["theta_e"].attrs["units"] = str(theta_e.metpy.units)

        theta_e_ds.to_netcdf(out_path)
        print(f"Saved {filename}")
