import io
import os
import tarfile

import requests

years = list(range(2002, 2025))
months = ["05", "06", "07", "08", "09", "10"]
base_outdir = "/mnt/drive2/janoski/StageIV"
tmp_dir = os.path.join(base_outdir, "tmp")
base_url = "https://data.rda.ucar.edu/d507005/stage4"

# Minimum expected hourly files per month (conservative: 28 days * 24h)
MIN_EXPECTED_FILES = 672

os.makedirs(tmp_dir, exist_ok=True)

for year in years:
    for month in months:
        outdir = os.path.join(base_outdir, str(year), month)
        os.makedirs(outdir, exist_ok=True)

        # Skip if already downloaded
        existing = [f for f in os.listdir(outdir) if ".01h" in f]
        if len(existing) >= MIN_EXPECTED_FILES:
            print(
                f"Skipping {year}-{month} — {len(existing)} hourly files already exist."
            )
            continue

        tar_name = f"stage4.{year}{month}.tar"
        tar_url = f"{base_url}/{tar_name}"
        tar_path = os.path.join(tmp_dir, tar_name)

        print(f"Downloading {tar_name}...")
        try:
            resp = requests.get(tar_url, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        except Exception as e:
            print(f"  Download failed for {year}-{month}: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            continue

        # Monthly tar contains daily tars (e.g. ST4.20210901), each holding
        # the individual hourly GRIB files. Extract CONUS 01h files only.
        print(f"  Extracting hourly files for {year}-{month}...")
        count = 0
        try:
            with tarfile.open(tar_path, "r") as outer:
                for daily_member in outer.getmembers():
                    daily_data = outer.extractfile(daily_member)
                    if daily_data is None:
                        continue
                    with tarfile.open(fileobj=io.BytesIO(daily_data.read())) as inner:
                        for member in inner.getmembers():
                            basename = os.path.basename(member.name)
                            if ".01h" not in basename:
                                continue
                            if not (
                                basename.startswith("ST4.")
                                or basename.startswith("st4_conus.")
                            ):
                                continue
                            if basename.endswith(".gif"):
                                continue
                            dest = os.path.join(outdir, basename)
                            with open(dest, "wb") as out:
                                out.write(inner.extractfile(member).read()) # pyright: ignore[reportOptionalMemberAccess]
                            count += 1
            print(f"  Extracted {count} hourly files for {year}-{month}.")
        except Exception as e:
            print(f"  Extraction failed for {year}-{month}: {e}")

        # Clean up tar
        if os.path.exists(tar_path):
            os.remove(tar_path)

# Summary
print("\n=== Download Summary ===")
for year in years:
    for month in months:
        outdir = os.path.join(base_outdir, str(year), month)
        if os.path.isdir(outdir):
            n = len([f for f in os.listdir(outdir) if ".01h" in f])
            print(f"  {year}-{month}: {n} files")
