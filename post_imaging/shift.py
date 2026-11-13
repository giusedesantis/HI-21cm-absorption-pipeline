import numpy as np
import pandas as pd
from pathlib import Path

from config_utils import load_config

# Load configuration
config = load_config()

# Directory containing resampled spectra
files = Path(config["post_imaging"]["resampling"]["output_dir"])

# Barycentric corrections generated previously
corrections_file = Path(config["post_imaging"]["barycentric_correct"]["output_file"])

# Verify inputs exist
if not files.is_dir():
    raise FileNotFoundError(f"Spectra directory not found: {files}")

if not corrections_file.is_file():
    raise FileNotFoundError(f"Barycentric corrections not found: {corrections_file}")

# Load corrections
corr = pd.read_csv(corrections_file, dtype={"obsid": str})

corr.columns = corr.columns.str.strip()
corr["obsid"] = corr["obsid"].astype(str).str.strip()

step = 0.010
for row in corr.itertuples(index=False):
    obsid = row.obsid
    #print(row._fields)
    fcorr = float(getattr(row, "_2"))   

    flux_csv = files / f"{obsid}_spec.csv"
    if flux_csv.exists():
        df = pd.read_csv(flux_csv)
        df.columns = df.columns.str.strip()
        df["Corrected Frequency (MHz)"] = df["Central Frequency(MHz)"] + step * np.rint(fcorr / step)            #df["Corrected Frequency (MHz)"] = df["Central Frequency(MHz)"] + step * np.rint(fcorr / step)
        
        print(f"\n{obsid}  (+{fcorr:.6f} MHz):")
        print(df[["Central Frequency(MHz)", "Corrected Frequency (MHz)"]])

        df.to_csv(files / f"{obsid}_spec_correct.csv", index=False)