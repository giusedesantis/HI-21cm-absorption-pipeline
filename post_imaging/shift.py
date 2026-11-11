import numpy as np
import pandas as pd
from pathlib import Path

#import barcorrect
#import source_find

base = Path("/scratch/mwasci/giuse/dip")
files = Path("/scratch/mwasci/giuse/spectral_cubes_0856_10kHz_resample")   #spectral_cubes_0856_10kHz_resample

corr = pd.read_csv(base / "corrections.csv", dtype={"obsid": str})
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