import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

output = Path("/scratch/mwasci/giuse/dip/final_spectra")

csv_file = "/scratch/mwasci/giuse/dip/boxfilter_output/box_754_20kHz_covariance.csv"

df = pd.read_csv(csv_file)

freq_col = "Freq(MHz)"    
flux_col = "boxcar_subtracted_flux_Jy"
rms_col = "boxcar_subtracted_unc_Jy"  

# Keep only the two columns needed for the spectrum
spec = df[[freq_col, flux_col, rms_col]].copy()

# Remove NaNs and sort by frequency
spec = spec.dropna().sort_values(freq_col)
nu_obs = 1420.4058 / (1 + 5.55)

# Get obsid automatically if the column exists
#if "obsid" in df.columns:
    #obsid = str(df["obsid"].iloc[0])
#else:
    #obsid = csv_file.stem.split("_")[0]

fig = plt.figure(figsize=(20, 4))

plt.step(
    spec[freq_col],
    spec[flux_col],
    where="mid",
    color='k',
    linewidth=0.62
)
# Coarse-channel edges
step = 1.28  # MHz

freq_min = spec[freq_col].min()
freq_max = spec[freq_col].max()

start = freq_min
n = int(np.floor((freq_max - start) / step)) + 1
grid = start + step * np.arange(n)
grid = np.round(grid, 6)

for f0 in grid:
    plt.axvline(
        f0,
        color="dimgrey",
        ls="--",
        linewidth=1,
        alpha=0.39
    )

plt.fill_between(
    spec[freq_col],
    -spec[rms_col],
    spec[rms_col],
    step='mid',
    alpha=0.29,
    color='#0000EE',
    edgecolor='none',
    label='RMS'
)

plt.axhline(0, color='k', lw=0.8, alpha=0.6)
plt.axvline(nu_obs, ls='--', color='brown', lw=1, alpha=0.65)
plt.xlim(spec[freq_col].min(), spec[freq_col].max())
plt.ylim(-0.15, 0.10)

plt.xlabel("Frequency [MHz]")
plt.ylabel("Flux density [Jy]")
# plt.title(f"Step-plot spectrum for obsid {obsid}")
plt.tight_layout()

plot_path = output / f"spectrum_754_20kHzResam_cov.png"
plt.savefig(plot_path, dpi=200)


# Zoom around the expected absorption line
# Find the coarse-channel boundary closest to the expected line
nearest_edge = grid[np.argmin(np.abs(grid - nu_obs))]

# Two coarse channels on each side
zoom_min = nearest_edge - 2 * step
zoom_max = nearest_edge + 2 * step

plt.xlim(zoom_min, zoom_max)

# Save zoomed spectrum
zoom_path = output / "spectrum_754_20kHzResam_cov_zoom.png"
plt.savefig(zoom_path, dpi=200)

print(f"Saved zoomed plot to: {zoom_path}")



print(f"Saved plot to: {plot_path}")

