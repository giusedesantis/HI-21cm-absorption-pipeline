import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

from config_utils import load_config

# Load configuration
config = load_config()
settings = config["post_imaging"]["stacking"]

# Input directory
base = Path(settings["input_dir"])

# Output files
out_path = Path(settings["output_csv"])
plot_path = Path(settings["plot_file"])

# Check input directory
if not base.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {base}")

# Create output directories
out_path.parent.mkdir(parents=True, exist_ok=True)
plot_path.parent.mkdir(parents=True, exist_ok=True)

# Find individual spectra
files = sorted(base.glob("spectrum_*.csv"))

if not files:
    raise FileNotFoundError(f"No spectrum_*.csv files found in {base}")

# Column names
FLUX_COL = "Subtracted flux(Jy)"
RMS_COL = "rms (Jy)"
FREQ_COL = "Freq(MHz)"

# Frequency settings
FREQ_DECIMALS = 3

# Observed HI frequency
z = config["target"]["redshift"]
nu_obs = 1420.4058 / (1 + z)

dfs = []
for f in files:
    df = pd.read_csv(f)

    if len(df) % 2 != 0:
        obsid = f.stem.removeprefix("spectrum_")
        print(f"Odd-length spectrum: obsid={obsid}, rows={len(df)}")

    df = df.rename(columns=lambda c: c.strip())
    df = df[[FREQ_COL, FLUX_COL, RMS_COL]].copy()
    df["source_file"] = f.name

    # Drop rows only if they are missing/invalid. Keep rows with RMS == 0.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[FREQ_COL, FLUX_COL, RMS_COL])

    if FREQ_DECIMALS is not None:
        df[FREQ_COL] = df[FREQ_COL].round(FREQ_DECIMALS)

    # Safe weighting:
    # - RMS > 0  -> normal inverse-variance weight
    # - RMS <= 0 -> zero weight, so the row is kept but does not influence the stack
    df["weight"] = np.where(df[RMS_COL] > 0, 1.0 / (df[RMS_COL] ** 2), 0.0)
    dfs.append(df)

if not dfs:
    raise RuntimeError("No files matched spectrum_*.csv")

all_rows = pd.concat(dfs, ignore_index=True)


def wmean_and_err(x: pd.DataFrame) -> pd.Series:
    # x is a sub-dataframe for one frequency
    w = x["weight"].to_numpy(dtype=float)
    y = x[FLUX_COL].to_numpy(dtype=float)

    valid = w > 0
    n_total = len(x)
    n_weighted = int(np.sum(valid))
    n_zero_rms = int(np.sum(~valid))

    if n_weighted == 0:
        # Nothing can be weighted safely at this frequency.
        # Keep the channel and leave it at zero rather than dropping it.
        return pd.Series({
            "weighted_mean_flux_Jy": 0.0,
            "weighted_unc_Jy": 0.0,
            "N": n_total,
            "N_weighted": n_weighted,
            "N_zero_rms": n_zero_rms,
        })

    w_valid = w[valid]
    y_valid = y[valid]

    wmean = np.average(y_valid, weights=w_valid)
    werr = np.sqrt(1.0 / np.sum(w_valid))

    return pd.Series({
        "weighted_mean_flux_Jy": wmean,
        "weighted_unc_Jy": werr,
        "N": n_total,
        "N_weighted": n_weighted,
        "N_zero_rms": n_zero_rms,
    })


out = (
    all_rows
    .groupby(FREQ_COL, as_index=False)
    .apply(wmean_and_err)
)

# optional for saving output
out = out.sort_values(FREQ_COL, ignore_index=True)
out.to_csv(out_path, index=False)

data = pd.read_csv(out_path)
freqData = data[FREQ_COL].to_numpy()
fluxData = data['weighted_mean_flux_Jy'].to_numpy()
errorData = data["weighted_unc_Jy"].to_numpy()

last = freqData[-1]
first = freqData[0]
step = 1.28

start = first
n = int(np.floor((last - start) / step)) + 1
grid = start + step * np.arange(n)
grid = np.round(grid, 6)

fig = plt.figure(figsize=(20, 4))
plt.fill_between(
    freqData,
    -errorData,
    +errorData,
    step='mid',
    alpha=0.29,
    color='#0000EE',
    edgecolor='none',
    label='RMS'
)
plt.step(freqData, fluxData, where='mid', linewidth=0.62, color='k')
for _, f0 in enumerate(grid):
    plt.axvline(f0, color='dimgrey', ls='--', linewidth=1, alpha=0.39)
plt.axhline(0, color='k', lw=0.8, alpha=0.6)
plt.axvline(nu_obs, ls='--', color='brown', lw=1, alpha=0.65)
v_patch = Patch(facecolor='brown', alpha=0.35,
                label=r'$\nu_{\mathrm{obs}}$ of 21-cm line')
chan_handle = Patch(facecolor='dimgrey', alpha=0.35, label=r'coarse channel edges')
noise_handle = Patch(facecolor='#0000EE', alpha=0.3, label=r'RMS')
#plt.legend(handles=[v_patch, chan_handle, noise_handle])
plt.xlim(freqData[0], freqData[-1])
plt.xlabel('Frequency (MHz)')
plt.ylabel('Flux (Jy)')
plt.tight_layout()

plt.savefig(plot_path, dpi=300)
plt.close(fig)