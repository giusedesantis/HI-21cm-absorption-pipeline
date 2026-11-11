from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# Directories
# --------------------------------------------------

input_dir = Path("/scratch/mwasci/giuse/spectral_cubes_0856_40kHz")

output_dir = Path(
    "/scratch/mwasci/giuse/spectral_cubes_0856_10kHz_resample"
)

output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Column names
# --------------------------------------------------

FREQ_COL = "Central Frequency(MHz)"
FLUX_COL = "flux_Jy"
RMS_COL = "ann_rms_jy"

# Original channel width = 20 kHz = 0.020 MHz
# New channel centres are displaced by +/- 5 kHz
FREQ_OFFSET_MHZ = 0.005

# --------------------------------------------------
# Find all raw spectrum files
# --------------------------------------------------

spec_files = sorted(input_dir.glob("*_spec.csv"))

print(f"Found {len(spec_files)} spectra")

# --------------------------------------------------
# Resample each spectrum
# --------------------------------------------------

for file in spec_files:

    df = pd.read_csv(file)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Check required columns exist
    required_cols = [FREQ_COL, FLUX_COL, RMS_COL]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"{file.name}: missing columns {missing_cols}"
        )

    # Convert the required columns to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Number of original channels
    n_channels = len(df)

    # --------------------------------------------------
    # Duplicate every row
    # --------------------------------------------------

    df_new = df.iloc[
        np.repeat(np.arange(n_channels), 2)
    ].copy().reset_index(drop=True)

    # --------------------------------------------------
    # Create the new frequency centres
    # --------------------------------------------------

    # For each original frequency:
    # first new channel  = original - 0.005 MHz
    # second new channel = original + 0.005 MHz

    offsets = np.tile(
        [-FREQ_OFFSET_MHZ, FREQ_OFFSET_MHZ],
        n_channels
    )

    df_new[FREQ_COL] = (
        df_new[FREQ_COL].to_numpy() + offsets
    )

    # --------------------------------------------------
    # Flux remains unchanged
    # --------------------------------------------------

    # Already duplicated from the original dataframe.
    # No additional operation is needed.

    # --------------------------------------------------
    # RMS increases by sqrt(2)
    # --------------------------------------------------

    df_new[RMS_COL] = (
        df_new[RMS_COL] * np.sqrt(2)
    )

    # --------------------------------------------------
    # Save the new spectrum
    # --------------------------------------------------

    output_file = output_dir / file.name

    df_new.to_csv(output_file, index=False)

    print(
        f"{file.name}: "
        f"{n_channels} -> {len(df_new)} channels"
    )

print("Done.")