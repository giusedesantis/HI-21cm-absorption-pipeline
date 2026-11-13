from pathlib import Path
import pandas as pd
import numpy as np
from config_utils import load_config

# --------------------------------------------------
# Directories
# --------------------------------------------------

config = load_config()
settings = config["post_imaging"]["resampling"]

input_dir = Path(settings["input_dir"])
output_dir = Path(settings["output_dir"])

# Verify the input directory exists
if not input_dir.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

# Create the output directory if necessary
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

if not spec_files:
    raise FileNotFoundError(f"No *_spec.csv files found in {input_dir}")

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