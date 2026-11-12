#!/usr/bin/env python3
"""Flux-correct MWA spectra while preserving valid sigma-clipped line channels.

Input : <obsid>_spec_correct.csv, produced after barycentric shifting.
Output: spectrum_<obsid>.csv with the same frequency/flux/RMS columns required by
        safe_stacking.py.

Important: sigma clipping determines which channels fit the continuum; it does
NOT decide which otherwise valid channels are saved in the output spectrum.
Channels flagged by channel_flagged, zero-flux sentinels (configurable),
invalid values, or non-positive input RMS keep output flux=0 and RMS=0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from config_utils import load_config

config = load_config()
settings = config["post_imaging"]["flux_fitting"]

output = Path(settings["output_dir"])

S_max = 0.85282
alp = -0.98555
q_val = -0.5097

S_pow = 2.4943
alp_pow = -0.65984

TREAT_ZERO_INPUT_FLUX_AS_FLAGGED = True

POLYNOMIAL_DEGREE = 3
CLIP_SIGMA = 3
MAX_CLIP_ITER = 5


def curvepowlaw(freq, S_norm, alpha, q):
    return S_norm * ((freq / 150) ** alpha) * np.exp(
        q * np.power(np.log(freq / 150), 2)
    )


def powlaw(freq, S_norm, alpha):
    return S_norm * (freq / 150) ** alpha


def fit_poly_sigma_clip(x, y, degree=3, sigma=3, max_iter=5):
    """Your existing iterative sigma-clipped polynomial fit, unchanged."""
    coeff = np.polyfit(x, y, degree)
    mask = np.ones_like(x, dtype=bool)

    for _ in range(max_iter):
        y_hat = np.polyval(coeff, x)
        r = y - y_hat
        s = np.std(r[mask], ddof=1)

        if not np.isfinite(s) or s == 0:
            break

        new_mask = np.abs(r) <= sigma * s
        if np.array_equal(new_mask, mask):
            break

        mask = new_mask
        if mask.sum() < degree + 1:
            break

        coeff = np.polyfit(x[mask], y[mask], degree)

    return coeff, mask


def read_explicit_flags(df):
    """Interpret channel_flagged=1/True as flagged, if supplied."""
    if "channel_flagged" not in df.columns:
        return np.zeros(len(df), dtype=bool)

    raw = df["channel_flagged"]
    values = raw.astype("string").str.strip().str.lower()
    flags = values.isin(["1", "1.0", "true", "t", "yes"])
    known = values.isin(["1", "1.0", "true", "t", "yes",
                         "0", "0.0", "false", "f", "no", "", "nan", "<na>"])
    if (~known).any():
        example = raw.loc[~known].iloc[0]
        raise ValueError(f"Unrecognised channel_flagged value: {example!r}")
    return flags.fillna(False).to_numpy(dtype=bool)


def process_file(file: Path, output_dir: Path):
    df = pd.read_csv(file).copy()
    df.columns = df.columns.str.strip()

    required = ["Corrected Frequency (MHz)", "flux_Jy", "ann_rms_jy"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{file.name}: missing columns {missing}")

    x_all = pd.to_numeric(df[required[0]], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(df[required[1]], errors="coerce").to_numpy(dtype=float)
    rms_all = pd.to_numeric(df[required[2]], errors="coerce").to_numpy(dtype=float)

    input_flags = read_explicit_flags(df)
    zero_flux_flags = ((y_all == 0.0) if TREAT_ZERO_INPUT_FLUX_AS_FLAGGED
                       else np.zeros(len(df), dtype=bool))

    # These channels must not influence the fit OR the subsequent stack.
    # In particular, flagged 40-kHz edge channels stay zero flux / zero RMS.
    invalid_input = (
        input_flags
        | zero_flux_flags
        | ~np.isfinite(x_all)
        | (x_all <= 0)  # necessary for the curved power-law model
        | ~np.isfinite(y_all)
        | ~np.isfinite(rms_all)
        | (rms_all <= 0)
    )
    valid = ~invalid_input

    n = len(df)
    observed_flux_out = np.zeros(n, dtype=float)
    observed_fit_out = np.zeros(n, dtype=float)
    model_fit_out = np.zeros(n, dtype=float)
    corrected_flux_out = np.zeros(n, dtype=float)
    subtracted_flux_out = np.zeros(n, dtype=float)
    rms_corr_out = np.zeros(n, dtype=float)
    sigma_clipped_from_fit = np.zeros(n, dtype=bool)
    invalid_final = invalid_input.copy()

    # Preserve uncorrected flux for valid channels, even if sigma-clipped.
    observed_flux_out[valid] = y_all[valid]

    if valid.sum() >= POLYNOMIAL_DEGREE + 1:
        valid_indices = np.flatnonzero(valid)
        x_valid = x_all[valid]
        y_valid = y_all[valid]
        rms_valid = rms_all[valid]

        # Sigma clipping is used ONLY to estimate the polynomial continuum.
        coeff, sigma_mask = fit_poly_sigma_clip(
            x_valid, y_valid, degree=POLYNOMIAL_DEGREE,
            sigma=CLIP_SIGMA, max_iter=MAX_CLIP_ITER
        )
        sigma_clipped_from_fit[valid_indices[~sigma_mask]] = True

        # Apply the continuum fit/model to *all* valid channels, including
        # the sigma-clipped ones. This preserves potential absorption lines.
        y_hat = np.polyval(coeff, x_valid)
        mod = curvepowlaw(x_valid, S_max, alp, q_val)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = mod / y_hat
            s_corr = y_valid * ratio
            rms_corr = np.abs(ratio) * rms_valid
            s_sub = s_corr - mod

        # Pathological model predictions must not acquire positive weights.
        corrected_ok = (
            np.isfinite(y_hat) & (y_hat > 0)
            & np.isfinite(mod) & np.isfinite(ratio)
            & np.isfinite(s_corr) & np.isfinite(s_sub)
            & np.isfinite(rms_corr) & (rms_corr > 0)
        )
        keep_indices = valid_indices[corrected_ok]
        rejected_indices = valid_indices[~corrected_ok]
        invalid_final[rejected_indices] = True

        observed_fit_out[keep_indices] = y_hat[corrected_ok]
        model_fit_out[keep_indices] = mod[corrected_ok]
        corrected_flux_out[keep_indices] = s_corr[corrected_ok]
        subtracted_flux_out[keep_indices] = s_sub[corrected_ok]
        rms_corr_out[keep_indices] = rms_corr[corrected_ok]
        observed_flux_out[rejected_indices] = 0.0
    else:
        # Insufficient data to fit a cubic: zero weight for the entire file.
        invalid_final[:] = True
        observed_flux_out[:] = 0.0
        print(f"WARNING {file.name}: insufficient valid channels for cubic fit; "
              "output flux/RMS set to zero.")

    # Identical column names to the old file, plus diagnostic columns.
    # safe_stacking.py selects only Freq(MHz), Subtracted flux(Jy), rms (Jy).
    out_df = pd.DataFrame({
        "Freq(MHz)": x_all,
        "Observed flux(Jy)": observed_flux_out,
        "Observed fit flux(Jy)": observed_fit_out,
        "Model fit flux(Jy)": model_fit_out,
        "Corrected flux(Jy)": corrected_flux_out,
        "Subtracted flux(Jy)": subtracted_flux_out,
        "rms (Jy)": rms_corr_out,
        "channel_flagged": invalid_final.astype(int),
        "sigma_clipped_from_fit": sigma_clipped_from_fit.astype(int),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"spectrum_{file.stem.split('_')[0]}.csv"
    out_df.to_csv(out_path, index=False)

    # Defensive checks: any invalid/flagged channels must have zero flux/RMS.
    assert np.all(subtracted_flux_out[invalid_final] == 0)
    assert np.all(rms_corr_out[invalid_final] == 0)
    print(f"{file.name} -> {out_path.name}: "
          f"{np.count_nonzero(~invalid_final)} valid, "
          f"{np.count_nonzero(invalid_final)} flagged/invalid, "
          f"{np.count_nonzero(sigma_clipped_from_fit & ~invalid_final)} "
          "sigma-clipped from fit but retained in corrected spectrum")
    return out_df


def main():
    files = sorted(output.glob("*spec_correct.csv"))
    print(f"Found {len(files)} corrected spectra")
    for f in files:
        process_file(f, output)
    print("Done.")


if __name__ == "__main__":
    main()
