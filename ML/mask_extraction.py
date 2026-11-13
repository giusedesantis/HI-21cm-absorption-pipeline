#!/usr/bin/env python3
"""
Build spectral cubes exactly as before, but extract each fine-channel spectrum
using the source pixels predicted on the MFS image by source_unet_mask_heatmap_32.py.

For each obsid this script:
  1. Finds the saved U-Net source-mask NPZ for that obsid.
  2. Builds the fine-channel cube from *XX-image.fits files (excluding MFS-XX).
  3. Maps/validates the saved MFS source pixels onto the cube spatial grid.
  4. Applies the SAME source mask to every fine-channel plane.
  5. Sums the source-mask pixels, optionally subtracting a local background.
  6. Converts the summed Jy/beam-pixels to Jy using the per-channel pixels/beam.
  7. Estimates a local RMS from a background ring around the source mask and
     converts it to the integrated-flux uncertainty over the masked area.
  8. Also saves the sigma-clipped RMS of the source pixels themselves as a
     diagnostic (this is NOT the preferred noise estimate because source
     structure contributes to it).
  9. Writes one CSV per obsid and keeps the baseline-subtracted spectral RMS
     calculation from the previous extraction workflow.

IMPORTANT
---------
The U-Net inference output must contain, at minimum:
    mask_y_full, mask_x_full
and preferably:
    mask_ra_deg, mask_dec_deg, full_image_shape
as produced by source_unet_mask_heatmap_32.py.

The source mask is fixed across all channel planes. This assumes the source
footprint defined by the MFS image is appropriate across the fine-channel band.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
import astropy.units as u


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
ROOT = Path("/scratch/mwasci/giuse/dip/MLtest/train_images")
OUTPUT = Path("/scratch/mwasci/giuse/dip/MLtest/test_cubes_spec")

# Directory containing the *_source_mask.npz files written by U-Net inference.
# Example:
# MASK_DIR = Path("/scratch/mwasci/giuse/dip/MLtest/full_masks")
MASK_DIR = Path("/scratch/mwasci/giuse/dip/MLtest/labels")

OUTPUT.mkdir(parents=True, exist_ok=True)

# Fine-channel naming
CHANNEL_GLOB = "*XX-image.fits"
MFS_SUFFIX = "MFS-XX-image.fits"

# Cube output naming
CUBE_SUFFIX = "_cube.fits"

# Mask-based local-background settings.
# The background ring is made by dilating the irregular source mask.
# Pixels within BKG_INNER_DILATE_PIX of the source are excluded, and the ring
# extends out to BKG_OUTER_DILATE_PIX from the source boundary.
BKG_INNER_DILATE_PIX = 2
BKG_OUTER_DILATE_PIX = 8
MIN_BKG_PIXELS = 20
DO_BKG_SUB = True

# Grid-alignment checks.
# If the saved MFS mask sky coordinates do not land at the same cube pixels
# within this tolerance, the code remaps the mask through RA/Dec.
MASK_WCS_TOL_PIX = 0.10
CHANNEL_WCS_TOL_PIX = 0.05


# --------------------------------------------------
# FITS / WCS helpers
# --------------------------------------------------
def read_2d_image_and_wcs(fits_path: Path):
    """Return 2D image array (float32), celestial WCS, and header."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    while data is not None and data.ndim > 2:
        data = data[0]

    if data is None or data.ndim != 2:
        raise ValueError(
            f"Not a 2D image after collapsing: {fits_path} "
            f"(shape={None if data is None else data.shape})"
        )

    img = np.asarray(data, dtype=np.float32)
    wcs = WCS(header).celestial
    return img, wcs, header


def central_freq_hz(header) -> float:
    """Central frequency in Hz from CRVAL3."""
    if "CRVAL3" in header:
        return float(header["CRVAL3"])
    return float("nan")


def beam_area_in_pixels(header) -> float:
    """
    Return pixels per beam for a Jy/beam image.

    Gaussian beam area = pi/(4 ln 2) * BMAJ * BMIN.
    Dividing by the pixel area gives pixels/beam.
    """
    bmaj = header.get("BMAJ", None)
    bmin = header.get("BMIN", None)
    cdelt1 = header.get("CDELT1", None)
    cdelt2 = header.get("CDELT2", None)

    if bmaj is None or bmin is None or cdelt1 is None or cdelt2 is None:
        return float("nan")

    beam_area_deg2 = (np.pi / (4.0 * np.log(2.0))) * float(bmaj) * float(bmin)
    pix_area_deg2 = abs(float(cdelt1) * float(cdelt2))
    if not np.isfinite(pix_area_deg2) or pix_area_deg2 <= 0:
        return float("nan")

    return beam_area_deg2 / pix_area_deg2


def channel_is_flagged(beam_info: dict, k: int) -> bool:
    """Flag channels with invalid per-channel beam metadata."""
    bmaj = beam_info["bmaj_deg"][k]
    bmin = beam_info["bmin_deg"][k]
    pixpb = beam_info["pix_per_beam"][k]

    return (
        (not np.isfinite(bmaj)) or (bmaj <= 0) or
        (not np.isfinite(bmin)) or (bmin <= 0) or
        (not np.isfinite(pixpb)) or (pixpb <= 0)
    )


def read_beam_table(cube_path: Path):
    """Read per-channel beam metadata from the cube BEAMS table."""
    with fits.open(cube_path) as hdul:
        if "BEAMS" not in hdul:
            raise KeyError(f"No BEAMS extension found in {cube_path}")

        tab = hdul["BEAMS"].data
        return {
            "chan": np.asarray(tab["CHAN"], dtype=np.int64),
            "freq_hz": np.asarray(tab["FREQ_HZ"], dtype=np.float64),
            "bmaj_deg": np.asarray(tab["BMAJ_DEG"], dtype=np.float64),
            "bmin_deg": np.asarray(tab["BMIN_DEG"], dtype=np.float64),
            "bpa_deg": np.asarray(tab["BPA_DEG"], dtype=np.float64),
            "cdelt1_deg": np.asarray(tab["CDELT1_DEG"], dtype=np.float64),
            "cdelt2_deg": np.asarray(tab["CDELT2_DEG"], dtype=np.float64),
            "pix_per_beam": np.asarray(tab["PIX_PER_BEAM"], dtype=np.float64),
        }


def spatial_wcs_max_delta_pix(reference_wcs: WCS, other_wcs: WCS, shape: tuple[int, int]) -> float:
    """
    Compare two celestial WCS grids at centre + four interior corners.

    Returns the maximum displacement, in pixels, when positions from the
    reference grid are transformed into the other grid.
    """
    ny, nx = shape
    xs = np.asarray([0.25 * (nx - 1), 0.75 * (nx - 1),
                     0.25 * (nx - 1), 0.75 * (nx - 1), 0.5 * (nx - 1)])
    ys = np.asarray([0.25 * (ny - 1), 0.25 * (ny - 1),
                     0.75 * (ny - 1), 0.75 * (ny - 1), 0.5 * (ny - 1)])

    sky = reference_wcs.pixel_to_world(xs, ys)
    xo, yo = other_wcs.world_to_pixel(sky)
    dr = np.hypot(np.asarray(xo) - xs, np.asarray(yo) - ys)
    if not np.all(np.isfinite(dr)):
        return float("inf")
    return float(np.max(dr))


# --------------------------------------------------
# Source-mask helpers
# --------------------------------------------------
def index_mask_files(mask_dir: Path) -> dict[str, Path]:
    """
    Build obsid -> source-mask-file lookup.

    U-Net inference writes names such as:
        <obsid>_<MFS stem>_source_mask.npz
    """
    if not mask_dir.exists():
        raise FileNotFoundError(f"MASK_DIR does not exist: {mask_dir}")

    mapping: dict[str, Path] = {}
    for fp in sorted(mask_dir.rglob("*.npz")):    #_source_mask.npz
        obsid = fp.name.split("_", 1)[0]
        if obsid in mapping:
            raise RuntimeError(
                f"More than one source-mask file found for obsid {obsid}:\n"
                f"  {mapping[obsid]}\n  {fp}"
            )
        mapping[obsid] = fp

    if not mapping:
        raise RuntimeError(f"No *_source_mask.npz files found under {mask_dir}")

    return mapping


def _unique_valid_indices(y: np.ndarray, x: np.ndarray, shape: tuple[int, int]):
    """Remove out-of-bounds and duplicate (y,x) pixels."""
    ny, nx = shape
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)

    valid = (y >= 0) & (y < ny) & (x >= 0) & (x < nx)
    y = y[valid]
    x = x[valid]

    if y.size == 0:
        return y, x

    flat = np.ravel_multi_index((y, x), dims=shape)
    _, first = np.unique(flat, return_index=True)
    first = np.sort(first)
    return y[first], x[first]


def load_source_mask_for_cube(mask_path: Path, cube_shape: tuple[int, int], cube_wcs: WCS):
    """
    Load the MFS U-Net source mask and map it to the cube spatial grid.

    Direct saved full-image indices are used when they agree with the cube WCS.
    If the grids differ, saved mask RA/Dec values are remapped to cube pixels.
    """
    with np.load(mask_path, allow_pickle=False) as z:
        if "mask_y_full" not in z or "mask_x_full" not in z:
            raise KeyError(
                f"{mask_path} does not contain mask_y_full/mask_x_full. "
                "Run inference with the updated U-Net script first."
            )

        y_saved = np.asarray(z["mask_y_full"], dtype=np.int64)
        x_saved = np.asarray(z["mask_x_full"], dtype=np.int64)

        saved_shape = tuple(np.asarray(z["full_image_shape"], dtype=int).tolist()) \
            if "full_image_shape" in z else None

        ra = np.asarray(z["mask_ra_deg"], dtype=np.float64) if "mask_ra_deg" in z else None
        dec = np.asarray(z["mask_dec_deg"], dtype=np.float64) if "mask_dec_deg" in z else None

        x_final = float(z["x_final"]) if "x_final" in z else float("nan")
        y_final = float(z["y_final"]) if "y_final" in z else float("nan")
        xw = float(z["xw"]) if "xw" in z else float("nan")
        yw = float(z["yw"]) if "yw" in z else float("nan")

    if x_saved.size == 0:
        raise RuntimeError(f"Saved source mask is empty: {mask_path}")

    mapping_mode = "direct_indices"
    max_wcs_delta_pix = float("nan")

    can_check_world = (
        ra is not None and dec is not None and
        len(ra) == len(x_saved) and len(dec) == len(y_saved) and
        len(ra) > 0
    )

    direct_shape_ok = (saved_shape is None) or (tuple(saved_shape) == tuple(cube_shape))

    if direct_shape_ok and can_check_world:
        sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        x_world, y_world = cube_wcs.world_to_pixel(sky)
        delta = np.hypot(np.asarray(x_world) - x_saved, np.asarray(y_world) - y_saved)
        finite = np.isfinite(delta)
        max_wcs_delta_pix = float(np.max(delta[finite])) if np.any(finite) else float("inf")

        if (not np.isfinite(max_wcs_delta_pix)) or max_wcs_delta_pix > MASK_WCS_TOL_PIX:
            mapping_mode = "radec_remap"
            y_use = np.rint(y_world).astype(np.int64)
            x_use = np.rint(x_world).astype(np.int64)
        else:
            y_use = y_saved
            x_use = x_saved

    elif direct_shape_ok:
        # No world coordinates were saved, but image dimensions agree.
        y_use = y_saved
        x_use = x_saved
        mapping_mode = "direct_indices_no_wcs_check"

    else:
        if not can_check_world:
            raise RuntimeError(
                f"Mask MFS shape {saved_shape} differs from cube shape {cube_shape}, "
                "and no saved mask RA/Dec coordinates are available for remapping."
            )
        sky = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        x_world, y_world = cube_wcs.world_to_pixel(sky)
        y_use = np.rint(y_world).astype(np.int64)
        x_use = np.rint(x_world).astype(np.int64)
        mapping_mode = "radec_remap_shape_mismatch"

    y_use, x_use = _unique_valid_indices(y_use, x_use, cube_shape)
    if x_use.size == 0:
        raise RuntimeError(f"No valid mask pixels remain after mapping {mask_path} to cube grid")

    source_mask = np.zeros(cube_shape, dtype=bool)
    source_mask[y_use, x_use] = True

    meta = {
        "mask_path": str(mask_path),
        "mapping_mode": mapping_mode,
        "max_wcs_delta_pix": max_wcs_delta_pix,
        "n_mask_pix": int(x_use.size),
        "x_final": x_final,
        "y_final": y_final,
        "xw": xw,
        "yw": yw,
    }
    return source_mask, meta


def dilate_binary(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """8-connected binary dilation with NumPy only."""
    out = np.asarray(binary, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        new = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                new |= padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = new
    return out


def make_background_ring(source_mask: np.ndarray, inner_dilate: int, outer_dilate: int) -> np.ndarray:
    """Create an irregular annulus/ring around the source mask."""
    if inner_dilate < 0 or outer_dilate <= inner_dilate:
        raise ValueError("Require 0 <= inner_dilate < outer_dilate")

    inner = dilate_binary(source_mask, inner_dilate) if inner_dilate > 0 else source_mask.copy()
    outer = dilate_binary(source_mask, outer_dilate)
    return outer & (~inner)


def masked_flux_and_rms(
    plane: np.ndarray,
    source_mask: np.ndarray,
    background_mask: np.ndarray,
    pix_per_beam: float,
    do_bkg_sub: bool = True,
):
    """
    Extract integrated source flux and local RMS from one fine-channel plane.

    Flux:
        sum(source pixels - local background median) / pixels_per_beam

    Preferred integrated-flux RMS:
        local_background_rms_Jy/beam * sqrt(N_source_pixels / pixels_per_beam)

    The sigma-clipped RMS *inside* the source pixels is also returned as a
    diagnostic, but should not normally be interpreted as the thermal/image
    noise because real source structure contributes to the pixel variation.
    """
    src_vals = np.asarray(plane[source_mask], dtype=np.float64)
    src_vals = src_vals[np.isfinite(src_vals)]

    bkg_vals = np.asarray(plane[background_mask], dtype=np.float64)
    bkg_vals = bkg_vals[np.isfinite(bkg_vals)]

    n_src = int(src_vals.size)
    n_bkg = int(bkg_vals.size)

    if n_src == 0:
        return {
            "mask_sum_raw_jybeam_pix": float("nan"),
            "mask_sum_bkgsub_jybeam_pix": float("nan"),
            "flux_Jy": float("nan"),
            "mask_npix_valid": 0,
            "mask_nbeams": float("nan"),
            "bkg_npix": n_bkg,
            "bkg_mean_jybeam": float("nan"),
            "bkg_median_jybeam": float("nan"),
            "bkg_rms_jybeam": float("nan"),
            "mask_flux_rms_Jy": float("nan"),
            "source_pixel_mean_jybeam": float("nan"),
            "source_pixel_median_jybeam": float("nan"),
            "source_pixel_rms_jybeam": float("nan"),
            "source_pixel_rms_scaled_Jy": float("nan"),
        }

    raw_sum = float(np.sum(src_vals))

    # Descriptive statistics of the source pixels themselves.
    src_mean, src_median, src_std = sigma_clipped_stats(
        src_vals, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std"
    )

    if n_bkg >= MIN_BKG_PIXELS:
        bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(
            bkg_vals, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std"
        )
        bkg_mean = float(bkg_mean)
        bkg_median = float(bkg_median)
        bkg_std = float(bkg_std)
    else:
        bkg_mean = float("nan")
        bkg_median = float("nan")
        bkg_std = float("nan")

    if do_bkg_sub and np.isfinite(bkg_median):
        bkgsub_sum = raw_sum - bkg_median * n_src
    else:
        bkgsub_sum = raw_sum

    if np.isfinite(pix_per_beam) and pix_per_beam > 0:
        n_beams = float(n_src / pix_per_beam)
        flux_jy = float(bkgsub_sum / pix_per_beam)

        # Preferred uncertainty: local background RMS scaled by independent beams.
        if np.isfinite(bkg_std):
            mask_flux_rms_jy = float(bkg_std * np.sqrt(n_beams))
        else:
            mask_flux_rms_jy = float("nan")

        # Diagnostic only: variation of values inside source mask, scaled similarly.
        if np.isfinite(src_std):
            src_rms_scaled_jy = float(src_std * np.sqrt(n_beams))
        else:
            src_rms_scaled_jy = float("nan")
    else:
        n_beams = float("nan")
        flux_jy = float("nan")
        mask_flux_rms_jy = float("nan")
        src_rms_scaled_jy = float("nan")

    return {
        "mask_sum_raw_jybeam_pix": raw_sum,
        "mask_sum_bkgsub_jybeam_pix": float(bkgsub_sum),
        "flux_Jy": flux_jy,
        "mask_npix_valid": n_src,
        "mask_nbeams": n_beams,
        "bkg_npix": n_bkg,
        "bkg_mean_jybeam": bkg_mean,
        "bkg_median_jybeam": bkg_median,
        "bkg_rms_jybeam": bkg_std,
        "mask_flux_rms_Jy": mask_flux_rms_jy,
        "source_pixel_mean_jybeam": float(src_mean),
        "source_pixel_median_jybeam": float(src_median),
        "source_pixel_rms_jybeam": float(src_std),
        "source_pixel_rms_scaled_Jy": src_rms_scaled_jy,
    }


# --------------------------------------------------
# Cube creation
# --------------------------------------------------
def find_channel_images(obsdir: Path):
    """
    Find XX fine-channel images, excluding the MFS-XX image, and sort by CRVAL3.
    """
    files = sorted([
        fp for fp in obsdir.glob(CHANNEL_GLOB)
        if fp.name.endswith("XX-image.fits") and not fp.name.endswith(MFS_SUFFIX)
    ])

    out = []
    for fp in files:
        with fits.open(fp) as hdul:
            hdr = hdul[0].header
            f = central_freq_hz(hdr)
        if np.isfinite(f):
            out.append((fp, float(f)))

    out.sort(key=lambda t: t[1])
    return out


def build_cube_fits(obsid: str, chan_list: list[tuple[Path, float]], cube_path: Path):
    """
    Build a 3D FITS cube and attach a BEAMS table with per-channel metadata.

    This preserves the previous cube-building approach, with one added safety
    check that all fine-channel images share the same spatial WCS grid.
    """
    if len(chan_list) == 0:
        raise RuntimeError("No channel images found to build cube.")

    first_fp, _ = chan_list[0]
    first_img, first_wcs, first_hdr = read_2d_image_and_wcs(first_fp)

    ny, nx = first_img.shape
    nchan = len(chan_list)
    freqs_hz = np.array([f for _, f in chan_list], dtype=np.float64)

    cube = np.zeros((nchan, ny, nx), dtype=np.float32)

    pix_per_beam_ch = np.full(nchan, np.nan, dtype=np.float64)
    bmaj_deg = np.full(nchan, np.nan, dtype=np.float64)
    bmin_deg = np.full(nchan, np.nan, dtype=np.float64)
    bpa_deg = np.full(nchan, np.nan, dtype=np.float64)
    cdelt1_deg = np.full(nchan, np.nan, dtype=np.float64)
    cdelt2_deg = np.full(nchan, np.nan, dtype=np.float64)

    for k, (fp, f) in enumerate(chan_list):
        img2d, wcs2d, hdr = read_2d_image_and_wcs(fp)

        if img2d.shape != (ny, nx):
            raise ValueError(f"Shape mismatch: {fp} has {img2d.shape}, expected {(ny, nx)}")

        # Same pixel indices must represent the same sky position in every plane.
        wcs_delta = spatial_wcs_max_delta_pix(first_wcs, wcs2d, (ny, nx))
        if wcs_delta > CHANNEL_WCS_TOL_PIX:
            raise ValueError(
                f"Spatial WCS mismatch in {fp}: max displacement {wcs_delta:.4f} pix "
                f"> tolerance {CHANNEL_WCS_TOL_PIX:.4f} pix"
            )

        cube[k, :, :] = img2d

        bmaj = hdr.get("BMAJ", np.nan)
        bmin = hdr.get("BMIN", np.nan)
        bpa = hdr.get("BPA", np.nan)
        cd1 = hdr.get("CDELT1", np.nan)
        cd2 = hdr.get("CDELT2", np.nan)

        bmaj_deg[k] = float(bmaj) if np.isfinite(bmaj) else np.nan
        bmin_deg[k] = float(bmin) if np.isfinite(bmin) else np.nan
        bpa_deg[k] = float(bpa) if np.isfinite(bpa) else np.nan
        cdelt1_deg[k] = float(cd1) if np.isfinite(cd1) else np.nan
        cdelt2_deg[k] = float(cd2) if np.isfinite(cd2) else np.nan
        pix_per_beam_ch[k] = beam_area_in_pixels(hdr)

    # Build 3D WCS header using first channel's spatial WCS.
    w3 = WCS(naxis=3)
    w2 = WCS(first_hdr).celestial

    w3.wcs.ctype[0] = w2.wcs.ctype[0]
    w3.wcs.ctype[1] = w2.wcs.ctype[1]
    w3.wcs.cunit[0] = w2.wcs.cunit[0]
    w3.wcs.cunit[1] = w2.wcs.cunit[1]
    w3.wcs.crval[0] = w2.wcs.crval[0]
    w3.wcs.crval[1] = w2.wcs.crval[1]
    w3.wcs.crpix[0] = w2.wcs.crpix[0]
    w3.wcs.crpix[1] = w2.wcs.crpix[1]
    w3.wcs.cdelt[0] = w2.wcs.cdelt[0]
    w3.wcs.cdelt[1] = w2.wcs.cdelt[1]

    w3.wcs.ctype[2] = "FREQ"
    w3.wcs.cunit[2] = "Hz"
    w3.wcs.crpix[2] = 1.0
    w3.wcs.crval[2] = freqs_hz[0]
    w3.wcs.cdelt[2] = float(np.median(np.diff(freqs_hz))) if nchan > 1 else 1.0

    hdr3 = w3.to_header()
    hdr3["NAXIS"] = 3
    hdr3["NAXIS1"] = nx
    hdr3["NAXIS2"] = ny
    hdr3["NAXIS3"] = nchan
    hdr3["BUNIT"] = first_hdr.get("BUNIT", "Jy/beam")

    finite_beam = np.isfinite(bmaj_deg) & np.isfinite(bmin_deg)
    if np.any(finite_beam):
        hdr3["BMAJ"] = float(np.nanmedian(bmaj_deg[finite_beam]))
        hdr3["BMIN"] = float(np.nanmedian(bmin_deg[finite_beam]))
        if np.any(np.isfinite(bpa_deg)):
            hdr3["BPA"] = float(np.nanmedian(bpa_deg[np.isfinite(bpa_deg)]))

    hdu0 = fits.PrimaryHDU(data=cube, header=hdr3)

    chan = np.arange(nchan, dtype=np.int32)
    cols = fits.ColDefs([
        fits.Column(name="CHAN",         format="J", array=chan),
        fits.Column(name="FREQ_HZ",      format="D", array=freqs_hz.astype(np.float64)),
        fits.Column(name="BMAJ_DEG",     format="D", array=bmaj_deg.astype(np.float64)),
        fits.Column(name="BMIN_DEG",     format="D", array=bmin_deg.astype(np.float64)),
        fits.Column(name="BPA_DEG",      format="D", array=bpa_deg.astype(np.float64)),
        fits.Column(name="CDELT1_DEG",   format="D", array=cdelt1_deg.astype(np.float64)),
        fits.Column(name="CDELT2_DEG",   format="D", array=cdelt2_deg.astype(np.float64)),
        fits.Column(name="PIX_PER_BEAM", format="D", array=pix_per_beam_ch.astype(np.float64)),
    ])
    hdu1 = fits.BinTableHDU.from_columns(cols, name="BEAMS")

    fits.HDUList([hdu0, hdu1]).writeto(cube_path, overwrite=True)
    return cube, freqs_hz, hdr3


# --------------------------------------------------
# Baseline-subtracted spectral RMS
# --------------------------------------------------
def add_spectral_baseline_rms_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the previous sigma-clipped linear spectral-baseline RMS calculation."""
    freq_mhz = df["Central Frequency(MHz)"].to_numpy(dtype=float)
    flux_jy = df["flux_Jy"].to_numpy(dtype=float)
    flagged = df["channel_flagged"].to_numpy(dtype=int) != 0

    # Important improvement: flagged channels are excluded even if their stored
    # flux value is 0.0 for compatibility with the previous CSV convention.
    good = np.isfinite(freq_mhz) & np.isfinite(flux_jy) & (~flagged)

    baseline_deg = 1
    rms_spec_jy = np.nan
    std_spec_jy_clipped = np.nan
    n_spec_total = int(np.sum(good))
    n_spec_used = 0
    baseline_coeffs = [np.nan] * (baseline_deg + 1)

    baseline_model = np.full_like(flux_jy, np.nan, dtype=float)
    residual_jy = np.full_like(flux_jy, np.nan, dtype=float)

    if np.sum(good) >= (baseline_deg + 2):
        x = freq_mhz[good]
        y = flux_jy[good]

        x0 = np.median(x)
        xs = x - x0
        keep = np.ones_like(y, dtype=bool)

        for _ in range(5):
            if np.sum(keep) < (baseline_deg + 2):
                break

            coeff = np.polyfit(xs[keep], y[keep], deg=baseline_deg)
            yfit = np.polyval(coeff, xs)
            resid = y - yfit

            clipped = sigma_clip(
                resid[keep], sigma=3.0, maxiters=1,
                cenfunc="median", stdfunc="std"
            )
            new_keep = keep.copy()
            new_keep_indices = np.where(keep)[0]
            new_keep[new_keep_indices] = ~clipped.mask

            if np.array_equal(new_keep, keep):
                keep = new_keep
                break
            keep = new_keep

        if np.sum(keep) >= (baseline_deg + 2):
            coeff = np.polyfit(xs[keep], y[keep], deg=baseline_deg)
            yfit = np.polyval(coeff, xs)
            resid = y - yfit

            resid_kept = resid[keep]
            med_resid = np.median(resid_kept)
            mad_resid = np.median(np.abs(resid_kept - med_resid))
            rms_spec_jy = 1.4826 * mad_resid
            std_spec_jy_clipped = np.std(resid_kept)

            n_spec_used = int(np.sum(keep))
            baseline_coeffs = coeff.tolist()
            baseline_model[good] = yfit
            residual_jy[good] = resid

    df["rms_spec_jy"] = rms_spec_jy
    df["std_spec_jy_clipped"] = std_spec_jy_clipped
    df["n_spec_total"] = n_spec_total
    df["n_spec_used_rms"] = n_spec_used
    df["baseline_deg"] = baseline_deg
    df["baseline_flux_jy"] = baseline_model
    df["residual_flux_jy"] = residual_jy
    df["baseline_x0_mhz"] = np.nanmedian(freq_mhz[good]) if np.any(good) else np.nan

    for i in range(baseline_deg + 1):
        df[f"baseline_coeff_{i}"] = baseline_coeffs[i]

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    obsid_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(obsid_dirs)} obsid directories under {ROOT}")

    mask_lookup = index_mask_files(MASK_DIR)
    print(f"Found {len(mask_lookup)} U-Net source masks under {MASK_DIR}")

    for obsdir in obsid_dirs:
        obsid = obsdir.name

        try:
            # A) find the U-Net source mask for this obsid
            mask_path = mask_lookup.get(obsid)
            if mask_path is None:
                print(f"[{obsid}] No U-Net source mask found, skipping.")
                continue

            # B) build cube exactly as in the previous workflow
            cube_path = OUTPUT / f"{obsid}{CUBE_SUFFIX}"
            chan_list = find_channel_images(obsdir)
            if len(chan_list) == 0:
                print(f"[{obsid}] No fine-channel XX images found, skipping cube/spectrum.")
                continue

            cube_data, freqs_hz, cube_hdr = build_cube_fits(obsid, chan_list, cube_path)
            beam_info = read_beam_table(cube_path)

            if len(beam_info["chan"]) != cube_data.shape[0]:
                raise RuntimeError(
                    f"BEAMS table length {len(beam_info['chan'])} does not match "
                    f"cube planes {cube_data.shape[0]}"
                )

            # C) map/validate saved MFS mask onto cube grid once for the whole obsid
            cube_wcs = WCS(cube_hdr).celestial
            cube_shape = tuple(cube_data.shape[1:])
            source_mask, mask_meta = load_source_mask_for_cube(mask_path, cube_shape, cube_wcs)

            background_mask = make_background_ring(
                source_mask,
                inner_dilate=BKG_INNER_DILATE_PIX,
                outer_dilate=BKG_OUTER_DILATE_PIX,
            )

            n_mask_pix_total = int(np.sum(source_mask))
            n_bkg_pix_total = int(np.sum(background_mask))

            print(
                f"[{obsid}] mask={mask_path.name} npix={n_mask_pix_total} "
                f"bkgpix={n_bkg_pix_total} mapping={mask_meta['mapping_mode']} "
                f"wcs_delta={mask_meta['max_wcs_delta_pix']:.4g} pix"
            )

            # D) apply the SAME source mask to every fine-channel plane
            rows = []

            for k in range(cube_data.shape[0]):
                plane = cube_data[k, :, :]
                bad_chan = channel_is_flagged(beam_info, k)

                common = {
                    "obsid": obsid,
                    "Central Frequency(MHz)": freqs_hz[k] / 1e6,
                    "channel_index": int(beam_info["chan"][k]),
                    "channel_flagged": int(bad_chan),
                    "pix_per_beam": beam_info["pix_per_beam"][k],
                    "BMAJ_deg": beam_info["bmaj_deg"][k],
                    "BMIN_deg": beam_info["bmin_deg"][k],
                    "BPA_deg": beam_info["bpa_deg"][k],
                    "CDELT1_deg": beam_info["cdelt1_deg"][k],
                    "CDELT2_deg": beam_info["cdelt2_deg"][k],
                    "mask_npix_total": n_mask_pix_total,
                    "background_ring_npix_total": n_bkg_pix_total,
                    "mask_npz": str(mask_path),
                    "mask_mapping_mode": mask_meta["mapping_mode"],
                    "mask_wcs_max_delta_pix": mask_meta["max_wcs_delta_pix"],
                    "x_final": mask_meta["x_final"],
                    "y_final": mask_meta["y_final"],
                    "xw": mask_meta["xw"],
                    "yw": mask_meta["yw"],
                    "cube_path": str(cube_path),
                }

                if bad_chan:
                    # Keep 0.0 convention for compatibility, but baseline fitting
                    # explicitly excludes channel_flagged == 1.
                    rows.append({
                        **common,
                        "flux_Jy": 0.0,
                        "mask_sum_raw_jybeam_pix": 0.0,
                        "mask_sum_bkgsub_jybeam_pix": 0.0,
                        "mask_npix_valid": 0,
                        "mask_nbeams": np.nan,
                        "bkg_npix": 0,
                        "bkg_mean_jybeam": np.nan,
                        "bkg_median_jybeam": np.nan,
                        "bkg_rms_jybeam": 0.0,
                        "mask_flux_rms_Jy": 0.0,
                        "source_pixel_mean_jybeam": np.nan,
                        "source_pixel_median_jybeam": np.nan,
                        "source_pixel_rms_jybeam": 0.0,
                        "source_pixel_rms_scaled_Jy": 0.0,
                        "plane_mean_jybeam": np.nan,
                        "plane_median_jybeam": np.nan,
                        "plane_std_jybeam": np.nan,
                    })
                    continue

                pixpb = float(beam_info["pix_per_beam"][k])
                ext = masked_flux_and_rms(
                    plane,
                    source_mask=source_mask,
                    background_mask=background_mask,
                    pix_per_beam=pixpb,
                    do_bkg_sub=DO_BKG_SUB,
                )

                plane_mean, plane_median, plane_std = sigma_clipped_stats(
                    plane, sigma=3.0, maxiters=5,
                    cenfunc="median", stdfunc="std"
                )

                rows.append({
                    **common,
                    **ext,
                    "plane_mean_jybeam": float(plane_mean),
                    "plane_median_jybeam": float(plane_median),
                    "plane_std_jybeam": float(plane_std),
                })

            df = pd.DataFrame(rows)
            df = add_spectral_baseline_rms_columns(df)

            out_csv = OUTPUT / f"{obsid}_spec.csv"
            df.to_csv(out_csv, index=False)

            print(
                f"[{obsid}] cube={cube_path.name} rows={len(df)} wrote {out_csv.name} "
                f"maskpix={n_mask_pix_total} mapping={mask_meta['mapping_mode']}"
            )

        except Exception as e:
            print(f"[{obsid}] failed: {e}")
            continue


if __name__ == "__main__":
    main()