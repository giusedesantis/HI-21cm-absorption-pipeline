#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

from centroid_utils import extract_cutout, local_centroid


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
ROOT = Path("/scratch/mwasci/giuse/small_cube_images_0856_40kHzflagchan")
output = Path("/scratch/mwasci/giuse/spectral_cubes_0856_40kHz")
output.mkdir(parents=True, exist_ok=True)

TARGET_COORD = SkyCoord("08h56m14.73s", "+02d23m59.6s", frame="fk5")
#TARGET_COORD = SkyCoord("09h24m19.92s", "-22d01m41.5s", frame="fk5")

CUTOUT_SIZE = 48
CENTROID_NSIGMA = 4.0
CENTROID_R_MAX = 8.0
PEAK_RADIUS = 6.0

# Aperture settings (pixels)
AP_R_FWHM_FACTOR = 0.7
AP_R_FALLBACK_PIX = 3.0
ANN_R_IN = 9.0
ANN_R_OUT = 13.0
DO_BKG_SUB = True

# Cube output naming
CUBE_SUFFIX = "_cube.fits"   # saved in output/<obsid>_cube.fits


# --------------------------------------------------
# FITS helpers
# --------------------------------------------------
def read_2d_image_and_wcs(fits_path: Path):
    """Return 2D image array (float32), celestial WCS, and header."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # collapse to 2D if needed
    while data is not None and data.ndim > 2:
        data = data[0]

    if data is None or data.ndim != 2:
        raise ValueError(f"Not a 2D image after collapsing: {fits_path} (shape={None if data is None else data.shape})")

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
    Return pixels per beam (pix/beam) for Jy/beam images using,
    BMAJ/BMIN (beam FWHM in degrees) and CDELT1/CDELT2 (pixel size in deg/pixel).
    Gaussian beam area = (pi / (4 ln(2))) * BMAJ * BMIN in deg^2.
    Images are in Jy/beam, but aperture sums are over many pixels,
    so the aperture sum is in approximately (Jy/beam) * pix. so diving by pix/beam gives Jy. 
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

def effective_pixel_scale_deg(cdelt1_deg: float, cdelt2_deg: float) -> float:
    """Return an effective pixel scale in deg/pix from possibly non-square pixels."""
    if not np.isfinite(cdelt1_deg) or not np.isfinite(cdelt2_deg):
        return float("nan")
    pix_area_deg2 = abs(float(cdelt1_deg) * float(cdelt2_deg))
    if pix_area_deg2 <= 0:
        return float("nan")
    return float(np.sqrt(pix_area_deg2))


def beam_fwhm_eff_pixels(bmaj_deg: float, bmin_deg: float, cdelt1_deg: float, cdelt2_deg: float) -> float:
    """Return the area-equivalent beam FWHM, sqrt(BMAJ*BMIN), in pixels."""
    if not np.isfinite(bmaj_deg) or not np.isfinite(bmin_deg):
        return float("nan")
    if bmaj_deg <= 0 or bmin_deg <= 0:
        return float("nan")

    pix_scale_deg = effective_pixel_scale_deg(cdelt1_deg, cdelt2_deg)
    if not np.isfinite(pix_scale_deg) or pix_scale_deg <= 0:
        return float("nan")

    fwhm_eff_deg = float(np.sqrt(float(bmaj_deg) * float(bmin_deg)))
    return fwhm_eff_deg / pix_scale_deg


def aperture_radius_from_beam(bmaj_deg: float, bmin_deg: float, cdelt1_deg: float, cdelt2_deg: float,
                              factor: float = AP_R_FWHM_FACTOR) -> float:
    """Return the circular aperture radius in pixels as factor × effective beam FWHM."""
    fwhm_eff_pix = beam_fwhm_eff_pixels(bmaj_deg, bmin_deg, cdelt1_deg, cdelt2_deg)
    if not np.isfinite(fwhm_eff_pix) or fwhm_eff_pix <= 0:
        return float("nan")
    return float(factor) * fwhm_eff_pix

def channel_is_flagged(beam_info: dict, k: int) -> bool:
    """
    Flag channels with invalid per-channel beam metadata.
    These are the channels we want to force to zero immediately.
    """
    bmaj = beam_info["bmaj_deg"][k]
    bmin = beam_info["bmin_deg"][k]
    pixpb = beam_info["pix_per_beam"][k]

    return (
        (not np.isfinite(bmaj)) or (bmaj <= 0) or
        (not np.isfinite(bmin)) or (bmin <= 0) or
        (not np.isfinite(pixpb)) or (pixpb <= 0)
    )

def read_beam_table(cube_path: Path):
    """
    Read per-channel beam metadata from the BEAMS table extension
    inside the cube FITS file.
    """
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

# --------------------------------------------------
# Photutils aperture
# --------------------------------------------------
def photutils_aperture_flux(img2d: np.ndarray, x0: float, y0: float,
                            ap_r: float,
                            ann_r_in: float | None = None,
                            ann_r_out: float | None = None,
                            do_bkg_sub: bool = True):
    """
    Photutils aperture photometry at float centre (x0,y0).
    Creating a circular aperture with CirccularAperture,
    computes the raw aperture sum using aperture_photometry, 
    and optionally estimates local background from a CircularAnnulus.
    Subtracts bkg_per_pix * ap_area_pix from aperture sum to get bkg-subtracted flux.
    If annulus fails (no finite pixels), it falls back to raw aperture sum 
    and sets background-related outputs to NaN/0.
    Returns dict:
      ap_sum_raw, ap_sum_bkgsub, bkg_per_pix, ap_area_pix, ann_npix
    """
    pos = [(x0, y0)]

    ap = CircularAperture(pos, r=ap_r)
    tab = aperture_photometry(img2d, ap, method="exact")
    ap_sum_raw = float(tab["aperture_sum"][0])
    ap_area_pix = float(ap.area)

    if (not do_bkg_sub) or (ann_r_in is None) or (ann_r_out is None):
        return {
            "ap_sum_raw": ap_sum_raw,
            "ap_sum_bkgsub": ap_sum_raw,
            "bkg_per_pix": float("nan"),
            "ap_area_pix": ap_area_pix,
            "ann_npix": 0,
            "ann_rms_jybeam": float("nan"),
        }

    ann = CircularAnnulus(pos, r_in=ann_r_in, r_out=ann_r_out)

    # Use annulus mask to collect annulus pixels
    m = ann.to_mask(method="center")[0]
    ann_data = m.multiply(img2d)
    ann_mask = m.data.astype(bool)
    ann_vals = ann_data[ann_mask]
    ann_vals = ann_vals[np.isfinite(ann_vals)]

    if ann_vals.size == 0:
        return {
            "ap_sum_raw": ap_sum_raw,
            "ap_sum_bkgsub": ap_sum_raw,
            "bkg_per_pix": float("nan"),
            "ap_area_pix": ap_area_pix,
            "ann_npix": 0,
            "ann_rms_jybeam": float("nan"),
        }

    ####new part:
    _, _, ann_rms_jybeam = sigma_clipped_stats(ann_vals, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
    ####

    bkg_per_pix = float(np.median(ann_vals))
    ap_sum_bkgsub = ap_sum_raw - bkg_per_pix * ap_area_pix

    return {
        "ap_sum_raw": ap_sum_raw,
        "ap_sum_bkgsub": float(ap_sum_bkgsub),
        "bkg_per_pix": bkg_per_pix,
        "ap_area_pix": ap_area_pix,
        "ann_npix": int(ann_vals.size),
        "ann_rms_jybeam": float(ann_rms_jybeam),
    }


# --------------------------------------------------
# Hybrid locator helpers
# --------------------------------------------------
def cutout_peak_near_center(cut: np.ndarray, radius: float) -> tuple[float, float]:
    """Last resort fallback: defines a circular shape around the cutout centre,
    find peak pixel within radius of cutout centre, and return its offset (dx, dy) from centre.
    When centroid fails.
    """
    size = cut.shape[0]
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0

    yy, xx = np.indices(cut.shape)
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    win = rr2 <= radius * radius

    masked = np.where(win, cut, -np.inf)
    if not np.isfinite(masked).any():
        return 0.0, 0.0

    flat = np.nanargmax(masked)
    y_peak, x_peak = np.unravel_index(flat, cut.shape)
    dx = float(x_peak - cx)
    dy = float(y_peak - cy)
    return dx, dy


def choose_pixel_hybrid(mfs_img: np.ndarray, mfs_wcs: WCS):
    """This is the core source-positioning function. The steps are:
    1) Convert TARGET_COORD to pixel coordinates (xw, yw) on the MFS image,
    2) Extract cutout around (xw, yw) with extract_cutout,
    3) Run local_centroid on the cutout. If successful and within CENTROID_R_MAX, use it,
    4) Otherwise, run cutout_peak_near_center as the fallback,
    5) Convert final (dx, dy) to absolute pixel coordinates (x_final, y_final),
    6) Build a diagnostic dict with all relevant info for later inspection.
    """
    xw, yw = mfs_wcs.world_to_pixel(TARGET_COORD)
    cut = extract_cutout(mfs_img, xw, yw, size=CUTOUT_SIZE)

    cres = local_centroid(cut, nsigma=CENTROID_NSIGMA)
    if cres.success and np.isfinite(cres.r_pix) and cres.r_pix <= CENTROID_R_MAX:
        dx, dy = cres.dx, cres.dy
        method = "centroid"
        diag = {"r": cres.r_pix, "snr_peak": cres.snr_peak, "reason": ""}
    else:
        dx, dy = cutout_peak_near_center(cut, radius=PEAK_RADIUS)
        method = "peak"
        diag = {"r": float(np.hypot(dx, dy)), "snr_peak": getattr(cres, "snr_peak", np.nan), "reason": getattr(cres, "reason", "centroid_fail")}

    x_final = float(xw + dx)
    y_final = float(yw + dy)

    diag.update({
        "xw": float(xw), "yw": float(yw),
        "dx": float(dx), "dy": float(dy),
        "x_final": x_final, "y_final": y_final,
        "x_round": int(np.rint(x_final)),
        "y_round": int(np.rint(y_final)),
    })
    return x_final, y_final, method, diag


# --------------------------------------------------
# Cube creation
# --------------------------------------------------
def find_channel_images(obsdir: Path):
    """
    Find all channel images in obsdir matching "*image.fits" but not "*MFS-image.fits",
    read each candidate's header to get central frequency (CRVAL3),
    and return a sorted list of (filepath, freq_hz).
    This ensures the cube planes are stacked in correct frequency order, not just alphabetical filename order.
    """
    files = sorted([fp for fp in obsdir.glob("*XX-image.fits")
                    if fp.name.endswith("XX-image.fits") and not fp.name.endswith("MFS-XX-image.fits")])  #before it was "image.fits" and "MFS-image.fits"

    out = []
    for fp in files:
        with fits.open(fp) as hdul:
            hdr = hdul[0].header
            f = central_freq_hz(hdr)
        if np.isfinite(f):
            out.append((fp, float(f)))
    # sort by frequency
    out.sort(key=lambda t: t[1])
    return out

def build_cube_fits(obsid: str, chan_list: list[tuple[Path, float]], cube_path: Path):
    """
    Build a 3D FITS cube and attach a BEAMS table extension containing
    per-channel beam metadata.

    Returns:
        cube_data, freqs_hz, cube_header
    """
    if len(chan_list) == 0:
        raise RuntimeError("No channel images found to build cube.")

    # Use first channel for spatial WCS
    first_fp, _ = chan_list[0]
    first_img, first_wcs, first_hdr = read_2d_image_and_wcs(first_fp)

    ny, nx = first_img.shape
    nchan = len(chan_list)
    freqs_hz = np.array([f for _, f in chan_list], dtype=np.float64)

    # Stack data
    cube = np.zeros((nchan, ny, nx), dtype=np.float32)

    # Per-channel metadata arrays
    pix_per_beam_ch = np.full(nchan, np.nan, dtype=np.float64)
    bmaj_deg = np.full(nchan, np.nan, dtype=np.float64)
    bmin_deg = np.full(nchan, np.nan, dtype=np.float64)
    bpa_deg = np.full(nchan, np.nan, dtype=np.float64)
    cdelt1_deg = np.full(nchan, np.nan, dtype=np.float64)
    cdelt2_deg = np.full(nchan, np.nan, dtype=np.float64)

    for k, (fp, f) in enumerate(chan_list):
        img2d, _, hdr = read_2d_image_and_wcs(fp)
        if img2d.shape != (ny, nx):
            raise ValueError(f"Shape mismatch: {fp} has {img2d.shape}, expected {(ny, nx)}")

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

    # Build 3D WCS header
    w3 = WCS(naxis=3)
    w2 = WCS(first_hdr).celestial

    # Spatial axes
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

    # Spectral axis
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

    # Optional: store one representative beam in the main cube header
    finite_beam = np.isfinite(bmaj_deg) & np.isfinite(bmin_deg)
    if np.any(finite_beam):
        hdr3["BMAJ"] = float(np.nanmedian(bmaj_deg[finite_beam]))
        hdr3["BMIN"] = float(np.nanmedian(bmin_deg[finite_beam]))
        if np.any(np.isfinite(bpa_deg)):
            hdr3["BPA"] = float(np.nanmedian(bpa_deg[np.isfinite(bpa_deg)]))

    # Primary HDU for cube
    hdu0 = fits.PrimaryHDU(data=cube, header=hdr3)

    # BEAMS table extension
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
# Main
# --------------------------------------------------
def main():
    """This is the main function that orchestrates the processing:
    - It lists obsid directories under ROOT;
    - For each obsid directory gets obsid = obsdir.name, it locates the MFS image, skipping if not found;
    - It reads the MFS image/WCS/header, runs choose_pixel_hybrid to get (x_final, y_final, method, diagnostics) - this gives source centre for entire obsid;
    - It computes pix_per_beam (calls beam_area_in_pixels(mfs_hdr)) from MFS header for later convert aperture sums to Jy;
    - Builds the cube from channel images: sets up cube_path, finds channel images with find_channel_images(obsdir), skipping if none found,
     then calls build_cube_fits to create the cube and get cube_data, freqs_hz, cube_hdr;
    - The cubes are then possible to be visualised in CARTA or similar, and also the cube in memory (cube_data) for photometry;
    Then. there is the extraction of the aperture spectrum from the cube,
    for each plane channel:
        - It extracts the 2D plane (plane = cube_data[k, :, :]),
        - Runs photutils_aperture_flux with the final (x_final, y_final) centre, AP_R, ANN_R_IN/OUT, DO_BKG_SUB,
        - Converts to Jy by dividing by pix_per_beam if possible,
        - Computes background stats with sigma_clipped_stats on the full 2D plane,
        - Appends a dict of results to rows, including obsid, frequency, flux, raw/bkg-sub aperture sums, annulus background info,
         aperture area, annulus pixel count, background stats, pix_per_beam, locator method/diagnostics, and cube path for later inspection.
    """
    obsid_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(obsid_dirs)} obsid directories under {ROOT}")

    for obsdir in obsid_dirs:
        obsid = obsdir.name

        # --- MFS file ---
        mfs_files = sorted(obsdir.glob("*MFS-XX-image.fits")) #before it was "*MFS-image.fits"
        if not mfs_files:
            print(f"[{obsid}] No *MFS-image.fits found, skipping.")
            continue
        mfs_fp = mfs_files[0]

        try:
            # A) locate on MFS
            mfs_img, mfs_wcs, mfs_hdr = read_2d_image_and_wcs(mfs_fp)
            x_final, y_final, method, diag = choose_pixel_hybrid(mfs_img, mfs_wcs)

            # Optional MFS fallback beam
            mfs_pix_per_beam = beam_area_in_pixels(mfs_hdr)
            mfs_ap_r_pix = aperture_radius_from_beam(
                mfs_hdr.get("BMAJ", np.nan),
                mfs_hdr.get("BMIN", np.nan),
                mfs_hdr.get("CDELT1", np.nan),
                mfs_hdr.get("CDELT2", np.nan),
                factor=AP_R_FWHM_FACTOR,
            )
            if (not np.isfinite(mfs_ap_r_pix)) or (mfs_ap_r_pix <= 0):
                mfs_ap_r_pix = AP_R_FALLBACK_PIX

            # B) build cube
            cube_path = output / f"{obsid}{CUBE_SUFFIX}"
            chan_list = find_channel_images(obsdir)
            if len(chan_list) == 0:
                print(f"[{obsid}] No channel images found, skipping cube/spectrum.")
                continue

            cube_data, freqs_hz, cube_hdr = build_cube_fits(obsid, chan_list, cube_path)

            # Read back the BEAMS table from the cube itself
            beam_info = read_beam_table(cube_path)

            if len(beam_info["chan"]) != cube_data.shape[0]:
                raise RuntimeError(
                    f"[{obsid}] BEAMS table length {len(beam_info['chan'])} does not match cube planes {cube_data.shape[0]}"
                )

            # C) extract aperture spectrum from cube
            # Prepare per-row outputs (one row per channel plane)
            rows = []

            for k in range(cube_data.shape[0]):
                plane = cube_data[k, :, :]

                bad_chan = channel_is_flagged(beam_info, k)

                if bad_chan:
                    rows.append({
                        "obsid": obsid,
                        "Central Frequency(MHz)": freqs_hz[k] / 1e6,
                        "flux_Jy": 0.0,
                        "ap_sum_raw": 0.0,
                        "ap_sum_bkgsub": 0.0,
                        "bkg_per_pix(Jy/beam)": float("nan"),
                        "ap_area_pix": float("nan"),
                        "ann_npix": 0,
                        "ann_rms_jybeam": 0.0,
                        "ann_rms_jy": 0.0,
                        "mean (Jy/beam)": float("nan"),
                        "median (Jy/beam)": float("nan"),
                        "std (Jy/beam)": float("nan"),
                        "pix_per_beam": beam_info["pix_per_beam"][k],
                        "mfs_pix_per_beam": mfs_pix_per_beam,
                        "BMAJ_deg": beam_info["bmaj_deg"][k],
                        "BMIN_deg": beam_info["bmin_deg"][k],
                        "BPA_deg": beam_info["bpa_deg"][k],
                        "CDELT1_deg": beam_info["cdelt1_deg"][k],
                        "CDELT2_deg": beam_info["cdelt2_deg"][k],
                        "channel_index": int(beam_info["chan"][k]),
                        "channel_flagged": 1,
                        "ap_r_pix": float("nan"),
                        "ap_r_source": "flagged_zeroed",
                        "locator_method": method,
                        "dx_pix": diag.get("dx", np.nan),
                        "dy_pix": diag.get("dy", np.nan),
                        "r_pix": diag.get("r", np.nan),
                        "x_final": x_final,
                        "y_final": y_final,
                        "xw": diag.get("xw", np.nan),
                        "yw": diag.get("yw", np.nan),
                        "cube_path": str(cube_path),
                    })
                    continue

                # Set aperture radius from the per-channel beam size.
                ap_r_pix = aperture_radius_from_beam(
                    beam_info["bmaj_deg"][k],
                    beam_info["bmin_deg"][k],
                    beam_info["cdelt1_deg"][k],
                    beam_info["cdelt2_deg"][k],
                    factor=AP_R_FWHM_FACTOR,
                )
                ap_r_source = "channel_beam"
                if (not np.isfinite(ap_r_pix)) or (ap_r_pix <= 0):
                    ap_r_pix = mfs_ap_r_pix
                    ap_r_source = "mfs_beam_or_fallback"

                # aperture photometry
                aph = photutils_aperture_flux(
                    plane, x_final, y_final,
                    ap_r=ap_r_pix,
                    ann_r_in=ANN_R_IN, ann_r_out=ANN_R_OUT,
                    do_bkg_sub=DO_BKG_SUB
                )

                # convert to Jy using the per-channel value stored in the cube BEAMS table
                pixpb = beam_info["pix_per_beam"][k]

                # optional fallback to MFS beam if per-channel value is missing
                if (not np.isfinite(pixpb)) or (pixpb <= 0):
                    pixpb = mfs_pix_per_beam

                if np.isfinite(pixpb) and pixpb > 0:
                    flux_jy = aph["ap_sum_bkgsub"] / pixpb
                else:
                    flux_jy = float("nan")

                ####new part 
                if np.isfinite(pixpb) and pixpb > 0 and np.isfinite(aph["ann_rms_jybeam"]):
                    ann_rms_jy = aph["ann_rms_jybeam"] * np.sqrt(aph["ap_area_pix"] / pixpb)
                else:
                    ann_rms_jy = float("nan")
                #######

                # stats 
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(
                    plane, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std"
                )

                rows.append({
                    "obsid": obsid,
                    "Central Frequency(MHz)": freqs_hz[k] / 1e6,
                    "flux_Jy": flux_jy,
                    "ap_sum_raw": aph["ap_sum_raw"],
                    "ap_sum_bkgsub": aph["ap_sum_bkgsub"],
                    "bkg_per_pix(Jy/beam)": aph["bkg_per_pix"],
                    "ap_area_pix": aph["ap_area_pix"],
                    "ann_npix": aph["ann_npix"],
                    "ann_rms_jybeam": aph["ann_rms_jybeam"], #new
                    "ann_rms_jy": ann_rms_jy,  #new
                    "mean (Jy/beam)": float(bkg_mean),
                    "median (Jy/beam)": float(bkg_median),
                    "std (Jy/beam)": float(bkg_std),
                    "pix_per_beam": pixpb,
                    "mfs_pix_per_beam": mfs_pix_per_beam,
                    "BMAJ_deg": beam_info["bmaj_deg"][k],
                    "BMIN_deg": beam_info["bmin_deg"][k],
                    "BPA_deg": beam_info["bpa_deg"][k],
                    "CDELT1_deg": beam_info["cdelt1_deg"][k],
                    "CDELT2_deg": beam_info["cdelt2_deg"][k],
                    "channel_index": int(beam_info["chan"][k]),
                    "channel_flagged": 0,
                    "locator_method": method,
                    "dx_pix": diag.get("dx", np.nan),
                    "dy_pix": diag.get("dy", np.nan),
                    "r_pix": diag.get("r", np.nan),
                    "x_final": x_final,
                    "y_final": y_final,
                    "xw": diag.get("xw", np.nan),
                    "yw": diag.get("yw", np.nan),
                    "cube_path": str(cube_path),
                    "ap_r_pix": ap_r_pix,
                    "ap_r_source": ap_r_source,
                })

            df = pd.DataFrame(rows)

            #------UNDER DEVELOPMENT ------
            # --------------------------------------------------
            # Baseline-subtracted spectral RMS (better estimate)
            # --------------------------------------------------

            freq_mhz = df["Central Frequency(MHz)"].to_numpy(dtype=float)
            flux_jy = df["flux_Jy"].to_numpy(dtype=float)

            good = np.isfinite(freq_mhz) & np.isfinite(flux_jy)

            # Defaults
            baseline_deg = 1
            rms_spec_jy = np.nan
            std_spec_jy_clipped = np.nan
            n_spec_total = int(np.sum(good))
            n_spec_used = 0
            baseline_coeffs = [np.nan] * (baseline_deg + 1)

            # Optional columns to save per-channel baseline/residuals
            baseline_model = np.full_like(flux_jy, np.nan, dtype=float)
            residual_jy = np.full_like(flux_jy, np.nan, dtype=float)

            if np.sum(good) >= (baseline_deg + 2):
                x = freq_mhz[good]
                y = flux_jy[good]

                # Improve numerical stability of polyfit
                x0 = np.median(x)
                xs = x - x0

                # Iterative sigma-clipped baseline fit
                keep = np.ones_like(y, dtype=bool)

                for _ in range(5):
                    if np.sum(keep) < (baseline_deg + 2):
                        break

                    coeff = np.polyfit(xs[keep], y[keep], deg=baseline_deg)
                    yfit = np.polyval(coeff, xs)
                    resid = y - yfit

                    clipped = sigma_clip(resid[keep], sigma=3.0, maxiters=1, cenfunc="median", stdfunc="std")
                    new_keep = keep.copy()
                    new_keep_indices = np.where(keep)[0]
                    new_keep[new_keep_indices] = ~clipped.mask

                    # stop if unchanged
                    if np.array_equal(new_keep, keep):
                        keep = new_keep
                        break
                    keep = new_keep

                # Final baseline on kept points
                if np.sum(keep) >= (baseline_deg + 2):
                    coeff = np.polyfit(xs[keep], y[keep], deg=baseline_deg)
                    yfit = np.polyval(coeff, xs)
                    resid = y - yfit

                    # RMS on kept residuals
                    resid_kept = resid[keep]

                    med_resid = np.median(resid_kept)
                    mad_resid = np.median(np.abs(resid_kept - med_resid))
                    rms_spec_jy = 1.4826 * mad_resid
                    std_spec_jy_clipped = np.std(resid_kept)

                    n_spec_used = int(np.sum(keep))
                    baseline_coeffs = coeff.tolist()

                    # Save baseline/residual arrays back to full-length columns
                    baseline_model[good] = yfit
                    residual_jy[good] = resid

            # Save summary columns (same value repeated for convenience)
            df["rms_spec_jy"] = rms_spec_jy
            df["std_spec_jy_clipped"] = std_spec_jy_clipped
            df["n_spec_total"] = n_spec_total
            df["n_spec_used_rms"] = n_spec_used
            df["baseline_deg"] = baseline_deg

            # Save baseline and residual per channel 
            df["baseline_flux_jy"] = baseline_model
            df["residual_flux_jy"] = residual_jy

            # Save baseline coefficients (same value repeated, easier to inspect later)
            # For deg=1, coeff[0]=slope, coeff[1]=intercept in shifted-frequency coords (x - x0)
            df["baseline_x0_mhz"] = np.nanmedian(freq_mhz[good]) if np.any(good) else np.nan
            for i in range(baseline_deg + 1):
                df[f"baseline_coeff_{i}"] = baseline_coeffs[i]
            #----------------------------------------------


            out_csv = output / f"{obsid}_spec.csv"
            df.to_csv(out_csv, index=False)

            print(
                f"[{obsid}] cube={cube_path.name} rows={len(df)} wrote {out_csv.name} "
                f"centre=({x_final:.3f},{y_final:.3f}) method={method} r={diag.get('r', np.nan):.2f}"
            )

        except Exception as e:
            print(f"[{obsid}] failed: {e}")
            continue


if __name__ == "__main__":
    main()