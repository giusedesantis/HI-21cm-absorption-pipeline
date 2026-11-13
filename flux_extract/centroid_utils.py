#!/usr/bin/env python3
"""
centroid_utils.py

Shared utilities for:
  - extracting cutouts around a WCS-predicted pixel
  - estimating a local centroid in a robust way
  - normalising cutouts consistently for CNN training/inference

Dependencies:
  numpy
  astropy (sigma_clipped_stats)

Typical usage:
  from centroid_utils import extract_cutout, local_centroid, normalise_cutout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from astropy.stats import sigma_clipped_stats


def cutout_center(size: int) -> Tuple[float, float]:
    """Return (cx, cy) = centre coordinate in pixel units for an array of shape (size,size)."""
    c = (size - 1) / 2.0
    return c, c


def extract_cutout(
    img: np.ndarray,
    xc: float,
    yc: float,
    size: int,
    pad_value: float = np.nan,
) -> np.ndarray:
    """
    Extract a size×size cutout centred on (xc, yc) in pixel coords.
    If the cutout extends beyond the image boundary, it pads with pad_value (default NaN).

    Parameters
    ----------
    img : np.ndarray
        2D image, shape (Ny, Nx)
    xc, yc : float
        Pixel coordinates (x=columns, y=rows) of cutout centre
    size : int
        Output cutout size, e.g. 64
    pad_value : float
        Value to pad outside image boundaries (NaN recommended)

    Returns
    -------
    cut : np.ndarray
        2D array of shape (size, size)
    """
    if img.ndim != 2:
        raise ValueError(f"extract_cutout expects a 2D image, got shape {img.shape}")

    h = size // 2
    x0, x1 = int(xc) - h, int(xc) + h
    y0, y1 = int(yc) - h, int(yc) + h

    # padding needed?
    pad_l = max(0, -x0)
    pad_r = max(0, x1 - img.shape[1])
    pad_b = max(0, -y0)
    pad_t = max(0, y1 - img.shape[0])

    if pad_l or pad_r or pad_b or pad_t:
        imgp = np.pad(
            img,
            ((pad_b, pad_t), (pad_l, pad_r)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        imgp = img

    # shift slice indices into padded image coords
    x0 += pad_l
    x1 += pad_l
    y0 += pad_b
    y1 += pad_b

    cut = imgp[y0:y1, x0:x1]

    # Ensure exact shape (size, size)
    if cut.shape != (size, size):
        # pad or crop to exact size
        out = np.full((size, size), pad_value, dtype=imgp.dtype)
        ys = min(size, cut.shape[0])
        xs = min(size, cut.shape[1])
        out[:ys, :xs] = cut[:ys, :xs]
        cut = out

    return cut


@dataclass
class CentroidResult:
    """Output of local_centroid()."""
    success: bool
    xcen: float = np.nan   # centroid x in cutout coords
    ycen: float = np.nan   # centroid y in cutout coords
    dx: float = np.nan     # centroid offset from cutout centre (x)
    dy: float = np.nan     # centroid offset from cutout centre (y)
    r_pix: float = np.nan  # radial offset sqrt(dx^2+dy^2)

    med: float = np.nan
    std: float = np.nan
    peak: float = np.nan
    snr_peak: float = np.nan
    nmask: int = 0

    reason: str = ""       # why it failed (if success=False)


def local_centroid(
    cut: np.ndarray,
    nsigma: float = 5.0,
    keep_percentile: float = 40.0,
    min_pixels: int = 5,
    clip_positive: bool = True,
) -> CentroidResult:
    """
    Robust centroid estimator near cutout centre.

    Steps:
      1) sigma-clipped median/std of cutout background
      2) threshold mask: cut > med + nsigma*std
      3) keep only the closest keep_percentile% pixels to centre (by distance)
      4) compute flux-weighted centroid (background-subtracted)

    Parameters
    ----------
    cut : np.ndarray
        2D cutout, shape (S,S)
    nsigma : float
        threshold in sigma above median
    keep_percentile : float
        keep closest pixels to centre up to this distance percentile
    min_pixels : int
        require at least this many thresholded pixels
    clip_positive : bool
        clip (cut-med) weights to >=0 before centroiding

    Returns
    -------
    CentroidResult
        success + centroid coords and useful diagnostics
    """
    if cut.ndim != 2 or cut.shape[0] != cut.shape[1]:
        return CentroidResult(False, reason=f"cutout not square: {cut.shape}")

    size = cut.shape[0]
    cx, cy = cutout_center(size)

    # Robust background stats
    mean, med, std = sigma_clipped_stats(cut, sigma=3.0, maxiters=5)
    med = float(med)
    std = float(std)

    if not np.isfinite(std) or std <= 0:
        return CentroidResult(False, med=med, std=std, reason="nonfinite_or_zero_std")

    thresh = med + nsigma * std
    mask = np.isfinite(ceren := cut) & (cut > thresh)
    nmask = int(mask.sum())
    if nmask < min_pixels:
        peak = float(np.nanmax(cut)) if np.isfinite(cut).any() else np.nan
        snr_peak = float((peak - med) / std) if np.isfinite(peak) else np.nan
        return CentroidResult(
            False, med=med, std=std, peak=peak, snr_peak=snr_peak, nmask=nmask,
            reason="too_few_pixels_above_threshold"
        )

    yy, xx = np.where(mask)

    # keep closest fraction to centre
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    d2_thr = np.percentile(d2, keep_percentile)
    keep = d2 <= d2_thr
    yy = yy[keep]
    xx = xx[keep]

    vals = cut[yy, xx] - med
    if clip_positive:
        vals = np.clip(vals, 0.0, None)

    s = float(np.sum(vals))
    if not np.isfinite(s) or s <= 0:
        peak = float(np.nanmax(cut))
        snr_peak = float((peak - med) / std)
        return CentroidResult(
            False, med=med, std=std, peak=peak, snr_peak=snr_peak, nmask=nmask,
            reason="nonpositive_weights"
        )

    xcen = float(np.sum(xx * vals) / s)
    ycen = float(np.sum(yy * vals) / s)

    dx = xcen - cx
    dy = ycen - cy
    r = float(np.sqrt(dx * dx + dy * dy))

    peak = float(np.nanmax(cut))
    snr_peak = float((peak - med) / std)

    return CentroidResult(
        True,
        xcen=xcen, ycen=ycen, dx=float(dx), dy=float(dy), r_pix=r,
        med=med, std=std, peak=peak, snr_peak=snr_peak, nmask=nmask,
        reason=""
    )


def normalise_cutout(
    cut: np.ndarray,
    med: Optional[float] = None,
    std: Optional[float] = None,
    clip: Optional[float] = 8.0,
    fill_value: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalise cutout for CNN input.

    Default behaviour:
      - if med/std not provided: compute sigma-clipped stats
      - replace non-finite pixels with med
      - output = (cut - med) / (std + eps)
      - optionally clip to [-clip, +clip]
      - replace any remaining non-finite with fill_value
      - return float32

    IMPORTANT: Use this exact function in both:
      - make_dataset.py (when saving X)
      - maxpixel.py (before CNN inference)

    Parameters
    ----------
    cut : np.ndarray
        2D cutout
    med, std : float or None
        if provided, use these; otherwise estimate from cut
    clip : float or None
        clip absolute value to this (set None to disable)
    fill_value : float
        value to fill any non-finite values at the end
    eps : float
        small number to avoid divide-by-zero

    Returns
    -------
    x : np.ndarray
        float32 array, same shape as cut
    """
    x = np.array(cut, dtype=np.float32, copy=True)

    if med is None or std is None:
        _, med_est, std_est = sigma_clipped_stats(x, sigma=3.0, maxiters=5)
        med = float(med_est)
        std = float(std_est)

    std = float(std)
    if not np.isfinite(std) or std <= 0:
        std = 1.0

    # Fill NaNs with median before normalising
    bad = ~np.isfinite(x)
    if np.any(bad):
        x[bad] = float(med)

    x = (x - float(med)) / (std + eps)

    if clip is not None:
        x = np.clip(x, -float(clip), float(clip))

    # final safety
    bad2 = ~np.isfinite(x)
    if np.any(bad2):
        x[bad2] = float(fill_value)

    return x.astype(np.float32, copy=False)
