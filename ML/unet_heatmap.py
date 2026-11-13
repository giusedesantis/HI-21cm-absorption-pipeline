#!/usr/bin/env python3
"""
Multi-task U-Net for MWA source localisation:
    input  : 32x32 MFS cutout around expected WCS/catalogue position
    output : (1) source-mask probability map
             (2) centre heatmap probability map

The script has three modes:

1) make-labels
   Creates semi-automatic labels from MFS images:
       - extracts a cutout around TARGET_COORD
       - estimates robust background/RMS
       - thresholds positive emission near the expected centre
       - keeps the connected component closest to the expected centre
       - saves image, mask, centre heatmap, and diagnostics to an NPZ file
       - optionally saves PNG overlays for manual inspection

2) train
   Trains a small U-Net with two output heads:
       - mask head for source/core/real-extension segmentation
       - heatmap head for source-centre localisation

3) infer
   Applies the trained model to MFS images, writes predicted centre diagnostics,
   and saves one NPZ per MFS image containing the source mask plus explicit
   cutout/full-image pixel indices for later cube-channel extraction.

Recommended workflow:
    python source_unet_mask_heatmap.py make-labels --root /scratch/... --out labels_npz --png review_png
    # inspect review_png manually and remove bad .npz label files
    python source_unet_mask_heatmap.py train --labels labels_npz --model-out source_unet.pt
    python source_unet_mask_heatmap.py infer --root /scratch/... --model source_unet.pt --out predictions.csv --save-png pred_png

Important:
    Semi-automatic labels are not ground truth. They are a first pass.
    For bad/ambiguous fields, inspect the PNGs and remove or correct bad labels before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

TARGET_COORD_DEFAULT = "08h56m14.73s +02d23m59.6s"
CUTOUT_SIZE_DEFAULT = 32

# Semi-label settings
LABEL_THRESHOLD_SIGMA_DEFAULT = 4.0
LABEL_GROW_THRESHOLD_SIGMA_DEFAULT = 2.0
LABEL_MAX_RADIUS_PIX_DEFAULT = 14.0
LABEL_MIN_PIXELS_DEFAULT = 4
LABEL_CENTRE_HEATMAP_SIGMA_PIX_DEFAULT = 1.5

# Training settings
BATCH_SIZE_DEFAULT = 32
EPOCHS_DEFAULT = 80
LR_DEFAULT = 1e-3
WEIGHT_DECAY_DEFAULT = 1e-4
MASK_LOSS_WEIGHT_DEFAULT = 1.0
HEATMAP_LOSS_WEIGHT_DEFAULT = 2.0


# -----------------------------------------------------------------------------
# FITS and image utilities
# -----------------------------------------------------------------------------

def read_2d_image_and_wcs(fits_path: Path) -> tuple[np.ndarray, WCS, fits.Header]:
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    while data is not None and data.ndim > 2:
        data = data[0]

    if data is None or data.ndim != 2:
        raise ValueError(f"Not a 2D image after collapsing: {fits_path}; shape={None if data is None else data.shape}")

    image = np.asarray(data, dtype=np.float32)
    wcs = WCS(header).celestial
    return image, wcs, header


def extract_cutout(
    image: np.ndarray,
    x: float,
    y: float,
    size: int = CUTOUT_SIZE_DEFAULT,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """
    Extract a square cutout around the expected floating-point position.

    Returns:
        cutout, x0, y0

    where (x0, y0) is the full-image pixel coordinate corresponding to
    cutout pixel (0, 0).  Saving this origin is important because it lets
    us map the predicted source-mask pixels back to the exact MFS/cube
    pixel indices later.
    """
    if size <= 0 or size % 2 != 0:
        raise ValueError(f"cutout size must be a positive even integer; got {size}")

    half = size // 2
    xi = int(np.rint(x))
    yi = int(np.rint(y))

    y0, y1 = yi - half, yi - half + size
    x0, x1 = xi - half, xi - half + size

    out = np.full((size, size), pad_value, dtype=np.float32)

    src_y0 = max(0, y0)
    src_y1 = min(image.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(image.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return out, x0, y0


def robust_normalise(cutout: np.ndarray) -> np.ndarray:
    """Robust normalisation for radio images with outliers and artefacts."""
    arr = np.asarray(cutout, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[finite]
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    out = (arr - med) / sigma
    out[~finite] = 0.0
    out = np.clip(out, -8.0, 20.0)
    return out.astype(np.float32)


def circular_prior_mask(
    size: int,
    radius_pix: float,
    centre_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Circular positional prior, optionally centred on the exact WCS sub-pixel position."""
    if centre_xy is None:
        cx = cy = (size - 1) / 2.0
    else:
        cx, cy = centre_xy
    yy, xx = np.indices((size, size))
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_pix**2


def gaussian_heatmap(size: int, x0: float, y0: float, sigma: float) -> np.ndarray:
    yy, xx = np.indices((size, size))
    heat = np.exp(-0.5 * (((xx - x0) / sigma) ** 2 + ((yy - y0) / sigma) ** 2))
    heat = heat / max(float(np.max(heat)), 1e-8)
    return heat.astype(np.float32)


def connected_components(binary: np.ndarray) -> list[np.ndarray]:
    """Simple 8-connected components without scipy/skimage dependency."""
    binary = np.asarray(binary, dtype=bool)
    visited = np.zeros(binary.shape, dtype=bool)
    components: list[np.ndarray] = []
    h, w = binary.shape

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pts = []

            while stack:
                yy, xx = stack.pop()
                pts.append((yy, xx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = yy + dy, xx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            comp = np.zeros(binary.shape, dtype=bool)
            ys, xs = zip(*pts)
            comp[np.array(ys), np.array(xs)] = True
            components.append(comp)

    return components


def dilate_binary(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small 8-connected binary dilation without scipy/skimage dependency."""
    out = np.asarray(binary, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        new = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                new |= padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = new
    return out


def component_score(component: np.ndarray, snr_img: np.ndarray, centre_xy: tuple[float, float]) -> float:
    """Prefer bright components close to the expected centre."""
    ys, xs = np.where(component)
    if len(xs) == 0:
        return -np.inf

    flux_score = float(np.nansum(np.clip(snr_img[component], 0, None)))
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    dx = cx - centre_xy[0]
    dy = cy - centre_xy[1]
    dist = math.hypot(dx, dy)

    # Penalise very distant components so a bright sidelobe does not always win.
    return flux_score / (1.0 + 0.15 * dist**2)


@dataclass
class LabelDiagnostics:
    source_file: str
    status: str
    reason: str
    xw: float
    yw: float
    centre_x_cutout: float
    centre_y_cutout: float
    n_mask_pix: int
    peak_snr: float
    total_positive_snr: float
    centroid_x_cutout: float
    centroid_y_cutout: float


def make_semiauto_label(
    cutout: np.ndarray,
    threshold_sigma: float = LABEL_THRESHOLD_SIGMA_DEFAULT,
    grow_threshold_sigma: float = LABEL_GROW_THRESHOLD_SIGMA_DEFAULT,
    max_radius_pix: float = LABEL_MAX_RADIUS_PIX_DEFAULT,
    min_pixels: int = LABEL_MIN_PIXELS_DEFAULT,
    heatmap_sigma_pix: float = LABEL_CENTRE_HEATMAP_SIGMA_PIX_DEFAULT,
    expected_xy_cutout: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelDiagnostics]:
    """
    Create a semi-automatic mask and centre heatmap from one MFS cutout.

    The method is deliberately conservative:
      - use robust background/RMS
      - threshold only inside a circular prior region
      - keep the connected component that is both bright and close to the expected centre
      - grow the mask into lower-SNR neighbouring pixels to include real extension
    """
    size = cutout.shape[0]
    if expected_xy_cutout is None:
        centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    else:
        centre = (float(expected_xy_cutout[0]), float(expected_xy_cutout[1]))

    mean, median, std = sigma_clipped_stats(cutout, sigma=3.0, maxiters=5, cenfunc="median", stdfunc="std")
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(cutout))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    if not np.isfinite(median):
        median = 0.0

    snr = (cutout - float(median)) / float(std)
    snr[~np.isfinite(snr)] = 0.0

    prior = circular_prior_mask(size, max_radius_pix, centre_xy=centre)

    # Seed mask: confident source pixels.
    seed = (snr >= threshold_sigma) & prior
    components = connected_components(seed)

    if len(components) == 0:
        mask = np.zeros_like(seed, dtype=np.float32)
        heat = gaussian_heatmap(size, centre[0], centre[1], heatmap_sigma_pix)
        diag = LabelDiagnostics(
            source_file="",
            status="reject",
            reason="no_component_above_threshold",
            xw=np.nan,
            yw=np.nan,
            centre_x_cutout=centre[0],
            centre_y_cutout=centre[1],
            n_mask_pix=0,
            peak_snr=float(np.nanmax(snr)),
            total_positive_snr=0.0,
            centroid_x_cutout=np.nan,
            centroid_y_cutout=np.nan,
        )
        return mask, heat, diag

    best = max(components, key=lambda c: component_score(c, snr, centre))

    # Grow around the seed to include lower-SNR extension connected to the source.
    grow_allowed = (snr >= grow_threshold_sigma) & prior
    grown = best.copy()
    for _ in range(5):
        candidate = dilate_binary(grown, iterations=1) & grow_allowed
        if np.array_equal(candidate, grown):
            break
        grown = candidate

    n_pix = int(np.sum(grown))
    peak_snr = float(np.nanmax(snr[grown])) if n_pix > 0 else float(np.nanmax(snr))
    total_positive_snr = float(np.nansum(np.clip(snr[grown], 0, None))) if n_pix > 0 else 0.0

    if n_pix < min_pixels:
        status = "reject"
        reason = "too_few_mask_pixels"
    else:
        status = "candidate"
        reason = "semi_auto"

    ys, xs = np.where(grown)
    if len(xs) > 0:
        weights = np.clip(snr[grown], 0.0, None)
        if np.sum(weights) > 0:
            xcen = float(np.average(xs, weights=weights))
            ycen = float(np.average(ys, weights=weights))
        else:
            xcen = float(np.mean(xs))
            ycen = float(np.mean(ys))
    else:
        xcen, ycen = centre

    mask = grown.astype(np.float32)
    heat = gaussian_heatmap(size, xcen, ycen, heatmap_sigma_pix)

    diag = LabelDiagnostics(
        source_file="",
        status=status,
        reason=reason,
        xw=np.nan,
        yw=np.nan,
        centre_x_cutout=centre[0],
        centre_y_cutout=centre[1],
        n_mask_pix=n_pix,
        peak_snr=peak_snr,
        total_positive_snr=total_positive_snr,
        centroid_x_cutout=xcen,
        centroid_y_cutout=ycen,
    )
    return mask, heat, diag


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceUNet(nn.Module):
    """Small U-Net with two heads: source mask and centre heatmap."""

    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.mask_head = nn.Conv2d(base, 1, kernel_size=1)
        self.heat_head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return {
            "mask_logits": self.mask_head(d1),
            "heat_logits": self.heat_head(d1),
        }


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    dims = (1, 2, 3)
    inter = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    return loss.mean()


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    heat: torch.Tensor,
    mask_weight: float = MASK_LOSS_WEIGHT_DEFAULT,
    heat_weight: float = HEATMAP_LOSS_WEIGHT_DEFAULT,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_logits = outputs["mask_logits"]
    heat_logits = outputs["heat_logits"]

    mask_loss = focal_bce_with_logits(mask_logits, mask) + dice_loss_with_logits(mask_logits, mask)
    heat_prob = torch.sigmoid(heat_logits)
    # The centre occupies very few pixels. Weight the Gaussian target region more
    # strongly so the trivial all-background solution is less attractive.
    heat_pixel_weight = 1.0 + 4.0 * heat
    heat_loss = torch.mean(heat_pixel_weight * (heat_prob - heat) ** 2)
    total = mask_weight * mask_loss + heat_weight * heat_loss

    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "heat_loss": float(heat_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class SourceLabelDataset(Dataset):
    def __init__(
        self,
        label_dir: Path,
        include_rejects: bool = False,
        augment: bool = True,
        files: list[Path] | None = None,
        expected_size: int = CUTOUT_SIZE_DEFAULT,
    ):
        self.include_rejects = include_rejects
        self.augment = augment
        self.expected_size = expected_size

        candidates = sorted(label_dir.glob("*.npz")) if files is None else [Path(fp) for fp in files]
        kept: list[Path] = []
        for fp in candidates:
            try:
                with np.load(fp, allow_pickle=True) as z:
                    diag = json.loads(str(z["diagnostics"]))
                    shape = tuple(z["image"].shape)
                if shape != (expected_size, expected_size):
                    continue
                if include_rejects or diag.get("status") == "candidate":
                    kept.append(fp)
            except Exception:
                continue
        self.files = kept

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        fp = self.files[idx]
        with np.load(fp, allow_pickle=True) as z:
            image = z["image"].astype(np.float32)
            mask = z["mask"].astype(np.float32)
            heat = z["heatmap"].astype(np.float32)

        image = robust_normalise(image)

        if self.augment:
            image, mask, heat = self.apply_augmentation(image, mask, heat)

        return {
            "image": torch.from_numpy(image[None, :, :]).float(),
            "mask": torch.from_numpy(mask[None, :, :]).float(),
            "heat": torch.from_numpy(heat[None, :, :]).float(),
            "path": str(fp),
        }

    @staticmethod
    def apply_augmentation(
        image: np.ndarray,
        mask: np.ndarray,
        heat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Rotations/flips improve orientation robustness while preserving the
        # pixel-wise relationship among image, mask and heatmap.
        k = int(np.random.randint(0, 4))
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
            heat = np.rot90(heat, k).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            heat = np.flip(heat, axis=1).copy()

        # Mild intensity/noise augmentation.  The mask/heatmap remain unchanged.
        scale = float(np.random.uniform(0.90, 1.10))
        bias = float(np.random.normal(0.0, 0.03))
        noise_sigma = float(np.random.uniform(0.0, 0.05))
        image = image * scale + bias
        if noise_sigma > 0:
            image = image + np.random.normal(0.0, noise_sigma, size=image.shape)
        return image.astype(np.float32), mask.astype(np.float32), heat.astype(np.float32)


# -----------------------------------------------------------------------------
# Label generation and inference utilities
# -----------------------------------------------------------------------------

def find_mfs_images(root: Path) -> list[Path]:
    """Find MFS images recursively without performing two directory scans."""
    return sorted(root.rglob("*MFS-XX-image.fits"))


def parse_target_coord(coord_text: str) -> SkyCoord:
    parts = coord_text.strip().split()
    if len(parts) == 2:
        return SkyCoord(parts[0], parts[1], frame="fk5")
    return SkyCoord(coord_text, frame="fk5")


def save_review_png(path: Path, image: np.ndarray, mask: np.ndarray, heat: np.ndarray, title: str = "") -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    im0 = axes[0].imshow(robust_normalise(image), origin="lower")
    axes[0].set_title("normalised cutout")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(robust_normalise(image), origin="lower")
    if np.any(mask >= 0.5):
        axes[1].contour(mask, levels=[0.5], colors="r", linewidths=1.0)
    axes[1].set_title("source mask")

    axes[2].imshow(robust_normalise(image), origin="lower")
    axes[2].contour(heat, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[2].set_title("centre heatmap")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_labels(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.png) if args.png else None
    target = parse_target_coord(args.target)

    mfs_files = find_mfs_images(root)
    print(f"Found {len(mfs_files)} MFS images under {root}")

    summary_rows = []
    for mfs_fp in mfs_files:
        try:
            image, wcs, header = read_2d_image_and_wcs(mfs_fp)
            xw, yw = wcs.world_to_pixel(target)
            cut, cut_x0, cut_y0 = extract_cutout(image, xw, yw, size=args.cutout_size)
            expected_xy_cutout = (float(xw - cut_x0), float(yw - cut_y0))

            mask, heat, diag = make_semiauto_label(
                cut,
                threshold_sigma=args.threshold_sigma,
                grow_threshold_sigma=args.grow_threshold_sigma,
                max_radius_pix=args.max_radius_pix,
                min_pixels=args.min_pixels,
                heatmap_sigma_pix=args.heatmap_sigma_pix,
                expected_xy_cutout=expected_xy_cutout,
            )
            diag.source_file = str(mfs_fp)
            diag.xw = float(xw)
            diag.yw = float(yw)

            mask_y_cut, mask_x_cut = np.where(mask >= 0.5)
            mask_y_full = mask_y_cut.astype(np.int64) + int(cut_y0)
            mask_x_full = mask_x_cut.astype(np.int64) + int(cut_x0)
            valid = (
                (mask_y_full >= 0) & (mask_y_full < image.shape[0]) &
                (mask_x_full >= 0) & (mask_x_full < image.shape[1])
            )
            mask_y_full = mask_y_full[valid]
            mask_x_full = mask_x_full[valid]

            if mask_x_full.size:
                sky = wcs.pixel_to_world(mask_x_full, mask_y_full)
                mask_ra_deg = np.asarray(sky.ra.deg, dtype=np.float64)
                mask_dec_deg = np.asarray(sky.dec.deg, dtype=np.float64)
            else:
                mask_ra_deg = np.empty(0, dtype=np.float64)
                mask_dec_deg = np.empty(0, dtype=np.float64)

            obsid = mfs_fp.parent.name
            stem = f"{obsid}_{mfs_fp.stem}".replace("/", "_")
            npz_path = out / f"{stem}.npz"

            np.savez_compressed(
                npz_path,
                image=cut.astype(np.float32),
                mask=mask.astype(np.uint8),
                heatmap=heat.astype(np.float32),
                # Explicit source-mask indices for later channel-plane extraction.
                mask_y_cutout=mask_y_cut.astype(np.int16),
                mask_x_cutout=mask_x_cut.astype(np.int16),
                mask_y_full=mask_y_full.astype(np.int32),
                mask_x_full=mask_x_full.astype(np.int32),
                mask_ra_deg=mask_ra_deg,
                mask_dec_deg=mask_dec_deg,
                cutout_x0=np.int32(cut_x0),
                cutout_y0=np.int32(cut_y0),
                full_image_shape=np.asarray(image.shape, dtype=np.int32),
                diagnostics=json.dumps(asdict(diag)),
                source_file=str(mfs_fp),
            )

            if png_dir is not None:
                status_prefix = "OK" if diag.status == "candidate" else "REJECT"
                save_review_png(
                    png_dir / f"{status_prefix}_{stem}.png",
                    cut, mask, heat,
                    title=f"{obsid}: {diag.status} {diag.reason}",
                )

            row = asdict(diag)
            row["npz"] = str(npz_path)
            row["n_valid_full_mask_pix"] = int(mask_x_full.size)
            summary_rows.append(row)
            print(f"[{obsid}] {diag.status}: {diag.reason}; n_pix={diag.n_mask_pix}; peak_snr={diag.peak_snr:.2f}")

        except Exception as exc:
            print(f"[WARN] failed {mfs_fp}: {exc}")

    summary_csv = out / "label_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote label summary: {summary_csv}")


def train(args: argparse.Namespace) -> None:
    label_dir = Path(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # Fixed 32x32 inputs benefit from cuDNN autotuning.
        torch.backends.cudnn.benchmark = True

    all_ds = SourceLabelDataset(
        label_dir, include_rejects=False, augment=False, expected_size=args.cutout_size
    )
    if len(all_ds) < 10:
        raise RuntimeError(
            f"Only {len(all_ds)} candidate {args.cutout_size}x{args.cutout_size} labels found. "
            "Inspect labels or regenerate them with the same cutout size before training."
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_ds))
    n_val = max(1, int(round(0.2 * len(all_ds))))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_files = [all_ds.files[i] for i in train_idx]
    val_files = [all_ds.files[i] for i in val_idx]

    # Critical: augmentation is ON only for training, OFF for validation.
    train_ds = SourceLabelDataset(
        label_dir, augment=True, files=train_files, expected_size=args.cutout_size
    )
    val_ds = SourceLabelDataset(
        label_dir, augment=False, files=val_files, expected_size=args.cutout_size
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = SourceUNet(in_ch=1, base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, args.patience // 3)
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {len(train_ds)} labels; validating on {len(val_ds)}; "
        f"device={device}; cutout={args.cutout_size}x{args.cutout_size}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            heat = batch["heat"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outputs = model(image)
            loss, loss_info = multitask_loss(
                outputs, mask, heat, args.mask_loss_weight, args.heatmap_loss_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_losses.append(loss_info["total_loss"])

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                heat = batch["heat"].to(device, non_blocking=True)
                outputs = model(image)
                loss, loss_info = multitask_loss(
                    outputs, mask, heat, args.mask_loss_weight, args.heatmap_loss_weight
                )
                val_losses.append(loss_info["total_loss"])

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        print(f"epoch {epoch:03d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_channels": args.base_channels,
                "cutout_size": args.cutout_size,
                "best_val_loss": best_val,
                "args": vars(args),
            }, out_path)
            print(f"  saved best model: {out_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation improvement.")
                break

    print(f"Finished training. Best validation loss: {best_val:.5f}")


@torch.inference_mode()
def predict_one(model: SourceUNet, cutout: np.ndarray, device: torch.device, mask_threshold: float = 0.5) -> dict[str, object]:
    x = torch.from_numpy(robust_normalise(cutout)[None, None, :, :]).float().to(device)
    outputs = model(x)
    mask_prob = torch.sigmoid(outputs["mask_logits"])[0, 0].cpu().numpy()
    heat_prob = torch.sigmoid(outputs["heat_logits"])[0, 0].cpu().numpy()

    # Centre from heatmap peak.
    iy, ix = np.unravel_index(np.nanargmax(heat_prob), heat_prob.shape)
    heat_peak = float(heat_prob[iy, ix])

    # Optional sub-pixel centre from heatmap local weighted average.
    r = 2
    y0, y1 = max(0, iy - r), min(heat_prob.shape[0], iy + r + 1)
    x0, x1 = max(0, ix - r), min(heat_prob.shape[1], ix + r + 1)
    patch = heat_prob[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    if np.sum(patch) > 0:
        x_sub = float(np.sum((xx + x0) * patch) / np.sum(patch))
        y_sub = float(np.sum((yy + y0) * patch) / np.sum(patch))
    else:
        x_sub, y_sub = float(ix), float(iy)

    mask = mask_prob >= mask_threshold
    n_mask = int(np.sum(mask))
    mask_conf = float(np.nanmean(mask_prob[mask])) if n_mask > 0 else 0.0

    return {
        "mask_prob": mask_prob,
        "heat_prob": heat_prob,
        "mask": mask.astype(np.float32),
        "x_cutout": x_sub,
        "y_cutout": y_sub,
        "heat_peak": heat_peak,
        "n_mask_pix": n_mask,
        "mask_mean_prob": mask_conf,
    }


def infer(args: argparse.Namespace) -> None:
    root = Path(args.root)
    target = parse_target_coord(args.target)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.model, map_location=device)
    base_channels = int(ckpt.get("base_channels", args.base_channels))
    trained_cutout_size = int(ckpt.get("cutout_size", args.cutout_size))
    if trained_cutout_size != args.cutout_size:
        print(
            f"[INFO] model was trained on {trained_cutout_size}x{trained_cutout_size}; "
            f"using that instead of requested/default {args.cutout_size}."
        )
    cutout_size = trained_cutout_size

    model = SourceUNet(in_ch=1, base=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    png_dir = Path(args.save_png) if args.save_png else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else out_csv.parent / f"{out_csv.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mfs_files = find_mfs_images(root)
    for mfs_fp in mfs_files:
        try:
            image, wcs, header = read_2d_image_and_wcs(mfs_fp)
            xw, yw = wcs.world_to_pixel(target)
            cut, cut_x0, cut_y0 = extract_cutout(image, xw, yw, size=cutout_size)
            pred = predict_one(model, cut, device, mask_threshold=args.mask_threshold)

            # Map heatmap centre exactly from cutout coordinates to full-image coordinates.
            # This avoids the half-pixel ambiguity that can occur for even-sized cutouts.
            x_final = float(cut_x0 + float(pred["x_cutout"]))
            y_final = float(cut_y0 + float(pred["y_cutout"]))
            dx = float(x_final - xw)
            dy = float(y_final - yw)

            mask_bool = np.asarray(pred["mask"] >= 0.5, dtype=bool)
            mask_y_cut, mask_x_cut = np.where(mask_bool)
            mask_y_full = mask_y_cut.astype(np.int64) + int(cut_y0)
            mask_x_full = mask_x_cut.astype(np.int64) + int(cut_x0)
            valid = (
                (mask_y_full >= 0) & (mask_y_full < image.shape[0]) &
                (mask_x_full >= 0) & (mask_x_full < image.shape[1])
            )
            mask_y_full = mask_y_full[valid]
            mask_x_full = mask_x_full[valid]

            # Save sky coordinates too.  Direct full-image indices are fastest when
            # the MFS and channel planes share a grid; RA/Dec provide a safe fallback
            # if a future channel image has a different WCS/grid.
            if mask_x_full.size:
                sky = wcs.pixel_to_world(mask_x_full, mask_y_full)
                mask_ra_deg = np.asarray(sky.ra.deg, dtype=np.float64)
                mask_dec_deg = np.asarray(sky.dec.deg, dtype=np.float64)
            else:
                mask_ra_deg = np.empty(0, dtype=np.float64)
                mask_dec_deg = np.empty(0, dtype=np.float64)

            obsid = mfs_fp.parent.name
            mask_npz = mask_dir / f"{obsid}_{mfs_fp.stem}_source_mask.npz"
            np.savez_compressed(
                mask_npz,
                # 32x32 ML outputs
                mask=mask_bool.astype(np.uint8),
                mask_prob=np.asarray(pred["mask_prob"], dtype=np.float32),
                heat_prob=np.asarray(pred["heat_prob"], dtype=np.float32),
                # Explicit source pixel indices
                mask_y_cutout=mask_y_cut.astype(np.int16),
                mask_x_cutout=mask_x_cut.astype(np.int16),
                mask_y_full=mask_y_full.astype(np.int32),
                mask_x_full=mask_x_full.astype(np.int32),
                # World coordinates for WCS-safe remapping if needed
                mask_ra_deg=mask_ra_deg,
                mask_dec_deg=mask_dec_deg,
                cutout_x0=np.int32(cut_x0),
                cutout_y0=np.int32(cut_y0),
                full_image_shape=np.asarray(image.shape, dtype=np.int32),
                xw=np.float64(xw),
                yw=np.float64(yw),
                x_final=np.float64(x_final),
                y_final=np.float64(y_final),
                mask_threshold=np.float32(args.mask_threshold),
                mfs_path=str(mfs_fp),
            )

            row = {
                "obsid": obsid,
                "mfs_path": str(mfs_fp),
                "mask_npz": str(mask_npz),
                "xw": float(xw),
                "yw": float(yw),
                "dx_pix": dx,
                "dy_pix": dy,
                "r_pix": float(math.hypot(dx, dy)),
                "x_final": x_final,
                "y_final": y_final,
                "x_cutout": float(pred["x_cutout"]),
                "y_cutout": float(pred["y_cutout"]),
                "cutout_x0": int(cut_x0),
                "cutout_y0": int(cut_y0),
                "heat_peak": float(pred["heat_peak"]),
                "n_mask_pix": int(mask_x_full.size),
                "mask_mean_prob": float(pred["mask_mean_prob"]),
                "quality_flag": quality_flag(pred, dx, dy, args),
            }
            rows.append(row)
            print(
                f"[{obsid}] dx={dx:.2f} dy={dy:.2f} heat={row['heat_peak']:.2f} "
                f"maskpix={row['n_mask_pix']} {row['quality_flag']} mask={mask_npz.name}"
            )

            if png_dir is not None:
                save_review_png(
                    png_dir / f"PRED_{obsid}_{mfs_fp.stem}.png",
                    cut,
                    pred["mask"],
                    pred["heat_prob"],
                    title=f"{obsid}: {row['quality_flag']} dx={dx:.2f}, dy={dy:.2f}",
                )

        except Exception as exc:
            print(f"[WARN] failed {mfs_fp}: {exc}")

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote predictions: {out_csv}")
    print(f"Saved source masks/indices under: {mask_dir}")


def quality_flag(pred: dict[str, object], dx: float, dy: float, args: argparse.Namespace) -> str:
    r = math.hypot(dx, dy)
    if float(pred["heat_peak"]) < args.min_heat_peak:
        return "bad_low_heat_peak"
    if int(pred["n_mask_pix"]) < args.min_pred_mask_pixels:
        return "bad_too_few_mask_pixels"
    if r > args.max_pred_shift_pix:
        return "warning_large_shift"
    if float(pred["mask_mean_prob"]) < args.min_mask_mean_prob:
        return "warning_low_mask_confidence"
    return "good"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/apply source mask + centre heatmap U-Net for MWA MFS cutouts.")
    sub = p.add_subparsers(dest="mode", required=True)

    q = sub.add_parser("make-labels", help="Create semi-automatic labels from MFS images.")
    q.add_argument("--root", required=True, help="Root directory containing obsid directories or MFS images.")
    q.add_argument("--out", required=True, help="Output directory for .npz label files.")
    q.add_argument("--png", default=None, help="Optional directory for PNG review plots.")
    q.add_argument("--target", default=TARGET_COORD_DEFAULT, help="Target coordinate, e.g. '08h56m14.73s +02d23m59.6s'.")
    q.add_argument("--cutout-size", type=int, default=CUTOUT_SIZE_DEFAULT)
    q.add_argument("--threshold-sigma", type=float, default=LABEL_THRESHOLD_SIGMA_DEFAULT)
    q.add_argument("--grow-threshold-sigma", type=float, default=LABEL_GROW_THRESHOLD_SIGMA_DEFAULT)
    q.add_argument("--max-radius-pix", type=float, default=LABEL_MAX_RADIUS_PIX_DEFAULT)
    q.add_argument("--min-pixels", type=int, default=LABEL_MIN_PIXELS_DEFAULT)
    q.add_argument("--heatmap-sigma-pix", type=float, default=LABEL_CENTRE_HEATMAP_SIGMA_PIX_DEFAULT)
    q.set_defaults(func=make_labels)

    t = sub.add_parser("train", help="Train the U-Net from label .npz files.")
    t.add_argument("--labels", required=True, help="Directory containing .npz label files.")
    t.add_argument("--model-out", required=True, help="Output .pt model path.")
    t.add_argument("--cutout-size", type=int, default=CUTOUT_SIZE_DEFAULT)
    t.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    t.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    t.add_argument("--lr", type=float, default=LR_DEFAULT)
    t.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY_DEFAULT)
    t.add_argument("--mask-loss-weight", type=float, default=MASK_LOSS_WEIGHT_DEFAULT)
    t.add_argument("--heatmap-loss-weight", type=float, default=HEATMAP_LOSS_WEIGHT_DEFAULT)
    t.add_argument("--base-channels", type=int, default=32)
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--patience", type=int, default=15, help="Early-stopping patience in epochs.")
    t.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/validation split.")
    t.add_argument("--cpu", action="store_true")
    t.set_defaults(func=train)

    i = sub.add_parser("infer", help="Apply trained U-Net to MFS images.")
    i.add_argument("--root", required=True, help="Root directory containing obsid directories or MFS images.")
    i.add_argument("--model", required=True, help="Trained .pt model path.")
    i.add_argument("--out", required=True, help="Output prediction CSV.")
    i.add_argument("--save-png", default=None, help="Optional directory for prediction review PNGs.")
    i.add_argument("--mask-dir", default=None, help="Directory for per-obsid source-mask NPZ files. Defaults beside --out.")
    i.add_argument("--target", default=TARGET_COORD_DEFAULT)
    i.add_argument("--cutout-size", type=int, default=CUTOUT_SIZE_DEFAULT)
    i.add_argument("--base-channels", type=int, default=32)
    i.add_argument("--mask-threshold", type=float, default=0.5)
    i.add_argument("--min-heat-peak", type=float, default=0.35)
    i.add_argument("--min-mask-mean-prob", type=float, default=0.55)
    i.add_argument("--min-pred-mask-pixels", type=int, default=4)
    i.add_argument("--max-pred-shift-pix", type=float, default=12.0)
    i.add_argument("--cpu", action="store_true")
    i.set_defaults(func=infer)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()