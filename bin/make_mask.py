#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
from pathlib import Path
from config_utils import load_config

import sys

#obsid = sys.argv[1]

config = load_config()

# Directories specified by the config file
images_dir = Path(config["paths"]["images_dir"])
masks_dir = Path(config["paths"]["masks_dir"])

# Input FITS image for the observation
template_fits = images_dir / str(obsid) / f"{obsid}_deep-MFS-image.fits"

# For GLEAM J0856+0224 from the config file
coord = config["target"]["coordinates"]
# Output directory for the generated masks
output_dir = masks_dir
output_dir.mkdir(parents=True, exist_ok=True)

# command line option is a template file, e.g. what to use to make the beam files
inp = Path(template_fits)
#mask_file = "{0}_mask.fits".format(inp[0:10])
mask_file = output_dir / (inp.stem + "_mask.fits")
hdu = fits.open(template_fits)
w = wcs.WCS(hdu[0].header, naxis=2)

# and coordinates of the source of interest
coords = coord
loc = w.world_to_pixel(SkyCoord(coords, unit=(u.hour, u.deg), frame="fk5"))

hdu[0].data = np.ones(hdu[0].data.shape, dtype=np.float32)
hdu[0].data[:,:,int(loc[1])-32:int(loc[1])+32,int(loc[0])-32:int(loc[0])+32] = 0.0

hdu.writeto(mask_file, overwrite=True)