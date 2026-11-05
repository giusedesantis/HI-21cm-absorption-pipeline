#!/usr/bin/env python3

import os
import sys
import report
import subprocess

from astropy.io import fits


if len(sys.argv) != 6:
    print('ERROR: Incorrect number of parameters.')
    exit(-1)

projectdir = sys.argv[1]
obsid = sys.argv[2]
robust = sys.argv[3]
tukey = sys.argv[4]
reportCsv = sys.argv[5]

# Define relavant file names and paths.
metafits = obsid + '.metafits'
measurementSet = obsid + '.ms'
obs = obsid
workms = obsid + '_sub_work.ms'
msigma=4                              # S/N Level at which to choose masked pixels for deepclean
autoThreshold = 1                   # S/N Threshold at which to stop cleaning
min_uv = 0
nmitter = 10
basescale=0.6
imsize=64
chans = 1536
mgain = 0.75
robust = 0.0


# Import header information from the metafits file.
metaHdu = fits.open(metafits)
metadata = metaHdu[0].header
metaHdu.close()


scale = basescale / metadata['CENTCHAN']
scale = round(scale, 8)

# Clean the two main linear polizations.
cleanCmd = f'''wsclean \
    -name /scratch/mwasci/user/images/{obsid}/{obsid}_minicube_submodel \
    -abs-mem 100 \
    -subtract-model \
    -mgain {mgain} \
    -nmiter {nmitter} \
    -niter 1000000 \
    -no-mf-weighting \
    -auto-mask {msigma} \
    -auto-threshold {autoThreshold} \
    -multiscale \
    -size {imsize} {imsize} \
    -scale {scale} \
    -weight briggs {robust} \
    -taper-inner-tukey {tukey} \
    -minuv-l {min_uv} \
    -pol xx,yy \
    -join-channels \
    -channels-out {chans} \
    -fit-spectral-pol 4 \
    -deconvolution-channels 96 \
    -channel-range 0 1536 \
    -data-column DATA \
    -shift 08h56m14.7s +02d23m59.6s \
    {workms}'''

print(cleanCmd)
subprocess.run(cleanCmd, shell=True, check=True)

report.updateObs(reportCsv, obsid, 'image', 'Success')