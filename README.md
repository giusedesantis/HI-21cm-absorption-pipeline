# H I 21-cm Absorption Pipeline

A Python high-performance computing (HPC) pipeline for searching for neutral hydrogen (H I) 21-cm absorption in high-redshift radio galaxies using observations from the Murchison Widefield Array (MWA).

## Overview

This repository contains the computational pipeline developed for the processing and analysis of low-frequency radio observations, with the aim of identifying H I 21-cm absorption signatures in distant radio galaxies.

Detecting these absorption features provides a way to investigate the presence and properties of neutral hydrogen in the early Universe. However, searching for weak spectral signatures requires careful calibration, imaging, spectral extraction, frequency corrections, and combination of observations.

The pipeline brings together radio-interferometric data processing, automated spectral analysis, uncertainty propagation, and machine-learning methods. It is designed to support the processing of large observational datasets on HPC systems, with development and execution carried out on the Pawsey Supercomputing Research Centre's Setonix supercomputer.

## Pipeline Components

The repository is organised into the following modules:

| Directory | Description |
|---|---|
| `download/` | Observation downloading through the MWA ASVO service and processing-status management |
| `bin/` | Imaging preparation, image masking, and WSClean-based image processing |
| `flux_extract/` | Conventional source flux-density extraction routines |
| `ML/` | Machine-learning-based approach to source localisation and spatial extent estimation, developed as an alternative to the conventional extraction method |
| `post_imaging/` | Spectral corrections, resampling, continuum subtraction, stacking, covariance propagation, filtering, and plotting |
| `slurm/` | SLURM scripts for coordinating and scheduling processing jobs on HPC systems |
| `config/` | Centralised configuration files for data locations, target information, and processing settings |

## Processing Workflow

The workflow consists of the following main stages:

1. **Observation management:** Download MWA observations and track their processing status.
2. **Imaging:** Produce radio images and spectral products using WSClean, including mask generation and image deconvolution.
3. **Source analysis:** Extract source information and investigate machine-learning approaches to source localisation and spatial extent estimation.
4. **Spectral processing:** Apply frequency corrections, resampling, and continuum fitting and subtraction.
5. **Spectral stacking:** Combine individual spectra using inverse-variance weighting to improve sensitivity.
6. **Uncertainty propagation:** Account for spectral-channel covariance in subsequent resampling and filtering.
7. **Analysis and visualisation:** Generate final stacked spectra and diagnostic figures.

### Post-imaging execution order

The post-imaging SLURM workflow is organised as follows:

1. `barycentric_correct.py`
2. `resampling.py`
3. `shift.py`
4. `flux_fitting.py`
5. `stacking.py`
6. `back_resampling.py`
7. `boxcar_filtering.py`
8. `plotting.py`

Each stage is intended to provide the data products required for subsequent analysis.

## Technologies and Dependencies

The pipeline uses several scientific computing and astronomical software tools:

- **Python:** NumPy, pandas, SciPy, Astropy, Matplotlib, and PyYAML
- **Radio astronomy:** WSClean and associated MWA processing tools
- **Machine learning:** U-Net-based source analysis
- **HPC infrastructure:** SLURM and Singularity/Apptainer containers
- **Data products:** FITS images, measurement sets, CSV spectra, and sparse covariance matrices

Additional dependencies vary by processing stage and execution environment.

## Configuration

Pipeline parameters and input/output paths are defined in `config/config.yaml`.

An example configuration is provided in `config/config.example.yaml`.

To create a local configuration:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config.yaml` to specify the relevant data directories, target parameters, and processing settings.

The local configuration is excluded from Git version control. Large observational datasets, intermediate data products, and processing logs are also intended to remain outside the repository.

## Running on Setonix

The repository contains SLURM submission scripts for imaging and post-imaging processing.

Example submission commands, from the repository root:

```bash
sbatch slurm/imaging_auto.sbatch
```

```bash
sbatch slurm/post_imaging.sbatch
```

Before submission, the user must configure the required input files, container paths, software environment, and SLURM resource settings.

**Note:** The repository is research software under active development. Some scripts and workflows remain specific to the Setonix environment, and the complete end-to-end workflow is undergoing integration and validation. The commands above are examples, not a guarantee that the pipeline will run without further environment-specific configuration.

## Scientific Considerations

Particular attention is given to several aspects of low-frequency absorption-line analysis:

- Identifying and accounting for instrumental and spectral-channel effects.
- Applying barycentric corrections and maintaining consistent frequency grids across observations.
- Handling flagged channels and invalid measurements.
- Performing continuum subtraction while preserving potential absorption features.
- Combining observations using noise-based weighting.
- Propagating covariance introduced by spectral resampling and filtering.

These procedures are intended to support reliable interpretation of the resulting spectra and their uncertainties.

## Project Status

This repository is under active development as part of an astrophysical research project.

Current work includes consolidating the processing modules, improving configuration portability, integrating the analysis stages, and validating the processing and uncertainty-propagation methods.

No claim of an H I absorption detection is made by this repository.

## Acknowledgements

This work uses observations from the Murchison Widefield Array and computing resources provided by the Pawsey Supercomputing Research Centre.

Relevant software, observational datasets, and collaborative contributions will be acknowledged in the associated research outputs.

## Usage and Attribution

This repository is currently maintained as research software. No open-source licence has been assigned. Permission and appropriate attribution should be established before external redistribution or reuse of code developed collaboratively.