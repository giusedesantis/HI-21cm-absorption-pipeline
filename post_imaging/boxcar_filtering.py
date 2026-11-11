#!/usr/bin/env python3
"""Apply the original 1-D boxcar *flux* subtraction and propagate covariance.

The input covariance is a scipy.sparse CSR matrix in exactly the same row
order as the input CSV. A fixed valid-channel mask is used (as in the supplied
boxcar_filtering.py). For this linear filter, C_residual = (I-B) C (I-B)^T,
where B is the masked boxcar-averaging operator.

This reproduces the original unweighted boxcar windows (default width 16),
including the first/last half-window edge treatment. Original flux/RMS/count
columns are retained. boxcar_average_unc_Jy and the new
boxcar_subtracted_unc_Jy are now covariance-propagated uncertainties.

Usage example:
  python boxcar_filtering_with_covariance.py \
      /scratch/mwasci/user/final_spec/stack.csv \
      --covariance /scratch/mwasci/user/final_spec/stack.npz \
      --binwidth 16
"""
#Fix line 40 & 48
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--covariance', type=Path, required=True,
                        help='NPZ from back_resampling_with_covariance.py')
    parser.add_argument('--binwidth', type=int, default=16)
    parser.add_argument('-o', '--output', type=Path, default=None)
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    table.columns = table.columns.str.strip()
    required = {'Freq(MHz)', 'weighted_mean_flux_Jy', 'weighted_unc_Jy', 'N_weighted'}    #Freq(MHz)  Corrected Frequency (MHz)
    if not required.issubset(table.columns):
        raise ValueError(f'Missing columns: {sorted(required.difference(table.columns))}')
    n = len(table)
    half = args.binwidth // 2
    if args.binwidth <= 0 or args.binwidth % 2 or args.binwidth > n:
        raise ValueError('binwidth must be a positive even integer <= number of channels')

    freqs = table['Freq(MHz)'].to_numpy(float)  #Freq(MHz)
    if np.any(np.diff(freqs) <= 0):
        raise ValueError('Frequencies must be strictly increasing')
    flux = table['weighted_mean_flux_Jy'].to_numpy(float)
    uncertainty = table['weighted_unc_Jy'].to_numpy(float)
    number = table['N_weighted'].to_numpy(float)
    invalid = (~np.isfinite(flux) | ~np.isfinite(uncertainty) |
               ~np.isfinite(number) | (uncertainty <= 0) | (number <= 0))

    C = sparse.load_npz(args.covariance).tocsr()
    if C.shape != (n, n):
        raise ValueError(f'Covariance matrix shape {C.shape} != ({n},{n})')
    if not np.isfinite(C.data).all():
        raise ValueError('Non-finite entries in input covariance')

    # B maps the input flux to its boxcar average. As in the original script,
    # invalid input channels are skipped when forming each window's mean.
    r, c, d = [], [], []
    for i in range(n):
        if invalid[i]:
            continue
        if i < half:
            candidates = np.arange(0, half)
        elif i >= n - half:
            candidates = np.arange(n - half, n)
        else:
            candidates = np.arange(i - half, i + half)
        selected = candidates[~invalid[candidates]]
        if selected.size == 0:
            raise ValueError(f'No valid samples in boxcar window for channel {i}')
        r.extend([i] * len(selected))
        c.extend(selected.tolist())
        d.extend([1.0 / len(selected)] * len(selected))
    B = sparse.csr_matrix((d, (r, c)), shape=(n, n))

    # Invalid output rows are masked, so their residuals/covariances are zero.
    good_I = sparse.diags((~invalid).astype(float), format='csr')
    R = good_I - B
    clean_flux = np.where(invalid, 0.0, flux)
    model = B @ clean_flux
    residual = R @ clean_flux
    C_model = (B @ C @ B.T).tocsr()
    C_resid = (R @ C @ R.T).tocsr()

    result = table.copy()
    result['boxcar_average_Jy'] = np.where(invalid, np.nan, model)
    result['boxcar_average_unc_Jy'] = np.where(
        invalid, 0.0, np.sqrt(np.maximum(C_model.diagonal(), 0.0)))
    result['boxcar_subtracted_flux_Jy'] = np.where(invalid, 0.0, residual)
    result['boxcar_subtracted_unc_Jy'] = np.where(
        invalid, 0.0, np.sqrt(np.maximum(C_resid.diagonal(), 0.0)))
    result['boxcar_flagged'] = invalid

    output = args.output or args.input_csv.with_name(
        f'{args.input_csv.stem}_boxcar{args.binwidth}_covariance.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    model_cov_path = output.with_name(output.stem + '_model_covariance.npz')
    residual_cov_path = output.with_name(output.stem + '_residual_covariance.npz')
    result.to_csv(output, index=False)
    sparse.save_npz(model_cov_path, C_model)
    sparse.save_npz(residual_cov_path, C_resid)
    print(f'Boxcar output: {output}')
    print(f'Boxcar model covariance: {model_cov_path}')
    print(f'Boxcar residual covariance: {residual_cov_path}')
    print('Note: assumes fixed masks/window and input covariance model; '
          'not a replacement for noise characterization of fitted continuum or imaging.')


if __name__ == '__main__':
    main()
