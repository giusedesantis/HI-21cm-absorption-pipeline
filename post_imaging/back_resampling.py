#!/usr/bin/env python3
"""Rebin a barycentrically corrected 10-kHz MWA stack to 20 kHz and
propagate covariance caused by duplicating native 20-kHz measurements.

This version accommodates DIFFERENT corrected RMS values in the two halves
of each parent 20-kHz channel, caused by fitting_modified.py. The fitted
continuum/correction is treated as FIXED: uncertainty in the polynomial fit
and the resulting cross-channel covariance are not propagated.

This reproduces the fixed-grid pairing and keeps the counts from
back_resampling_fixed_grid_with_N_weighted.py. Unlike that script, it MUST
read the SAME individual 10-kHz spectrum CSVs used to build the 10-kHz stack.

ASSUMPTIONS (important):
* Each native 20-kHz measurement was duplicated at native centre +/-5 kHz;
  assigned RMS in each duplicate = sqrt(2) * native RMS.
* Each observation received a single integer * 10 kHz barycentric shift,
  as in shift.py and dip/corrections.csv.
* Before duplication, noise in distinct native channels/OBSIDs is independent.
* Flux-correction factors are treated as fixed, conditional on the observed
  continuum polynomial fit. Fit-derived and imaging noise correlations are
  NOT included.
* The per-OBSID CSVs and the corrections file are exactly those used for the
  stacked 10-kHz CSV. We verify their weights, counts and flux against it.

Output:
- usual rebinned 20-kHz CSV, with covariance-correct RMS and N_weighted
- original 10-kHz CSV left untouched
- covariance-corrected 10-kHz CSV alongside the preserved original stack
- full sparse 10-kHz and 20-kHz covariance matrices (.npz, SciPy format)
- unpaired 10-kHz channels, as before
- a 20-kHz covariance summary CSV (diagonal and next-bin covariance)

This is the conditional covariance induced by DUPLICATION AND REBINNING
under the independent-native-channel and fixed-continuum-fit model; it is
not a measurement of the complete noise covariance from imaging or fitting.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from config_utils import load_config
#FIX LINE 238 AS WELL AS 299
# ------------------------ INPUTS ------------------------------
config = load_config()
settings = config["post_imaging"]["back_resampling"]

input_file = Path(settings["input_file"])
output_file = Path(settings["output_file"])
original_dir = Path(settings["original_dir"])

# Must contain the exact source files used to create input_file.
individual_dir = Path(settings["individual_dir"])
individual_glob = settings["individual_glob"]

corrections_file = Path(settings["corrections_file"])

# Names in the individual spectrum CSVs used in safe_stacking.py.
IND_FREQ_COL = 'Freq(MHz)' #Freq(MHz) Corrected Frequency (MHz)
IND_FLUX_COL = 'Subtracted flux(Jy)'  #Simulated flux(Jy)  Subtracted flux(Jy)
IND_RMS_COL = 'rms (Jy)'

GRID_OFFSET_KHZ = 0    # 0 = original 20-kHz grid; 10 = alternate grid
GRID_ANCHOR_MHZ = None # original 20-kHz centre, or read from original_dir
ORIGINAL_FREQ_COL = 'Central Frequency(MHz)'
FREQ_COL = 'Freq(MHz)'      # Corrected Frequency (MHz)  # Freq(MHz)
FLUX_COL = 'weighted_mean_flux_Jy'
RMS_COL = 'weighted_unc_Jy'
COUNT_COLS = ['N', 'N_weighted', 'N_zero_rms']
STEP_10_MHZ = .010
WIDTH_20_MHZ = .020
HALF_WIDTH_MHZ = .005
GRID_TOLERANCE_MHZ = .00055
PAIR_TOLERANCE_MHZ = .00010


def fkey(freq_mhz):
    """Integer kHz key, matching safe_stacking.py rounding to 0.001 MHz."""
    return int(np.rint(float(freq_mhz) * 1000.0))


def read_observation(file):
    """Mirror safe_stacking.py and inspect fitting_modified.py validity markers.

    A flagged or sigma-clipped row with zero corrected flux but *positive* RMS
    is erroneously treated as valid by the original stacking script. Refuse to
    propagate a covariance for those inputs until the RMS-mask bug is fixed.
    """
    df0 = pd.read_csv(file)
    df0.columns = df0.columns.str.strip()
    cols = [IND_FREQ_COL, IND_FLUX_COL, IND_RMS_COL]
    missing = [c for c in cols if c not in df0.columns]
    if missing:
        raise KeyError(f'{file}: missing {missing}')
    extra = ['Observed fit flux(Jy)', 'Model fit flux(Jy)']
    available = [c for c in extra if c in df0.columns]
    df0 = df0[cols + available].copy()
    for col in cols + available:
        df0[col] = pd.to_numeric(df0[col], errors='raise')
    df0 = df0.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    if 'Observed fit flux(Jy)' in df0:
        # The continuum-fit script leaves zeros for rejected/flagged channels,
        # but *currently* initializes their output RMS to the input RMS.
        invalid_but_weighted = (
            (df0[IND_RMS_COL] > 0)
            & (~np.isfinite(df0['Observed fit flux(Jy)'])
               | (df0['Observed fit flux(Jy)'] == 0))
        )
        if invalid_but_weighted.any():
            raise ValueError(
                f'{file.name}: {int(invalid_but_weighted.sum())} channels have '
                'no fitted/processed flux but positive RMS. '
                'In fitting_modified.py initialise rms_corr_out to zeros '
                'instead of rms_all, regenerate spectra, and restack.'
            )

    # Preserve the sign of the multiplicative correction, if present.
    # Model continuum is positive in the provided fitting script; a negative
    # fitted baseline would yield a negative multiplicative correction.
    if all(c in df0 for c in extra):
        ratio = (df0['Model fit flux(Jy)'].to_numpy(dtype=float)
                 / df0['Observed fit flux(Jy)'].replace(0, np.nan).to_numpy(dtype=float))
        signs = np.sign(ratio)
        signs[~np.isfinite(signs)] = 1.0  # zero-RMS rows are skipped downstream
    else:
        signs = np.ones(len(df0), dtype=float)
        print(f'WARNING {file.name}: no fit columns; assuming positive corrections.')
    df0['_correction_sign'] = signs
    df0['_key'] = np.rint(df0[IND_FREQ_COL].to_numpy() * 1000).astype(np.int64)
    if df0['_key'].duplicated().any():
        raise ValueError(f'{file}: same OBSID supplies duplicate 10-kHz frequencies')
    return df0


# ------------------ FIXED GRID / 20-kHz PAIRS ----------------------
if GRID_OFFSET_KHZ not in (0, 10):
    raise ValueError('GRID_OFFSET_KHZ must be 0 or 10')
if GRID_ANCHOR_MHZ is None:
    orig_files = sorted(original_dir.glob('*_spec.csv'))
    if not orig_files:
        raise FileNotFoundError(f'No original spectra in {original_dir}')
    first = pd.read_csv(orig_files[0], nrows=1)
    first.columns = first.columns.str.strip()
    if first.empty or ORIGINAL_FREQ_COL not in first.columns:
        raise ValueError(f'Cannot get original 20-kHz grid from {orig_files[0]}')
    original_anchor = float(first[ORIGINAL_FREQ_COL].iloc[0])
else:
    original_anchor = float(GRID_ANCHOR_MHZ)
anchor = original_anchor + GRID_OFFSET_KHZ / 1000.0
print(f'Original 20-kHz grid anchor: {original_anchor:.6f} MHz')
print(f'Chosen output grid anchor:   {anchor:.6f} MHz')

stack = pd.read_csv(input_file)
stack.columns = stack.columns.str.strip()
required = [FREQ_COL, FLUX_COL, RMS_COL, 'N_weighted']
missing = [c for c in required if c not in stack.columns]
if missing:
    raise KeyError(f'Missing stacked spectrum columns: {missing}')
if stack.empty:
    raise ValueError('Stacked CSV is empty')
for col in required:
    stack[col] = pd.to_numeric(stack[col], errors='raise')
if (not np.isfinite(stack[FREQ_COL]).all()
        or not np.isfinite(stack['N_weighted']).all()
        or (stack['N_weighted'] < 0).any()):
    raise ValueError('Invalid stacked frequencies or N_weighted')
stack = stack.sort_values(FREQ_COL).reset_index(drop=True)
if stack[FREQ_COL].duplicated().any():
    raise ValueError('Duplicate 10-kHz frequency in stack')

freqs = stack[FREQ_COL].to_numpy(float)
full_10_keys = [fkey(f) for f in freqs]
if len(set(full_10_keys)) != len(full_10_keys):
    raise ValueError('Frequencies collide after rounding to 0.001 MHz')
keys_to_10idx = {k: i for i, k in enumerate(full_10_keys)}

bin_id = np.rint((freqs - anchor) / WIDTH_20_MHZ).astype(np.int64)
centres = anchor + bin_id * WIDTH_20_MHZ
delta = freqs - centres
if not np.isclose(np.abs(delta), HALF_WIDTH_MHZ,
                  rtol=0, atol=GRID_TOLERANCE_MHZ).all():
    raise ValueError('Input 10-kHz channels do not lie on chosen 20-kHz grid')
stack['_bin_id'] = bin_id
stack['_side'] = np.where(delta < 0, 'lower', 'upper')
if stack.duplicated(['_bin_id', '_side']).any():
    raise ValueError('Duplicate 10-kHz position in an output bin')
low = stack[stack['_side'].eq('lower')].set_index('_bin_id')
up = stack[stack['_side'].eq('upper')].set_index('_bin_id')
complete = low.index.intersection(up.index).sort_values()
if len(complete) == 0:
    raise ValueError('No complete output 20-kHz pairs')
ch1, ch2 = low.loc[complete], up.loc[complete]
if not np.allclose(ch2[FREQ_COL].to_numpy() - ch1[FREQ_COL].to_numpy(),
                   STEP_10_MHZ, rtol=0, atol=PAIR_TOLERANCE_MHZ):
    raise ValueError('Pair includes nonadjacent 10-kHz channels')

out = pd.DataFrame({
    FREQ_COL: anchor + complete.to_numpy() * WIDTH_20_MHZ,
    FLUX_COL: (ch1[FLUX_COL].to_numpy() + ch2[FLUX_COL].to_numpy()) / 2.,
})
for col in COUNT_COLS:
    if col in stack.columns:
        out[f'{col}_lower10kHz'] = ch1[col].to_numpy()
        out[f'{col}_upper10kHz'] = ch2[col].to_numpy()
out['N_weighted'] = np.minimum(ch1['N_weighted'].to_numpy(),
                               ch2['N_weighted'].to_numpy()).astype(int)

# Which 10-kHz frequencies feed each 20-kHz bin?
complete_to_outidx = {int(b): i for i, b in enumerate(complete)}
keys_to_outidx = {}
for b, ii in complete_to_outidx.items():
    keys_to_outidx[fkey(ch1.loc[b, FREQ_COL])] = ii
    keys_to_outidx[fkey(ch2.loc[b, FREQ_COL])] = ii

unpaired = stack[~stack['_bin_id'].isin(complete)].copy()
unpaired = unpaired.drop(columns=['_bin_id', '_side'])

# ------------------ INDIVIDUAL OBSID SPECTRA -----------------------
individual_files = sorted(individual_dir.glob(individual_glob))
if not individual_files:
    raise FileNotFoundError(f'No {individual_glob} in {individual_dir}. '
                            'Covariance cannot be recovered from stack alone.')
corrections = pd.read_csv(corrections_file, dtype={'obsid': str})
corrections.columns = corrections.columns.str.strip()
corrections['obsid'] = corrections['obsid'].astype(str).str.strip()
corr_col = 'frequency correction(MHz)'
if corr_col not in corrections.columns:
    raise KeyError(f'Missing {corr_col!r} in {corrections_file}')
if corrections['obsid'].duplicated().any():
    raise ValueError('Duplicate obsid in corrections.csv')
correction_by_obsid = dict(zip(corrections['obsid'],
                               corrections[corr_col].astype(float)))

# First pass: reconstruct the exact 10-kHz inverse-variance stacking weights.
weight_sums = defaultdict(float)
weighted_flux_sums = defaultdict(float)
valid_counts = defaultdict(int)
seen_obsids = set()
for file in individual_files:
    obsid = file.stem.removeprefix('spectrum_') # file.stem.removeprefix('spectrum_').removesuffix('_sim_correct')     file.stem.removeprefix('spectrum_')
    if obsid in seen_obsids:
        raise ValueError(f'Duplicate OBSID: {obsid}')
    seen_obsids.add(obsid)
    if obsid not in correction_by_obsid:
        raise ValueError(f'Missing barycentric correction for OBSID {obsid}')
    sp = read_observation(file)
    for key, flux, rms in sp[['_key', IND_FLUX_COL, IND_RMS_COL]].itertuples(index=False, name=None):
        key = int(key)
        if key not in keys_to_10idx:
            raise ValueError(f'{file.name}: frequency {key/1000:.3f} not in 10-kHz stack. '
                             'Ensure same OBSIDs and inputs were used.')
        if rms > 0:
            w = 1.0 / float(rms)**2
            weight_sums[key] += w
            weighted_flux_sums[key] += w * float(flux)
            valid_counts[key] += 1

# Fail loudly if the original observations do not exactly reproduce the stack.
for idx, key in enumerate(full_10_keys):
    observed_count = int(stack['N_weighted'].iloc[idx])
    if valid_counts[key] != observed_count:
        raise ValueError(f'Input mismatch at {key/1000:.3f} MHz: '
                         f'saved N_weighted={observed_count}, sources={valid_counts[key]}. '
                         'Use exactly the individual spectra included in the stack.')
    if weight_sums[key] > 0:
        calculated_flux = weighted_flux_sums[key] / weight_sums[key]
        calculated_unc = np.sqrt(1.0 / weight_sums[key])
        if not np.isclose(calculated_flux, stack[FLUX_COL].iloc[idx],
                          rtol=1e-5, atol=1e-6):
            raise ValueError(f'Flux mismatch at {key/1000:.3f} MHz. '
                             'Cannot compute trustworthy covariance.')
        if not np.isclose(calculated_unc, stack[RMS_COL].iloc[idx],
                          rtol=1e-4, atol=1e-7):
            raise ValueError(f'RMS mismatch at {key/1000:.3f} MHz. '
                             'Check input RMS columns / stacking weights.')
    elif observed_count != 0:
        raise ValueError(f'Missing weighted data at {key/1000:.3f} MHz')
print(f'Validated all {len(full_10_keys)} stacked channels against '
      f'{len(individual_files)} individual OBSID spectra.')

# Second pass: for each native 20-kHz datum, collect how much of it contributes
# to each 10-kHz output and each rebinned 20-kHz output. Then accumulate a*C*a^T.
var10 = np.zeros(len(stack), dtype=float)
var20 = np.zeros(len(out), dtype=float)
# Covariances between DISTINCT output channels (triangular part only).
off10 = defaultdict(float)
off20 = defaultdict(float)


def accumulate(terms, native_variance, diagonal, off_diagonal):
    # terms maps OUTPUT CHANNEL INDEX -> coefficient in linear estimator.
    entries = sorted(terms.items())
    for idx, coeff in entries:
        diagonal[idx] += native_variance * coeff * coeff
    for i, (idx, coeff) in enumerate(entries):
        for jdx, other in entries[i+1:]:
            off_diagonal[(idx, jdx)] += native_variance * coeff * other


for file in individual_files:
    obsid = file.stem.removeprefix('spectrum_') #file.stem.removeprefix('spectrum_').removesuffix('_sim_correct')   file.stem.removeprefix('spectrum_')
    shift = STEP_10_MHZ * np.rint(correction_by_obsid[obsid] / STEP_10_MHZ)
    sp = read_observation(file)
    # Every original 20-kHz datum has one underlying noise realisation Z
    # (unit variance). The two daughter bins have corrected noise amplitudes
    # sign(ratio) * corrected_rms / sqrt(2), which need not be identical.
    # For each original channel, accumulate its coefficient in each output bin.
    parents = {}
    for freq, key, rms, correction_sign in sp[[
        IND_FREQ_COL, '_key', IND_RMS_COL, '_correction_sign'
    ]].itertuples(index=False, name=None):
        if rms <= 0:
            continue
        key = int(key)
        native_n = int(np.rint((float(freq) - shift - original_anchor) / WIDTH_20_MHZ))
        native_centre = original_anchor + native_n * WIDTH_20_MHZ
        error = abs(abs(float(freq) - shift - native_centre) - HALF_WIDTH_MHZ)
        if error > GRID_TOLERANCE_MHZ:
            raise ValueError(f'{file.name}, {float(freq):.6f} MHz: cannot map '
                             f'to native 20-kHz channel; error={error:.6f} MHz. '
                             'Check corrections.csv and per-OBSID files.')
        weight = 1.0 / float(rms)**2
        coefficient10 = weight / weight_sums[key]
        # Physical uncertainty amplitude per corrected daughter, given the
        # original *artificial* sqrt(2) RMS scaling convention.
        noise_amplitude = float(correction_sign) * float(rms) / np.sqrt(2.0)
        parent = parents.setdefault(native_n, {
            'a10': defaultdict(float), 'a20': defaultdict(float)
        })
        parent['a10'][keys_to_10idx[key]] += coefficient10 * noise_amplitude
        if key in keys_to_outidx:
            parent['a20'][keys_to_outidx[key]] += coefficient10 * noise_amplitude / 2.0

    # Each parent has a unit-variance underlying random variable. Any shared
    # contribution across bins therefore creates covariance.
    for parent in parents.values():
        accumulate(parent['a10'], 1.0, var10, off10)
        accumulate(parent['a20'], 1.0, var20, off20)

out[RMS_COL] = np.sqrt(np.maximum(0.0, var20))
# Zero RMS for invalid bins, as expected by boxcar_filtering.py.
out.loc[out['N_weighted'].eq(0), RMS_COL] = 0.0


def sparse_covariance(diagonal, offdiag):
    n = len(diagonal)
    rows = list(range(n))
    cols = list(range(n))
    data = list(diagonal)
    for (i, j), v in offdiag.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([v, v])
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


cov10 = sparse_covariance(var10, off10)
cov20 = sparse_covariance(var20, off20)
# For the 10-kHz output, we must also correct the formal RMS values: the
# 10-kHz CSV's original RMS was assigned using a sqrt(2) resampling convention.
# We preserve that original file, and produce a separate covariance-corrected
# 10-kHz CSV with otherwise identical flux and count columns.
stack10_corrected = stack.drop(columns=['_bin_id', '_side']).copy()
stack10_corrected[RMS_COL] = np.sqrt(np.maximum(0.0, var10))
stack10_corrected.loc[stack10_corrected['N_weighted'].eq(0), RMS_COL] = 0.0
# Any invalid output channels should be masked when fitting a model.

# Diagnostics: adjoining 20-kHz frequency bins only (gaps give NaN).
adj_cov = np.full(len(out), np.nan)
for i in range(len(out) - 1):
    if np.isclose(float(out[FREQ_COL].iloc[i+1] - out[FREQ_COL].iloc[i]),
                  WIDTH_20_MHZ, rtol=0, atol=PAIR_TOLERANCE_MHZ):
        adj_cov[i] = off20.get((i, i+1), 0.0)
summary = pd.DataFrame({
    FREQ_COL: out[FREQ_COL],
    'variance_Jy2': var20,
    'RMS_covariance_Jy': np.sqrt(np.maximum(var20, 0)),
    'covariance_with_next_20kHz_Jy2': adj_cov,
    'N_weighted': out['N_weighted'],
})

output_file.parent.mkdir(parents=True, exist_ok=True)
stack10_corrected_file = input_file.with_name(input_file.stem + '_covariance.csv')
file10 = input_file.with_name(input_file.stem + '_covariance_matrix.npz')
file20 = output_file.with_name(output_file.stem + '_matrix.npz')
summary_file = output_file.with_name(output_file.stem + '_covariance_summary.csv')
unpaired_file = output_file.with_name(output_file.stem + '_unpaired_10kHz.csv')

out.to_csv(output_file, index=False)
stack10_corrected.to_csv(stack10_corrected_file, index=False)
unpaired.to_csv(unpaired_file, index=False)
summary.to_csv(summary_file, index=False)
sparse.save_npz(file10, cov10)
sparse.save_npz(file20, cov20)

print('\nDone: conditional duplication-induced covariance propagated.')
print(f'Input 10-kHz channels: {len(stack)}')
print(f'Rebinned 20-kHz channels: {len(out)}')
print(f'Unpaired 10-kHz channels: {len(unpaired)}')
print(f'Rebinned spectrum: {output_file}')
print(f'10-kHz stack with covariance-corrected RMS: {stack10_corrected_file}')
print(f'10-kHz covariance (same row order as input): {file10}')
print(f'20-kHz covariance (same row order as output): {file20}')
print(f'20-kHz covariance summary: {summary_file}')
print(f'Unpaired 10-kHz report: {unpaired_file}')
print('IMPORTANT: conditional covariance of duplication/shift/rebin operations only; '
      'additional correlations from imaging/continuum subtraction are not modelled.')
