#!/usr/bin/env python3
"""Preprocess raw LUMINA spectra: interpolate, adaptive smooth, asinh transform, PCA.

v2: asinh transform + adaptive smoothing + feature reconstruction validation.

Usage:
  python3 scripts/02_preprocess.py
  python3 scripts/02_preprocess.py --variance 0.999
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumina_ml import config as cfg
from lumina_ml.preprocessing import (
    SpectralPCA, ParamScaler, preprocess_spectrum, validate_spectrum,
    measure_spectral_features, asinh_inverse,
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess spectra for emulator training")
    parser.add_argument('--variance', type=float, default=cfg.PCA_VARIANCE_THRESHOLD,
                        help=f'PCA variance threshold (default: {cfg.PCA_VARIANCE_THRESHOLD})')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Validation split fraction (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split')
    args = parser.parse_args()

    # Load raw data
    print("Loading raw data...")
    params_all = np.load(str(cfg.DATA_RAW / "params_all.npy"))
    spectra_raw = np.load(str(cfg.DATA_RAW / "spectra_all.npy"))
    waves_raw = np.load(str(cfg.DATA_RAW / "waves_all.npy"))

    print(f"  Loaded: {params_all.shape[0]} models, {params_all.shape[1]}D params, "
          f"{spectra_raw.shape[1]} wavelength bins")

    # Filter out failed models (all-zero spectra)
    valid_mask = np.any(spectra_raw != 0, axis=1)
    print(f"  Non-zero spectra: {valid_mask.sum()}/{len(valid_mask)}")

    params_valid = params_all[valid_mask]
    spectra_valid = spectra_raw[valid_mask]
    waves_valid = waves_raw[valid_mask]

    # Preprocess: interpolate -> adaptive smooth -> normalize -> asinh
    print(f"\nPreprocessing {len(params_valid)} spectra...")
    print(f"  Adaptive SG smoothing: {cfg.SG_REGIONS}")
    print(f"  asinh softening: {cfg.ASINH_SOFTENING}")

    n_bins = len(cfg.SPECTRUM_GRID)
    spectra_processed = np.zeros((len(params_valid), n_bins))
    keep = []

    for i in range(len(params_valid)):
        spec = preprocess_spectrum(waves_valid[i], spectra_valid[i], use_asinh=True)
        if validate_spectrum(spec, use_asinh=True):
            spectra_processed[i] = spec
            keep.append(i)

    keep = np.array(keep)
    spectra_processed = spectra_processed[keep]
    params_valid = params_valid[keep]
    print(f"  Valid after preprocessing: {len(keep)}")

    # Check asinh range
    print(f"\n  asinh spectrum stats:")
    print(f"    Range: [{spectra_processed.min():.3f}, {spectra_processed.max():.3f}]")
    print(f"    Expected peak: asinh(1/{cfg.ASINH_SOFTENING}) = "
          f"{np.arcsinh(1.0/cfg.ASINH_SOFTENING):.3f}")

    # Feature measurements on a sample
    print("\n  Feature measurements (10 random samples):")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(keep), min(10, len(keep)), replace=False)
    all_features = {}
    for idx in sample_idx:
        feats = measure_spectral_features(spectra_processed[idx], is_asinh=True)
        for k, v in feats.items():
            all_features.setdefault(k, []).append(v)
    for k, vals in all_features.items():
        print(f"    {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # PCA
    print(f"\nFitting PCA (variance threshold = {args.variance})...")
    pca = SpectralPCA(variance_threshold=args.variance)
    pca.fit(spectra_processed)

    # PCA coefficients
    pca_coeffs = pca.transform(spectra_processed)
    print(f"  PCA coefficients shape: {pca_coeffs.shape}")

    # Reconstruction errors
    recon_err = pca.reconstruction_error(spectra_processed)
    print(f"  Global reconstruction RMS: mean={recon_err.mean():.6f}, max={recon_err.max():.6f}")

    # Feature-specific reconstruction errors
    feat_err = pca.feature_reconstruction_error(spectra_processed)
    print(f"\n  Feature reconstruction errors (asinh space):")
    for name, stats in feat_err.items():
        print(f"    {name:15s}: mean={stats['mean']:.6f}  max={stats['max']:.6f}")

    # Convert to linear flux for interpretable RMS
    recon_asinh = pca.inverse_transform(pca.transform(spectra_processed))
    spec_linear = asinh_inverse(spectra_processed)
    recon_linear = asinh_inverse(recon_asinh)
    rms_linear = np.sqrt(np.mean((spec_linear - recon_linear)**2, axis=1))
    print(f"\n  Linear flux reconstruction RMS: mean={rms_linear.mean():.6f}, "
          f"max={rms_linear.max():.6f}")

    # Normalize parameters to [0, 1]
    param_scaler = ParamScaler()
    params_norm = param_scaler.transform(params_valid)

    # Train/val split
    rng = np.random.default_rng(args.seed)
    n_total = len(params_valid)
    n_val = int(n_total * args.val_fraction)
    n_train = n_total - n_val

    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = params_norm[train_idx]
    X_val = params_norm[val_idx]
    Y_train = pca_coeffs[train_idx]
    Y_val = pca_coeffs[val_idx]

    # Also save raw params + spectra for SBI
    params_train = params_valid[train_idx]
    params_val = params_valid[val_idx]
    spectra_train = spectra_processed[train_idx]
    spectra_val = spectra_processed[val_idx]

    print(f"\nSplit: {n_train} train, {n_val} val")

    # Save everything
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    np.save(str(cfg.DATA_PROCESSED / "X_train.npy"), X_train)
    np.save(str(cfg.DATA_PROCESSED / "X_val.npy"), X_val)
    np.save(str(cfg.DATA_PROCESSED / "Y_train.npy"), Y_train)
    np.save(str(cfg.DATA_PROCESSED / "Y_val.npy"), Y_val)
    np.save(str(cfg.DATA_PROCESSED / "params_train.npy"), params_train)
    np.save(str(cfg.DATA_PROCESSED / "params_val.npy"), params_val)
    np.save(str(cfg.DATA_PROCESSED / "spectra_train.npy"), spectra_train)
    np.save(str(cfg.DATA_PROCESSED / "spectra_val.npy"), spectra_val)

    pca.save(cfg.DATA_PROCESSED / "pca_model.pkl")
    param_scaler.save(cfg.DATA_PROCESSED / "param_scaler.pkl")

    print(f"\nSaved to {cfg.DATA_PROCESSED}/")
    print(f"  X_train:  {X_train.shape}  (normalized params)")
    print(f"  Y_train:  {Y_train.shape}  (standardized PCA coeffs)")
    print(f"  X_val:    {X_val.shape}")
    print(f"  Y_val:    {Y_val.shape}")
    print(f"  params_train:  {params_train.shape}  (physical params for SBI)")
    print(f"  spectra_train: {spectra_train.shape}  (asinh-transformed spectra)")
    print(f"  pca_model.pkl, param_scaler.pkl")


if __name__ == '__main__':
    main()
