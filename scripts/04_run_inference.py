#!/usr/bin/env python3
"""Run Bayesian inference on observed SN 2011fe spectrum.

v2: Feature-aware likelihood with wavelength-dependent sigma.

Usage:
  python3 scripts/04_run_inference.py --method mcmc
  python3 scripts/04_run_inference.py --method sbi
  python3 scripts/04_run_inference.py --method dynesty
  python3 scripts/04_run_inference.py --method all
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumina_ml import config as cfg
from lumina_ml.preprocessing import (
    SpectralPCA, ParamScaler, interpolate_to_grid, adaptive_smooth,
    peak_normalize, asinh_transform, measure_spectral_features,
)
from lumina_ml.emulator import Emulator
from lumina_ml.inference import (
    FeatureAwareLikelihood, compute_sigma_array,
    run_mcmc, run_nested, run_sbi, summary_statistics,
)


def load_observed_spectrum():
    """Load and preprocess the observed SN 2011fe B-max spectrum.

    Applies same pipeline as training data: interpolate -> adaptive smooth ->
    peak normalize -> asinh transform.
    """
    obs = np.genfromtxt(str(cfg.OBS_FILE_BMAX), delimiter=',', names=True)
    wave = obs['wavelength_angstrom']
    flux = obs['flux_erg_s_cm2_angstrom']

    # Interpolate to standard grid
    grid_flux = np.interp(cfg.SPECTRUM_GRID, wave, flux, left=0.0, right=0.0)

    # Adaptive smooth (same as training data)
    grid_flux = adaptive_smooth(grid_flux)

    # Peak normalize
    normalized, peak = peak_normalize(grid_flux)

    # asinh transform (same as training data)
    return asinh_transform(normalized)


def print_summary(method_name, results):
    """Print posterior summary statistics."""
    stats = summary_statistics(results['samples'])
    print(f"\n{'='*60}")
    print(f"{method_name} -- Posterior Summary")
    print(f"{'='*60}")
    print(f"  {'Parameter':22s} {'Median':>10s} {'Std':>8s} {'16%':>10s} {'84%':>10s}")
    print(f"  {'-'*58}")
    for name in cfg.PARAM_NAMES:
        s = stats[name]
        print(f"  {name:22s} {s['median']:10.3f} {s['std']:8.3f} "
              f"{s['CI_16']:10.3f} {s['CI_84']:10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian inference on SN 2011fe")
    parser.add_argument('--method', choices=['mcmc', 'dynesty', 'sbi', 'all'],
                        default='all', help='Inference method (default: all)')
    parser.add_argument('--mcmc-walkers', type=int, default=cfg.MCMC_N_WALKERS)
    parser.add_argument('--mcmc-burn', type=int, default=cfg.MCMC_N_BURN)
    parser.add_argument('--mcmc-production', type=int, default=cfg.MCMC_N_PRODUCTION)
    parser.add_argument('--sbi-samples', type=int, default=10000)
    parser.add_argument('--dynesty-live', type=int, default=cfg.DYNESTY_LIVE_POINTS)
    args = parser.parse_args()

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load observed spectrum (asinh-transformed)
    print("Loading observed spectrum...")
    obs_spectrum = load_observed_spectrum()
    print(f"  Grid: {len(obs_spectrum)} bins ({cfg.WAVE_MIN}-{cfg.WAVE_MAX} A)")
    print(f"  asinh range: [{obs_spectrum.min():.3f}, {obs_spectrum.max():.3f}]")

    # Measure observed features for feature-aware likelihood
    obs_features = measure_spectral_features(obs_spectrum, is_asinh=True)
    print(f"\n  Observed features:")
    for k, v in obs_features.items():
        print(f"    {k}: {v:.4f}")

    # Wavelength-dependent sigma
    sigma_array = compute_sigma_array()
    print(f"\n  Sigma array: min={sigma_array.min():.4f}, max={sigma_array.max():.4f}")

    methods = [args.method] if args.method != 'all' else ['mcmc', 'sbi', 'dynesty']

    # Load emulator for MCMC/dynesty
    emulator = None
    if 'mcmc' in methods or 'dynesty' in methods:
        print("\nLoading trained emulator...")
        emulator = Emulator.load(cfg.MODELS_DIR, cfg.DATA_PROCESSED)
        print(f"  Device: {emulator.device}")

    # ===== MCMC =====
    if 'mcmc' in methods:
        print(f"\n{'='*60}")
        print("MCMC Inference (emcee) -- Feature-aware likelihood")
        print(f"{'='*60}")

        likelihood = FeatureAwareLikelihood(
            emulator, obs_spectrum,
            sigma_array=sigma_array,
            mode='spectrum',
            obs_features=obs_features,
        )

        t0 = time.time()
        mcmc_results = run_mcmc(
            likelihood,
            n_walkers=args.mcmc_walkers,
            n_burn=args.mcmc_burn,
            n_production=args.mcmc_production,
        )
        print(f"  Total MCMC time: {time.time()-t0:.1f}s")

        np.save(str(cfg.RESULTS_DIR / "mcmc_samples.npy"), mcmc_results['samples'])
        np.save(str(cfg.RESULTS_DIR / "mcmc_log_prob.npy"), mcmc_results['log_prob'])
        print(f"  Saved to {cfg.RESULTS_DIR}/mcmc_*.npy")
        print_summary("MCMC", mcmc_results)

    # ===== SBI =====
    if 'sbi' in methods:
        print(f"\n{'='*60}")
        print("SBI Inference (Neural Posterior Estimation)")
        print(f"{'='*60}")

        params_train = np.load(str(cfg.DATA_PROCESSED / "params_train.npy"))
        spectra_train = np.load(str(cfg.DATA_PROCESSED / "spectra_train.npy"))
        print(f"  Training pairs: {len(params_train)}")

        t0 = time.time()
        sbi_results = run_sbi(
            params_train, spectra_train, obs_spectrum,
            n_posterior_samples=args.sbi_samples,
        )
        print(f"  Total SBI time: {time.time()-t0:.1f}s")

        np.save(str(cfg.RESULTS_DIR / "sbi_samples.npy"), sbi_results['samples'])
        np.save(str(cfg.RESULTS_DIR / "sbi_log_prob.npy"), sbi_results['log_prob'])
        print(f"  Saved to {cfg.RESULTS_DIR}/sbi_*.npy")
        print_summary("SBI", sbi_results)

    # ===== Nested Sampling =====
    if 'dynesty' in methods:
        print(f"\n{'='*60}")
        print("Nested Sampling (dynesty) -- Feature-aware likelihood")
        print(f"{'='*60}")

        likelihood = FeatureAwareLikelihood(
            emulator, obs_spectrum,
            sigma_array=sigma_array,
            mode='spectrum',
            obs_features=obs_features,
        )

        t0 = time.time()
        nested_results = run_nested(likelihood, n_live=args.dynesty_live)
        print(f"  Total dynesty time: {time.time()-t0:.1f}s")

        np.save(str(cfg.RESULTS_DIR / "dynesty_samples.npy"), nested_results['samples'])
        print(f"  log(Z) = {nested_results['logz']:.2f} +/- {nested_results['logzerr']:.2f}")
        print(f"  Saved to {cfg.RESULTS_DIR}/dynesty_*.npy")
        print_summary("Nested Sampling", nested_results)

    # Comparison
    if len(methods) > 1:
        print(f"\n{'='*60}")
        print("Method Comparison (Posterior Medians)")
        print(f"{'='*60}")
        headers = ['Parameter'] + [m.upper() for m in methods]
        print(f"  {headers[0]:22s}" + "".join(f" {h:>12s}" for h in headers[1:]))
        print(f"  {'-'*58}")

        all_stats = {}
        for m in methods:
            samples = np.load(str(cfg.RESULTS_DIR / f"{m}_samples.npy"))
            all_stats[m] = summary_statistics(samples)

        for name in cfg.PARAM_NAMES:
            row = f"  {name:22s}"
            for m in methods:
                row += f" {all_stats[m][name]['median']:12.3f}"
            print(row)

    print("\nDone! Run 05_plot_results.py to generate plots.")


if __name__ == '__main__':
    main()
