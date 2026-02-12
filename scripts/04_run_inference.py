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
    FeatureAwareLikelihood, compute_sigma_array, log_prior_stage2,
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


def print_summary(method_name, results, param_names=None):
    """Print posterior summary statistics."""
    if param_names is None:
        param_names = cfg.PARAM_NAMES
    stats = summary_statistics(results['samples'], param_names=param_names)
    print(f"\n{'='*60}")
    print(f"{method_name} -- Posterior Summary")
    print(f"{'='*60}")
    print(f"  {'Parameter':22s} {'Median':>10s} {'Std':>8s} {'16%':>10s} {'84%':>10s}")
    print(f"  {'-'*58}")
    for name in param_names:
        if name not in stats:
            continue
        s = stats[name]
        print(f"  {name:22s} {s['median']:10.4f} {s['std']:8.4f} "
              f"{s['CI_16']:10.4f} {s['CI_84']:10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian inference on SN 2011fe")
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3],
                        help='Pipeline stage (default: 1)')
    parser.add_argument('--method', choices=['mcmc', 'dynesty', 'sbi', 'all'],
                        default='all', help='Inference method (default: all)')
    parser.add_argument('--mcmc-walkers', type=int, default=cfg.MCMC_N_WALKERS)
    parser.add_argument('--mcmc-burn', type=int, default=cfg.MCMC_N_BURN)
    parser.add_argument('--mcmc-production', type=int, default=cfg.MCMC_N_PRODUCTION)
    parser.add_argument('--sbi-samples', type=int, default=10000)
    parser.add_argument('--dynesty-live', type=int, default=cfg.DYNESTY_LIVE_POINTS)
    args = parser.parse_args()

    # Stage-specific config
    _, data_processed, models_dir, results_dir = cfg.get_stage_paths(args.stage)
    if args.stage == 2:
        param_names = cfg.STAGE2_PARAM_NAMES
        param_ranges = cfg.STAGE2_PARAM_RANGES
        n_params = cfg.STAGE2_N_PARAMS
        prior_fn = log_prior_stage2
    else:
        param_names = cfg.PARAM_NAMES
        param_ranges = cfg.PARAM_RANGES
        n_params = cfg.N_PARAMS
        prior_fn = None  # default (Stage 1)

    print(f"Stage: {args.stage} ({n_params}D)")
    results_dir.mkdir(parents=True, exist_ok=True)

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
        emulator = Emulator.load(models_dir, data_processed)
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
            prior_fn=prior_fn,
        )

        t0 = time.time()
        mcmc_results = run_mcmc(
            likelihood,
            n_walkers=args.mcmc_walkers,
            n_burn=args.mcmc_burn,
            n_production=args.mcmc_production,
            n_params=n_params,
            param_ranges=param_ranges,
        )
        print(f"  Total MCMC time: {time.time()-t0:.1f}s")

        np.save(str(results_dir / "mcmc_samples.npy"), mcmc_results['samples'])
        np.save(str(results_dir / "mcmc_log_prob.npy"), mcmc_results['log_prob'])
        print(f"  Saved to {results_dir}/mcmc_*.npy")
        print_summary("MCMC", mcmc_results, param_names=param_names)

    # ===== SBI =====
    if 'sbi' in methods:
        print(f"\n{'='*60}")
        print("SBI Inference (Neural Posterior Estimation)")
        print(f"{'='*60}")

        params_train = np.load(str(data_processed / "params_train.npy"))
        spectra_train = np.load(str(data_processed / "spectra_train.npy"))
        print(f"  Training pairs: {len(params_train)}")

        t0 = time.time()
        sbi_results = run_sbi(
            params_train, spectra_train, obs_spectrum,
            n_posterior_samples=args.sbi_samples,
            param_ranges=param_ranges,
        )
        print(f"  Total SBI time: {time.time()-t0:.1f}s")

        np.save(str(results_dir / "sbi_samples.npy"), sbi_results['samples'])
        np.save(str(results_dir / "sbi_log_prob.npy"), sbi_results['log_prob'])
        print(f"  Saved to {results_dir}/sbi_*.npy")
        print_summary("SBI", sbi_results, param_names=param_names)

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
            prior_fn=prior_fn,
        )

        t0 = time.time()
        nested_results = run_nested(likelihood, n_live=args.dynesty_live,
                                     n_params=n_params, param_ranges=param_ranges)
        print(f"  Total dynesty time: {time.time()-t0:.1f}s")

        np.save(str(results_dir / "dynesty_samples.npy"), nested_results['samples'])
        print(f"  log(Z) = {nested_results['logz']:.2f} +/- {nested_results['logzerr']:.2f}")
        print(f"  Saved to {results_dir}/dynesty_*.npy")
        print_summary("Nested Sampling", nested_results, param_names=param_names)

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
            samples = np.load(str(results_dir / f"{m}_samples.npy"))
            all_stats[m] = summary_statistics(samples, param_names=param_names)

        for name in param_names:
            if name not in all_stats[methods[0]]:
                continue
            row = f"  {name:22s}"
            for m in methods:
                row += f" {all_stats[m][name]['median']:12.4f}"
            print(row)

    print("\nDone! Run 05_plot_results.py to generate plots.")


if __name__ == '__main__':
    main()
