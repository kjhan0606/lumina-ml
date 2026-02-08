#!/usr/bin/env python3
"""Generate diagnostic plots from inference results.

v2: Feature-specific diagnostics (Si II, S II, Ca II, UV ratio).

1. Corner plot: 15x15 marginalized posteriors
2. Best-fit spectrum: observed + emulator median + posterior draws (linear flux)
3. Si II + S II detail zoom
4. UV region zoom
5. Feature diagnostics: Si II vel/depth, UV ratio, Ca II depth
6. Emulator accuracy per feature window
7. Method comparison

Usage:
  python3 scripts/05_plot_results.py
  python3 scripts/05_plot_results.py --method mcmc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumina_ml import config as cfg
from lumina_ml.inference import summary_statistics
from lumina_ml.preprocessing import asinh_inverse, measure_spectral_features


def load_observed():
    """Load and preprocess observed spectrum (same pipeline as training)."""
    from lumina_ml.preprocessing import (
        interpolate_to_grid, adaptive_smooth, peak_normalize, asinh_transform,
    )
    obs = np.genfromtxt(str(cfg.OBS_FILE_BMAX), delimiter=',', names=True)
    wave = obs['wavelength_angstrom']
    flux = obs['flux_erg_s_cm2_angstrom']
    grid_flux = np.interp(cfg.SPECTRUM_GRID, wave, flux, left=0.0, right=0.0)
    grid_flux = adaptive_smooth(grid_flux)
    normalized, peak = peak_normalize(grid_flux)
    return asinh_transform(normalized)


def plot_corner(samples, method_name, output_path):
    """15x15 corner plot of posterior samples."""
    try:
        import corner
    except ImportError:
        print("  corner package not installed, skipping corner plot")
        return

    labels = [
        r'$\log L$', r'$v_{\rm inner}$', r'$\log \rho_0$', r'$n_{\rm inner}$',
        r'$T_e/T_{\rm rad}$', r'$v_{\rm core}$', r'$v_{\rm wall}$',
        r'$X_{\rm Fe,core}$', r'$X_{\rm Si,wall}$',
        r'$v_{\rm break}$', r'$n_{\rm outer}$', r'$t_{\rm exp}$',
        r'$X_{\rm Fe,wall}$', r'$X_{\rm Ni}$', r'$X_{\rm Fe,outer}$',
    ]

    fig = corner.corner(
        samples, labels=labels, show_titles=True,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
        quantiles=[0.16, 0.5, 0.84],
        title_fmt='.3f',
    )
    fig.suptitle(f'{method_name} Posterior -- SN 2011fe (15D)', fontsize=14, y=1.02)
    fig.savefig(str(output_path), dpi=80, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spectrum_fit(samples, emulator, obs_spectrum_asinh, method_name, output_path):
    """Best-fit spectrum + posterior draws in LINEAR flux space."""
    grid = cfg.SPECTRUM_GRID
    n_draws = 100

    # Convert observed to linear
    obs_linear = asinh_inverse(obs_spectrum_asinh)

    # Draw random posterior spectra
    rng = np.random.default_rng(42)
    draw_idx = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
    draw_spectra_asinh = emulator.predict_spectrum(samples[draw_idx])
    draw_spectra = asinh_inverse(draw_spectra_asinh)

    # Median spectrum
    median_params = np.median(samples, axis=0)
    median_asinh = emulator.predict_spectrum(median_params)
    median_linear = asinh_inverse(median_asinh)

    fig, axes = plt.subplots(4, 1, figsize=(16, 18),
                              gridspec_kw={'height_ratios': [3, 2, 2, 1.5]})

    # Panel 1: Full spectrum (linear flux)
    ax = axes[0]
    ax.plot(grid, obs_linear, 'k-', lw=1.2, label='Observed (SN 2011fe)', alpha=0.9)
    lo = np.percentile(draw_spectra, 16, axis=0)
    hi = np.percentile(draw_spectra, 84, axis=0)
    lo95 = np.percentile(draw_spectra, 2.5, axis=0)
    hi95 = np.percentile(draw_spectra, 97.5, axis=0)
    ax.fill_between(grid, lo95, hi95, alpha=0.15, color='royalblue', label='95% posterior')
    ax.fill_between(grid, lo, hi, alpha=0.3, color='royalblue', label='68% posterior')
    ax.plot(grid, median_linear, 'r-', lw=1.0, alpha=0.8, label='Emulator median')
    ax.set_xlim(2500, 9500)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'{method_name}: Full Spectrum (Linear Flux)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Si II + S II zoom (5000-7000 A)
    ax = axes[1]
    mask = (grid >= 5000) & (grid <= 7000)
    ax.plot(grid[mask], obs_linear[mask], 'k-', lw=1.2, label='Observed')
    ax.fill_between(grid[mask], lo[mask], hi[mask], alpha=0.3, color='royalblue')
    ax.plot(grid[mask], median_linear[mask], 'r-', lw=1.0, alpha=0.8)
    # Feature markers
    ax.axvline(6355, color='green', ls='--', alpha=0.3, label='Si II 6355 rest')
    for v in [10000, 12000, 15000]:
        w = 6355 * (1 - v / 3e5)
        ax.axvline(w, color='orange', ls=':', alpha=0.3)
        ax.text(w, 0.12, f'{v//1000}k', fontsize=7, color='orange', ha='center')
    # S II markers
    ax.axvspan(5300, 5700, alpha=0.05, color='purple', label='S II W-feature')
    ax.set_xlim(5000, 7000)
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Si II 6355 + S II W-feature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: UV + Ca H&K zoom (2500-4500 A)
    ax = axes[2]
    mask = (grid >= 2500) & (grid <= 4500)
    ax.plot(grid[mask], obs_linear[mask], 'k-', lw=1.2, label='Observed')
    ax.fill_between(grid[mask], lo[mask], hi[mask], alpha=0.3, color='royalblue')
    ax.plot(grid[mask], median_linear[mask], 'r-', lw=1.0, alpha=0.8)
    ax.axvspan(3600, 4000, alpha=0.05, color='blue', label='Ca II H&K')
    ax.axvspan(2500, 3500, alpha=0.05, color='red', label='UV band')
    ax.set_xlim(2500, 4500)
    ax.set_ylabel('Normalized Flux')
    ax.set_title('UV + Ca II H&K Region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Residuals
    ax = axes[3]
    residual = median_linear - obs_linear
    ax.plot(grid, residual, 'r-', lw=0.8, alpha=0.8, label='Median - Obs')
    ax.fill_between(grid, lo - obs_linear, hi - obs_linear,
                     alpha=0.3, color='royalblue', label='68% band')
    ax.axhline(0, color='gray', lw=0.5)
    ax.fill_between(grid, -0.05, 0.05, color='green', alpha=0.1, label='5% band')
    ax.set_xlim(2500, 9500)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Residual')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_method_comparison(output_path):
    """Compare MCMC vs SBI vs dynesty posteriors."""
    methods_available = []
    samples_dict = {}

    for m in ['mcmc', 'sbi', 'dynesty']:
        f = cfg.RESULTS_DIR / f"{m}_samples.npy"
        if f.exists():
            samples_dict[m] = np.load(str(f))
            methods_available.append(m)

    if len(methods_available) < 2:
        print("  Need at least 2 methods for comparison, skipping")
        return

    n_params = cfg.N_PARAMS
    n_rows = (n_params + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3 * n_rows))
    axes_flat = axes.flatten()
    colors = {'mcmc': 'royalblue', 'sbi': 'orangered', 'dynesty': 'forestgreen'}

    for i, name in enumerate(cfg.PARAM_NAMES):
        ax = axes_flat[i]
        for m in methods_available:
            s = samples_dict[m][:, i]
            lo, hi = cfg.PARAM_RANGES[i]
            bins = np.linspace(lo, hi, 50)
            ax.hist(s, bins=bins, alpha=0.4, density=True, color=colors[m], label=m.upper())
        ax.set_xlabel(name, fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle('Posterior Comparison: MCMC vs SBI vs Dynesty', fontsize=14)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_emulator_diagnostics(output_path):
    """Plot emulator accuracy per feature window on validation set."""
    spectra_val = np.load(str(cfg.DATA_PROCESSED / "spectra_val.npy"))
    params_val = np.load(str(cfg.DATA_PROCESSED / "params_val.npy"))

    from lumina_ml.emulator import Emulator
    emulator = Emulator.load(cfg.MODELS_DIR, cfg.DATA_PROCESSED)

    pred_spectra = emulator.predict_spectrum(params_val)
    grid = cfg.SPECTRUM_GRID

    # Convert to linear for feature measurements
    spec_true_lin = asinh_inverse(spectra_val)
    spec_pred_lin = asinh_inverse(pred_spectra)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # RMS histogram (asinh space)
    rms_asinh = np.sqrt(np.mean((pred_spectra - spectra_val)**2, axis=1))
    ax = axes[0, 0]
    ax.hist(rms_asinh, bins=50, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(np.median(rms_asinh), color='red', ls='--',
               label=f'Median: {np.median(rms_asinh):.4f}')
    ax.set_xlabel('Spectral RMS (asinh)')
    ax.set_ylabel('Count')
    ax.set_title('Emulator Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature-specific RMS
    ax = axes[0, 1]
    feat_names = []
    feat_rms_means = []
    for name, (wmin, wmax) in cfg.FEATURE_WINDOWS.items():
        mask = (grid >= wmin) & (grid <= wmax)
        if mask.sum() > 0:
            f_rms = np.sqrt(np.mean((pred_spectra[:, mask] - spectra_val[:, mask])**2, axis=1))
            feat_names.append(name.replace('_', '\n'))
            feat_rms_means.append(f_rms.mean())
    ax.barh(range(len(feat_names)), feat_rms_means, color='steelblue', edgecolor='k')
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel('Mean RMS (asinh)')
    ax.set_title('Per-Feature Accuracy')
    ax.grid(True, alpha=0.3, axis='x')

    # UV/opt ratio scatter
    uv_mask = (grid >= cfg.UV_BAND[0]) & (grid <= cfg.UV_BAND[1])
    opt_mask = (grid >= cfg.OPT_BAND[0]) & (grid <= cfg.OPT_BAND[1])
    ratio_true = spec_true_lin[:, uv_mask].mean(axis=1) / (spec_true_lin[:, opt_mask].mean(axis=1) + 1e-30)
    ratio_pred = spec_pred_lin[:, uv_mask].mean(axis=1) / (spec_pred_lin[:, opt_mask].mean(axis=1) + 1e-30)
    ax = axes[0, 2]
    ax.scatter(ratio_true, ratio_pred, s=5, alpha=0.3, c='steelblue')
    lims = [0, max(ratio_true.max(), ratio_pred.max()) * 1.1]
    ax.plot(lims, lims, 'k--', lw=0.5)
    ax.set_xlabel('True UV/opt ratio')
    ax.set_ylabel('Predicted UV/opt ratio')
    ax.set_title('UV/Optical Ratio Accuracy')
    ax.grid(True, alpha=0.3)

    # Si II velocity accuracy
    rng = np.random.default_rng(42)
    n_sample = min(200, len(spectra_val))
    sample_idx = rng.choice(len(spectra_val), n_sample, replace=False)
    si_vel_true, si_vel_pred = [], []
    si_depth_true, si_depth_pred = [], []
    for idx in sample_idx:
        ft = measure_spectral_features(spectra_val[idx], is_asinh=True)
        fp = measure_spectral_features(pred_spectra[idx], is_asinh=True)
        si_vel_true.append(ft['si_ii_vel'])
        si_vel_pred.append(fp['si_ii_vel'])
        si_depth_true.append(ft['si_ii_depth'])
        si_depth_pred.append(fp['si_ii_depth'])

    ax = axes[1, 0]
    ax.scatter(si_vel_true, si_vel_pred, s=10, alpha=0.5, c='steelblue')
    lims = [min(min(si_vel_true), min(si_vel_pred)) - 500,
            max(max(si_vel_true), max(si_vel_pred)) + 500]
    ax.plot(lims, lims, 'k--', lw=0.5)
    vel_err = np.abs(np.array(si_vel_true) - np.array(si_vel_pred))
    ax.set_xlabel('True Si II velocity (km/s)')
    ax.set_ylabel('Predicted Si II velocity (km/s)')
    ax.set_title(f'Si II Velocity (mean err: {vel_err.mean():.0f} km/s)')
    ax.grid(True, alpha=0.3)

    # Si II depth accuracy
    ax = axes[1, 1]
    ax.scatter(si_depth_true, si_depth_pred, s=10, alpha=0.5, c='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
    depth_err = np.abs(np.array(si_depth_true) - np.array(si_depth_pred))
    ax.set_xlabel('True Si II depth')
    ax.set_ylabel('Predicted Si II depth')
    ax.set_title(f'Si II Depth (mean err: {depth_err.mean():.4f})')
    ax.grid(True, alpha=0.3)

    # Best/worst examples
    sorted_idx = np.argsort(rms_asinh)
    examples = {
        'Best': sorted_idx[0],
        'Median': sorted_idx[len(sorted_idx)//2],
        'Worst': sorted_idx[-1],
    }
    ax = axes[1, 2]
    for label, idx in examples.items():
        ax.plot(grid, spec_true_lin[idx], 'k-', alpha=0.4, lw=0.5)
        ax.plot(grid, spec_pred_lin[idx], '--', alpha=0.7, lw=0.8,
                label=f'{label} (RMS={rms_asinh[idx]:.4f})')
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Normalized Flux (linear)')
    ax.set_xlim(2500, 9500)
    ax.set_title('Example Predictions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RMS vs key params
    ax = axes[2, 0]
    ax.scatter(params_val[:, 0], rms_asinh, s=5, alpha=0.3, c='steelblue')
    ax.set_xlabel('log_L')
    ax.set_ylabel('RMS (asinh)')
    ax.set_title('Accuracy vs Luminosity')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.scatter(params_val[:, 11], rms_asinh, s=5, alpha=0.3, c='steelblue')
    ax.set_xlabel('t_exp (days)')
    ax.set_ylabel('RMS (asinh)')
    ax.set_title('Accuracy vs Epoch')
    ax.grid(True, alpha=0.3)

    # RMS vs X_Fe_core (new 15D param)
    ax = axes[2, 2]
    ax.scatter(params_val[:, 7], rms_asinh, s=5, alpha=0.3, c='steelblue')
    ax.set_xlabel('X_Fe_core')
    ax.set_ylabel('RMS (asinh)')
    ax.set_title('Accuracy vs Core Fe Abundance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot inference results")
    parser.add_argument('--method', choices=['mcmc', 'sbi', 'dynesty', 'all'],
                        default='all')
    args = parser.parse_args()

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    obs_spectrum = load_observed()

    methods = [args.method] if args.method != 'all' else ['mcmc', 'sbi', 'dynesty']

    # Load emulator
    emulator = None
    try:
        from lumina_ml.emulator import Emulator
        emulator = Emulator.load(cfg.MODELS_DIR, cfg.DATA_PROCESSED)
    except Exception as e:
        print(f"  Could not load emulator: {e}")

    for method in methods:
        samples_file = cfg.RESULTS_DIR / f"{method}_samples.npy"
        if not samples_file.exists():
            print(f"  {method} samples not found, skipping")
            continue

        samples = np.load(str(samples_file))
        print(f"\n{method.upper()} ({len(samples)} samples)")

        # Corner plot
        plot_corner(samples, method.upper(), cfg.RESULTS_DIR / f"{method}_corner.png")

        # Spectrum fit (linear flux)
        if emulator is not None:
            plot_spectrum_fit(
                samples, emulator, obs_spectrum, method.upper(),
                cfg.RESULTS_DIR / f"{method}_spectrum_fit.png",
            )

    # Comparison plot
    plot_method_comparison(cfg.RESULTS_DIR / "method_comparison.png")

    # Emulator diagnostics
    try:
        plot_emulator_diagnostics(cfg.RESULTS_DIR / "emulator_diagnostics.png")
    except Exception as e:
        print(f"  Emulator diagnostics failed: {e}")

    print("\nAll plots saved to", cfg.RESULTS_DIR)


if __name__ == '__main__':
    main()
