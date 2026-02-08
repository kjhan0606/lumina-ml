"""Bayesian inference: MCMC (emcee), Nested Sampling (dynesty), and SBI (Neural Posterior Estimation).

v2: Feature-aware, wavelength-dependent likelihood.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg


# ===== Priors =====
def log_prior(theta: np.ndarray) -> float:
    """Uniform prior within PARAM_RANGES + physical validity."""
    for i, (lo, hi) in enumerate(cfg.PARAM_RANGES):
        if theta[i] < lo or theta[i] > hi:
            return -np.inf

    v_inner = theta[1]
    v_core = theta[5]
    v_wall = theta[6]
    X_Fe_core = theta[7]
    X_Si_wall = theta[8]
    v_break = theta[9]
    X_Fe_wall = theta[12]
    X_Ni = theta[13]
    X_Fe_outer = theta[14]

    if v_core + 1000 >= v_wall:
        return -np.inf
    if v_core <= v_inner:
        return -np.inf
    if v_wall >= cfg.V_OUTER:
        return -np.inf
    if v_break <= v_inner + 1000:
        return -np.inf
    if v_break >= cfg.V_OUTER - 1000:
        return -np.inf
    if X_Fe_wall + X_Si_wall > 0.65:
        return -np.inf

    # Oxygen filler must be positive in all zones
    for zone in ('core', 'wall', 'outer'):
        if zone == 'core':
            X_Si, X_Fe = 0.05, X_Fe_core
        elif zone == 'wall':
            X_Si, X_Fe = X_Si_wall, X_Fe_wall
        else:
            X_Si, X_Fe = 0.02, X_Fe_outer
        X_S = cfg.ZONE_S[zone]
        X_Ca = cfg.ZONE_CA[zone]
        X_O = 1.0 - cfg.FIXED_SPECIES_SUM_BASE - X_Si - X_Fe - X_S - X_Ca - X_Ni
        if X_O < 0.03:
            return -np.inf

    return 0.0


# ===== Wavelength-dependent sigma =====
def compute_sigma_array(grid: np.ndarray = None,
                        sigma_obs_base: float = 0.03,
                        sigma_emu_base: float = 0.02) -> np.ndarray:
    """Compute wavelength-dependent sigma for the likelihood.

    - UV (< 3500 A): higher noise (sigma_obs * 2.0)
    - Optical (3500-7500 A): base noise
    - NIR (> 7500 A): moderate noise (sigma_obs * 1.5)
    - Feature windows: slightly reduced sigma (upweight by 1/sigma)

    Returns sigma array on SPECTRUM_GRID [n_bins].
    """
    if grid is None:
        grid = cfg.SPECTRUM_GRID

    sigma_obs = np.full(len(grid), sigma_obs_base)
    sigma_emu = np.full(len(grid), sigma_emu_base)

    # UV has higher noise
    uv = grid < 3500
    sigma_obs[uv] *= 2.0
    sigma_emu[uv] *= 1.5

    # NIR has moderate noise
    nir = grid > 7500
    sigma_obs[nir] *= 1.5
    sigma_emu[nir] *= 1.2

    # Upweight feature windows (reduce sigma by factor)
    feature_boost = 0.7  # effectively 1/0.7 = 1.43x weight
    for name, (wmin, wmax) in cfg.FEATURE_WINDOWS.items():
        mask = (grid >= wmin) & (grid <= wmax)
        sigma_obs[mask] *= feature_boost
        sigma_emu[mask] *= feature_boost

    return np.sqrt(sigma_obs**2 + sigma_emu**2)


# ===== Likelihood =====
class FeatureAwareLikelihood:
    """Feature-aware Gaussian likelihood with wavelength-dependent sigma.

    chi² = sum((obs - model)² / sigma(λ)²)
         + w_si * (si_vel_obs - si_vel_pred)² / sigma_vel²
         + w_uv * (uv_ratio_obs - uv_ratio_pred)² / sigma_ratio²
    """

    def __init__(self, emulator, obs_spectrum: np.ndarray,
                 sigma_array: np.ndarray = None,
                 mode: str = 'spectrum',
                 obs_features: dict = None):
        """
        Args:
            emulator: Emulator instance
            obs_spectrum: observed spectrum on SPECTRUM_GRID (asinh-transformed)
            sigma_array: wavelength-dependent sigma [n_bins]
            mode: 'spectrum' or 'pca'
            obs_features: dict with measured features of observed spectrum
                          (si_ii_vel, si_ii_depth, uv_opt_ratio, etc.)
        """
        self.emulator = emulator
        self.obs = obs_spectrum
        self.mode = mode
        self.grid = cfg.SPECTRUM_GRID

        if sigma_array is None:
            sigma_array = compute_sigma_array()
        self.sigma2 = sigma_array**2

        # Minimum flux mask: only compare where obs has signal
        # In asinh space, asinh(0.05/0.05) = 0.88
        self.mask = obs_spectrum > 0.5

        # Feature measurements of observed spectrum
        self.obs_features = obs_features or {}

        if mode == 'pca':
            self.obs_pca = emulator.spectral_pca.transform(obs_spectrum[np.newaxis, :])[0]

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Compute log-likelihood for parameters theta."""
        try:
            if self.mode == 'pca':
                pred = self.emulator.predict_pca(theta)
                # Simple MSE in PCA space (uniform sigma)
                chi2 = np.sum((self.obs_pca - pred)**2) / 0.001
            else:
                pred = self.emulator.predict_spectrum(theta)
                # Wavelength-dependent chi2
                mask = self.mask
                chi2 = np.sum((self.obs[mask] - pred[mask])**2 / self.sigma2[mask])

            logL = -0.5 * chi2

            # Feature-based terms
            if self.obs_features and self.mode == 'spectrum':
                from .preprocessing import measure_spectral_features
                pred_features = measure_spectral_features(pred, self.grid, is_asinh=True)

                # Si II velocity penalty
                if 'si_ii_vel' in self.obs_features and 'si_ii_vel' in pred_features:
                    dv = self.obs_features['si_ii_vel'] - pred_features['si_ii_vel']
                    logL -= 0.5 * (dv / 500.0)**2  # sigma_vel = 500 km/s

                # UV/optical ratio penalty
                if 'uv_opt_ratio' in self.obs_features and 'uv_opt_ratio' in pred_features:
                    dr = self.obs_features['uv_opt_ratio'] - pred_features['uv_opt_ratio']
                    logL -= 0.5 * (dr / 0.05)**2  # sigma_ratio = 0.05

            return logL
        except Exception:
            return -np.inf

    def log_probability(self, theta: np.ndarray) -> float:
        """Log posterior = log prior + log likelihood."""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


# Legacy alias for backward compatibility
GaussianLikelihood = FeatureAwareLikelihood


# ===== MCMC with emcee =====
def run_mcmc(likelihood,
             initial_guess: np.ndarray = None,
             n_walkers: int = cfg.MCMC_N_WALKERS,
             n_burn: int = cfg.MCMC_N_BURN,
             n_production: int = cfg.MCMC_N_PRODUCTION,
             verbose: bool = True) -> dict:
    """Run MCMC inference using emcee."""
    import emcee

    ndim = cfg.N_PARAMS

    if initial_guess is None:
        initial_guess = np.array([(lo + hi) / 2 for lo, hi in cfg.PARAM_RANGES])

    # Find MAP with differential evolution
    if verbose:
        print("  Finding MAP estimate with differential evolution...")
    try:
        from scipy.optimize import differential_evolution
        bounds = cfg.PARAM_RANGES

        def neg_logprob(theta):
            lp = likelihood.log_probability(theta)
            return -lp if np.isfinite(lp) else 1e30

        result = differential_evolution(neg_logprob, bounds, maxiter=200, seed=42,
                                        tol=1e-4, polish=False)
        if result.success or result.fun < 1e29:
            initial_guess = result.x
            if verbose:
                print(f"  MAP found: -logP = {result.fun:.2f}")
    except Exception as e:
        if verbose:
            print(f"  DE optimization failed: {e}, using default initial guess")

    # Initialize walkers
    scales = np.array([hi - lo for lo, hi in cfg.PARAM_RANGES]) * 0.01
    pos = np.array([initial_guess + scales * np.random.randn(ndim) for _ in range(n_walkers)])

    for i in range(n_walkers):
        attempts = 0
        while log_prior(pos[i]) == -np.inf and attempts < 1000:
            pos[i] = initial_guess + scales * np.random.randn(ndim)
            attempts += 1

    sampler = emcee.EnsembleSampler(n_walkers, ndim, likelihood.log_probability)

    # Burn-in
    if verbose:
        print(f"  Running burn-in ({n_burn} steps, {n_walkers} walkers)...")
    t0 = time.time()
    state = sampler.run_mcmc(pos, n_burn, progress=verbose)
    if verbose:
        print(f"  Burn-in complete ({time.time() - t0:.1f}s)")

    sampler.reset()

    # Production
    if verbose:
        print(f"  Running production ({n_production} steps)...")
    t0 = time.time()
    sampler.run_mcmc(state, n_production, progress=verbose)
    if verbose:
        print(f"  Production complete ({time.time() - t0:.1f}s)")

    flat_samples = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    results = {
        'samples': flat_samples,
        'log_prob': flat_log_prob,
        'chain': sampler.get_chain(),
        'acceptance_fraction': sampler.acceptance_fraction,
    }

    try:
        autocorr = sampler.get_autocorr_time(quiet=True)
        results['autocorr_time'] = autocorr
        if verbose:
            print(f"  Autocorrelation time: {np.mean(autocorr):.1f} steps")
    except Exception:
        results['autocorr_time'] = None
        if verbose:
            print("  Autocorrelation time: not converged")

    if verbose:
        print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        best_idx = np.argmax(flat_log_prob)
        print(f"  Best log-prob: {flat_log_prob[best_idx]:.2f}")
        print(f"  Best params: {flat_samples[best_idx]}")

    return results


# ===== Nested Sampling with dynesty =====
def run_nested(likelihood,
               n_live: int = cfg.DYNESTY_LIVE_POINTS,
               dlogz: float = cfg.DYNESTY_DLOGZ,
               verbose: bool = True) -> dict:
    """Run nested sampling with dynesty."""
    import dynesty

    ndim = cfg.N_PARAMS

    def prior_transform(u):
        theta = np.empty(ndim)
        for i, (lo, hi) in enumerate(cfg.PARAM_RANGES):
            theta[i] = lo + u[i] * (hi - lo)
        return theta

    def loglike(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -1e30
        return likelihood.log_likelihood(theta)

    if verbose:
        print(f"  Running dynesty ({n_live} live points, dlogz={dlogz})...")

    t0 = time.time()
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim,
                                     nlive=n_live)
    sampler.run_nested(dlogz=dlogz, print_progress=verbose)
    res = sampler.results

    if verbose:
        print(f"  Nested sampling complete ({time.time() - t0:.1f}s)")
        print(f"  log(Z) = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}")
        print(f"  N_iterations = {res.niter}")

    from dynesty.utils import resample_equal
    weights = np.exp(res.logwt - res.logz[-1])
    weights /= weights.sum()
    samples = resample_equal(res.samples, weights)

    return {
        'samples': samples,
        'weights': weights,
        'logz': res.logz[-1],
        'logzerr': res.logzerr[-1],
        'results': res,
    }


# ===== SBI: Neural Posterior Estimation =====
def run_sbi(training_params: np.ndarray, training_spectra: np.ndarray,
            obs_spectrum: np.ndarray,
            n_posterior_samples: int = 10000,
            verbose: bool = True) -> dict:
    """Run Simulation-Based Inference using Neural Posterior Estimation (NPE)."""
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    ndim = cfg.N_PARAMS

    if verbose:
        print("  Setting up SBI (Neural Posterior Estimation)...")

    low = torch.tensor([lo for lo, hi in cfg.PARAM_RANGES], dtype=torch.float32)
    high = torch.tensor([hi for lo, hi in cfg.PARAM_RANGES], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    theta_tensor = torch.tensor(training_params, dtype=torch.float32)
    x_tensor = torch.tensor(training_spectra, dtype=torch.float32)

    inference = SNPE(prior=prior)

    if verbose:
        print(f"  Appending {len(training_params)} simulation pairs...")
    inference.append_simulations(theta_tensor, x_tensor)

    if verbose:
        print("  Training neural posterior estimator...")
    t0 = time.time()
    density_estimator = inference.train(show_train_summary=verbose)
    if verbose:
        print(f"  Training complete ({time.time() - t0:.1f}s)")

    posterior = inference.build_posterior(density_estimator)

    obs_tensor = torch.tensor(obs_spectrum, dtype=torch.float32).unsqueeze(0)

    if verbose:
        print(f"  Drawing {n_posterior_samples} posterior samples...")
    samples = posterior.sample((n_posterior_samples,), x=obs_tensor)
    log_prob = posterior.log_prob(samples, x=obs_tensor)

    samples_np = samples.numpy()
    log_prob_np = log_prob.numpy()

    if verbose:
        best_idx = np.argmax(log_prob_np)
        print(f"  Best log-prob: {log_prob_np[best_idx]:.2f}")
        print(f"  Best params: {samples_np[best_idx]}")

    return {
        'samples': samples_np,
        'log_prob': log_prob_np,
        'posterior': posterior,
    }


def summary_statistics(samples: np.ndarray) -> dict:
    """Compute summary statistics from posterior samples."""
    result = {}
    for i, name in enumerate(cfg.PARAM_NAMES):
        s = samples[:, i]
        result[name] = {
            'median': np.median(s),
            'mean': np.mean(s),
            'std': np.std(s),
            'CI_16': np.percentile(s, 16),
            'CI_84': np.percentile(s, 84),
            'CI_2.5': np.percentile(s, 2.5),
            'CI_97.5': np.percentile(s, 97.5),
        }
    return result
