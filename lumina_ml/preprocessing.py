"""Preprocessing pipeline: interpolation, adaptive smoothing, asinh transform, PCA.

Key improvements over v1:
  - asinh(flux/F_ref) transform: equalizes UV and optical dynamic range for PCA
  - Adaptive SG smoothing: narrower in optical (preserve Si II, S II), wider in UV/NIR
  - Feature reconstruction validation
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def interpolate_to_grid(wave: np.ndarray, flux: np.ndarray,
                        grid: np.ndarray = None) -> np.ndarray:
    """Interpolate a LUMINA spectrum onto the standard wavelength grid.

    LUMINA outputs flux in erg/s/cm; convert to erg/s/A (divide by bin width in cm).
    """
    if grid is None:
        grid = cfg.SPECTRUM_GRID

    # LUMINA flux is per cm of wavelength; convert to per Angstrom
    flux_A = flux / 1e8
    return np.interp(grid, wave, flux_A, left=0.0, right=0.0)


def adaptive_smooth(flux: np.ndarray, grid: np.ndarray = None,
                    order: int = None) -> np.ndarray:
    """Apply region-dependent Savitzky-Golay smoothing.

    UV (2000-3500): 155A window (31 pts) — heavy, noisy line-blanketed region
    NUV (3500-4500): 105A window (21 pts) — moderate, Ca H&K
    Optical (4500-7500): 55A window (11 pts) — light, preserve Si II/S II profiles
    NIR (7500-10000): 105A window (21 pts) — moderate, Ca IR triplet
    """
    if grid is None:
        grid = cfg.SPECTRUM_GRID
    if order is None:
        order = cfg.SG_ORDER

    result = flux.copy()
    for wave_min, wave_max, window_A in cfg.SG_REGIONS:
        mask = (grid >= wave_min) & (grid < wave_max)
        n_pts = mask.sum()
        if n_pts < 5:
            continue
        # Convert Angstrom window to points (grid spacing = WAVE_STEP)
        window_pts = int(round(window_A / cfg.WAVE_STEP))
        if window_pts % 2 == 0:
            window_pts += 1
        window_pts = max(window_pts, order + 2)
        window_pts = min(window_pts, n_pts - 1)
        if window_pts % 2 == 0:
            window_pts -= 1
        if window_pts >= order + 2:
            result[mask] = savgol_filter(flux[mask], window_pts, order)
    return result


def smooth_spectrum(flux: np.ndarray, window: int = None,
                    order: int = None) -> np.ndarray:
    """Legacy uniform smoothing (kept for backward compatibility)."""
    if window is None:
        window = 51  # fallback
    if order is None:
        order = cfg.SG_ORDER
    return savgol_filter(flux, window, order)


def peak_normalize(flux: np.ndarray, grid: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """Normalize spectrum by peak flux in 4000-7000 A range.

    Returns (normalized_flux, peak_value).
    """
    if grid is None:
        grid = cfg.SPECTRUM_GRID

    opt = (grid >= 4000) & (grid <= 7000)
    peak = flux[opt].max() if opt.sum() > 0 else flux.max()
    if peak <= 0:
        return flux, 0.0
    return flux / peak, peak


def asinh_transform(flux_norm: np.ndarray, softening: float = None) -> np.ndarray:
    """Apply asinh transform to equalize UV and optical dynamic range.

    asinh(x) ~ x for |x| << 1 (linear near zero)
    asinh(x) ~ ln(2x) for |x| >> 1 (logarithmic for bright regions)

    This gives UV (flux ~ 0.01-0.1) and optical (flux ~ 0.5-1.0) comparable
    weight in PCA space.
    """
    if softening is None:
        softening = cfg.ASINH_SOFTENING
    return np.arcsinh(flux_norm / softening)


def asinh_inverse(flux_asinh: np.ndarray, softening: float = None) -> np.ndarray:
    """Inverse asinh transform: recover peak-normalized flux."""
    if softening is None:
        softening = cfg.ASINH_SOFTENING
    return softening * np.sinh(flux_asinh)


def preprocess_spectrum(wave: np.ndarray, flux: np.ndarray,
                        use_asinh: bool = True) -> np.ndarray:
    """Full preprocessing: interpolate -> adaptive smooth -> normalize -> asinh.

    Returns preprocessed spectrum on standard grid.
    """
    grid_flux = interpolate_to_grid(wave, flux)
    smoothed = adaptive_smooth(grid_flux)
    normalized, peak = peak_normalize(smoothed)
    if use_asinh:
        return asinh_transform(normalized)
    return normalized


def preprocess_batch(raw_waves: list, raw_fluxes: list,
                     use_asinh: bool = True) -> np.ndarray:
    """Preprocess a batch of spectra.

    Args:
        raw_waves: List of wavelength arrays from LUMINA
        raw_fluxes: List of flux arrays from LUMINA
        use_asinh: Apply asinh transform (default True)

    Returns:
        spectra: [N, n_bins] array of preprocessed spectra
    """
    n = len(raw_waves)
    n_bins = len(cfg.SPECTRUM_GRID)
    spectra = np.zeros((n, n_bins))
    for i in range(n):
        spectra[i] = preprocess_spectrum(raw_waves[i], raw_fluxes[i],
                                          use_asinh=use_asinh)
    return spectra


def validate_spectrum(flux_preprocessed: np.ndarray, use_asinh: bool = True) -> bool:
    """Check if a preprocessed spectrum is physically reasonable."""
    if np.any(np.isnan(flux_preprocessed)) or np.any(np.isinf(flux_preprocessed)):
        return False
    if np.all(flux_preprocessed == 0):
        return False

    if use_asinh:
        # In asinh space: peak ~ asinh(1/softening) ~ 3.7 for softening=0.05
        peak_expected = np.arcsinh(1.0 / cfg.ASINH_SOFTENING)
        if flux_preprocessed.max() < 0.3 * peak_expected:
            return False
        if flux_preprocessed.max() > 2.0 * peak_expected:
            return False
    else:
        if flux_preprocessed.max() > 2.0 or flux_preprocessed.max() < 0.5:
            return False
    return True


def measure_spectral_features(spectrum: np.ndarray, grid: np.ndarray = None,
                               is_asinh: bool = True) -> dict:
    """Measure key SN Ia spectral features for validation.

    Returns dict with feature measurements:
      - si_ii_depth: Si II 6355 absorption depth (0=no absorption, 1=complete)
      - si_ii_vel: Si II 6355 velocity (km/s) from trough minimum
      - s_ii_ratio: S II W-feature depth ratio (two-dip structure indicator)
      - uv_opt_ratio: mean UV flux / mean optical flux
      - ca_hk_depth: Ca II H&K absorption depth
      - ca_ir_depth: Ca II IR triplet depth
    """
    if grid is None:
        grid = cfg.SPECTRUM_GRID

    # Convert back to linear flux if in asinh space
    if is_asinh:
        flux = asinh_inverse(spectrum)
    else:
        flux = spectrum.copy()

    result = {}

    # Si II 6355: depth and velocity
    si_mask = (grid >= 5900) & (grid <= 6500)
    if si_mask.sum() > 0:
        si_flux = flux[si_mask]
        si_wave = grid[si_mask]
        # Pseudo-continuum: max in red side (6300-6500)
        red_mask = (si_wave >= 6300) & (si_wave <= 6500)
        if red_mask.sum() > 0:
            continuum = si_flux[red_mask].max()
        else:
            continuum = si_flux.max()
        trough_idx = np.argmin(si_flux)
        trough_flux = si_flux[trough_idx]
        trough_wave = si_wave[trough_idx]

        result['si_ii_depth'] = 1.0 - trough_flux / max(continuum, 1e-30)
        # v = c * (1 - lambda_obs / lambda_rest)
        result['si_ii_vel'] = 3e5 * (1 - trough_wave / cfg.SI_II_REST)
    else:
        result['si_ii_depth'] = 0.0
        result['si_ii_vel'] = 0.0

    # S II W-feature: look for double-dip structure
    s_mask = (grid >= 5300) & (grid <= 5700)
    if s_mask.sum() > 10:
        s_flux = flux[s_mask]
        s_wave = grid[s_mask]
        # Find the two deepest local minima
        from scipy.signal import argrelmin
        minima = argrelmin(s_flux, order=3)[0]
        if len(minima) >= 2:
            depths = s_flux[minima]
            two_deepest = np.argsort(depths)[:2]
            result['s_ii_ratio'] = depths[two_deepest[0]] / max(depths[two_deepest[1]], 1e-30)
        else:
            result['s_ii_ratio'] = 1.0  # single dip = no W structure
    else:
        result['s_ii_ratio'] = 1.0

    # UV/optical ratio
    uv_mask = (grid >= cfg.UV_BAND[0]) & (grid <= cfg.UV_BAND[1])
    opt_mask = (grid >= cfg.OPT_BAND[0]) & (grid <= cfg.OPT_BAND[1])
    if uv_mask.sum() > 0 and opt_mask.sum() > 0:
        uv_mean = np.mean(flux[uv_mask])
        opt_mean = np.mean(flux[opt_mask])
        result['uv_opt_ratio'] = uv_mean / max(opt_mean, 1e-30)
    else:
        result['uv_opt_ratio'] = 0.0

    # Ca II H&K depth
    ca_mask = (grid >= 3600) & (grid <= 4000)
    if ca_mask.sum() > 0:
        ca_flux = flux[ca_mask]
        # Continuum from blue side of Ca (4000-4200)
        cont_mask = (grid >= 4000) & (grid <= 4200)
        if cont_mask.sum() > 0:
            ca_cont = flux[cont_mask].max()
        else:
            ca_cont = flux.max()
        result['ca_hk_depth'] = 1.0 - ca_flux.min() / max(ca_cont, 1e-30)
    else:
        result['ca_hk_depth'] = 0.0

    # Ca II IR triplet depth
    cair_mask = (grid >= 8000) & (grid <= 8800)
    if cair_mask.sum() > 0:
        cair_flux = flux[cair_mask]
        # Continuum from before triplet
        cont_mask = (grid >= 7800) & (grid <= 8000)
        if cont_mask.sum() > 0:
            cair_cont = flux[cont_mask].mean()
        else:
            cair_cont = flux.max()
        result['ca_ir_depth'] = 1.0 - cair_flux.min() / max(cair_cont, 1e-30)
    else:
        result['ca_ir_depth'] = 0.0

    return result


class SpectralPCA:
    """PCA compression for spectra with auto-component selection."""

    def __init__(self, variance_threshold: float = None, max_components: int = None):
        self.variance_threshold = variance_threshold or cfg.PCA_VARIANCE_THRESHOLD
        self.max_components = max_components or cfg.PCA_MAX_COMPONENTS
        self.pca = None
        self.n_components = None
        self.scaler = StandardScaler()

    def fit(self, spectra: np.ndarray) -> 'SpectralPCA':
        """Fit PCA to training spectra.

        Args:
            spectra: [N, n_bins] preprocessed spectra (asinh-transformed)
        """
        # Fit full PCA first to determine n_components
        pca_full = PCA(n_components=min(self.max_components, spectra.shape[0], spectra.shape[1]))
        pca_full.fit(spectra)

        # Find n_components for desired variance
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = np.searchsorted(cumvar, self.variance_threshold) + 1
        n_comp = min(n_comp, self.max_components)
        self.n_components = int(n_comp)

        # Refit with exact n_components
        self.pca = PCA(n_components=self.n_components)
        coeffs = self.pca.fit_transform(spectra)

        # Fit scaler on PCA coefficients
        self.scaler.fit(coeffs)

        print(f"PCA: {self.n_components} components capture "
              f"{cumvar[self.n_components - 1]:.4%} variance")

        return self

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        """Project spectra to standardized PCA coefficients."""
        coeffs = self.pca.transform(spectra)
        return self.scaler.transform(coeffs)

    def inverse_transform(self, coeffs_std: np.ndarray) -> np.ndarray:
        """Reconstruct spectra from standardized PCA coefficients."""
        coeffs = self.scaler.inverse_transform(coeffs_std)
        return self.pca.inverse_transform(coeffs)

    def reconstruction_error(self, spectra: np.ndarray) -> np.ndarray:
        """Compute per-spectrum RMS reconstruction error."""
        coeffs_std = self.transform(spectra)
        recon = self.inverse_transform(coeffs_std)
        return np.sqrt(np.mean((spectra - recon)**2, axis=1))

    def feature_reconstruction_error(self, spectra: np.ndarray,
                                      grid: np.ndarray = None) -> dict:
        """Compute reconstruction error per spectral feature window.

        Returns dict of {feature_name: mean_rms} for each FEATURE_WINDOWS entry.
        """
        if grid is None:
            grid = cfg.SPECTRUM_GRID

        coeffs_std = self.transform(spectra)
        recon = self.inverse_transform(coeffs_std)
        diff = spectra - recon

        result = {}
        for name, (wmin, wmax) in cfg.FEATURE_WINDOWS.items():
            mask = (grid >= wmin) & (grid <= wmax)
            if mask.sum() > 0:
                rms = np.sqrt(np.mean(diff[:, mask]**2, axis=1))
                result[name] = {'mean': rms.mean(), 'max': rms.max(),
                                'median': np.median(rms)}
        return result

    def save(self, filepath: Path):
        """Save PCA model to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'scaler': self.scaler,
                'n_components': self.n_components,
                'variance_threshold': self.variance_threshold,
            }, f)

    @classmethod
    def load(cls, filepath: Path) -> 'SpectralPCA':
        """Load PCA model from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.pca = data['pca']
        obj.scaler = data['scaler']
        obj.n_components = data['n_components']
        obj.variance_threshold = data.get('variance_threshold', cfg.PCA_VARIANCE_THRESHOLD)
        return obj


class ParamScaler:
    """Normalize parameters to [0, 1] based on PARAM_RANGES."""

    def __init__(self, param_ranges=None):
        ranges = param_ranges or cfg.PARAM_RANGES
        self.mins = np.array([r[0] for r in ranges])
        self.maxs = np.array([r[1] for r in ranges])
        self.spans = self.maxs - self.mins

    def transform(self, params: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]. Input/output shape: [..., n_params]."""
        return (params - self.mins) / self.spans

    def inverse_transform(self, params_norm: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1]."""
        return params_norm * self.spans + self.mins

    def save(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump({'mins': self.mins, 'maxs': self.maxs}, f)

    @classmethod
    def load(cls, filepath: Path) -> 'ParamScaler':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.mins = data['mins']
        obj.maxs = data['maxs']
        obj.spans = obj.maxs - obj.mins
        return obj
