"""Configuration for Lumina-ML: parameter ranges, paths, constants."""

from pathlib import Path
import numpy as np

# ===== Paths =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LUMINA_ROOT = PROJECT_ROOT.parent / "Lumina-sn"

# LUMINA binaries (GPU preferred, CPU fallback)
LUMINA_CUDA = LUMINA_ROOT / "lumina_cuda"
LUMINA_CPU = LUMINA_ROOT / "lumina"
REF_DIR = LUMINA_ROOT / "data" / "tardis_reference"

# Observed data
OBS_DIR = LUMINA_ROOT / "data" / "sn2011fe"
OBS_FILE_BMAX = OBS_DIR / "sn2011fe_observed_Bmax.csv"

# Output directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Temporary directory for LUMINA runs
TMPDIR_BASE = Path("/tmp/lumina_ml")

# ===== Physical constants =====
SIGMA_SB = 5.6704e-5       # Stefan-Boltzmann (erg/cm^2/s/K^4)
C_LIGHT = 2.99792458e10    # Speed of light (cm/s)
V_OUTER = 25000.0           # Outer velocity boundary (km/s)
N_SHELLS = 30               # Number of radial shells
T_REF = 19.0                # Reference epoch (days) — SN 2011fe B-max

# Ni56 → Co56 → Fe56 decay chain
TAU_NI = 8.8    # Ni56 half-life (days)
TAU_CO = 111.4  # Co56 half-life (days)
LN2 = 0.6931471805599453
LAMBDA_NI = LN2 / TAU_NI   # 0.07876 /day (Ni56 decay rate)
LAMBDA_CO = LN2 / TAU_CO   # 0.006223 /day (Co56 decay rate)

# ===== 15D Parameter Space =====
PARAM_NAMES = [
    'log_L', 'v_inner', 'log_rho_0', 'density_exp', 'T_e_ratio',
    'v_core', 'v_wall', 'X_Fe_core', 'X_Si_wall',
    'v_break', 'density_exp_outer', 't_exp',
    'X_Fe_wall', 'X_Ni', 'X_Fe_outer',
]

PARAM_RANGES = [
    (42.50, 43.50),    # log_L (erg/s) — expanded both sides
    (7000, 15000),     # v_inner (km/s) — expanded both sides
    (-14.0, -12.3),    # log_rho_0 (g/cm^3) — expanded both sides
    (-10, -4),         # density_exp (inner slope)
    (0.7, 1.0),        # T_e_ratio
    (9000, 17000),     # v_core (km/s) — expanded both sides
    (12000, 24000),    # v_wall (km/s) — lower bound expanded
    (0.05, 0.85),      # X_Fe_core — lower bound expanded
    (0.05, 0.75),      # X_Si_wall — lower bound expanded
    (10000, 22000),    # v_break (km/s) — expanded both sides
    (-14, -4),         # density_exp_outer — upper bound expanded
    (10.0, 28.0),      # t_exp (days) — expanded both sides
    (0.001, 0.50),     # X_Fe_wall (Fe in wall zone) — expanded both sides
    (0.005, 0.25),     # X_Ni (Ni abundance, all zones) — expanded both sides
    (0.001, 0.15),     # X_Fe_outer (Fe in outer zone) — expanded both sides
]

PARAM_RANGES_DICT = dict(zip(PARAM_NAMES, PARAM_RANGES))

N_PARAMS = len(PARAM_NAMES)  # 15

# ===== Fixed abundances (not free parameters) =====
# Co is now computed from Ni56 decay chain (not fixed)
FIXED_SPECIES = {
    6:  0.02,   # C (fixed — mostly burned, weak optical lines)
}
FIXED_SPECIES_SUM_BASE = sum(FIXED_SPECIES.values())  # 0.02

# Fixed abundances per zone (Stage 1 — freed in Stage 2)
ZONE_S  = {'core': 0.05, 'wall': 0.05, 'outer': 0.02}
ZONE_CA = {'core': 0.03, 'wall': 0.03, 'outer': 0.01}
ZONE_MG = {'core': 0.005, 'wall': 0.01, 'outer': 0.02}   # IME, outer layers
ZONE_TI = {'core': 0.001, 'wall': 0.002, 'outer': 0.0005} # trace, Si-burning
ZONE_CR = {'core': 0.002, 'wall': 0.003, 'outer': 0.001}  # trace, Fe-group

# Element ordering in abundances.csv (must match atom_masses.csv)
# C=6, O=8, Mg=12, Si=14, S=16, Ca=20, Ti=22, Cr=24, Fe=26, Co=27, Ni=28
ELEMENT_ORDER = [6, 8, 12, 14, 16, 20, 22, 24, 26, 27, 28]

# Files regenerated per model (rest are symlinked from reference)
REGEN_FILES = {
    'config.json', 'geometry.csv', 'density.csv', 'abundances.csv',
    'electron_densities.csv', 'plasma_state.csv',
}

# ===== Spectrum grid =====
WAVE_MIN = 2000.0   # Angstrom
WAVE_MAX = 10000.0  # Angstrom
WAVE_STEP = 5.0     # Angstrom
SPECTRUM_GRID = np.arange(WAVE_MIN, WAVE_MAX + WAVE_STEP, WAVE_STEP)  # 1101 bins

# LUMINA raw output grid (500-20000 A, 2000 bins)
LUMINA_N_BINS = 2000
LUMINA_WAVE_MIN = 500.0
LUMINA_WAVE_MAX = 20000.0

# ===== PCA config =====
PCA_VARIANCE_THRESHOLD = 0.999  # Keep 99.9% variance
PCA_MAX_COMPONENTS = 300        # Upper bound on components

# ===== Preprocessing: asinh transform =====
# asinh(flux / ASINH_SOFTENING) equalizes UV and optical in PCA space
# F_ref chosen so that asinh(peak_flux / F_ref) ~ 3-5 (good dynamic range)
ASINH_SOFTENING = 0.05   # In peak-normalized units; asinh(1/0.05) = 3.69

# ===== Adaptive Savitzky-Golay smoothing =====
SG_ORDER = 3
# Region-dependent smoothing windows (Angstrom -> points at 5A/pt)
SG_REGIONS = [
    # (wave_min, wave_max, window_angstrom)
    (2000, 3500,  155),   # UV: heavy smoothing (31 pts) — noisy, blanketed
    (3500, 4500,  105),   # NUV: moderate (21 pts) — Ca H&K region
    (4500, 7500,   55),   # Optical: light (11 pts) — Si II, S II, Fe II
    (7500, 10000, 105),   # NIR: moderate (21 pts) — Ca IR triplet, O I
]

# ===== Spectral feature windows for loss weighting =====
FEATURE_WINDOWS = {
    'Si_II_6355': (5900, 6500),    # Si II 6355 trough + blue wing
    'Si_II_5972': (5700, 6000),    # Si II 5972 (temperature indicator)
    'S_II_W':     (5300, 5700),    # S II "W" doublet (5454 + 5640)
    'Ca_HK':      (3600, 4000),    # Ca II H&K (3934/3968)
    'Fe_blend':   (4500, 5200),    # Fe II + blends
    'Ca_IR':      (8000, 8800),    # Ca II IR triplet (8498/8542/8662)
    'O_I':        (7400, 7900),    # O I 7774
}
# UV/optical ratio bands
UV_BAND = (2500, 3500)
OPT_BAND = (5000, 6000)

# Feature loss weights
FEATURE_LOSS_LAMBDA = 0.5     # Weight for feature windows in composite loss
UV_RATIO_LOSS_LAMBDA = 0.3    # Weight for UV/optical ratio loss

# ===== Neural network config =====
NN_HIDDEN_LAYERS = [512, 512, 256, 256, 128]
NN_ACTIVATION = 'SiLU'
NN_DROPOUT = 0.01
NN_LEARNING_RATE = 5e-4
NN_WEIGHT_DECAY = 1e-5
NN_BATCH_SIZE = 128
NN_MAX_EPOCHS = 3000
NN_PATIENCE = 300       # Early stopping patience
NN_T0 = 100             # CosineAnnealing T_0
NN_T_MULT = 2           # CosineAnnealing T_mult
NN_GRAD_CLIP = 1.0

# ===== MCMC config =====
MCMC_N_WALKERS = 64
MCMC_N_BURN = 2000
MCMC_N_PRODUCTION = 5000

# ===== Nested sampling config =====
DYNESTY_LIVE_POINTS = 500
DYNESTY_DLOGZ = 0.1

# ===== Training data generation =====
DEFAULT_N_MODELS = 50000
DEFAULT_N_PACKETS = 200000
DEFAULT_N_ITERS = 10
BATCH_SAVE_INTERVAL = 100

# ===== Feature windows for Si II measurement =====
SI_II_REST = 6355.0  # Si II 6355 rest wavelength (Angstrom)

# ===== Multi-stage refinement =====
RELAXATION_MARGIN = 0.10   # ±10% around previous-stage best-fit

# Stage 2: 6-zone heavy-element composition
STAGE2_N_ZONES = 6
STAGE2_SPECIES = [26, 14, 16, 20, 28, 12, 22, 24]  # Fe, Si, S, Ca, Ni, Mg, Ti, Cr
STAGE2_SPECIES_NAMES = ['Fe', 'Si', 'S', 'Ca', 'Ni', 'Mg', 'Ti', 'Cr']
STAGE2_ABUNDANCE_RANGES = {
    26: (0.001, 0.85),  # Fe
    14: (0.001, 0.75),  # Si
    16: (0.001, 0.10),  # S
    20: (0.001, 0.10),  # Ca
    28: (0.001, 0.40),  # Ni (initial Ni56)
    12: (0.001, 0.10),  # Mg
    22: (0.0001, 0.02), # Ti (trace, but strong blue lines)
    24: (0.0001, 0.02), # Cr (trace, Fe-group blend)
}

# Stage 3: 15-zone (finer granularity)
STAGE3_N_ZONES = 15


# Parameters where ±margin applies to the full prior width (not the value itself)
# — log-scale params have large absolute values but small meaningful ranges
_RANGE_BASED_MARGIN = {'log_L', 'log_rho_0', 'density_exp', 'density_exp_outer'}


def relaxed_ranges(best_fit, param_names, param_ranges, margin=RELAXATION_MARGIN):
    """Create parameter ranges relaxed ±margin around previous-stage best-fit.

    Args:
        best_fit: dict {param_name: best_value} from previous stage
        param_names: list of parameter names for this stage
        param_ranges: list of (lo, hi) full prior ranges for this stage
        margin: fractional relaxation (default 0.10 = ±10%)

    Returns:
        list of (lo, hi) tuples — narrowed for inherited params, full for new params
    """
    relaxed = []
    for name, (full_lo, full_hi) in zip(param_names, param_ranges):
        if name in best_fit:
            val = best_fit[name]
            if name in _RANGE_BASED_MARGIN:
                # Log/exponent params: ±margin of the full prior width
                delta = margin * (full_hi - full_lo)
            elif abs(val) < 1e-10:
                # Near-zero: fallback to range-based
                delta = margin * (full_hi - full_lo)
            else:
                # Normal: ±margin of the value itself
                delta = abs(val) * margin
            lo = max(full_lo, val - delta)
            hi = min(full_hi, val + delta)
            # Ensure at least a tiny range
            if hi <= lo:
                mid = (lo + hi) / 2
                lo = mid - 1e-6
                hi = mid + 1e-6
            relaxed.append((lo, hi))
        else:
            # New parameter: use full prior range
            relaxed.append((full_lo, full_hi))
    return relaxed
