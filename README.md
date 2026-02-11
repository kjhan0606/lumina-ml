# LUMINA-ML: Spectral Emulator & Bayesian Inference for SN Ia

ML-based spectral emulator and Bayesian inverse-problem pipeline for the LUMINA-SN Monte Carlo radiative transfer code.

## Overview

A 3-stage recursive fitting system to estimate physical parameters from the observed spectrum of SN 2011fe (Type Ia supernova):

```
                    Stage 1                    Stage 2                    Stage 3
              ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  Observed ──>│ 15D Global Search│──────>│ 6-Zone Comp.    │──────>│ 15-Zone Comp.   │──> Final Model
  Spectrum    │  LHS → MLP → MCMC│ ±10%  │ LHS → MLP → MCMC│ ±10%  │ LHS → MLP → MCMC│
              └─────────────────┘ relax └─────────────────┘ relax └─────────────────┘
```

Each stage passes its best-fit results to the next stage with **±10% relaxation**, preventing the search from being trapped in a local minimum.

## 3-Stage Recursive Fitting

### Stage 1: Global Parameters (15D)

Explores macroscopic physical quantities: density structure, velocity boundaries, and luminosity.

| Parameter | Description | Range |
|-----------|-------------|-------|
| `log_L` | log10(luminosity) [erg/s] | 42.50 -- 43.50 |
| `v_inner` | Inner velocity [km/s] | 7,000 -- 15,000 |
| `log_rho_0` | log10(reference density) [g/cm3] | -14.0 -- -12.3 |
| `density_exp` | Inner density exponent | -10 -- -4 |
| `T_e_ratio` | T_e / T_rad | 0.7 -- 1.0 |
| `v_core` | Core/wall boundary [km/s] | 9,000 -- 17,000 |
| `v_wall` | Wall/outer boundary [km/s] | 12,000 -- 24,000 |
| `X_Fe_core` | Fe mass fraction (core) | 0.05 -- 0.85 |
| `X_Si_wall` | Si mass fraction (wall) | 0.05 -- 0.75 |
| `v_break` | Density slope break [km/s] | 10,000 -- 22,000 |
| `density_exp_outer` | Outer density exponent | -14 -- -4 |
| `t_exp` | Time since explosion [days] | 10 -- 28 |
| `X_Fe_wall` | Fe mass fraction (wall) | 0.001 -- 0.50 |
| `X_Ni` | Initial Ni56 fraction (all zones) | 0.005 -- 0.25 |
| `X_Fe_outer` | Fe mass fraction (outer) | 0.001 -- 0.15 |

- Density model: **Broken power-law** (v < v_break: inner exponent, v >= v_break: outer exponent)
- Composition model: **3-zone** (Core / Si-wall / Outer), O as filler element
- Sampling: 25,000--50,000 Latin Hypercube samples

### Stage 2: 6-Zone Composition (~63D)

Fixes global parameters from Stage 1 within ±10% and explores 8 heavy-element abundances independently across 6 radial zones.

**6 Zones** (30 shells divided into 6 zones):
| Zone | Shells | Region |
|------|--------|--------|
| 1 | 0--4 | Innermost core |
| 2 | 5--9 | Outer core |
| 3 | 10--14 | Inner Si wall |
| 4 | 15--19 | Outer Si wall |
| 5 | 20--24 | Transition |
| 6 | 25--29 | Outer envelope |

**8 Species** per zone:
| Element | Z | Range | Role |
|---------|---|-------|------|
| Fe | 26 | 0.001 -- 0.85 | Line blanketing (dominant) |
| Si | 14 | 0.001 -- 0.75 | SN Ia classification feature |
| S  | 16 | 0.001 -- 0.10 | "W" diagnostic doublet |
| Ca | 20 | 0.001 -- 0.10 | H&K, IR triplet |
| Ni | 28 | 0.001 -- 0.40 | UV opacity, radioactive source |
| Mg | 12 | 0.001 -- 0.10 | IME diagnostic |
| Ti | 22 | 0.0001 -- 0.02 | Blue suppression (4000--4500 A) |
| Cr | 24 | 0.0001 -- 0.02 | Fe-group blend (4500--4800 A) |

Total dimension: 15 (global, relaxed) + 6 x 8 (composition) = **63D**

### Stage 3: 15-Zone Composition (~135D)

Refines Stage 2 results from 6 zones to 15 zones. Dimensionality is kept tractable via PCA-based reduction or parametric abundance profiles.

## Elements (11 species)

All elements included in LUMINA simulations (confirmed in atomic database):

| Z | Element | Lines | Levels | Macro-atom | Status |
|---|---------|-------|--------|------------|--------|
| 6 | C | 5,762 | -- | -- | Fixed (0.02) |
| 8 | O | 4,824 | -- | -- | Filler (1 - rest) |
| 12 | Mg | 3,738 | 663 | 663 | Fixed in Stage 1, free in Stage 2 |
| 14 | Si | 2,344 | -- | -- | Free (Stage 1: per-zone) |
| 16 | S | 4,110 | -- | -- | Fixed in Stage 1, free in Stage 2 |
| 20 | Ca | 7,877 | -- | -- | Fixed in Stage 1, free in Stage 2 |
| 22 | Ti | 18,688 | 1,444 | 1,444 | Fixed in Stage 1, free in Stage 2 |
| 24 | Cr | 19,681 | 1,700 | 1,700 | Fixed in Stage 1, free in Stage 2 |
| 26 | Fe | 20,510 | -- | -- | Free (Stage 1: per-zone) |
| 27 | Co | 33,115 | -- | -- | Computed from Ni56 decay |
| 28 | Ni | 58,693 | -- | -- | Free (initial Ni56) |

Total: **271,741 lines** in atomic database (kurucz_cd23_chianti_H_He.h5)

## Ni56 Decay Chain

Co abundance is not a fixed value but is physically computed from the Ni56 radioactive decay chain:

```
  Ni56 ──(t_half=8.8d)──> Co56 ──(t_half=111.4d)──> Fe56
```

At time t_exp (Bateman equations):
- `X_Ni(t) = X_Ni_initial * exp(-lambda_Ni * t)`
- `X_Co(t) = X_Ni_initial * lambda_Ni/(lambda_Co - lambda_Ni) * (exp(-lambda_Ni*t) - exp(-lambda_Co*t))`
- `X_Fe_decay(t) = X_Ni_initial - X_Ni(t) - X_Co(t)`

Example at B-max (t ~ 19 days, X_Ni_initial = 0.10):
| Species | Fraction | Mass |
|---------|----------|------|
| Ni56 (remaining) | 22.4% | 0.0224 |
| Co56 (from decay) | 72.2% | 0.0722 |
| Fe56 (from decay) | 5.5% | 0.0054 |

Fe56 from decay is added to the zone's Fe abundance. Total mass is conserved.

## Pipeline

### Scripts

```
scripts/
  01_generate_training_data.py   # LHS sampling -> LUMINA simulation -> (params, spectrum) pairs
  02_preprocess.py               # Savitzky-Golay smoothing -> asinh transform -> PCA
  03_train_emulator.py           # MLP emulator: params -> PCA coefficients
  04_run_inference.py            # Bayesian inference: MCMC / Nested / SBI
  05_plot_results.py             # Corner plots, spectrum comparison, diagnostics
```

### Step 1: Generate Training Data

```bash
# Auto-detect GPU+CPU, dynamic work queue
python3 scripts/01_generate_training_data.py --n-models 25000 --mode both

# Resume from checkpoint
python3 scripts/01_generate_training_data.py --resume

# CPU only
python3 scripts/01_generate_training_data.py --mode cpu --omp-threads 64
```

In `--mode both`, GPU and CPU workers share a **dynamic work queue**:
- All models are placed into a shared queue
- GPU and CPU threads each pull the next available model from the queue
- Faster devices naturally process more models
- Automatic checkpoint every 100 models

### Step 2: Preprocess

```bash
python3 scripts/02_preprocess.py
```

- Adaptive Savitzky-Golay smoothing (UV: 155A, Optical: 55A, NIR: 105A)
- asinh transform (softening=0.05, equalizes UV/optical dynamic range)
- PCA dimensionality reduction (99.9% variance retained)

### Step 3: Train Emulator

```bash
python3 scripts/03_train_emulator.py
```

- MLP: [512, 512, 256, 256, 128] with SiLU activation
- Feature-weighted loss: Si II, Ca H&K, S II, Fe blend, Ca IR, O I
- CosineAnnealing LR schedule, early stopping (patience=300)

### Step 4: Bayesian Inference

```bash
python3 scripts/04_run_inference.py
```

Three methods for posterior estimation:
- **MCMC** (emcee): 64 walkers, 2000 burn-in, 5000 production
- **Nested sampling** (dynesty): 500 live points
- **SBI** (sbi/SNPE): Neural posterior estimation

### Step 5: Plot Results

```bash
python3 scripts/05_plot_results.py
```

## Relaxed Parameter Inheritance

Each stage uses ±10% of the previous stage's best-fit as its search range:

```python
from lumina_ml.config import relaxed_ranges

stage1_best = {'log_L': 43.10, 'v_inner': 11500, ...}
stage2_ranges = relaxed_ranges(stage1_best, param_names, param_ranges, margin=0.10)
# log_L: 43.10 -> [43.00, 43.20]  (range-based, 10% of prior width)
# v_inner: 11500 -> [10350, 12650]  (value-based, 10% of the value)
# New parameters: full prior range retained
```

Log/exponent parameters use **10% of the prior width**, while linear parameters use **10% of the value itself**.

## Project Structure

```
Lumina-ML/
  lumina_ml/
    config.py          # All settings: parameter ranges, elements, network, inference
    data_utils.py      # ModelParams, LuminaRunner, LHS sampling
    preprocessing.py   # SG smoothing, asinh, PCA
    emulator.py        # MLP emulator (PyTorch)
    inference.py       # MCMC, Nested, SBI wrapper
  scripts/             # 5-step pipeline scripts
  data/
    raw/               # Generated (params, spectra, waves).npy
    processed/         # PCA-transformed data
  models/              # Trained emulator checkpoints
  results/             # Posterior samples, corner plots
  docs/
    LUMINA_ML_GUIDE.md # Comprehensive technical guide
```

## Dependencies

- LUMINA-SN: `../Lumina-sn/lumina_cuda` (GPU) or `../Lumina-sn/lumina` (CPU)
- Atomic data: `../Lumina-sn/data/atomic/kurucz_cd23_chianti_H_He.h5`
- Python: see `requirements.txt`

```bash
pip install -r requirements.txt
```

## Hardware

- GPU: NVIDIA RTX 5000 Ada (sm_89), CUDA 13.0
- ~3s/model (GPU), ~10s/model (CPU with 64 OMP threads)
- 25,000 models: ~21h (GPU+CPU dynamic queue)

## References

- LUMINA-SN: Monte Carlo radiative transfer for Type Ia supernovae
- TARDIS: Kerzendorf & Sim (2014), reference implementation
- SN 2011fe: Nugent et al. (2011), well-observed nearby SN Ia
