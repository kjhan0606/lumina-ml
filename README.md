# LUMINA-ML: Spectral Emulator & Bayesian Inference for SN Ia

LUMINA-SN Monte Carlo 복사전달 코드의 ML 기반 스펙트럼 에뮬레이터 및 베이지안 역문제 풀이 파이프라인.

## Overview

SN 2011fe (Type Ia 초신성)의 관측 스펙트럼에서 물리적 파라미터를 추정하기 위한 3단계 재귀적 피팅 시스템:

```
                    Stage 1                    Stage 2                    Stage 3
              ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  관측 ──────>│ 15D 글로벌 탐색  │──────>│ 6존 조성 세분화  │──────>│ 15존 조성 세분화 │──> 최종 모델
  스펙트럼    │  LHS → MLP → MCMC│ ±10%  │ LHS → MLP → MCMC│ ±10%  │ LHS → MLP → MCMC│
              └─────────────────┘ relax └─────────────────┘ relax └─────────────────┘
```

각 단계의 결과가 다음 단계의 초기 범위로 전달되며, **±10% 여유**를 둬서 이전 단계의 local minimum에 갇히지 않도록 합니다.

## 3-Stage Recursive Fitting

### Stage 1: Global Parameters (15D)

밀도 구조, 속도, 광도 등 거시적 물리량을 탐색합니다.

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

- 밀도 모델: **Broken power-law** (v < v_break: exp_inner, v >= v_break: exp_outer)
- 조성 모델: **3-zone** (Core / Si-wall / Outer), O filler
- 샘플링: 25,000--50,000 Latin Hypercube samples

### Stage 2: 6-Zone Composition (~48D)

Stage 1에서 글로벌 파라미터를 ±10% 범위로 고정하고, 6개 존에서 8종 중원소 조성을 개별 탐색합니다.

**6 Zones** (셸 30개를 6개 존으로 분할):
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

Dimension: 15 (global, relaxed) + 6 x 8 (composition) = **63D**

### Stage 3: 15-Zone Composition (~135D)

Stage 2의 6존 결과를 15존으로 세분화. PCA 기반 차원 축소 또는 parametric abundance profile을 통해 tractable하게 유지합니다.

## Elements (11 species)

LUMINA 시뮬레이션에 포함되는 원소 목록 (atomic data에서 확인됨):

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

Co 함량은 고정값이 아니라 Ni56 방사성 붕괴 체인에서 물리적으로 계산됩니다:

```
  Ni56 ──(t_half=8.8d)──> Co56 ──(t_half=111.4d)──> Fe56
```

시간 t_exp에서:
- `X_Ni(t) = X_Ni_initial * exp(-lambda_Ni * t)`
- `X_Co(t) = X_Ni_initial * lambda_Ni/(lambda_Co - lambda_Ni) * (exp(-lambda_Ni*t) - exp(-lambda_Co*t))`
- `X_Fe_decay(t) = X_Ni_initial - X_Ni(t) - X_Co(t)`

B-max (t ~ 19 days) 예시 (X_Ni_initial = 0.10):
| Species | Fraction | Mass |
|---------|----------|------|
| Ni56 (remaining) | 22.4% | 0.0224 |
| Co56 (from decay) | 72.2% | 0.0722 |
| Fe56 (from decay) | 5.5% | 0.0054 |

Fe56 from decay는 zone Fe에 합산됩니다. 전체 질량은 보존됩니다.

## Pipeline

### Scripts

```
scripts/
  01_generate_training_data.py   # LHS sampling → LUMINA simulation → (params, spectrum) pairs
  02_preprocess.py               # Savitzky-Golay smoothing → asinh transform → PCA
  03_train_emulator.py           # MLP emulator: params → PCA coefficients
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

`--mode both`에서 GPU와 CPU는 **동적 작업 큐**로 작동합니다:
- 모든 모델이 공유 큐에 들어감
- GPU/CPU 스레드가 각자 큐에서 다음 모델을 가져감
- 빠른 디바이스가 자연스럽게 더 많은 모델을 처리
- 100개 모델마다 자동 체크포인트

### Step 2: Preprocess

```bash
python3 scripts/02_preprocess.py
```

- Adaptive Savitzky-Golay smoothing (UV: 155A, Optical: 55A, NIR: 105A)
- asinh transform (softening=0.05, equalizes UV/optical dynamic range)
- PCA 차원 축소 (99.9% variance retained)

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

3가지 방법으로 posterior 추정:
- **MCMC** (emcee): 64 walkers, 2000 burn-in, 5000 production
- **Nested sampling** (dynesty): 500 live points
- **SBI** (sbi/SNPE): Neural posterior estimation

### Step 5: Plot Results

```bash
python3 scripts/05_plot_results.py
```

## Relaxed Parameter Inheritance

각 stage에서 이전 결과의 ±10%를 탐색 범위로 사용합니다:

```python
from lumina_ml.config import relaxed_ranges

stage1_best = {'log_L': 43.10, 'v_inner': 11500, ...}
stage2_ranges = relaxed_ranges(stage1_best, param_names, param_ranges, margin=0.10)
# log_L: 43.10 → [43.00, 43.20]  (range-based, prior width의 10%)
# v_inner: 11500 → [10350, 12650]  (value-based, 값의 10%)
# 새 파라미터: 전체 범위 유지
```

로그/지수 파라미터는 **prior width의 10%**, 일반 파라미터는 **값의 10%**를 사용합니다.

## Project Structure

```
Lumina-ML/
  lumina_ml/
    config.py          # 모든 설정: 파라미터 범위, 원소, 네트워크, 추론
    data_utils.py      # ModelParams, LuminaRunner, LHS sampling
    preprocessing.py   # SG smoothing, asinh, PCA
    emulator.py        # MLP 에뮬레이터 (PyTorch)
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
