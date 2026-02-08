# LUMINA-ML: Guide for Dummies

**A Complete Methodological Guide to Neural Spectral Emulation & Bayesian Inference for Type Ia Supernovae**

---

## Table of Contents

1. [The Big Picture: Why Do We Need This?](#1-the-big-picture)
2. [Step 0: The Forward Problem — LUMINA Simulator](#2-the-forward-problem)
3. [Step 1: Latin Hypercube Sampling — Designing the Training Set](#3-latin-hypercube-sampling)
4. [Step 2: Preprocessing — Taming the Raw Spectra](#4-preprocessing)
   - 4a. Adaptive Savitzky-Golay Smoothing
   - 4b. Peak Normalization
   - 4c. Inverse Hyperbolic Sine (asinh) Transform
5. [Step 3: PCA — Compressing 1,101 Bins to ~60 Numbers](#5-pca)
6. [Step 4: Neural Network Emulator — The Fast Surrogate](#6-neural-network)
   - 6a. Architecture (MLP)
   - 6b. Activation Functions (SiLU)
   - 6c. Training (AdamW, Cosine Annealing)
   - 6d. Feature-Weighted Composite Loss
7. [Step 5: Bayesian Inference — The Inverse Problem](#7-bayesian-inference)
   - 7a. Bayes' Theorem
   - 7b. The Likelihood Function
   - 7c. MCMC with emcee
   - 7d. Nested Sampling with dynesty
   - 7e. Simulation-Based Inference (SBI/SNPE)
8. [Method Comparison: MCMC vs Nested vs SBI](#8-method-comparison)
9. [Glossary](#9-glossary)
10. [References](#10-references)

---

# 1. The Big Picture

## The Problem

We observe the spectrum of a Type Ia supernova (SN 2011fe). This spectrum encodes
physical information: how bright the explosion was, how fast the ejecta moves, what
elements are present, etc. We want to **extract** these physical parameters from the
observed spectrum.

## The Difficulty

LUMINA is a Monte Carlo radiative transfer code that can simulate a supernova spectrum
given a set of physical parameters. But each simulation takes **4-8 seconds** (GPU) or
**30+ seconds** (CPU). To explore a 15-dimensional parameter space, we need millions
of evaluations. At 5 seconds each, that's **years** of compute time.

## The Solution: Emulator + Bayesian Inference

```
                    TRAINING PHASE (one-time, ~2 days)
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Physical Parameters ──→ LUMINA ──→ Synthetic Spectrum   │
  │  (50,000 samples)       (slow)     (training data)       │
  │         │                                │               │
  │         ▼                                ▼               │
  │  Normalize [0,1]          Smooth → asinh → PCA → Scale   │
  │         │                                │               │
  │         ▼                                ▼               │
  │    X_train [N×15]                  Y_train [N×60]        │
  │         │                                │               │
  │         └──────── Neural Network ────────┘               │
  │                   (15 → 60, MLP)                         │
  │                   Training: ~10 min                      │
  └──────────────────────────────────────────────────────────┘

                    INFERENCE PHASE (fast, ~1 minute)
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Observed Spectrum (SN 2011fe)                           │
  │         │                                                │
  │         ▼                                                │
  │  Same preprocessing (smooth → asinh)                     │
  │         │                                                │
  │         ▼                                                │
  │  MCMC / Nested / SBI                                     │
  │    "Which parameters θ make the emulator                 │
  │     output match the observed spectrum?"                  │
  │         │                                                │
  │         ▼                                                │
  │  Posterior Distribution P(θ | data)                      │
  │    → Best-fit parameters + uncertainties                 │
  └──────────────────────────────────────────────────────────┘
```

The neural network replaces LUMINA's 5-second simulation with a **0.0001-second**
prediction. This makes Bayesian inference feasible: 500,000 likelihood evaluations
take ~50 seconds instead of ~29 days.

---

# 2. The Forward Problem — LUMINA Simulator

LUMINA solves the radiative transfer equation for supernova ejecta using Monte Carlo
methods. Millions of "photon packets" are launched from the photosphere and propagated
through expanding ejecta shells, interacting with atoms along the way.

## The 15 Physical Parameters

Our parameter space has 15 dimensions. Think of each simulation as a point in this
15D space, and the output spectrum as a function of that point:

```
spectrum(λ) = f(θ₁, θ₂, ..., θ₁₅)
```

| # | Parameter | Symbol | Range | What It Controls |
|---|-----------|--------|-------|------------------|
| 1 | Luminosity | log L | 42.80-43.15 | Overall brightness |
| 2 | Inner velocity | v_inner | 8,000-13,000 km/s | Photosphere location |
| 3 | Reference density | log ρ₀ | -13.5 to -12.7 | How much matter |
| 4 | Density exponent | n_inner | -10 to -4 | Density profile slope |
| 5 | Temperature ratio | T_e/T_rad | 0.7-1.0 | Ionization balance |
| 6 | Core boundary | v_core | 10,000-16,000 km/s | Where Fe-rich core ends |
| 7 | Wall boundary | v_wall | 14,000-22,000 km/s | Where Si-rich wall ends |
| 8 | Core Fe fraction | X_Fe_core | 0.2-0.8 | Iron in the core |
| 9 | Wall Si fraction | X_Si_wall | 0.2-0.7 | Silicon in the wall |
| 10 | Break velocity | v_break | 12,000-20,000 km/s | Density slope change |
| 11 | Outer exponent | n_outer | -14 to -6 | Outer density falloff |
| 12 | Time since explosion | t_exp | 12-25 days | Epoch of observation |
| 13 | Wall Fe fraction | X_Fe_wall | 0.01-0.40 | Iron contamination |
| 14 | Nickel fraction | X_Ni | 0.02-0.20 | Radioactive power |
| 15 | Outer Fe fraction | X_Fe_outer | 0.005-0.10 | High-velocity iron |

### The Ejecta Structure (3-Zone Model)

```
         Core              Wall              Outer
    ◄──────────────►◄───────────────►◄──────────────────►
    │   Fe-rich     │   Si-rich      │   O-rich          │
    │   Fe: 20-80%  │   Si: 20-70%   │   O: filler       │
    │   Si: 5%      │   Fe: 1-40%    │   Fe: 0.5-10%     │
    │               │   S:  5%       │   Si: 2%           │
    │               │   Ca: 3%       │                    │
    │               │                │                    │
  v_inner         v_core           v_wall              v_outer
  (photosphere)                                      (25000 km/s)
```

### Broken Power-Law Density Profile

The density of the ejecta drops with velocity, but the rate of drop can change:

```
            log ρ
              │
              │╲  slope = n_inner
              │ ╲    (e.g., -7)
              │  ╲
              │   ╲
              │    ╲___
              │        ╲  slope = n_outer
              │         ╲   (e.g., -12)
              │          ╲
              │           ╲
              └────────────╲──────────── log v
                   v_break
```

**Mathematical form:**

For v < v_break:

    ρ(v) = ρ₀ × (v / v_inner)^n_inner

For v ≥ v_break:

    ρ(v) = ρ_break × (v / v_break)^n_outer

where ρ_break = ρ₀ × (v_break / v_inner)^n_inner ensures continuity.

**Time evolution:** as the supernova expands, density drops as t⁻³:

    ρ(v, t) = ρ(v, t_ref) × (t_ref / t)³

---

# 3. Latin Hypercube Sampling (LHS)

## The Challenge of High Dimensions

We need to sample 15D parameter space efficiently. Simple approaches fail:

- **Grid sampling**: 10 points per dimension → 10¹⁵ = 1 quadrillion models. Impossible.
- **Random sampling**: Leaves gaps and clusters. Wastes samples.

## How LHS Works

Latin Hypercube Sampling is like a multi-dimensional version of a Sudoku puzzle.

### 1D Analogy: Latin Square

Imagine you want N=5 samples of a parameter x in [0, 1]:

```
Random sampling:        Latin Hypercube:
  x: 0.12, 0.15, 0.18,    x: 0.13, 0.38, 0.52, 0.71, 0.95
     0.45, 0.91
                            │   │   │   │   │
  ──┼─┼─┼────┼──────┼──  ──┼───┼───┼───┼───┼──
  0  0.2 0.4 0.6 0.8 1   0  0.2 0.4 0.6 0.8 1
     ▲▲▲ clustered!        Each bin has exactly 1 sample
```

LHS divides [0, 1] into N equal bins and places **exactly one sample per bin**.
This guarantees uniform marginal coverage.

### Multi-Dimensional Extension

For 15 parameters and N=50,000 samples:

1. For each dimension d = 1, ..., 15:
   - Create a random permutation π_d of {0, 1, ..., 49999}
   - Sample i gets bin π_d(i) in dimension d
   - Within that bin, sample uniformly: x_d(i) ~ U[π_d(i)/N, (π_d(i)+1)/N]

2. Scale from [0, 1] to physical range:
   x_physical = x_min + x_lhs × (x_max - x_min)

3. Validate physical constraints:
   - v_core + 1000 < v_wall (zones must be separated)
   - v_core > v_inner (core outside photosphere)
   - X_Fe_wall + X_Si_wall ≤ 0.65 (can't exceed 100% with filler)
   - etc.

### Why LHS is Better Than Random

```
Random (N=25, 2D):           LHS (N=25, 2D):

  1 │  · ·                     1 │ ·   ·  · ·  ·
    │    ·  ·                    │  · ·   ·  ·
    │  · ·                       │ ·  · ·   · ·
    │      ·                     │  ·  ·  · ·
    │  · ·   · ·                 │ · ·  · ·   ·
  0 └──────────── 1            0 └──────────────── 1

  Clusters & gaps!              Every row AND column
                                has exactly 1 sample
```

For the same number of samples, LHS provides:
- **Better coverage** of extreme regions (corners of parameter space)
- **Lower variance** in estimated statistics
- **No gaps**: every "slice" of parameter space is represented

### Our Implementation

```python
# From data_utils.py
def latin_hypercube_valid(n_samples, seed=42):
    rng = np.random.default_rng(seed)

    # Over-generate to account for rejected invalid samples
    n_try = int(n_samples * 1.5)

    # Standard LHS
    samples = np.zeros((n_try, 15))
    for d in range(15):
        perm = rng.permutation(n_try)
        samples[:, d] = (perm + rng.uniform(size=n_try)) / n_try

    # Scale to physical ranges
    for d, (lo, hi) in enumerate(PARAM_RANGES.values()):
        samples[:, d] = lo + samples[:, d] * (hi - lo)

    # Keep only physically valid samples
    valid = [s for s in samples if ModelParams(*s).is_valid()]
    return np.array(valid[:n_samples])
```

---

# 4. Preprocessing — Taming the Raw Spectra

Raw LUMINA spectra are noisy (Monte Carlo noise), span a huge dynamic range (UV is
50× dimmer than optical), and have 1,101 bins. We need to clean and compress them.

## 4a. Adaptive Savitzky-Golay Smoothing

### The Problem: Monte Carlo Noise

LUMINA uses random photon packets, so the output spectrum has statistical noise:

```
Raw spectrum (200K packets):

  Flux │    ╱╲    ╱╲╱╲     True signal
       │ ╱╲╱  ╲╱╲╱    ╲╱╲
       │╱                  ╲╱╲  ╱╲
       │                       ╲╱  ╲  Noisy wiggles
       └────────────────────────────── λ
```

We need to smooth out the noise without destroying real spectral features.

### The Savitzky-Golay (SG) Filter

The SG filter fits a **local polynomial** to a sliding window of data points, then
evaluates the polynomial at the center. It's like a moving average, but smarter.

**How it works:**

For each point i, take its neighbors [i-w, ..., i+w] (window size = 2w+1):

1. Fit a polynomial of degree p to these 2w+1 points (least squares)
2. Evaluate the polynomial at the center point i
3. This gives the smoothed value

```
Window of 11 points:      Polynomial fit (order 3):

  · ·                        ___
  ·   · ·                  _/   \___
              · · ·       /         \___
        ·          ·     │              │
                     ·   │              │
  ────────────────────   ────────────────────
     ← 11 points →           Smooth curve
```

**Why SG is better than a simple moving average:**
- Preserves the **shape** of peaks and troughs (doesn't flatten them)
- Preserves the **position** of features (no phase shift)
- Can handle asymmetric features faithfully

### Adaptive: Different Windows for Different Regions

Key insight: UV photons are scarce (noisy) but have broad features, while optical
features are narrow (Si II trough is ~100 Å wide). We use **different window sizes**:

| Wavelength Region | Window Size | Why |
|-------------------|-------------|-----|
| UV (2000-3500 Å) | 155 Å (31 points) | Very noisy, features are broad |
| Near-UV (3500-4500 Å) | 105 Å (21 points) | Moderate noise, Ca H&K is ~200 Å |
| Optical (4500-7500 Å) | 55 Å (11 points) | Key features! Si II, S II are narrow |
| Near-IR (7500-10000 Å) | 105 Å (21 points) | Ca IR triplet is broad |

**Critical design choice:** The optical window (55 Å) is small enough to preserve:
- Si II 6355 trough (~100 Å wide): ratio = 100/55 ≈ 1.8 → well resolved
- S II "W" double-dip (5454 Å and 5640 Å, separated by 186 Å) → not blurred together

## 4b. Peak Normalization

Each spectrum is divided by its peak flux (in the 4000-7000 Å range):

```
F_normalized(λ) = F(λ) / max{F(λ) : 4000 ≤ λ ≤ 7000}
```

After normalization, the peak is exactly 1.0 and all other values are relative to it.
This removes the overall brightness (luminosity) from the spectral shape, which is
important because we want the neural network to learn **shape**, not amplitude.

## 4c. The asinh Transform

### The Dynamic Range Problem

After normalization, typical flux values are:

| Region | Typical Flux | Relative to Peak |
|--------|-------------|------------------|
| UV (3000 Å) | 0.01-0.05 | 1-5% |
| Si II trough (6150 Å) | 0.2-0.5 | 20-50% |
| Peak (5500 Å) | 1.0 | 100% |

The UV flux is 20-100× smaller. In a least-squares fit, UV contributes almost nothing
because (0.01)² = 0.0001 is negligible compared to (0.5)² = 0.25. But UV contains
crucial physics (iron blanketing, temperature).

### The Transform: f(x) = asinh(x/α)

The **inverse hyperbolic sine** is defined as:

    asinh(x) = ln(x + √(x² + 1))

With a softening parameter α = 0.05:

    f(x) = asinh(x / 0.05)

This function has beautiful properties:

```
For small x (x << α):   asinh(x/α) ≈ x/α         (linear — amplifies small values)
For large x (x >> α):   asinh(x/α) ≈ ln(2x/α)    (logarithmic — compresses large values)
```

**Numerical examples:**

| Linear Flux | asinh(x/0.05) | Compression Ratio |
|-------------|---------------|-------------------|
| 0.01 (UV) | 0.200 | 20× amplification |
| 0.05 | 0.962 | 19× amplification |
| 0.20 | 2.09 | 10.5× amplification |
| 0.50 | 2.99 | 6× amplification |
| 1.00 (peak) | 3.69 | 3.7× amplification |

The dynamic range is compressed from [0.01, 1.0] (100:1) to [0.2, 3.7] (18:1).
Now UV features have comparable weight to optical features in the PCA and loss function.

### Why asinh Instead of log?

- **log(x)** diverges to -∞ at x=0, so zero-flux regions cause NaN
- **asinh(x/α)** is smooth everywhere, even at x=0: asinh(0) = 0
- asinh acts like log for large values but linear for small values
- It's used extensively in astronomy (SDSS magnitudes, "luptitudes")

### The Inverse

To convert back to linear flux for plotting:

    x = α × sinh(f) = 0.05 × (e^f - e^(-f)) / 2

---

# 5. PCA — Compressing 1,101 Bins to ~60 Numbers

## What is PCA?

**Principal Component Analysis** finds the directions of maximum variance in data.
Think of it as finding the "main axes" of a cloud of data points.

### A 2D Analogy

Imagine measuring height and arm span for 1000 people:

```
Arm span │      · ·  · ·
         │    · · · ·· ·
         │  · · ·· ·· ·      ← Data is elongated
         │ · · ·· ·· ·          along a diagonal
         │  · · ·· ·
         │   · · ·
         └─────────────── Height
```

PCA finds:
- **PC1** (first principal component): the direction of maximum spread (the long axis)
- **PC2**: perpendicular to PC1 (the short axis)

If PC1 captures 99% of the variance, you can describe each person with just their
PC1 coordinate (one number instead of two), losing only 1% of the information.

### Applied to Spectra

Each spectrum has 1,101 flux values. Think of each spectrum as a point in
1,101-dimensional space. With 50,000 spectra, we have 50,000 points in this space.

PCA finds that these points lie near a ~60-dimensional hyperplane:

```
1,101 dimensions (wavelength bins)
         │
         │  All spectra live near a    Only ~60 PCA directions
         │  low-dimensional surface    capture 99.9% of the variance
         │
         ▼
  F(λ₁), F(λ₂), ..., F(λ₁₁₀₁)  ──→  c₁, c₂, ..., c₆₀
       (1,101 numbers)                   (60 numbers)
```

## The Math

### Step 1: Compute the Mean Spectrum

    μ(λ) = (1/N) Σᵢ Fᵢ(λ)

This is the "average supernova spectrum."

### Step 2: Center the Data

    F̃ᵢ(λ) = Fᵢ(λ) - μ(λ)

### Step 3: Compute the Covariance Matrix

    C(λ, λ') = (1/N) Σᵢ F̃ᵢ(λ) × F̃ᵢ(λ')

This is a 1101×1101 matrix. Entry C(λ, λ') tells you how much the flux at wavelength
λ co-varies with flux at λ'.

### Step 4: Eigendecomposition

    C × φₖ = σₖ² × φₖ

where:
- φₖ is the k-th **eigenvector** (principal component) — a spectrum-shaped basis vector
- σₖ² is the k-th **eigenvalue** — the variance captured by this component

The eigenvalues are sorted: σ₁² ≥ σ₂² ≥ ... ≥ σ₁₁₀₁².

### Step 5: Project

Each spectrum is represented as a weighted sum of basis spectra:

    Fᵢ(λ) ≈ μ(λ) + Σₖ cᵢₖ × φₖ(λ)

where cᵢₖ = ∫ F̃ᵢ(λ) × φₖ(λ) dλ is the **PCA coefficient** (dot product).

### Choosing the Number of Components

We keep enough components to explain 99.9% of the total variance:

    Σₖ₌₁ⁿ σₖ² / Σₖ₌₁¹¹⁰¹ σₖ² ≥ 0.999

Typically n ≈ 50-80 for our spectra.

### Visual Interpretation of PCA Components

```
PC1 (largest variance):   PC2:                  PC3:

  │     ╱╲                │  ╱╲    ╱╲           │╱╲
  │   ╱    ╲              │╱    ╲╱    ╲         │   ╲  ╱╲
  │─╱────────╲──          │──────────────╲──    │────╲╱───╲──
  │              ╲        │                ╲    │          ╲
  └──────────── λ         └──────────── λ       └──────────── λ

  Overall shape           Temperature           Absorption
  (bright/faint)          (hot/cool shift)      feature depth
```

- **PC1** typically captures the overall spectral slope (hot blue vs cool red)
- **PC2** often captures absorption line depths
- **Higher PCs** capture finer details (line profiles, velocity shifts)

### Standardization of PCA Coefficients

After projection, each coefficient cₖ has different mean and variance. We standardize:

    zₖ = (cₖ - mean(cₖ)) / std(cₖ)

This ensures all PCA coefficients are on the same scale (mean=0, std=1) for neural
network training.

### Reconstruction

To get a spectrum back from PCA coefficients:

```
Input:  z₁, z₂, ..., z₆₀  (standardized PCA coefficients)
                │
                ▼
Step 1: Unstandardize:  cₖ = zₖ × std(cₖ) + mean(cₖ)
                │
                ▼
Step 2: Linear combination:  F(λ) = μ(λ) + Σₖ cₖ × φₖ(λ)
                │
                ▼
Output: F(λ₁), F(λ₂), ..., F(λ₁₁₀₁)  (reconstructed spectrum in asinh space)
                │
                ▼
Step 3: Inverse asinh:  F_linear(λ) = 0.05 × sinh(F(λ))  (physical flux)
```

### Feature-Specific Reconstruction Error

We validate that PCA preserves key spectral features:

| Feature | Wavelength | Acceptable RMS |
|---------|-----------|----------------|
| Si II 6355 | 5900-6500 Å | < 0.01 |
| S II "W" | 5300-5700 Å | < 0.01 |
| Ca II H&K | 3600-4000 Å | < 0.02 |
| UV continuum | 2500-3500 Å | < 0.03 |

---

# 6. Neural Network Emulator — The Fast Surrogate

## The Goal

We want a function that maps 15 physical parameters to ~60 PCA coefficients:

```
f_NN: R¹⁵ → R⁶⁰
θ = (log_L, v_inner, ..., X_Fe_outer) ↦ (z₁, z₂, ..., z₆₀)
```

This replaces LUMINA's 5-second simulation with a 0.1-millisecond matrix multiplication.

## 6a. Architecture: Multi-Layer Perceptron (MLP)

An MLP is a sequence of **fully connected layers**, each performing:

    y = activation(W × x + b)

where:
- x is the input vector
- W is a weight matrix (learned)
- b is a bias vector (learned)
- activation is a nonlinear function

### Our Architecture

```
Input Layer          Hidden Layers                          Output Layer

   15 params    512      512      256      256      128     60 PCA coeffs

   ●──────────[■■■]───[■■■]───[■■■]───[■■■]───[■■■]──────●
   ●           │SiLU│   │SiLU│   │SiLU│   │SiLU│   │SiLU│  ●
   ●           │Drop│   │Drop│   │Drop│   │Drop│   │Drop│  ●
   ...         │0.01│   │0.01│   │0.01│   │0.01│   │0.01│  ...
   ●           └────┘   └────┘   └────┘   └────┘   └────┘  ●
   (15)                                                     (60)
```

Each hidden layer consists of:
1. **Linear transformation**: y = W × x + b
2. **SiLU activation**: σ(y) × y
3. **Dropout** (p=0.01): randomly zero out 1% of neurons during training

The output layer has **no activation** (linear), because PCA coefficients can be any
real number.

### Total Parameters

```
Layer 1:  15 × 512 + 512   =   8,192
Layer 2: 512 × 512 + 512   = 262,656
Layer 3: 512 × 256 + 256   = 131,328
Layer 4: 256 × 256 + 256   =  65,792
Layer 5: 256 × 128 + 128   =  32,896
Output:  128 × 60  + 60    =   7,740
                            ─────────
                Total:      ~508,604 parameters
```

## 6b. Activation Function: SiLU (Sigmoid Linear Unit)

The activation function introduces **nonlinearity**. Without it, stacking layers
would just be one big linear transformation (useless for complex functions).

### SiLU Definition

    SiLU(x) = x × σ(x) = x × 1/(1 + e⁻ˣ)

```
                SiLU(x) = x × sigmoid(x)

    y │
    3 │                              ╱
      │                            ╱
    2 │                          ╱
      │                       ╱╱
    1 │                    ╱╱
      │                ╱╱╱
    0 │──────────╱╱╱╱╱
      │        ╱
  -.2 │───╱───
      └───┬───┬───┬───┬───┬───── x
         -4  -2   0   2   4
```

### Why SiLU Over ReLU?

| Property | ReLU | SiLU |
|----------|------|------|
| Formula | max(0, x) | x × sigmoid(x) |
| Smooth? | No (kink at 0) | Yes (infinitely smooth) |
| Gradient at 0 | Undefined | 0.5 |
| Negative values | Always 0 | Slightly negative (then 0) |
| "Dying neurons" | Yes (permanently stuck at 0) | No |

SiLU is smooth and non-monotonic (it dips slightly below zero near x ≈ -1.28), which
gives the network more expressive power for modeling complex spectral dependencies.

## 6c. Training: AdamW + Cosine Annealing

### The Optimization Problem

We want to find weights W that minimize the loss function:

    W* = argmin_W  L(W)  =  argmin_W  (1/N) Σᵢ loss(f_NN(xᵢ; W), yᵢ)

where (xᵢ, yᵢ) are training pairs (parameters, PCA coefficients).

### AdamW Optimizer

AdamW (Adaptive Moment Estimation with decoupled Weight decay) maintains **per-parameter**
learning rates, adapting to the geometry of the loss landscape.

For each parameter w at step t:

```
1. Compute gradient:      gₜ = ∂L/∂w

2. Update first moment:   mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ          (momentum)
3. Update second moment:  vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²         (adaptive scaling)

4. Bias correction:       m̂ₜ = mₜ / (1-β₁ᵗ)
                          v̂ₜ = vₜ / (1-β₂ᵗ)

5. Parameter update:      wₜ = wₜ₋₁ - η × (m̂ₜ/(√v̂ₜ + ε) + λw × wₜ₋₁)
                                        ╰──── Adam step ────╯   ╰─ weight decay ─╯
```

**Hyperparameters:**
| Symbol | Value | Meaning |
|--------|-------|---------|
| η | 5×10⁻⁴ | Base learning rate |
| β₁ | 0.9 | Momentum decay |
| β₂ | 0.999 | Scaling decay |
| ε | 10⁻⁸ | Numerical stability |
| λw | 10⁻⁵ | Weight decay (L2 regularization) |

**Intuition:**
- **Momentum (m)**: "Keep going in the direction you've been going" → smoother convergence
- **Adaptive scaling (v)**: "Take smaller steps for parameters with large gradients" → stable
- **Weight decay**: "Penalize large weights" → prevents overfitting

### Cosine Annealing with Warm Restarts

The learning rate follows a cosine schedule that periodically resets:

```
Learning rate η(t):

  5e-4 │╲        ╱╲              ╱╲
       │  ╲      │  ╲            │  ╲
       │   ╲     │   ╲           │   ╲
       │    ╲    │    ╲          │    ╲
       │     ╲   │     ╲         │     ╲
       │      ╲  │      ╲        │      ╲
       │       ╲ │       ╲       │       ╲
       │        ╲│        ╲      │        ╲
  ~0   │         ╲─────────╲─────│─────────╲───
       └───┬─────┬──────────┬────┬──────────┬── Epoch
           0    100        300   500        900
           ◄─T₀─►◄──T₀×2──►    ◄──T₀×4───►
```

**Formula at epoch t within cycle:**

    η(t) = η_min + ½(η_max - η_min)(1 + cos(π × t_cycle / T_cycle))

**Configuration:**
- T₀ = 100 epochs (first cycle length)
- T_mult = 2 (each cycle is 2× longer)
- Cycles: [0-100], [100-300], [300-700], [700-1500], ...

**Why warm restarts?**
- Escapes local minima: when η resets to max, the optimizer can explore new regions
- Different learning rate phases: high η for coarse search, low η for fine-tuning
- Implicit ensemble effect: different minima found in different cycles

### Gradient Clipping

Before each optimizer step, we clip gradients to prevent exploding updates:

    if ‖∇L‖₂ > 1.0:  ∇L ← ∇L × (1.0 / ‖∇L‖₂)

This ensures no single batch can cause a catastrophically large weight change.

### Early Stopping

```
Track: best validation loss and patience counter

For each epoch:
    1. Compute val_loss
    2. If val_loss < best_loss - min_delta:
         best_loss = val_loss
         patience_counter = 0
         Save model weights (best checkpoint)
    3. Else:
         patience_counter += 1
    4. If patience_counter >= 300:
         STOP training
         Restore best model weights
```

This prevents overfitting: if the model hasn't improved on validation data for 300
epochs, we stop and use the best version.

### Dropout (p = 0.01)

During training, each neuron has a 1% chance of being "turned off" (output set to 0)
on each forward pass. This:

- Prevents **co-adaptation**: neurons can't rely on specific other neurons
- Acts like training an **ensemble** of slightly different networks
- Reduces overfitting

During inference, dropout is disabled and all neurons are active. The outputs are
implicitly scaled by (1-p) = 0.99 to compensate.

Note: We use very light dropout (1%) because our training set is large (50K) and
the mapping is relatively smooth.

## 6d. Feature-Weighted Composite Loss

### Why Not Just MSE?

Simple MSE (Mean Squared Error) on PCA coefficients treats all spectral features equally
in PCA space. But PCA space ≠ feature space:

- **PC1** might capture overall slope (unimportant for line identification)
- **PC47** might capture the Si II trough depth (critical for physics)

A small error in PC47 could destroy the Si II feature while barely changing the
overall MSE. We need a loss function that "knows" about spectral features.

### The Composite Loss

```
L_total = L_pca + 0.5 × L_features + 0.3 × L_UV_ratio
          ╰─┬──╯   ╰──────┬───────╯   ╰───────┬────────╯
       PCA-space MSE    Feature windows     UV/optical ratio
       (global shape)   (absorption lines)  (temperature/blanketing)
```

### Component 1: PCA Loss

    L_pca = (1/n_pca) Σₖ (ẑₖ - zₖ)²

This is standard MSE on the standardized PCA coefficients. It ensures global spectral
shape is correct.

### Component 2: Feature-Window Loss

This requires a **differentiable** path from PCA coefficients to spectrum:

```
PCA coeffs ──→ Unstandardize ──→ PCA inverse ──→ Spectrum (1,101 bins)
    zₖ           cₖ=zₖσ+μ      F=Σcₖφₖ+μ_spec     F(λᵢ)
```

All three steps are linear, so the composition is differentiable (a matrix multiply).

For each feature window w (e.g., Si II 6355: 5900-6500 Å):

    L_w = (1/n_w) Σ_{λ∈w} (F̂(λ) - F(λ))²

The total feature loss averages over all 7 windows:

    L_features = (1/7) Σ_{w=1}^{7} L_w

### Component 3: UV/Optical Ratio Loss

Measures the UV-to-optical flux ratio in **linear** (inverse-asinh) space:

    r = mean{0.05 × sinh(F(λ)) : λ ∈ UV} / mean{0.05 × sinh(F(λ)) : λ ∈ optical}

    L_UV_ratio = (r̂ - r)²

This ratio is sensitive to iron-group line blanketing (Fe, Ni, Co absorb UV photons)
and temperature (hotter → more UV). It can't be captured by any single PCA component.

### Loss Weights: Why 1.0, 0.5, 0.3?

These were chosen so that each component contributes roughly equally to the total loss
at the start of training:

| Component | Typical Value | Weight | Contribution |
|-----------|--------------|--------|--------------|
| L_pca | ~0.1 | 1.0 | 0.10 |
| L_features | ~0.2 | 0.5 | 0.10 |
| L_UV_ratio | ~0.3 | 0.3 | 0.09 |

If one component dominates, the network would ignore the others.

---

# 7. Bayesian Inference — The Inverse Problem

## 7a. Bayes' Theorem

We've trained a neural network that quickly predicts spectra. Now the question is:

> **Given an observed spectrum D, what are the physical parameters θ?**

This is the **inverse problem**, and Bayes' theorem provides the answer:

```
                   P(D|θ) × P(θ)
    P(θ|D)  =  ────────────────────
                      P(D)

    ╰──┬──╯     ╰──┬──╯   ╰─┬─╯    ╰─┬─╯
   Posterior   Likelihood  Prior    Evidence
   (answer)    (forward)   (belief) (normalization)
```

### Posterior P(θ|D)

This is what we want: the probability distribution of parameters given the data.
It tells us the best-fit values AND their uncertainties AND correlations.

### Likelihood P(D|θ)

"How probable is the observed data if the true parameters are θ?"

We compute a predicted spectrum from θ using the emulator, then compare to the
observed spectrum. The closer the match, the higher the likelihood.

### Prior P(θ)

"What did we believe about θ before seeing the data?"

We use **uniform priors** within physically allowed ranges. This means any valid
parameter combination is equally probable a priori. Invalid combinations (e.g.,
v_core > v_wall) have probability zero.

### Evidence P(D)

"How probable is the data under ALL possible parameter values?"

    P(D) = ∫ P(D|θ) × P(θ) dθ

This is a normalization constant. MCMC doesn't need it (works with ratios).
Nested sampling computes it explicitly (useful for model comparison).

## 7b. The Likelihood Function

### Basic Gaussian Likelihood

If the difference between model and data is due to Gaussian noise with variance σ²:

    log P(D|θ) = -½ Σᵢ (Dᵢ - Mᵢ(θ))² / σᵢ²  -  ½ Σᵢ log(2π σᵢ²)

where Dᵢ is the observed flux at wavelength λᵢ and Mᵢ(θ) is the emulator prediction.

### Our Feature-Aware Likelihood

We extend this with three components:

#### 1. Wavelength-Dependent σ(λ)

Different spectral regions have different noise levels:

```
σ(λ) = √[σ_obs(λ)² + σ_emu(λ)²]

σ_obs(λ):  observational noise (measurement error)
σ_emu(λ):  emulator error (imperfect neural network)
```

| Region | σ_obs | σ_emu | Reason |
|--------|-------|-------|--------|
| UV (< 3500 Å) | 0.06 | 0.03 | Low flux → high noise |
| Optical (general) | 0.03 | 0.02 | Standard |
| Feature windows | 0.021 | 0.014 | Boost weight (×0.7) |
| NIR (> 7500 Å) | 0.045 | 0.024 | Moderate noise |

The feature windows (Si II, S II, Ca, etc.) get **smaller σ**, which means **higher
weight** in the likelihood. This forces the fit to prioritize absorption features
over smooth continuum.

#### 2. Si II Velocity Penalty

If we measure the Si II 6355 trough position in both observed and model spectra:

    log L_vel = -½ (v_obs - v_model)² / σ_vel²

with σ_vel = 500 km/s. This adds a "soft constraint" that the model must reproduce
the correct blueshift velocity.

#### 3. UV/Optical Ratio Penalty

    log L_ratio = -½ (r_obs - r_model)² / σ_ratio²

with σ_ratio = 0.05. This ensures the UV-to-optical color is correct.

#### Total Log-Likelihood

```
log L(θ) = -½ Σ (obs - model)²/σ(λ)²     ← spectral match
           -½ (Δv_SiII / 500)²            ← velocity match
           -½ (Δr_UV/opt / 0.05)²         ← color match
```

## 7c. MCMC with emcee — Markov Chain Monte Carlo

### What is MCMC?

MCMC is an algorithm that **draws samples from a probability distribution** without
knowing its normalization constant. After enough samples, the histogram of sampled
points approximates the posterior distribution.

### The Key Idea

Imagine you're blindfolded on a mountain and want to map its shape. MCMC says:

1. Start somewhere
2. Propose a random step
3. If the new position is **higher** (more probable): always accept
4. If the new position is **lower** (less probable): accept with probability = height_new/height_old
5. Repeat millions of times
6. The density of your footsteps maps the mountain

### Formal Algorithm: Metropolis-Hastings

```
Initialize: θ₀ (starting position)
For t = 1, 2, ..., T:
    1. PROPOSE: θ* ~ q(θ*|θₜ₋₁)          (random perturbation)
    2. COMPUTE: α = P(θ*|D) / P(θₜ₋₁|D)  (acceptance ratio)
             = [P(D|θ*) × P(θ*)] / [P(D|θₜ₋₁) × P(θₜ₋₁)]
    3. ACCEPT: u ~ Uniform(0,1)
       if u < α: θₜ = θ*                  (move to new position)
       else:     θₜ = θₜ₋₁               (stay in place)
```

Note: P(D) cancels in the ratio α, so we never need to compute it!

### emcee: The Affine-Invariant Ensemble Sampler

Standard MCMC uses a single "walker" (random walk). **emcee** uses an **ensemble** of
walkers that communicate with each other.

```
Standard MCMC (1 walker):       emcee (64 walkers):

  · → · → · → ·                  ·₁ →  ·₁ →  ·₁
  slow random walk                ·₂ →  ·₂ →  ·₂
  in 15D space                    ·₃ →  ·₃ →  ·₃
  (very slow in high-D!)          ...   ...   ...
                                  ·₆₄→  ·₆₄→  ·₆₄
                                  Walkers stretch toward each other
```

#### The Stretch Move

For each walker j:

1. Pick a random partner walker k (from a complementary set)
2. Propose: θ* = θₖ + Z × (θⱼ - θₖ)
   where Z ~ 1/√g on [1/a, a] with a=2 (the "stretch factor")
3. Accept with probability: min(1, Z^(d-1) × L(θ*)/L(θⱼ))

```
Walker j and partner k:

  Before:    k──────────────j

  Z > 1:    k──────────────j────→ j*    (stretch away from k)
  Z < 1:    k────j*────────j            (compress toward k)
```

**Why this is brilliant:**
- **Affine invariant**: performance doesn't depend on parameter correlations
  (standard MCMC slows down dramatically with correlated parameters)
- **Parallel**: all walkers can be updated simultaneously on multi-core CPUs
- **Self-tuning**: the ensemble shape adapts to the posterior shape

### Our MCMC Setup

```
Step 1: MAP Estimation (find the peak)
        ├── Method: scipy.optimize.differential_evolution
        ├── Iterations: 200
        └── Result: θ_MAP (best-fit parameters)

Step 2: Initialize Walkers
        ├── 64 walkers
        ├── Positions: θ_MAP + N(0, 0.01 × range) per dimension
        └── Reject any walkers with log P = -∞ (invalid physics)

Step 3: Burn-in (2,000 steps)
        ├── Walkers explore from initial positions
        ├── Discarded (not part of posterior)
        └── Purpose: forget initial conditions

Step 4: Production (5,000 steps)
        ├── 64 walkers × 5,000 steps = 320,000 samples
        ├── These samples approximate P(θ|D)
        └── Saved for analysis
```

### Burn-in: Why Discard the First 2,000 Steps?

```
                                  Start of production
                                  ↓
  log P │                    ┌─── samples oscillate around
        │                    │    the typical set
        │               ╱────┘
        │          ╱────╱
        │     ╱───╱
        │╱───╱
        │     Burn-in: walkers
        │     are still finding
        │     the high-probability
        │     region
        └───────────────────── Step
            0    2000     7000
```

The walkers start near the MAP estimate but need time to spread out and fill the
posterior distribution. The burn-in period allows this "mixing" to happen before
we start recording samples.

### Convergence Diagnostics

#### Acceptance Fraction

    f_accept = (number of accepted proposals) / (total proposals)

- Too low (< 10%): steps are too large, most proposals are rejected
- Too high (> 90%): steps are too small, chain is barely moving
- Ideal: 20-50% for ensemble samplers

#### Autocorrelation Time τ

The number of steps between effectively independent samples:

    τ = 1 + 2 Σₖ₌₁^∞ ρ(k)

where ρ(k) is the autocorrelation at lag k.

Rule of thumb: need N_steps > 50 × τ for reliable posterior estimates.
If τ ≈ 50, we need 2,500+ production steps (we use 5,000).

### Reading the Results: Corner Plots

A corner plot shows all pairwise correlations between parameters:

```
          log_L     v_inner    log_ρ₀
         ┌─────┐   ┌─────┐   ┌─────┐
  log_L  │╱╲   │   │     │   │     │
         │  ╲  │   │     │   │     │
         └─────┘   │     │   │     │
         ┌─────┐   ┌─────┐   │     │
 v_inner │  ·  │   │╱╲   │   │     │
         │ ·:· │   │  ╲  │   │     │
         └─────┘   └─────┘   │     │
         ┌─────┐   ┌─────┐   ┌─────┐
 log_ρ₀  │  ·  │   │  ·  │   │╱╲   │
         │  ·  │   │ ·   │   │  ╲  │
         └─────┘   └─────┘   └─────┘

  Diagonal: 1D marginalized posteriors (histograms)
  Off-diagonal: 2D joint posteriors (scatter/contour)
```

- **Tight contours**: well-constrained parameter
- **Elongated contours**: degeneracy (parameters are correlated)
- **Multimodal**: multiple solutions exist

## 7d. Nested Sampling with dynesty

### What is Nested Sampling?

Nested sampling is an alternative to MCMC that was designed to compute the **evidence**
(marginal likelihood) P(D) as its primary output, with posterior samples as a byproduct.

### The Key Idea

Instead of exploring the posterior directly, nested sampling **shrinks** the prior
volume systematically, always keeping the highest-likelihood points:

```
Step 0:                        Step 50:
(500 live points in prior)     (points concentrated near peak)

  θ₂│· · · · · · · · · ·      θ₂│         · ·
    │ · · · · · · · · ·          │       · ·:· ·
    │· · · · · · · · · ·         │      · ·::·· ·
    │ · · · · · · · · ·          │       · ·:· ·
    │· · · · · · · · · ·         │         · ·
    └──────────────── θ₁         └──────────────── θ₁
    Uniform: all of prior        Concentrated: high L region
```

### The Algorithm

```
1. Draw n_live = 500 points uniformly from the prior
2. Evaluate likelihood L(θ) for all live points
3. Repeat:
   a. Find the point with LOWEST likelihood: L_worst
   b. Record it as a "dead point" with weight w
   c. Remove it from the live set
   d. Draw a NEW point with L(θ) > L_worst from the prior
      (this is the hard part — use random walks constrained to L > L_worst)
   e. The prior volume shrinks by factor ≈ n/(n+1) each iteration
4. Stop when the remaining evidence contribution is < dlogz (= 0.1)
```

### Computing the Evidence

At each step i, the "dead point" has likelihood Lᵢ and occupies a volume ΔVᵢ.
The evidence is:

    Z = P(D) = Σᵢ Lᵢ × ΔVᵢ

The volume shrinks geometrically: Vᵢ ≈ V₀ × (n/(n+1))^i

```
   L(θ)
     │          ╱╲          The integral under this curve
     │         ╱  ╲         is the evidence Z
     │        ╱    ╲
     │       ╱      ╲
     │      ╱        ╲
     │ ────╱──────────╲────
     └──────────────────── X (enclosed prior volume)
        1                0
        ◄── shrinking ────
```

### Why Nested Sampling?

| Advantage | Explanation |
|-----------|-------------|
| Evidence Z | Enables model comparison (which model fits better?) |
| No burn-in | All points contribute from the start |
| Handles multimodality | Naturally explores multiple peaks |
| Guaranteed convergence | Stopping criterion is well-defined (dlogz) |

| Disadvantage | Explanation |
|--------------|-------------|
| Slower | More likelihood evaluations than MCMC for same posterior quality |
| Constrained drawing | Generating new points with L > L_min is hard in high-D |
| Less mature | Fewer diagnostics than MCMC |

### Posterior Samples from Nested Sampling

The dead points are NOT posterior samples — they have non-uniform weights. To get
equal-weight posterior samples, we **resample**:

    weight_i = L_i × ΔV_i / Z

Then draw from {θᵢ} with probability proportional to weight_i. This is done
automatically by `dynesty.utils.resample_equal()`.

### Model Comparison via Bayes Factor

If we have two models M₁ and M₂ (e.g., 11D vs 15D parameter space):

    B₁₂ = Z₁ / Z₂ = P(D|M₁) / P(D|M₂)

| log B₁₂ | Interpretation |
|----------|---------------|
| < 1 | Not significant |
| 1-2.5 | Substantial |
| 2.5-5 | Strong |
| > 5 | Decisive |

## 7e. Simulation-Based Inference (SBI / SNPE)

### The Revolutionary Idea

MCMC and nested sampling both need to **evaluate the likelihood** at each step.
Our likelihood requires running the emulator and comparing spectra — simple, but
still involves design choices (sigma arrays, feature weights).

SBI takes a completely different approach:

> **Learn the posterior directly from simulated data, without ever writing down a
> likelihood function.**

### How SNPE (Sequential Neural Posterior Estimation) Works

```
TRAINING:

  θ₁ → Simulator → x₁  ╮
  θ₂ → Simulator → x₂  │  Training pairs {(θᵢ, xᵢ)}
  θ₃ → Simulator → x₃  │
  ...                   │
  θₙ → Simulator → xₙ  ╯
                         │
                         ▼
              Neural Density Estimator
              "Learn P(θ|x) directly"
                         │
INFERENCE:               ▼

  x_obs ──→ Trained Network ──→ P(θ|x_obs) ──→ Sample
                                                  │
                                                  ▼
                                          Posterior samples
```

### What is a Neural Density Estimator?

The neural density estimator is a neural network that outputs a **probability
distribution** (not a single prediction). Given an observed spectrum x, it produces
the conditional density P(θ|x).

Common architectures used by the `sbi` package:

#### Normalizing Flows

A normalizing flow transforms a simple distribution (e.g., Gaussian) into a complex
one through a sequence of invertible transformations:

```
  z ~ N(0, I)    Standard normal (simple)
       │
       ▼
  z₁ = f₁(z)    First transformation (learned)
       │
       ▼
  z₂ = f₂(z₁)   Second transformation (learned)
       │
       ▼
  ...
       │
       ▼
  θ = fₖ(zₖ₋₁)  Final output: complex posterior shape
```

Each transformation fᵢ must be:
1. **Invertible** (so we can compute probability densities)
2. **Differentiable** (so we can train via backpropagation)
3. **Efficient** (fast Jacobian computation)

The density of θ is computed via the change-of-variables formula:

    log P(θ) = log P(z) - Σᵢ log |det(∂fᵢ/∂zᵢ₋₁)|

### Why SBI is Different

| Property | MCMC / Nested | SBI |
|----------|---------------|-----|
| Needs likelihood function | Yes | **No** |
| Needs simulator | At inference time | Only at training time |
| Amortized | No (re-run for each obs) | **Yes** (train once, apply to many) |
| Design choices | sigma, weights, features | Implicit (learned from data) |
| Inference speed | Minutes | **Seconds** (forward pass) |
| Training cost | None | Hours (one-time) |

### The SBI Pipeline in Our Code

```python
# Step 1: Prepare training data
θ_train  = physical parameters  [N × 15]
x_train  = asinh spectra        [N × 1101]

# Step 2: Define prior (same as MCMC)
prior = BoxUniform(low=[42.8, 8000, ...], high=[43.15, 13000, ...])

# Step 3: Train neural density estimator
inference = SNPE(prior)
inference.append_simulations(θ_train, x_train)
density_estimator = inference.train()

# Step 4: Build posterior
posterior = inference.build_posterior(density_estimator)

# Step 5: Sample from posterior given observation
samples = posterior.sample((10000,), x=x_observed)
```

### Strengths and Weaknesses

**Strengths:**
- **No likelihood design**: avoids choosing sigma arrays, feature weights, etc.
- **Amortized**: once trained, instant inference for any observation
- **Flexible**: captures arbitrary posterior shapes (multimodal, skewed, etc.)
- **Scales well**: works for very complex simulators where likelihood is intractable

**Weaknesses:**
- **Training data limited**: posterior quality depends on training set coverage
- **No convergence guarantee**: hard to know if the neural network learned the true posterior
- **Black box**: less interpretable than explicit likelihood
- **Needs many simulations**: typically requires 10,000+ training pairs

### When to Use Which Method?

| Scenario | Best Method |
|----------|-------------|
| Single observation, well-understood noise | MCMC |
| Model comparison (which model is better?) | Nested Sampling |
| Many observations, same parameter space | SBI (amortized) |
| Intractable likelihood | SBI |
| Gold standard, publication quality | MCMC + Nested (cross-check) |

---

# 8. Method Comparison: MCMC vs Nested vs SBI

## Summary Table

| Property | MCMC (emcee) | Nested (dynesty) | SBI (SNPE) |
|----------|-------------|-------------------|------------|
| **Core algorithm** | Ensemble random walk | Shrinking prior volume | Neural density estimator |
| **Primary output** | Posterior samples | Evidence Z + posterior | Posterior samples |
| **Likelihood needed?** | Yes | Yes | No |
| **Convergence diagnostic** | Autocorrelation time | dlogz threshold | Training loss |
| **Handles multimodality** | Poorly (single mode) | Well | Well |
| **Evidence computation** | No | Yes (primary output) | No |
| **Amortized** | No | No | Yes |
| **Our config** | 64 walkers, 7000 steps | 500 live points | SNPE default |
| **Typical runtime** | ~1-2 min | ~5-10 min | Train: hours, Infer: seconds |
| **Samples produced** | 320,000 | ~10,000 (resampled) | 10,000 (configurable) |

## When Results Disagree

If the three methods give different posteriors, this is **informative**:

- **MCMC ≈ Nested ≠ SBI**: SBI may have insufficient training data or coverage
- **MCMC ≠ Nested**: MCMC may be stuck in a local mode (check chains)
- **All agree**: Strong evidence that the posterior is robust

## Visual Comparison Example

```
        log_L posterior from three methods:

  P(log_L) │
           │      ╱╲         MCMC (solid)
           │     ╱  ╲        Nested (dashed)
           │    ╱:╱╲:╲       SBI (dotted)
           │   ╱:╱  ╲:╲
           │  ╱:╱    ╲:╲
           │╱:╱........╲:╲
           └────────────────── log_L
              42.9  43.0  43.1

  All three peak near 43.0: robust result
  Width indicates uncertainty
```

---

# 9. Glossary

| Term | Definition |
|------|-----------|
| **Affine invariant** | An algorithm whose performance doesn't depend on linear transformations of parameters |
| **Amortized inference** | Training once, then applying to many observations (SBI) |
| **asinh** | Inverse hyperbolic sine: asinh(x) = ln(x + √(x²+1)) |
| **Autocorrelation time** | Number of MCMC steps between independent samples |
| **Bayes factor** | Ratio of evidence between two models: B₁₂ = Z₁/Z₂ |
| **Burn-in** | Initial MCMC steps discarded before the chain has converged |
| **Corner plot** | Grid of 1D (diagonal) and 2D (off-diagonal) posterior marginals |
| **Credible interval** | Bayesian analog of confidence interval (68% = 1σ equivalent) |
| **Dead point** | In nested sampling: the removed lowest-likelihood point at each step |
| **Degeneracy** | When multiple parameter combinations produce similar spectra |
| **dlogz** | Nested sampling stopping criterion (remaining evidence contribution) |
| **Dropout** | Randomly disabling neurons during training for regularization |
| **Early stopping** | Halting training when validation loss stops improving |
| **Eigenvalue** | Amount of variance captured by a PCA component |
| **Eigenvector** | Direction of maximum variance in PCA (basis spectrum) |
| **emcee** | Python ensemble MCMC sampler (Foreman-Mackey et al. 2013) |
| **Evidence** | P(D) = ∫ P(D\|θ)P(θ)dθ, the probability of data under a model |
| **Feature window** | Wavelength range containing a specific spectral absorption line |
| **Gradient clipping** | Limiting gradient magnitude to prevent training instability |
| **LHS** | Latin Hypercube Sampling: space-filling experimental design |
| **Likelihood** | P(D\|θ), probability of observing data given parameters |
| **Live point** | In nested sampling: a point in the current active set |
| **MAP** | Maximum A Posteriori: the single most probable parameter value |
| **Marginal posterior** | Posterior for one parameter, integrated over all others |
| **MLP** | Multi-Layer Perceptron: a fully connected neural network |
| **Normalizing flow** | Neural network that transforms simple → complex distributions |
| **PCA** | Principal Component Analysis: linear dimensionality reduction |
| **Posterior** | P(θ\|D), the probability of parameters given observed data |
| **Prior** | P(θ), our pre-data belief about parameter values |
| **Savitzky-Golay** | Local polynomial smoothing filter |
| **SBI** | Simulation-Based Inference: learning posteriors directly from simulations |
| **SiLU** | Sigmoid Linear Unit: x × sigmoid(x), smooth activation function |
| **SNPE** | Sequential Neural Posterior Estimation (SBI method) |
| **Stretch move** | emcee's proposal mechanism using walker pairs |
| **Walker** | One Markov chain in an ensemble MCMC sampler |
| **Weight decay** | L2 regularization: penalizing large neural network weights |

---

# 10. References

### MCMC & Bayesian Inference
1. **Foreman-Mackey, D. et al. (2013)** — "emcee: The MCMC Hammer"
   *PASP 125, 306.* The affine-invariant ensemble sampler we use.

2. **Goodman, J. & Weare, J. (2010)** — "Ensemble Samplers with Affine Invariance"
   *Comm. App. Math. Comp. Sci. 5, 65.* The mathematical foundation of emcee.

### Nested Sampling
3. **Skilling, J. (2004)** — "Nested Sampling"
   *AIP Conf. Proc. 735, 395.* Original nested sampling algorithm.

4. **Speagle, J. (2020)** — "dynesty: a dynamic nested sampling package"
   *MNRAS 493, 3132.* The nested sampling code we use.

### Simulation-Based Inference
5. **Cranmer, K., Brehmer, J., Louppe, G. (2020)** — "The frontier of simulation-based inference"
   *PNAS 117, 30055.* Review of SBI methods including SNPE.

6. **Tejero-Cantero, A. et al. (2020)** — "sbi: A toolkit for simulation-based inference"
   *JOSS 5, 2505.* The SBI Python package we use.

### PCA & Neural Network Emulators
7. **Jolliffe, I. T. (2002)** — "Principal Component Analysis" (2nd ed.)
   *Springer.* The definitive PCA reference.

8. **Alsing, J. et al. (2020)** — "SPECULATOR: Emulating stellar population synthesis"
   *ApJS 249, 5.* Neural network spectral emulator (similar approach).

### Supernova Physics
9. **Kerzendorf, W. & Sim, S. (2014)** — "A spectral synthesis code for rapid modelling of supernovae"
   *MNRAS 440, 387.* TARDIS, the reference code for SN Ia spectral synthesis.

10. **Pereira, R. et al. (2013)** — "Spectrophotometric time series of SN 2011fe"
    *A&A 554, A27.* The SN 2011fe observations we fit.

### Astronomical Applications of asinh
11. **Lupton, R. et al. (1999)** — "A Modified Magnitude System that Produces Well-Behaved Magnitudes"
    *AJ 118, 1406.* Introduction of asinh magnitudes ("luptitudes") for SDSS.

---

*This guide was written for the LUMINA-ML project.*
*Last updated: February 2026*
