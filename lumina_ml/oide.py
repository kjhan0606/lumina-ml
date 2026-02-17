"""OIDE: Overlapping Iterative Dimensional Expansion.

Multi-phase SBI approach where each phase explores ~8-10D with overlapping
"carry-over" dimensions from previous phases, determined by Constraint Ratio (CR).

CR_i = (q84 - q16) / (prior_hi - prior_lo)
  < 0.10: locked (well-constrained, fix at median)
  0.10-0.25: constrained (narrow prior in carry-over)
  0.25-0.50: weak (moderate prior in carry-over)
  > 0.50: unconstrained (full prior in carry-over)

Ti/Cr (12 params) are fixed throughout all phases at Stage 2.5 posterior medians.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfg

# Ti/Cr indices in the 63D parameter space (fixed throughout all phases)
# Zone 0: Ti=21, Cr=22; Zone 1: Ti=29, Cr=30; ... Zone 5: Ti=61, Cr=62
TICR_INDICES = [21, 22, 29, 30, 37, 38, 45, 46, 53, 54, 61, 62]

# CR classification thresholds
CR_LOCKED = 0.10
CR_CONSTRAINED = 0.25
CR_WEAK = 0.50

MAX_CARRY = 3


@dataclass
class PhaseConfig:
    """Configuration for a single OIDE phase."""
    phase_id: int
    new_param_indices: List[int]
    carry_indices: List[int] = field(default_factory=list)
    n_models: int = 12000
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 128])
    description: str = ""

    @property
    def free_indices(self):
        """All free dimensions: new + carry-over, sorted."""
        return sorted(set(self.new_param_indices + self.carry_indices))

    @property
    def n_free(self):
        return len(self.free_indices)


def get_default_phases():
    """Return the 9 OIDE phase definitions (carry-over filled dynamically)."""
    return [
        PhaseConfig(
            phase_id=1,
            new_param_indices=[0, 1, 2, 3, 4, 9, 10, 11],
            n_models=20000,
            description="Core physical: luminosity, density, timing",
        ),
        PhaseConfig(
            phase_id=2,
            new_param_indices=[5, 6, 7, 8, 12, 13, 14],
            n_models=15000,
            description="Zone boundaries + global abundances",
        ),
        PhaseConfig(
            phase_id=3,
            new_param_indices=[15, 16, 17, 18, 19, 20],
            n_models=12000,
            description="Zone 0 composition (innermost)",
        ),
        PhaseConfig(
            phase_id=4,
            new_param_indices=[23, 24, 25, 26, 27, 28],
            n_models=12000,
            description="Zone 1 composition",
        ),
        PhaseConfig(
            phase_id=5,
            new_param_indices=[31, 32, 33, 34, 35, 36],
            n_models=12000,
            description="Zone 2 composition",
        ),
        PhaseConfig(
            phase_id=6,
            new_param_indices=[39, 40, 41, 42, 43, 44],
            n_models=12000,
            description="Zone 3 composition",
        ),
        PhaseConfig(
            phase_id=7,
            new_param_indices=[47, 48, 49, 50, 51, 52],
            n_models=12000,
            description="Zone 4 composition",
        ),
        PhaseConfig(
            phase_id=8,
            new_param_indices=[55, 56, 57, 58, 59, 60],
            n_models=12000,
            description="Zone 5 composition (outermost)",
        ),
        PhaseConfig(
            phase_id=9,
            new_param_indices=[],  # Filled dynamically from worst CR params
            n_models=10000,
            description="Final refinement: top-10 weakest params",
        ),
    ]


def load_base_values(filepath):
    """Load base (fixed) values and initial CR from oide_base_values.json.

    Returns:
        base_values: (63,) array of median values
        initial_cr: dict {param_index: CR_value}
    """
    with open(filepath) as f:
        data = json.load(f)

    base = np.zeros(cfg.STAGE2_N_PARAMS)
    cr = {}
    for i, name in enumerate(cfg.STAGE2_PARAM_NAMES):
        if name in data:
            base[i] = data[name]['median']
            cr[i] = data[name]['CR']
        else:
            lo, hi = cfg.STAGE2_PARAM_RANGES[i]
            base[i] = (lo + hi) / 2.0
            cr[i] = 1.0
    return base, cr


def compute_cr(samples, free_indices, full_ranges=None):
    """Compute Constraint Ratio for each free parameter.

    CR_i = (q84 - q16) / (prior_hi - prior_lo)

    Args:
        samples: (N, n_free) posterior samples in free-dimension order
        free_indices: list of indices into the 63D space
        full_ranges: full 63D prior ranges (default: STAGE2_PARAM_RANGES)

    Returns:
        dict {param_index: CR_value}
    """
    if full_ranges is None:
        full_ranges = cfg.STAGE2_PARAM_RANGES

    cr = {}
    for j, idx in enumerate(free_indices):
        q16 = np.percentile(samples[:, j], 16)
        q84 = np.percentile(samples[:, j], 84)
        lo, hi = full_ranges[idx]
        cr[idx] = (q84 - q16) / (hi - lo) if (hi - lo) > 0 else 1.0
    return cr


def classify_cr(cr_value):
    """Classify a CR value into a category string."""
    if cr_value < CR_LOCKED:
        return 'locked'
    elif cr_value < CR_CONSTRAINED:
        return 'constrained'
    elif cr_value < CR_WEAK:
        return 'weak'
    else:
        return 'unconstrained'


def select_carry_over(cr_dict, max_carry=MAX_CARRY):
    """Select carry-over indices: params with CR > CR_CONSTRAINED.

    Returns up to max_carry indices, sorted by CR descending (worst first).
    Ti/Cr indices are excluded.
    """
    candidates = [(idx, cr) for idx, cr in cr_dict.items()
                  if cr > CR_CONSTRAINED and idx not in TICR_INDICES]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates[:max_carry]]


def build_phase_ranges(free_indices, base_values, cr_values, carry_indices):
    """Build prior ranges for free dimensions of a phase.

    - New params: full Stage 2 prior range
    - Carry-over params with CR < 0.25: narrow (±50% of estimated posterior width)
    - Carry-over params with CR 0.25-0.50: moderate (±100% of estimated posterior width)
    - Carry-over params with CR > 0.50: full prior range

    Returns:
        list of (lo, hi) tuples, one per free_index (same ordering as free_indices)
    """
    ranges = []
    for idx in free_indices:
        full_lo, full_hi = cfg.STAGE2_PARAM_RANGES[idx]

        if idx in carry_indices:
            cr = cr_values.get(idx, 1.0)
            if cr > CR_WEAK:
                # Unconstrained: full prior range
                ranges.append((full_lo, full_hi))
                continue

            # Estimated posterior width ~ CR * prior_width
            width = cr * (full_hi - full_lo)
            mid = base_values[idx]

            if cr < CR_CONSTRAINED:
                delta = width * 0.5
            else:
                delta = width * 1.0

            lo = max(full_lo, mid - delta)
            hi = min(full_hi, mid + delta)
            if hi <= lo:
                lo, hi = full_lo, full_hi
            ranges.append((lo, hi))
        else:
            # New param: full prior range
            ranges.append((full_lo, full_hi))

    return ranges


def generate_phase_samples(n_models, free_indices, phase_ranges, base_values, rng):
    """Generate full 63D parameter arrays where only free dims vary.

    Uses Latin Hypercube in free dimensions, fixes all others at base values.
    Applies zone abundance constraints and validity checking.

    Returns:
        (N_valid, 63) array of valid parameter sets (N_valid <= n_models)
    """
    from .data_utils import Stage2Params, _constrain_zone_abundances

    n_free = len(free_indices)

    # LHS in free dimensions (stratified random sampling)
    lhs = np.zeros((n_models, n_free))
    for d in range(n_free):
        perm = rng.permutation(n_models)
        for i in range(n_models):
            lo_frac = perm[i] / n_models
            hi_frac = (perm[i] + 1) / n_models
            lhs[i, d] = lo_frac + rng.random() * (hi_frac - lo_frac)

    # Scale to phase ranges
    for j, (lo, hi) in enumerate(phase_ranges):
        lhs[:, j] = lo + lhs[:, j] * (hi - lo)

    # Build full 63D arrays: tile base values, overwrite free dims
    full_params = np.tile(base_values, (n_models, 1))
    for j, idx in enumerate(free_indices):
        full_params[:, idx] = lhs[:, j]

    # Apply zone abundance constraints (O filler >= 0.03)
    for i in range(n_models):
        full_params[i] = _constrain_zone_abundances(full_params[i])

    # Validate and collect valid samples
    valid = []
    for i in range(n_models):
        p = Stage2Params.from_array(full_params[i])
        if p.is_valid():
            valid.append(full_params[i])

    n_rejected = n_models - len(valid)
    if n_rejected > 0:
        print(f"  LHS: {n_rejected}/{n_models} rejected, filling with random samples...")

    # Fill rejected slots with random valid samples
    max_attempts = n_models * 50
    attempts = 0
    while len(valid) < n_models and attempts < max_attempts:
        vals = base_values.copy()
        for j, idx in enumerate(free_indices):
            lo, hi = phase_ranges[j]
            vals[idx] = rng.uniform(lo, hi)
        vals = _constrain_zone_abundances(vals)
        p = Stage2Params.from_array(vals)
        if p.is_valid():
            valid.append(vals)
        attempts += 1

    if len(valid) < n_models:
        print(f"  WARNING: Only generated {len(valid)}/{n_models} valid samples "
              f"after {max_attempts} attempts")

    return np.array(valid[:n_models])


def select_phase9_params(all_cr, n_select=10):
    """Select top-N worst CR params across all phases for Phase 9.

    Excludes Ti/Cr (fixed throughout).
    """
    candidates = [(idx, cr) for idx, cr in all_cr.items()
                  if idx not in TICR_INDICES]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates[:n_select]]


def update_base_values(base_values, cr_dict, posterior_medians, free_indices):
    """Update base values for well-constrained params (CR < CR_CONSTRAINED).

    Modifies base_values in-place.
    """
    for j, idx in enumerate(free_indices):
        if cr_dict.get(idx, 1.0) < CR_CONSTRAINED:
            base_values[idx] = posterior_medians[j]
    return base_values
