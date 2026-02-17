"""LUMINA wrapper: ModelParams, Stage2Params, model directory creation, execution, LHS sampling."""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as cfg


@dataclass
class ModelParams:
    """15D physical parameters for a LUMINA SN Ia model."""
    log_L: float             # log10(luminosity in erg/s)
    v_inner: float           # inner velocity (km/s)
    log_rho_0: float         # log10(reference density in g/cm^3)
    density_exp: float       # inner density power-law exponent
    T_e_ratio: float         # T_e / T_rad ratio
    v_core: float            # core/wall boundary velocity (km/s)
    v_wall: float            # wall/outer boundary velocity (km/s)
    X_Fe_core: float         # Fe mass fraction in core zone
    X_Si_wall: float         # Si mass fraction in wall zone
    v_break: float           # density slope break velocity (km/s)
    density_exp_outer: float # outer density exponent (v > v_break)
    t_exp: float             # time since explosion (days)
    X_Fe_wall: float         # Fe mass fraction in wall zone
    X_Ni: float              # Ni mass fraction (all zones, UV blanketer)
    X_Fe_outer: float        # Fe mass fraction in outer zone

    @property
    def L_erg_s(self):
        return 10**self.log_L

    @property
    def rho_0(self):
        return 10**self.log_rho_0

    @property
    def t_exp_s(self):
        """Time since explosion in seconds."""
        return self.t_exp * 86400.0

    @property
    def v_inner_cm_s(self):
        return self.v_inner * 1e5

    @property
    def T_inner_estimate(self):
        """Stefan-Boltzmann estimate for initial T_inner."""
        R_inner = self.v_inner_cm_s * self.t_exp_s
        return (self.L_erg_s / (4 * np.pi * cfg.SIGMA_SB * R_inner**2))**0.25

    def density_at_v(self, v_mid):
        """Compute broken power-law density at velocity v_mid (km/s).

        Density is scaled by (t_ref/t_exp)^3 for time evolution:
        rho(v, t) = rho_0(v) * (t_ref / t_exp)^3
        """
        rho_0 = self.rho_0
        v_min = self.v_inner
        # Time scaling: density dilutes as t^-3 in homologous expansion
        t_scale = (cfg.T_REF / self.t_exp) ** 3

        if v_mid < self.v_break:
            rho = rho_0 * (v_mid / v_min) ** self.density_exp
        else:
            rho_break = rho_0 * (self.v_break / v_min) ** self.density_exp
            rho = rho_break * (v_mid / self.v_break) ** self.density_exp_outer

        return rho * t_scale

    @property
    def ni56_decay_fractions(self):
        """Ni56 → Co56 → Fe56 decay fractions at time t_exp.

        Returns (f_Ni, f_Co, f_Fe) where f_Ni + f_Co + f_Fe = 1.
        X_Ni parameter represents initial Ni56 mass fraction at explosion.
        """
        t = self.t_exp  # days
        f_Ni = np.exp(-cfg.LAMBDA_NI * t)
        f_Co = (cfg.LAMBDA_NI / (cfg.LAMBDA_CO - cfg.LAMBDA_NI)
                * (np.exp(-cfg.LAMBDA_NI * t) - np.exp(-cfg.LAMBDA_CO * t)))
        f_Fe = 1.0 - f_Ni - f_Co
        return f_Ni, f_Co, f_Fe

    def zone_abundances(self, zone):
        """Return dict {Z: mass_fraction} for a given zone.

        Fe includes zone Fe + Fe56 from Ni56 decay.
        Ni and Co are computed from Ni56 → Co56 → Fe56 decay chain at t_exp.
        O is the filler (1 - sum of all others).
        """
        if zone == 'core':
            X_Si, X_Fe_zone = 0.05, self.X_Fe_core
        elif zone == 'wall':
            X_Si, X_Fe_zone = self.X_Si_wall, self.X_Fe_wall
        else:  # outer
            X_Si, X_Fe_zone = 0.02, self.X_Fe_outer

        X_S  = cfg.ZONE_S[zone]
        X_Ca = cfg.ZONE_CA[zone]
        X_Mg = cfg.ZONE_MG[zone]
        X_Ti = cfg.ZONE_TI[zone]
        X_Cr = cfg.ZONE_CR[zone]

        # Ni56 decay chain: X_Ni is initial Ni56 at t=0
        f_Ni, f_Co, f_Fe = self.ni56_decay_fractions
        X_Ni = self.X_Ni * f_Ni      # remaining Ni56
        X_Co = self.X_Ni * f_Co      # Co56 from decay
        X_Fe = X_Fe_zone + self.X_Ni * f_Fe  # zone Fe + Fe56 from decay

        # O = filler (total mass = 1)
        X_O = (1.0 - cfg.FIXED_SPECIES_SUM_BASE
               - X_Si - X_Fe - X_S - X_Ca - X_Ni - X_Co
               - X_Mg - X_Ti - X_Cr)

        return {
            6: cfg.FIXED_SPECIES[6],  # C
            8: X_O, 12: X_Mg, 14: X_Si, 16: X_S, 20: X_Ca,
            22: X_Ti, 24: X_Cr, 26: X_Fe, 27: X_Co, 28: X_Ni,
        }

    def is_valid(self):
        """Check physical validity constraints."""
        # Zones must not overlap, with >=1000 km/s gap
        if self.v_core + 1000 >= self.v_wall:
            return False
        # Core boundary must be above inner boundary
        if self.v_core <= self.v_inner:
            return False
        # Wall boundary must be below outer boundary
        if self.v_wall >= cfg.V_OUTER:
            return False
        # Density break must be within ejecta
        if self.v_break <= self.v_inner + 1000:
            return False
        if self.v_break >= cfg.V_OUTER - 1000:
            return False
        # Oxygen filler must be positive in all zones
        for zone in ('core', 'wall', 'outer'):
            abund = self.zone_abundances(zone)
            if abund[8] < 0.03:  # O = filler
                return False
        # Wall Fe + Si must not exceed budget
        if self.X_Fe_wall + self.X_Si_wall > 0.65:
            return False
        return True

    def to_array(self):
        """Convert to numpy array in PARAM_NAMES order."""
        return np.array([
            self.log_L, self.v_inner, self.log_rho_0, self.density_exp,
            self.T_e_ratio, self.v_core, self.v_wall, self.X_Fe_core,
            self.X_Si_wall, self.v_break, self.density_exp_outer, self.t_exp,
            self.X_Fe_wall, self.X_Ni, self.X_Fe_outer,
        ])

    @classmethod
    def from_array(cls, arr):
        """Create from numpy array in PARAM_NAMES order."""
        return cls(
            log_L=float(arr[0]), v_inner=float(arr[1]),
            log_rho_0=float(arr[2]), density_exp=float(arr[3]),
            T_e_ratio=float(arr[4]), v_core=float(arr[5]),
            v_wall=float(arr[6]), X_Fe_core=float(arr[7]),
            X_Si_wall=float(arr[8]), v_break=float(arr[9]),
            density_exp_outer=float(arr[10]), t_exp=float(arr[11]),
            X_Fe_wall=float(arr[12]), X_Ni=float(arr[13]),
            X_Fe_outer=float(arr[14]),
        )


@dataclass
class Stage2Params:
    """63D parameters: 15 physical + 48 zone composition (6 zones × 8 species).

    Zone mapping: z0=shells 0-4, z1=5-9, ..., z5=25-29.
    Species per zone: Fe(26), Si(14), S(16), Ca(20), Ni(28), Mg(12), Ti(22), Cr(24).
    """
    # 15 physical parameters (same as ModelParams)
    log_L: float
    v_inner: float
    log_rho_0: float
    density_exp: float
    T_e_ratio: float
    v_core: float
    v_wall: float
    X_Fe_core: float
    X_Si_wall: float
    v_break: float
    density_exp_outer: float
    t_exp: float
    X_Fe_wall: float
    X_Ni: float
    X_Fe_outer: float

    # 48 zone composition parameters: zone_X[zone_id] = {Z: mass_fraction}
    zone_X: Dict[int, Dict[int, float]] = field(default_factory=dict)

    @property
    def L_erg_s(self):
        return 10**self.log_L

    @property
    def rho_0(self):
        return 10**self.log_rho_0

    @property
    def t_exp_s(self):
        return self.t_exp * 86400.0

    @property
    def v_inner_cm_s(self):
        return self.v_inner * 1e5

    @property
    def T_inner_estimate(self):
        R_inner = self.v_inner_cm_s * self.t_exp_s
        return (self.L_erg_s / (4 * np.pi * cfg.SIGMA_SB * R_inner**2))**0.25

    def density_at_v(self, v_mid):
        rho_0 = self.rho_0
        v_min = self.v_inner
        t_scale = (cfg.T_REF / self.t_exp) ** 3
        if v_mid < self.v_break:
            rho = rho_0 * (v_mid / v_min) ** self.density_exp
        else:
            rho_break = rho_0 * (self.v_break / v_min) ** self.density_exp
            rho = rho_break * (v_mid / self.v_break) ** self.density_exp_outer
        return rho * t_scale

    @property
    def ni56_decay_fractions(self):
        t = self.t_exp
        f_Ni = np.exp(-cfg.LAMBDA_NI * t)
        f_Co = (cfg.LAMBDA_NI / (cfg.LAMBDA_CO - cfg.LAMBDA_NI)
                * (np.exp(-cfg.LAMBDA_NI * t) - np.exp(-cfg.LAMBDA_CO * t)))
        f_Fe = 1.0 - f_Ni - f_Co
        return f_Ni, f_Co, f_Fe

    def zone_abundances(self, zone_id):
        """Return dict {Z: mass_fraction} for zone_id (0-5).

        Uses zone_X overrides for the 8 species. Ni56 decay is computed per-zone
        from the zone's Ni abundance. C is fixed, O is filler.
        """
        if zone_id not in self.zone_X:
            raise ValueError(f"No zone data for zone_id={zone_id}")

        zx = self.zone_X[zone_id]
        X_Fe_zone = zx[26]
        X_Si = zx[14]
        X_S = zx[16]
        X_Ca = zx[20]
        X_Ni_init = zx[28]  # initial Ni56
        X_Mg = zx[12]
        X_Ti = zx[22]
        X_Cr = zx[24]

        # Ni56 decay chain
        f_Ni, f_Co, f_Fe = self.ni56_decay_fractions
        X_Ni = X_Ni_init * f_Ni
        X_Co = X_Ni_init * f_Co
        X_Fe = X_Fe_zone + X_Ni_init * f_Fe

        # O = filler
        X_O = (1.0 - cfg.FIXED_SPECIES_SUM_BASE
               - X_Si - X_Fe - X_S - X_Ca - X_Ni - X_Co
               - X_Mg - X_Ti - X_Cr)

        return {
            6: cfg.FIXED_SPECIES[6],  # C
            8: X_O, 12: X_Mg, 14: X_Si, 16: X_S, 20: X_Ca,
            22: X_Ti, 24: X_Cr, 26: X_Fe, 27: X_Co, 28: X_Ni,
        }

    def is_valid(self):
        # Physical velocity constraints (same as Stage 1)
        if self.v_core + 1000 >= self.v_wall:
            return False
        if self.v_core <= self.v_inner:
            return False
        if self.v_wall >= cfg.V_OUTER:
            return False
        if self.v_break <= self.v_inner + 1000:
            return False
        if self.v_break >= cfg.V_OUTER - 1000:
            return False
        # O filler must be positive in all 6 zones
        for zi in range(cfg.STAGE2_N_ZONES):
            if zi not in self.zone_X:
                return False
            abund = self.zone_abundances(zi)
            if abund[8] < 0.03:
                return False
        return True

    def to_array(self):
        """Convert to 63-element numpy array: 15 physical + 48 zone composition."""
        phys = np.array([
            self.log_L, self.v_inner, self.log_rho_0, self.density_exp,
            self.T_e_ratio, self.v_core, self.v_wall, self.X_Fe_core,
            self.X_Si_wall, self.v_break, self.density_exp_outer, self.t_exp,
            self.X_Fe_wall, self.X_Ni, self.X_Fe_outer,
        ])
        zone_vals = []
        for zi in range(cfg.STAGE2_N_ZONES):
            zx = self.zone_X.get(zi, {})
            for sp_z in cfg.STAGE2_SPECIES:
                zone_vals.append(zx.get(sp_z, 0.0))
        return np.concatenate([phys, np.array(zone_vals)])

    @classmethod
    def from_array(cls, arr):
        """Create from 63-element numpy array."""
        arr = np.asarray(arr, dtype=float)
        phys = arr[:15]
        zone_X = {}
        idx = 15
        for zi in range(cfg.STAGE2_N_ZONES):
            zx = {}
            for sp_z in cfg.STAGE2_SPECIES:
                zx[sp_z] = float(arr[idx])
                idx += 1
            zone_X[zi] = zx
        return cls(
            log_L=phys[0], v_inner=phys[1], log_rho_0=phys[2],
            density_exp=phys[3], T_e_ratio=phys[4], v_core=phys[5],
            v_wall=phys[6], X_Fe_core=phys[7], X_Si_wall=phys[8],
            v_break=phys[9], density_exp_outer=phys[10], t_exp=phys[11],
            X_Fe_wall=phys[12], X_Ni=phys[13], X_Fe_outer=phys[14],
            zone_X=zone_X,
        )


def _constrain_zone_abundances(arr):
    """Rescale zone abundances in a 63D array so O filler >= 0.03 in all zones.

    For each zone, the 8 species (Fe, Si, S, Ca, Ni, Mg, Ti, Cr) must sum to
    <= 0.95 (since C=0.02 is fixed and O_min=0.03 is the filler floor).
    If the sum exceeds the budget, all 8 species are scaled proportionally.
    """
    MAX_SPECIES_SUM = 1.0 - cfg.FIXED_SPECIES_SUM_BASE - 0.03  # 0.95
    n_species = len(cfg.STAGE2_SPECIES)  # 8
    for zi in range(cfg.STAGE2_N_ZONES):
        base = 15 + zi * n_species
        zone_vals = arr[base:base + n_species]
        total = zone_vals.sum()
        if total > MAX_SPECIES_SUM:
            arr[base:base + n_species] = zone_vals * (MAX_SPECIES_SUM / total)
    return arr


def latin_hypercube(n_samples, param_ranges=None, rng=None, params_class=ModelParams):
    """Generate Latin Hypercube samples in N-dimensional parameter space.

    Args:
        n_samples: Number of samples to generate.
        param_ranges: List of (min, max) tuples. Default: cfg.PARAM_RANGES.
        rng: numpy random generator.
        params_class: ModelParams (15D) or Stage2Params (63D).

    Returns list of valid params_class instances.
    """
    if param_ranges is None:
        param_ranges = cfg.PARAM_RANGES
    if rng is None:
        rng = np.random.default_rng(42)

    ndim = len(param_ranges)
    is_stage2 = (params_class is Stage2Params)
    max_attempts = n_samples * 50 if is_stage2 else n_samples * 20

    # Generate LHS: stratified random sampling
    result = np.zeros((n_samples, ndim))
    for d in range(ndim):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            lo = perm[i] / n_samples
            hi = (perm[i] + 1) / n_samples
            result[i, d] = lo + rng.random() * (hi - lo)

    # Scale to parameter ranges
    for d in range(ndim):
        pmin, pmax = param_ranges[d]
        result[:, d] = pmin + result[:, d] * (pmax - pmin)

    # For Stage 2: constrain zone abundances before validity check
    if is_stage2:
        for i in range(n_samples):
            result[i] = _constrain_zone_abundances(result[i])

    # Convert to params_class, rejecting invalid
    valid = []
    for i in range(n_samples):
        p = params_class.from_array(result[i])
        if p.is_valid():
            valid.append(p)

    # Fill rejected slots with random valid samples
    attempts = 0
    while len(valid) < n_samples and attempts < max_attempts:
        vals = np.array([rng.uniform(lo, hi) for lo, hi in param_ranges])
        if is_stage2:
            vals = _constrain_zone_abundances(vals)
        p = params_class.from_array(vals)
        if p.is_valid():
            valid.append(p)
        attempts += 1

    if len(valid) < n_samples:
        print(f"WARNING: Only generated {len(valid)}/{n_samples} valid samples")

    return valid[:n_samples]


class LuminaRunner:
    """Manages LUMINA model execution: directory setup, binary invocation, output parsing."""

    def __init__(self, binary=None, ref_dir=None, nlte=False, nlte_start_iter=0):
        # Auto-detect binary: prefer CUDA, fall back to CPU
        if binary is not None:
            self.binary = Path(binary)
        elif cfg.LUMINA_CUDA.exists():
            self.binary = cfg.LUMINA_CUDA
        elif cfg.LUMINA_CPU.exists():
            self.binary = cfg.LUMINA_CPU
        else:
            raise FileNotFoundError(
                f"No LUMINA binary found. Tried:\n"
                f"  {cfg.LUMINA_CUDA}\n  {cfg.LUMINA_CPU}"
            )

        self.ref_dir = Path(ref_dir) if ref_dir else cfg.REF_DIR
        self.ref_files = [f.name for f in self.ref_dir.iterdir() if f.is_file()]
        self.nlte = nlte
        self.nlte_start_iter = nlte_start_iter

        cfg.TMPDIR_BASE.mkdir(parents=True, exist_ok=True)

        # Load reference density/n_e for scaling
        ref_density = np.genfromtxt(
            str(self.ref_dir / "density.csv"), delimiter=',', names=True
        )
        ref_ne = np.genfromtxt(
            str(self.ref_dir / "electron_densities.csv"), delimiter=',', names=True
        )
        self._ref_rho0 = float(ref_density['rho'][0])
        self._ref_ne0 = float(ref_ne['n_e'][0])

    def create_model_dir(self, params: ModelParams, tag: str = "") -> Path:
        """Create temporary reference directory with modified parameters."""
        dirname = f"model_{tag}_{id(params):x}"
        temp_dir = cfg.TMPDIR_BASE / dirname
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Symlink unchanged files
        for fname in self.ref_files:
            if fname not in cfg.REGEN_FILES:
                src = self.ref_dir / fname
                dst = temp_dir / fname
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)

        # Generate modified files
        self._write_config_json(temp_dir, params)
        self._write_geometry_csv(temp_dir, params)
        self._write_density_csv(temp_dir, params)
        self._write_abundances_csv(temp_dir, params)
        self._write_electron_densities_csv(temp_dir, params)
        self._write_plasma_state_csv(temp_dir, params)

        return temp_dir

    def _write_config_json(self, dirpath, params: ModelParams):
        config = {
            "time_explosion_s": params.t_exp_s,
            "T_inner_K": params.T_inner_estimate,
            "luminosity_inner_erg_s": params.L_erg_s,
            "n_shells": cfg.N_SHELLS,
            "n_lines": 137252,
            "n_packets": 200000,
            "n_iterations": 20,
            "seed": 23111963,
            "v_inner_min_cm_s": params.v_inner_cm_s,
            "v_outer_max_cm_s": cfg.V_OUTER * 1e5,
            "T_e_T_rad_ratio": params.T_e_ratio,
        }
        with open(dirpath / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _write_geometry_csv(self, dirpath, params: ModelParams):
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        t_exp_s = params.t_exp_s
        with open(dirpath / "geometry.csv", 'w') as f:
            f.write("shell_id,r_inner,r_outer,v_inner,v_outer\n")
            for i in range(cfg.N_SHELLS):
                vi = (v_min + i * dv) * 1e5        # cm/s
                vo = (v_min + (i + 1) * dv) * 1e5  # cm/s
                ri = vi * t_exp_s                    # cm
                ro = vo * t_exp_s                    # cm
                f.write(f"{i},{ri},{ro},{vi},{vo}\n")

    def _write_density_csv(self, dirpath, params: ModelParams):
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        with open(dirpath / "density.csv", 'w') as f:
            f.write("shell_id,rho\n")
            for i in range(cfg.N_SHELLS):
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                rho = params.density_at_v(v_mid)
                f.write(f"{i},{rho}\n")

    def _write_abundances_csv(self, dirpath, params):
        """Write per-shell abundances. Dispatches on params type:

        - Stage2Params (has zone_X): 6 zones by shell index (zone_id = shell_id // 5)
        - ModelParams: 3 zones by velocity (core/wall/outer)
        """
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        abundances = {z: np.zeros(cfg.N_SHELLS) for z in cfg.ELEMENT_ORDER}

        has_zone_X = hasattr(params, 'zone_X') and params.zone_X

        for i in range(cfg.N_SHELLS):
            if has_zone_X:
                # Stage 2: zone by shell index
                zone_id = i // 5
                zone_abund = params.zone_abundances(zone_id)
            else:
                # Stage 1: zone by velocity
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                if v_mid < params.v_core:
                    zone = 'core'
                elif v_mid < params.v_wall:
                    zone = 'wall'
                else:
                    zone = 'outer'
                zone_abund = params.zone_abundances(zone)

            for z in cfg.ELEMENT_ORDER:
                abundances[z][i] = zone_abund[z]

        with open(dirpath / "abundances.csv", 'w') as f:
            header = "atomic_number," + ",".join(str(i) for i in range(cfg.N_SHELLS))
            f.write(header + "\n")
            for z in cfg.ELEMENT_ORDER:
                vals = ",".join(f"{abundances[z][i]}" for i in range(cfg.N_SHELLS))
                f.write(f"{z},{vals}\n")

    def _write_electron_densities_csv(self, dirpath, params: ModelParams):
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        with open(dirpath / "electron_densities.csv", 'w') as f:
            f.write("shell_id,n_e\n")
            for i in range(cfg.N_SHELLS):
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                rho_new = params.density_at_v(v_mid)
                ne = self._ref_ne0 * (rho_new / self._ref_rho0)
                f.write(f"{i},{ne}\n")

    def _write_plasma_state_csv(self, dirpath, params: ModelParams):
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        T_inner = params.T_inner_estimate
        with open(dirpath / "plasma_state.csv", 'w') as f:
            f.write("shell_id,W,T_rad\n")
            for i in range(cfg.N_SHELLS):
                vi = v_min + i * dv
                vo = v_min + (i + 1) * dv
                v_mid = (vi + vo) / 2.0
                r_ratio = v_min / v_mid
                W = 0.5 * (1.0 - np.sqrt(1.0 - r_ratio**2))
                T_rad = T_inner * (v_min / v_mid)**0.45
                f.write(f"{i},{W},{T_rad}\n")

    def run_model(self, params, n_packets: int, n_iters: int,
                  tag: str = "", timeout: int = 600,
                  env: dict = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run LUMINA and return (wavelength, flux) arrays, or None on failure.

        Returns the rotation spectrum (Doppler-weighted observer-frame).

        Args:
            params: ModelParams or Stage2Params instance.
            env: Optional environment dict for subprocess (e.g., CUDA_VISIBLE_DEVICES).
                 If None, inherits parent environment.
        """
        temp_dir = self.create_model_dir(params, tag=tag)
        work_dir = cfg.TMPDIR_BASE / f"work_{tag}_{id(params):x}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build command: add 'nlte' arg if NLTE enabled
            cmd = [str(self.binary), str(temp_dir), str(n_packets), str(n_iters), "rotation"]
            if self.nlte:
                cmd.append("nlte")

            # Build environment: add LUMINA_NLTE=1 if enabled
            run_env = env
            if self.nlte:
                run_env = (env if env else os.environ).copy()
                run_env['LUMINA_NLTE'] = '1'
                if self.nlte_start_iter > 0:
                    run_env['LUMINA_NLTE_START_ITER'] = str(self.nlte_start_iter)

            proc = subprocess.run(
                cmd,
                capture_output=True, timeout=timeout, cwd=str(work_dir), text=True,
                env=run_env,
            )
            if proc.returncode != 0:
                return None

            # Read rotation spectrum
            spec_file = work_dir / "lumina_spectrum_rotation.csv"
            if not spec_file.exists():
                spec_file = work_dir / "lumina_spectrum.csv"
            if not spec_file.exists():
                return None

            spec = np.genfromtxt(str(spec_file), delimiter=',', names=True)
            wave = spec['wavelength_angstrom']
            flux = spec['flux']
            return wave, flux

        except (subprocess.TimeoutExpired, Exception):
            return None
        finally:
            self._cleanup(temp_dir)
            self._cleanup(work_dir)

    def _cleanup(self, dirpath):
        try:
            if dirpath.exists():
                shutil.rmtree(dirpath, ignore_errors=True)
        except Exception:
            pass
