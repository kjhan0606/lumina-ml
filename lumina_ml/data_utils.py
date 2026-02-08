"""LUMINA wrapper: ModelParams, model directory creation, execution, LHS sampling."""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

    def zone_abundances(self, zone):
        """Return (X_Si, X_Fe, X_S, X_Ca, X_Ni, X_O) for a given zone."""
        if zone == 'core':
            X_Si, X_Fe = 0.05, self.X_Fe_core
        elif zone == 'wall':
            X_Si, X_Fe = self.X_Si_wall, self.X_Fe_wall
        else:  # outer
            X_Si, X_Fe = 0.02, self.X_Fe_outer
        X_S = cfg.ZONE_S[zone]
        X_Ca = cfg.ZONE_CA[zone]
        X_Ni = self.X_Ni
        # O = filler: 1.0 - (Co + C) - Si - Fe - S - Ca - Ni
        X_O = 1.0 - cfg.FIXED_SPECIES_SUM_BASE - X_Si - X_Fe - X_S - X_Ca - X_Ni
        return X_Si, X_Fe, X_S, X_Ca, X_Ni, X_O

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
            *_, X_O = self.zone_abundances(zone)
            if X_O < 0.03:
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


def latin_hypercube(n_samples, param_ranges=None, rng=None):
    """Generate Latin Hypercube samples in 12D parameter space.

    Returns list of valid ModelParams.
    """
    if param_ranges is None:
        param_ranges = cfg.PARAM_RANGES
    if rng is None:
        rng = np.random.default_rng(42)

    ndim = len(param_ranges)
    max_attempts = n_samples * 20

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

    # Convert to ModelParams, rejecting invalid
    valid = []
    for i in range(n_samples):
        p = ModelParams.from_array(result[i])
        if p.is_valid():
            valid.append(p)

    # Fill rejected slots with random valid samples
    attempts = 0
    while len(valid) < n_samples and attempts < max_attempts:
        vals = [rng.uniform(lo, hi) for lo, hi in param_ranges]
        p = ModelParams.from_array(vals)
        if p.is_valid():
            valid.append(p)
        attempts += 1

    if len(valid) < n_samples:
        print(f"WARNING: Only generated {len(valid)}/{n_samples} valid samples")

    return valid[:n_samples]


class LuminaRunner:
    """Manages LUMINA model execution: directory setup, binary invocation, output parsing."""

    def __init__(self, binary=None, ref_dir=None):
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

    def _write_abundances_csv(self, dirpath, params: ModelParams):
        v_min = params.v_inner
        dv = (cfg.V_OUTER - v_min) / cfg.N_SHELLS
        abundances = {z: np.zeros(cfg.N_SHELLS) for z in cfg.ELEMENT_ORDER}

        for i in range(cfg.N_SHELLS):
            vi = v_min + i * dv
            vo = v_min + (i + 1) * dv
            v_mid = (vi + vo) / 2.0

            if v_mid < params.v_core:
                zone = 'core'
            elif v_mid < params.v_wall:
                zone = 'wall'
            else:
                zone = 'outer'

            X_Si, X_Fe, X_S, X_Ca, X_Ni, X_O = params.zone_abundances(zone)
            for z in cfg.ELEMENT_ORDER:
                if z == 8:
                    abundances[z][i] = X_O
                elif z == 14:
                    abundances[z][i] = X_Si
                elif z == 26:
                    abundances[z][i] = X_Fe
                elif z == 16:
                    abundances[z][i] = X_S
                elif z == 20:
                    abundances[z][i] = X_Ca
                elif z == 28:
                    abundances[z][i] = X_Ni
                else:
                    abundances[z][i] = cfg.FIXED_SPECIES[z]

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

    def run_model(self, params: ModelParams, n_packets: int, n_iters: int,
                  tag: str = "", timeout: int = 600) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run LUMINA and return (wavelength, flux) arrays, or None on failure.

        Returns the rotation spectrum (Doppler-weighted observer-frame).
        """
        temp_dir = self.create_model_dir(params, tag=tag)
        work_dir = cfg.TMPDIR_BASE / f"work_{tag}_{id(params):x}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                [str(self.binary), str(temp_dir), str(n_packets), str(n_iters), "rotation"],
                capture_output=True, timeout=timeout, cwd=str(work_dir), text=True,
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
