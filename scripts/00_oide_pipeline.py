#!/usr/bin/env python3
"""OIDE: Overlapping Iterative Dimensional Expansion pipeline.

Multi-phase SBI where each phase explores ~8-10D with carry-over dimensions
from previous phases. Total: 9 phases, ~117K models.

Each phase: LHS sampling → LUMINA runs → preprocess → train emulator → SBI → CR analysis.
Between phases: update base values, select carry-over dimensions.

Usage:
  python3 scripts/00_oide_pipeline.py                               # Full run (phases 1-9)
  python3 scripts/00_oide_pipeline.py --start-phase 3               # Resume from phase 3
  python3 scripts/00_oide_pipeline.py --start-phase 1 --end-phase 2 # Only phases 1-2
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumina_ml import config as cfg
from lumina_ml.data_utils import LuminaRunner, Stage2Params, _constrain_zone_abundances
from lumina_ml.preprocessing import (
    preprocess_spectrum, validate_spectrum,
    SpectralPCA, ParamScaler,
    interpolate_to_grid, adaptive_smooth, peak_normalize, asinh_transform,
)
from lumina_ml.emulator import SpectralMLP, EmulatorTrainer
from lumina_ml.inference import run_sbi
from lumina_ml.oide import (
    get_default_phases, load_base_values, compute_cr, classify_cr,
    select_carry_over, build_phase_ranges, generate_phase_samples,
    select_phase9_params, update_base_values, TICR_INDICES,
)


# ===== Multi-GPU data generation (reused from 01_generate_training_data.py) =====

def detect_gpu_count():
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().split('\n')
                        if l.startswith('GPU')])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def make_gpu_env(gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return env


def make_cpu_env(omp_threads):
    """Create environment dict with OMP_NUM_THREADS set."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_threads)
    return env


class SharedState:
    """Thread-safe shared state with dynamic work queue."""
    def __init__(self, n_total, remaining_indices):
        self.lock = threading.Lock()
        self.spectra_dict = {}
        self.completed_indices = set()
        self.n_success = 0
        self.n_fail = 0
        self.n_total = n_total
        self.device_counts = {}
        self.last_checkpoint = 0
        self.work_queue = queue.Queue()
        for idx in remaining_indices:
            self.work_queue.put(idx)

    def get_next(self):
        try:
            return self.work_queue.get_nowait()
        except queue.Empty:
            return None

    def add_result(self, idx, wave, flux, source):
        with self.lock:
            label = source.strip()
            self.device_counts[label] = self.device_counts.get(label, 0) + 1
            if wave is not None:
                self.spectra_dict[idx] = (wave, flux)
                self.completed_indices.add(idx)
                self.n_success += 1
                return f"  [{self.n_success}/{self.n_total}] {source} Model {idx}: OK"
            else:
                self.n_fail += 1
                return f"  [{self.n_success}/{self.n_total}] {source} Model {idx}: FAILED"


def _save_checkpoint(params_array, spectra_dict, completed_indices,
                     params_file, spectra_file, waves_file, checkpoint_file):
    """Save current state to disk."""
    if not spectra_dict:
        return
    sorted_indices = sorted(spectra_dict.keys())
    first_wave, _ = spectra_dict[sorted_indices[0]]
    n_wave = len(first_wave)

    waves = np.zeros((len(params_array), n_wave))
    spectra = np.zeros((len(params_array), n_wave))
    for idx in sorted_indices:
        w, f = spectra_dict[idx]
        if len(w) == n_wave:
            waves[idx] = w
            spectra[idx] = f

    np.save(str(params_file), params_array)
    np.save(str(spectra_file), spectra)
    np.save(str(waves_file), waves)
    np.savez(str(checkpoint_file),
             completed_indices=np.array(list(completed_indices)))


def generate_data(params_array, all_params_list, phase_dir,
                  n_packets, n_iters, n_gpus, nlte, nlte_start_iter,
                  timeout=1800, n_cpu_workers=0, omp_threads=32):
    """Run LUMINA for all models in a phase, with multi-GPU workers.

    Supports checkpoint/resume within a phase.
    Returns path to raw data directory.
    """
    raw_dir = phase_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    params_file = raw_dir / "params_all.npy"
    spectra_file = raw_dir / "spectra_all.npy"
    waves_file = raw_dir / "waves_all.npy"
    checkpoint_file = raw_dir / "checkpoint.npz"

    # Check for resume within phase
    completed_indices = set()
    spectra_dict = {}
    if checkpoint_file.exists():
        ckpt = np.load(str(checkpoint_file), allow_pickle=True)
        completed_indices = set(ckpt['completed_indices'].tolist())
        if spectra_file.exists() and waves_file.exists():
            prev_spectra = np.load(str(spectra_file))
            prev_waves = np.load(str(waves_file))
            for i in completed_indices:
                if i < len(prev_spectra):
                    spectra_dict[i] = (prev_waves[i], prev_spectra[i])
        print(f"  Resuming: {len(completed_indices)} already completed")

    remaining = [i for i in range(len(all_params_list)) if i not in completed_indices]
    print(f"  Remaining: {len(remaining)} models")

    if not remaining:
        print("  All models already completed!")
        return raw_dir

    t0 = time.time()

    def device_worker(state, device_name, runner, env):
        while True:
            idx = state.get_next()
            if idx is None:
                break
            params = all_params_list[idx]
            t1 = time.time()
            tag = f"oide_p{phase_dir.name.split('_')[-1]}_{device_name.lower()}_{idx}"
            result = runner.run_model(params, n_packets, n_iters, tag=tag,
                                      timeout=timeout, env=env)
            dt = time.time() - t1
            if result is not None:
                wave, flux = result
                msg = state.add_result(idx, wave, flux, device_name)
            else:
                msg = state.add_result(idx, None, None, device_name)
            print(f"{msg} ({dt:.1f}s)", flush=True)

            # Periodic checkpoint
            with state.lock:
                total_done = state.n_success + state.n_fail - len(completed_indices)
                if total_done > 0 and total_done >= state.last_checkpoint + 100:
                    state.last_checkpoint = total_done
                    _save_checkpoint(params_array, state.spectra_dict,
                                     state.completed_indices,
                                     params_file, spectra_file,
                                     waves_file, checkpoint_file)
                    elapsed = time.time() - t0
                    rate = total_done / elapsed
                    eta = (len(remaining) - total_done) / rate if rate > 0 else 0
                    devs = " ".join(f"{k}:{v}" for k, v in
                                    sorted(state.device_counts.items()))
                    print(f"  -- Checkpoint {total_done}/{len(remaining)}. "
                          f"Rate: {rate:.2f}/s, ETA: {eta/3600:.1f}h [{devs}] --",
                          flush=True)

    # Create state and launch GPU + CPU workers
    state = SharedState(len(all_params_list), remaining)
    state.spectra_dict = spectra_dict
    state.completed_indices = completed_indices
    state.n_success = len(completed_indices)

    threads = []

    # GPU workers
    if n_gpus > 0:
        cuda_runner = LuminaRunner(binary=cfg.LUMINA_CUDA, nlte=nlte,
                                   nlte_start_iter=nlte_start_iter)
        for gpu_id in range(n_gpus):
            env = make_gpu_env(gpu_id)
            t = threading.Thread(target=device_worker,
                                 args=(state, f"GPU{gpu_id}", cuda_runner, env))
            threads.append(t)

    # CPU workers
    if n_cpu_workers > 0:
        threads_per_cpu = max(1, omp_threads // n_cpu_workers)
        cpu_runner = LuminaRunner(binary=cfg.LUMINA_CPU, nlte=nlte,
                                  nlte_start_iter=nlte_start_iter)
        for cpu_id in range(n_cpu_workers):
            env = make_cpu_env(threads_per_cpu)
            label = f"CPU{cpu_id}" if n_cpu_workers > 1 else "CPU"
            t = threading.Thread(target=device_worker,
                                 args=(state, label, cpu_runner, env))
            threads.append(t)

    print(f"  Launching {len(threads)} workers ({n_gpus} GPU + {n_cpu_workers} CPU) "
          f"for {len(remaining)} models")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Final save
    _save_checkpoint(params_array, state.spectra_dict, state.completed_indices,
                     params_file, spectra_file, waves_file, checkpoint_file)

    total_time = time.time() - t0
    n_new = state.n_success - len(completed_indices)
    print(f"  Data generation complete: {state.n_success}/{len(all_params_list)} success, "
          f"{state.n_fail} fail, {total_time:.0f}s ({total_time/3600:.1f}h)")
    if total_time > 0 and n_new > 0:
        print(f"  Rate: {n_new/total_time:.2f} models/s "
              f"({total_time/n_new:.1f}s/model)")

    return raw_dir


def load_observed_spectrum():
    """Load and preprocess observed SN 2011fe B-max spectrum."""
    obs = np.genfromtxt(str(cfg.OBS_FILE_BMAX), delimiter=',', names=True)
    wave = obs['wavelength_angstrom']
    flux = obs['flux_erg_s_cm2_angstrom']
    grid_flux = np.interp(cfg.SPECTRUM_GRID, wave, flux, left=0.0, right=0.0)
    grid_flux = adaptive_smooth(grid_flux)
    normalized, _ = peak_normalize(grid_flux)
    return asinh_transform(normalized)


def run_phase(phase, base_values, all_cr, args):
    """Run a single OIDE phase: generate → preprocess → train → SBI → analyze.

    Returns:
        (cr_dict, carry_indices, posterior_medians)
    """
    phase_dir = args.oide_dir / f"phase_{phase.phase_id}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    free_indices = phase.free_indices
    free_names = [cfg.STAGE2_PARAM_NAMES[i] for i in free_indices]
    n_free = phase.n_free

    phase_t0 = time.time()

    print(f"\n{'='*70}")
    print(f"OIDE Phase {phase.phase_id}: {phase.description}")
    print(f"  Free dims: {n_free} ({', '.join(free_names)})")
    if phase.carry_indices:
        carry_names = [cfg.STAGE2_PARAM_NAMES[i] for i in phase.carry_indices]
        print(f"  Carry-over: {', '.join(carry_names)}")
    print(f"  Models: {phase.n_models}")
    print(f"  NN: {phase.hidden_layers}")
    print(f"{'='*70}")

    # Build prior ranges for free dimensions
    phase_ranges = build_phase_ranges(
        free_indices, base_values, all_cr, phase.carry_indices
    )

    print(f"\n  Prior ranges:")
    for idx, (lo, hi), name in zip(free_indices, phase_ranges, free_names):
        full_lo, full_hi = cfg.STAGE2_PARAM_RANGES[idx]
        compress = (full_hi - full_lo) / max(1e-10, hi - lo)
        tag = " [carry]" if idx in phase.carry_indices else ""
        print(f"    {name:22s}: [{lo:.6f}, {hi:.6f}]  ({compress:.1f}x){tag}")

    # Save phase config
    phase_config = {
        'phase_id': phase.phase_id,
        'description': phase.description,
        'free_indices': free_indices,
        'carry_indices': phase.carry_indices,
        'n_models': phase.n_models,
        'free_names': free_names,
        'phase_ranges': [(float(lo), float(hi)) for lo, hi in phase_ranges],
        'base_values': base_values.tolist(),
    }
    with open(phase_dir / "phase_config.json", 'w') as f:
        json.dump(phase_config, f, indent=2)

    # ===== Step 1: Generate LHS samples =====
    print(f"\n--- Step 1: Generating {phase.n_models} LHS samples in {n_free}D ---")
    rng = np.random.default_rng(42 + phase.phase_id)
    params_array = generate_phase_samples(
        phase.n_models, free_indices, phase_ranges, base_values, rng
    )
    print(f"  Valid samples: {len(params_array)}")

    # Convert to Stage2Params list for LuminaRunner
    all_params_list = [Stage2Params.from_array(params_array[i])
                       for i in range(len(params_array))]

    # ===== Step 2: Run LUMINA =====
    print(f"\n--- Step 2: Running LUMINA ({args.n_gpus} GPUs, "
          f"{args.n_packets} pkts, {args.n_iters} iters, "
          f"timeout={args.timeout}s) ---")
    raw_dir = generate_data(
        params_array, all_params_list, phase_dir,
        n_packets=args.n_packets, n_iters=args.n_iters,
        n_gpus=args.n_gpus, nlte=True, nlte_start_iter=5,
        timeout=args.timeout,
        n_cpu_workers=args.n_cpu_workers, omp_threads=args.omp_threads,
    )

    # ===== Step 3: Preprocess spectra =====
    print(f"\n--- Step 3: Preprocessing spectra ---")
    processed_dir = phase_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    spectra_raw = np.load(str(raw_dir / "spectra_all.npy"))
    waves_raw = np.load(str(raw_dir / "waves_all.npy"))
    params_raw = np.load(str(raw_dir / "params_all.npy"))

    # Find completed (non-zero) spectra
    completed_mask = np.any(spectra_raw != 0, axis=1)
    print(f"  Completed spectra: {completed_mask.sum()}/{len(spectra_raw)}")

    # Preprocess each spectrum
    spectra_preprocessed = []
    valid_mask = []
    for i in range(len(spectra_raw)):
        if not completed_mask[i]:
            valid_mask.append(False)
            spectra_preprocessed.append(np.zeros(len(cfg.SPECTRUM_GRID)))
            continue
        try:
            spec = preprocess_spectrum(waves_raw[i], spectra_raw[i])
            if validate_spectrum(spec):
                spectra_preprocessed.append(spec)
                valid_mask.append(True)
            else:
                spectra_preprocessed.append(np.zeros(len(cfg.SPECTRUM_GRID)))
                valid_mask.append(False)
        except Exception:
            spectra_preprocessed.append(np.zeros(len(cfg.SPECTRUM_GRID)))
            valid_mask.append(False)

    spectra_preprocessed = np.array(spectra_preprocessed)
    valid_mask = np.array(valid_mask)
    n_valid = valid_mask.sum()
    print(f"  Valid preprocessed: {n_valid}/{len(spectra_raw)}")

    if n_valid < 100:
        print(f"  ERROR: Too few valid spectra ({n_valid}). Skipping phase.")
        return {}, [], np.array([])

    # Extract free-dim params and valid spectra
    params_free = params_raw[:, free_indices]  # (N, n_free)
    params_valid = params_free[valid_mask]
    spectra_valid = spectra_preprocessed[valid_mask]

    # PCA
    pca = SpectralPCA(variance_threshold=cfg.PCA_VARIANCE_THRESHOLD,
                      max_components=50)
    pca.fit(spectra_valid)
    pca_coeffs = pca.transform(spectra_valid)
    n_pca = pca_coeffs.shape[1]
    print(f"  PCA: {n_pca} components ({cfg.PCA_VARIANCE_THRESHOLD*100:.1f}% variance)")

    recon_err = pca.reconstruction_error(spectra_valid)
    print(f"  PCA reconstruction error: median={np.median(recon_err):.6f}, "
          f"max={np.max(recon_err):.6f}")

    pca.save(processed_dir / "spectral_pca.pkl")

    # Param scaler (for free dims only)
    scaler = ParamScaler(phase_ranges)
    params_scaled = scaler.transform(params_valid)
    scaler.save(processed_dir / "param_scaler.pkl")

    # Save processed data
    np.save(str(processed_dir / "params_valid.npy"), params_valid)
    np.save(str(processed_dir / "spectra_valid.npy"), spectra_valid)
    np.save(str(processed_dir / "pca_coeffs.npy"), pca_coeffs)
    np.save(str(processed_dir / "params_scaled.npy"), params_scaled)

    # ===== Step 4: Train emulator =====
    print(f"\n--- Step 4: Training emulator ({n_free}D → {n_pca} PCA) ---")
    models_dir = phase_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train/val split (90/10)
    n_train = int(0.9 * n_valid)
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(n_valid)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train = params_scaled[train_idx]
    X_val = params_scaled[val_idx]
    Y_train = pca_coeffs[train_idx]
    Y_val = pca_coeffs[val_idx]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    model = SpectralMLP(n_input=n_free, n_output=n_pca,
                        hidden_layers=phase.hidden_layers,
                        dropout=cfg.NN_DROPOUT)
    trainer = EmulatorTrainer(model, pca_model=pca)

    t_train = time.time()
    history = trainer.fit(
        X_train, Y_train, X_val, Y_val,
        epochs=cfg.NN_MAX_EPOCHS,
        batch_size=128,
        patience=200,
    )
    print(f"  Training time: {time.time()-t_train:.1f}s")
    print(f"  Best val loss: {history.get('best_val_loss', float('inf')):.6f}")

    trainer.save(models_dir / "emulator.pt")

    # ===== Step 5: Run SBI =====
    print(f"\n--- Step 5: SBI inference ({n_free}D, {n_valid} training samples) ---")
    results_dir = phase_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    obs_spectrum = load_observed_spectrum()

    t_sbi = time.time()
    sbi_results = run_sbi(
        params_valid, spectra_valid, obs_spectrum,
        n_posterior_samples=10000,
        param_ranges=phase_ranges,
    )
    print(f"  SBI time: {time.time()-t_sbi:.1f}s")

    samples = sbi_results['samples']  # (10000, n_free)
    np.save(str(results_dir / "sbi_samples.npy"), samples)
    if 'log_prob' in sbi_results:
        np.save(str(results_dir / "sbi_log_prob.npy"), sbi_results['log_prob'])
    print(f"  SBI samples: {samples.shape}")

    # ===== Step 6: Constraint Ratio analysis =====
    print(f"\n--- Step 6: Constraint Ratio analysis ---")
    cr_dict = compute_cr(samples, free_indices)

    posterior_medians = np.median(samples, axis=0)

    print(f"\n  {'Parameter':22s}  {'Median':>10s}  {'q16':>10s}  "
          f"{'q84':>10s}  {'CR':>6s}  {'Classification':>14s}")
    print(f"  {'-'*82}")
    for j, idx in enumerate(free_indices):
        name = cfg.STAGE2_PARAM_NAMES[idx]
        med = np.median(samples[:, j])
        q16 = np.percentile(samples[:, j], 16)
        q84 = np.percentile(samples[:, j], 84)
        cr = cr_dict[idx]
        cls = classify_cr(cr)
        tag = " [carry]" if idx in phase.carry_indices else ""
        print(f"  {name:22s}  {med:10.4f}  {q16:10.4f}  "
              f"{q84:10.4f}  {cr:6.3f}  {cls:>14s}{tag}")

    # Save CR analysis
    cr_summary = {}
    for j, idx in enumerate(free_indices):
        name = cfg.STAGE2_PARAM_NAMES[idx]
        cr_summary[name] = {
            'index': idx,
            'median': float(np.median(samples[:, j])),
            'q16': float(np.percentile(samples[:, j], 16)),
            'q84': float(np.percentile(samples[:, j], 84)),
            'CR': float(cr_dict[idx]),
            'class': classify_cr(cr_dict[idx]),
        }
    with open(results_dir / "cr_analysis.json", 'w') as f:
        json.dump(cr_summary, f, indent=2)

    # Update base values for well-constrained params
    update_base_values(base_values, cr_dict, posterior_medians, free_indices)

    # Update global CR tracker (keep best/lowest CR per param)
    for idx, cr in cr_dict.items():
        if idx not in all_cr or cr < all_cr[idx]:
            all_cr[idx] = cr

    # Select carry-over for next phase
    carry = select_carry_over(cr_dict)
    carry_names = [cfg.STAGE2_PARAM_NAMES[i] for i in carry]

    n_locked = sum(1 for cr in cr_dict.values() if cr < 0.10)
    n_constrained = sum(1 for cr in cr_dict.values() if 0.10 <= cr < 0.25)
    n_weak = sum(1 for cr in cr_dict.values() if 0.25 <= cr < 0.50)
    n_uncon = sum(1 for cr in cr_dict.values() if cr >= 0.50)

    print(f"\n  Summary: {n_locked} locked, {n_constrained} constrained, "
          f"{n_weak} weak, {n_uncon} unconstrained")
    print(f"  Carry-over for next phase: {carry_names}")
    print(f"  Phase {phase.phase_id} total time: "
          f"{(time.time()-phase_t0)/3600:.1f}h")

    return cr_dict, carry, posterior_medians


def main():
    parser = argparse.ArgumentParser(
        description="OIDE: Overlapping Iterative Dimensional Expansion pipeline")
    parser.add_argument('--start-phase', type=int, default=1,
                        help='Start from this phase (default: 1)')
    parser.add_argument('--end-phase', type=int, default=9,
                        help='End at this phase (default: 9)')
    parser.add_argument('--base-values', type=str,
                        default='data/oide_base_values.json',
                        help='Path to base values JSON')
    parser.add_argument('--n-packets', type=int, default=200000,
                        help='Packets per model (default: 200000)')
    parser.add_argument('--n-iters', type=int, default=10,
                        help='Iterations per model (default: 10)')
    parser.add_argument('--n-gpus', type=int, default=0,
                        help='Number of GPUs (0=auto-detect)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Per-model timeout in seconds (default: 1800)')
    parser.add_argument('--n-cpu-workers', type=int, default=0,
                        help='Number of CPU workers (default: 0)')
    parser.add_argument('--omp-threads', type=int, default=32,
                        help='Total OMP threads for CPU workers (default: 32)')
    parser.add_argument('--oide-dir', type=str, default='data/oide',
                        help='OIDE output directory')
    args = parser.parse_args()

    args.oide_dir = Path(args.oide_dir)
    args.oide_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect GPUs
    if args.n_gpus <= 0:
        args.n_gpus = max(detect_gpu_count(), 1)

    total_t0 = time.time()
    print(f"{'='*70}")
    print(f"OIDE Pipeline: phases {args.start_phase} → {args.end_phase}")
    print(f"  GPUs: {args.n_gpus}, CPU workers: {args.n_cpu_workers} "
          f"(OMP: {args.omp_threads})")
    print(f"  Packets: {args.n_packets}, Iters: {args.n_iters}, Timeout: {args.timeout}s")
    print(f"  Output: {args.oide_dir}")
    print(f"  Base values: {args.base_values}")
    print(f"{'='*70}")

    # Load base values from Stage 2.5 posterior
    base_values, initial_cr = load_base_values(args.base_values)
    all_cr = dict(initial_cr)

    # Get phase definitions
    phases = get_default_phases()

    # If resuming from a later phase, reload state from previous phases
    carry_indices = []
    if args.start_phase > 1:
        print(f"\nReloading state from phases 1-{args.start_phase - 1}...")
        for p in range(1, args.start_phase):
            phase_dir = args.oide_dir / f"phase_{p}"
            cr_file = phase_dir / "results" / "cr_analysis.json"
            if cr_file.exists():
                with open(cr_file) as f:
                    prev_cr = json.load(f)
                cr_dict = {v['index']: v['CR'] for v in prev_cr.values()}
                # Update all_cr
                for idx, cr in cr_dict.items():
                    if idx not in all_cr or cr < all_cr[idx]:
                        all_cr[idx] = cr
                # Update base values from well-constrained params
                for name, v in prev_cr.items():
                    if v['CR'] < 0.25:
                        base_values[v['index']] = v['median']
                print(f"  Phase {p}: loaded {len(prev_cr)} CR values")
            else:
                print(f"  Phase {p}: no results found (cr_analysis.json missing)")

        # Load carry-over from the immediately previous phase
        prev_cr_file = (args.oide_dir / f"phase_{args.start_phase - 1}"
                        / "results" / "cr_analysis.json")
        if prev_cr_file.exists():
            with open(prev_cr_file) as f:
                prev_cr = json.load(f)
            cr_dict = {v['index']: v['CR'] for v in prev_cr.values()}
            carry_indices = select_carry_over(cr_dict)
            carry_names = [cfg.STAGE2_PARAM_NAMES[i] for i in carry_indices]
            print(f"  Carry-over from phase {args.start_phase - 1}: {carry_names}")

    # Run phases
    for phase_id in range(args.start_phase, args.end_phase + 1):
        phase = phases[phase_id - 1]  # 0-indexed

        # Set carry-over from previous phase
        phase.carry_indices = list(carry_indices)

        # Phase 9: dynamic param selection from worst CR across all phases
        if phase.phase_id == 9:
            phase.new_param_indices = select_phase9_params(all_cr, n_select=10)
            # Remove any that overlap with carry
            phase.new_param_indices = [i for i in phase.new_param_indices
                                       if i not in carry_indices]
            if not phase.new_param_indices:
                print(f"\n  Phase 9: No params to refine (all well-constrained).")
                break

        cr_dict, carry_indices, _ = run_phase(phase, base_values, all_cr, args)

        # Save global state after each phase
        np.save(str(args.oide_dir / "base_values_current.npy"), base_values)
        with open(args.oide_dir / "all_cr.json", 'w') as f:
            json.dump({cfg.STAGE2_PARAM_NAMES[k]: float(v)
                       for k, v in all_cr.items()}, f, indent=2)

    # Final summary
    total_time = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"OIDE Pipeline complete!")
    print(f"  Total time: {total_time/3600:.1f}h ({total_time/86400:.1f} days)")
    print(f"  Base values: {args.oide_dir}/base_values_current.npy")
    print(f"  All CR: {args.oide_dir}/all_cr.json")

    print(f"\n  Final CR summary (best across all phases):")
    print(f"  {'Parameter':22s}  {'CR':>6s}  {'Classification':>14s}")
    print(f"  {'-'*50}")
    for idx in sorted(all_cr.keys()):
        if idx not in TICR_INDICES:
            name = cfg.STAGE2_PARAM_NAMES[idx]
            cr = all_cr[idx]
            print(f"  {name:22s}  {cr:6.3f}  {classify_cr(cr):>14s}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
