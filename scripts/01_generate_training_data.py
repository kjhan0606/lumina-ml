#!/usr/bin/env python3
"""Generate training data for the spectral emulator.

Runs LUMINA models across a 15D Latin Hypercube to produce (params, spectrum) pairs.
Supports parallel execution via CUDA (multi-GPU), CPU (multi-worker OpenMP), or both.

All workers (GPU and CPU) pull from a single shared work queue for automatic load
balancing — whichever device finishes first picks up the next model.

Usage:
  python3 scripts/01_generate_training_data.py                          # auto-detect
  python3 scripts/01_generate_training_data.py --mode both              # all GPUs + CPU
  python3 scripts/01_generate_training_data.py --mode both --n-gpus 2 --n-cpu-workers 3
  python3 scripts/01_generate_training_data.py --mode cpu --n-cpu-workers 4 --omp-threads 64
  python3 scripts/01_generate_training_data.py --resume                 # Continue from checkpoint
"""

import argparse
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
from lumina_ml.data_utils import LuminaRunner, ModelParams, latin_hypercube


def detect_gpu_count():
    """Detect number of NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split('\n')
                     if l.startswith('GPU')]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def make_gpu_env(gpu_id):
    """Create environment dict with CUDA_VISIBLE_DEVICES set for one GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return env


def make_cpu_env(omp_threads):
    """Create environment dict with OMP_NUM_THREADS set."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(omp_threads)
    return env


class SharedState:
    """Thread-safe shared state with dynamic work queue for multi-device execution.

    All remaining model indices are placed in a shared queue. Each worker
    (GPU or CPU) pulls the next model when it finishes, ensuring perfect
    load balancing regardless of per-model runtime variation.
    """
    def __init__(self, n_total, remaining_indices):
        self.lock = threading.Lock()
        self.spectra_dict = {}
        self.completed_indices = set()
        self.n_success = 0
        self.n_fail = 0
        self.n_total = n_total
        # Per-device counters: key = device label, value = count
        self.device_counts = {}
        # Shared work queue: all remaining models
        self.work_queue = queue.Queue()
        for idx in remaining_indices:
            self.work_queue.put(idx)

    def get_next(self):
        """Get next model index from queue, or None if empty."""
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

    def device_summary(self):
        """Return formatted string of per-device counts."""
        parts = []
        for label in sorted(self.device_counts.keys()):
            parts.append(f"{label}:{self.device_counts[label]}")
        return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate LUMINA training data")
    parser.add_argument('--n-models', type=int, default=cfg.DEFAULT_N_MODELS,
                        help=f'Number of models (default: {cfg.DEFAULT_N_MODELS})')
    parser.add_argument('--n-packets', type=int, default=cfg.DEFAULT_N_PACKETS,
                        help=f'Packets per model (default: {cfg.DEFAULT_N_PACKETS})')
    parser.add_argument('--n-iters', type=int, default=cfg.DEFAULT_N_ITERS,
                        help=f'Iterations per model (default: {cfg.DEFAULT_N_ITERS})')
    parser.add_argument('--mode', choices=['cuda', 'cpu', 'both', 'auto'], default='auto',
                        help='Execution mode (default: auto)')
    parser.add_argument('--n-gpus', type=int, default=0,
                        help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--n-cpu-workers', type=int, default=1,
                        help='Number of CPU worker threads (default: 1)')
    parser.add_argument('--omp-threads', type=int, default=64,
                        help='Total OMP threads for CPU, divided among workers (default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for LHS (default: 42)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()

    # Check available binaries
    have_cuda = cfg.LUMINA_CUDA.exists()
    have_cpu = cfg.LUMINA_CPU.exists()

    # Auto-detect GPU count
    n_gpus_available = detect_gpu_count() if have_cuda else 0
    if args.n_gpus <= 0:
        args.n_gpus = max(n_gpus_available, 1) if have_cuda else 0

    if args.mode == 'auto':
        if have_cuda and have_cpu:
            args.mode = 'both'
        elif have_cuda:
            args.mode = 'cuda'
        elif have_cpu:
            args.mode = 'cpu'
        else:
            print("ERROR: No LUMINA binary found")
            sys.exit(1)

    if args.mode in ('both', 'cuda') and not have_cuda:
        print(f"ERROR: --mode {args.mode} requires CUDA binary")
        print(f"  CUDA: {cfg.LUMINA_CUDA} (MISSING)")
        sys.exit(1)
    if args.mode in ('both', 'cpu') and not have_cpu:
        print(f"ERROR: --mode {args.mode} requires CPU binary")
        print(f"  CPU:  {cfg.LUMINA_CPU} (MISSING)")
        sys.exit(1)

    # Compute OMP threads per CPU worker
    threads_per_cpu = max(1, args.omp_threads // args.n_cpu_workers)

    print(f"Mode: {args.mode}")
    print(f"Models: {args.n_models}, Packets: {args.n_packets}, Iters: {args.n_iters}")
    if args.mode in ('both', 'cuda'):
        print(f"GPUs: {args.n_gpus} (detected: {n_gpus_available})")
    if args.mode in ('both', 'cpu'):
        print(f"CPU workers: {args.n_cpu_workers}, OMP threads/worker: {threads_per_cpu}")

    # Output paths
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    params_file = cfg.DATA_RAW / "params_all.npy"
    spectra_file = cfg.DATA_RAW / "spectra_all.npy"
    waves_file = cfg.DATA_RAW / "waves_all.npy"
    checkpoint_file = cfg.DATA_RAW / "checkpoint.npz"

    # Generate LHS samples
    print(f"\nGenerating {args.n_models} Latin Hypercube samples in {cfg.N_PARAMS}D...")
    rng = np.random.default_rng(args.seed)
    all_params = latin_hypercube(args.n_models, rng=rng)
    print(f"  Valid samples: {len(all_params)}")

    params_array = np.array([p.to_array() for p in all_params])

    # Resume from checkpoint?
    completed_indices = set()
    spectra_dict = {}

    if args.resume and checkpoint_file.exists():
        ckpt = np.load(str(checkpoint_file), allow_pickle=True)
        completed_indices = set(ckpt['completed_indices'].tolist())
        if spectra_file.exists() and waves_file.exists():
            prev_spectra = np.load(str(spectra_file))
            prev_waves = np.load(str(waves_file))
            for i in completed_indices:
                if i < len(prev_spectra):
                    spectra_dict[i] = (prev_waves[i], prev_spectra[i])
        print(f"  Resuming: {len(completed_indices)} completed")

    remaining = [i for i in range(len(all_params)) if i not in completed_indices]
    print(f"  Remaining: {len(remaining)} models")

    if not remaining:
        print("  All models already completed!")
        return

    t0 = time.time()

    # ===== Shared worker function =====
    def device_worker(state, device_name, runner, env):
        """Generic worker: pull from shared queue until empty."""
        batch_local = 0
        while True:
            idx = state.get_next()
            if idx is None:
                break
            params = all_params[idx]
            t1 = time.time()
            tag = f"{device_name.lower().replace(' ', '')}_{idx}"
            result = runner.run_model(params, args.n_packets, args.n_iters,
                                      tag=tag, env=env)
            dt = time.time() - t1
            if result is not None:
                wave, flux = result
                msg = state.add_result(idx, wave, flux, device_name)
            else:
                msg = state.add_result(idx, None, None, device_name)
            print(f"{msg} ({dt:.1f}s)", flush=True)
            batch_local += 1
            # Checkpoint (use lock to avoid concurrent writes)
            with state.lock:
                total_done = state.n_success + state.n_fail - len(completed_indices)
                if total_done > 0 and total_done % cfg.BATCH_SAVE_INTERVAL == 0:
                    _save_checkpoint(params_array, state.spectra_dict,
                                     state.completed_indices,
                                     params_file, spectra_file,
                                     waves_file, checkpoint_file)
                    elapsed = time.time() - t0
                    rate = total_done / elapsed
                    eta = (len(remaining) - total_done) / rate if rate > 0 else 0
                    print(f"  -- Checkpoint. Rate: {rate:.2f}/s, ETA: {eta/3600:.1f}h"
                          f" [{state.device_summary()}] --", flush=True)

    # ===== Launch workers based on mode =====
    use_cuda = args.mode in ('both', 'cuda')
    use_cpu = args.mode in ('both', 'cpu')

    state = SharedState(len(all_params), remaining)
    state.spectra_dict = spectra_dict
    state.completed_indices = completed_indices
    state.n_success = len(completed_indices)

    threads = []

    # GPU workers
    if use_cuda:
        n_gpu_workers = args.n_gpus
        cuda_runner = LuminaRunner(binary=cfg.LUMINA_CUDA)
        for gpu_id in range(n_gpu_workers):
            env = make_gpu_env(gpu_id)
            label = f"GPU{gpu_id}" if n_gpu_workers > 1 else "CUDA"
            t = threading.Thread(target=device_worker,
                                 args=(state, label, cuda_runner, env),
                                 name=f"gpu-{gpu_id}")
            threads.append(t)

    # CPU workers
    if use_cpu:
        cpu_runner = LuminaRunner(binary=cfg.LUMINA_CPU)
        for cpu_id in range(args.n_cpu_workers):
            env = make_cpu_env(threads_per_cpu)
            label = f"CPU{cpu_id}" if args.n_cpu_workers > 1 else "CPU"
            t = threading.Thread(target=device_worker,
                                 args=(state, label, cpu_runner, env),
                                 name=f"cpu-{cpu_id}")
            threads.append(t)

    n_total_workers = len(threads)
    print(f"\n  Launching {n_total_workers} workers: "
          f"{args.n_gpus if use_cuda else 0} GPU + "
          f"{args.n_cpu_workers if use_cpu else 0} CPU")
    print(f"  Dynamic work queue: {len(remaining)} models")

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Summary
    total_done = state.n_success + state.n_fail - len(completed_indices)
    if total_done > 0:
        print(f"  Final split: {state.device_summary()}")
        total_device = sum(state.device_counts.values())
        if total_device > 0:
            for label in sorted(state.device_counts.keys()):
                cnt = state.device_counts[label]
                print(f"    {label}: {cnt} ({cnt/total_device*100:.0f}%)")

    spectra_dict = state.spectra_dict
    completed_indices = state.completed_indices
    n_success = state.n_success
    n_fail = state.n_fail

    # Final save
    _save_checkpoint(params_array, spectra_dict, completed_indices,
                     params_file, spectra_file, waves_file, checkpoint_file)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Data generation complete:")
    print(f"  Success: {n_success}/{len(all_params)}")
    print(f"  Failed:  {n_fail}")
    print(f"  Time:    {total_time:.0f}s ({total_time/3600:.1f}h)")
    if total_time > 0:
        print(f"  Rate:    {(n_success - len(completed_indices))/total_time:.2f} models/s")
    print(f"\nOutput files:")
    print(f"  {params_file}")
    print(f"  {spectra_file}")
    print(f"  {waves_file}")


def _save_checkpoint(params_array, spectra_dict, completed_indices,
                     params_file, spectra_file, waves_file, checkpoint_file):
    """Save current state to disk."""
    if not spectra_dict:
        return

    sorted_indices = sorted(spectra_dict.keys())
    first_wave, first_flux = spectra_dict[sorted_indices[0]]
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
    np.savez(str(checkpoint_file), completed_indices=np.array(list(completed_indices)))


if __name__ == '__main__':
    main()
