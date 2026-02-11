#!/usr/bin/env python3
"""Generate training data for the spectral emulator.

Runs LUMINA models across a 15D Latin Hypercube to produce (params, spectrum) pairs.
Supports parallel execution via CUDA (serial GPU), CPU (OpenMP), or both simultaneously.

In 'both' mode, a shared work queue dynamically distributes models between GPU and CPU
workers — whichever device finishes first picks up the next model automatically.

Usage:
  python3 scripts/01_generate_training_data.py                          # auto-detect
  python3 scripts/01_generate_training_data.py --mode both              # GPU + CPU simultaneously
  python3 scripts/01_generate_training_data.py --mode cpu --omp-threads 64
  python3 scripts/01_generate_training_data.py --resume                 # Continue from checkpoint
"""

import argparse
import os
import queue
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lumina_ml import config as cfg
from lumina_ml.data_utils import LuminaRunner, ModelParams, latin_hypercube


class SharedState:
    """Thread-safe shared state with dynamic work queue for CUDA+CPU dual execution.

    Instead of pre-splitting models between GPU and CPU, all remaining model indices
    are placed in a shared queue. Each worker (GPU or CPU) pulls the next model from
    the queue when it finishes, ensuring perfect load balancing regardless of per-model
    runtime variation.
    """
    def __init__(self, n_total, remaining_indices):
        self.lock = threading.Lock()
        self.spectra_dict = {}
        self.completed_indices = set()
        self.n_success = 0
        self.n_fail = 0
        self.n_total = n_total
        self.n_cuda = 0
        self.n_cpu = 0
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
            if source.strip() == "CUDA":
                self.n_cuda += 1
            else:
                self.n_cpu += 1
            if wave is not None:
                self.spectra_dict[idx] = (wave, flux)
                self.completed_indices.add(idx)
                self.n_success += 1
                return f"  [{self.n_success}/{self.n_total}] {source} Model {idx}: OK"
            else:
                self.n_fail += 1
                return f"  [{self.n_success}/{self.n_total}] {source} Model {idx}: FAILED"


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
    parser.add_argument('--omp-threads', type=int, default=64,
                        help='OMP threads for CPU (default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for LHS (default: 42)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()

    # Check available binaries
    have_cuda = cfg.LUMINA_CUDA.exists()
    have_cpu = cfg.LUMINA_CPU.exists()

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

    if args.mode == 'both' and (not have_cuda or not have_cpu):
        print(f"ERROR: --mode both requires both binaries")
        print(f"  CUDA: {cfg.LUMINA_CUDA} ({'found' if have_cuda else 'MISSING'})")
        print(f"  CPU:  {cfg.LUMINA_CPU} ({'found' if have_cpu else 'MISSING'})")
        sys.exit(1)

    print(f"Mode: {args.mode}")
    print(f"Models: {args.n_models}, Packets: {args.n_packets}, Iters: {args.n_iters}")

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

    # ===== BOTH mode: CUDA + CPU with dynamic work queue =====
    if args.mode == 'both':
        print(f"\n  BOTH mode: dynamic work queue ({len(remaining)} models)")
        print(f"  GPU and CPU pull from shared queue — no static split")
        print(f"  CPU: OMP_NUM_THREADS={args.omp_threads}")

        state = SharedState(len(all_params), remaining)
        state.spectra_dict = spectra_dict
        state.completed_indices = completed_indices
        state.n_success = len(completed_indices)

        batch_count = [0]  # mutable for closure

        def device_worker(device_name, runner):
            """Generic worker: pull from shared queue until empty."""
            while True:
                idx = state.get_next()
                if idx is None:
                    break
                params = all_params[idx]
                t1 = time.time()
                tag = f"{device_name.lower().strip()}_{idx}"
                result = runner.run_model(params, args.n_packets, args.n_iters, tag=tag)
                dt = time.time() - t1
                if result is not None:
                    wave, flux = result
                    msg = state.add_result(idx, wave, flux, device_name)
                else:
                    msg = state.add_result(idx, None, None, device_name)
                print(f"{msg} ({dt:.1f}s)", flush=True)
                batch_count[0] += 1
                if batch_count[0] % cfg.BATCH_SAVE_INTERVAL == 0:
                    with state.lock:
                        _save_checkpoint(params_array, state.spectra_dict, state.completed_indices,
                                         params_file, spectra_file, waves_file, checkpoint_file)
                    elapsed = time.time() - t0
                    rate = batch_count[0] / elapsed
                    eta = (len(remaining) - batch_count[0]) / rate if rate > 0 else 0
                    print(f"  -- Checkpoint. Rate: {rate:.2f}/s, ETA: {eta/3600:.1f}h"
                          f" [GPU:{state.n_cuda} CPU:{state.n_cpu}] --", flush=True)

        cuda_runner = LuminaRunner(binary=cfg.LUMINA_CUDA)
        os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
        cpu_runner = LuminaRunner(binary=cfg.LUMINA_CPU)

        cuda_thread = threading.Thread(target=device_worker, args=("CUDA", cuda_runner))
        cpu_thread = threading.Thread(target=device_worker, args=("CPU ", cpu_runner))

        cuda_thread.start()
        cpu_thread.start()

        cuda_thread.join()
        cpu_thread.join()

        print(f"  Final split: GPU={state.n_cuda}, CPU={state.n_cpu}"
              f" ({state.n_cuda/(state.n_cuda+state.n_cpu)*100:.0f}%/{state.n_cpu/(state.n_cuda+state.n_cpu)*100:.0f}%)")

        spectra_dict = state.spectra_dict
        completed_indices = state.completed_indices
        n_success = state.n_success
        n_fail = state.n_fail

    # ===== CUDA only =====
    elif args.mode == 'cuda':
        print(f"\n  CUDA mode: serial execution")
        runner = LuminaRunner(binary=cfg.LUMINA_CUDA)
        n_success = len(completed_indices)
        n_fail = 0
        batch_count = 0

        for idx in remaining:
            params = all_params[idx]
            t1 = time.time()
            result = runner.run_model(params, args.n_packets, args.n_iters, tag=f"gen_{idx}")
            elapsed = time.time() - t1
            if result is not None:
                wave, flux = result
                spectra_dict[idx] = (wave, flux)
                completed_indices.add(idx)
                n_success += 1
                print(f"  [{n_success}/{len(all_params)}] Model {idx}: OK ({elapsed:.1f}s)", flush=True)
            else:
                n_fail += 1
                print(f"  [{n_success}/{len(all_params)}] Model {idx}: FAILED ({elapsed:.1f}s)", flush=True)
            batch_count += 1
            if batch_count % cfg.BATCH_SAVE_INTERVAL == 0:
                _save_checkpoint(params_array, spectra_dict, completed_indices,
                                 params_file, spectra_file, waves_file, checkpoint_file)
                eta = (time.time() - t0) / batch_count * (len(remaining) - batch_count)
                print(f"  -- Checkpoint. ETA: {eta/3600:.1f}h --", flush=True)

    # ===== CPU only =====
    elif args.mode == 'cpu':
        os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)
        print(f"\n  CPU mode: OMP_NUM_THREADS={args.omp_threads}")
        runner = LuminaRunner(binary=cfg.LUMINA_CPU)
        n_success = len(completed_indices)
        n_fail = 0
        batch_count = 0

        for idx in remaining:
            params = all_params[idx]
            t1 = time.time()
            result = runner.run_model(params, args.n_packets, args.n_iters, tag=f"gen_{idx}")
            elapsed = time.time() - t1
            if result is not None:
                wave, flux = result
                spectra_dict[idx] = (wave, flux)
                completed_indices.add(idx)
                n_success += 1
                print(f"  [{n_success}/{len(all_params)}] Model {idx}: OK ({elapsed:.1f}s)", flush=True)
            else:
                n_fail += 1
                print(f"  [{n_success}/{len(all_params)}] Model {idx}: FAILED ({elapsed:.1f}s)", flush=True)
            batch_count += 1
            if batch_count % cfg.BATCH_SAVE_INTERVAL == 0:
                _save_checkpoint(params_array, spectra_dict, completed_indices,
                                 params_file, spectra_file, waves_file, checkpoint_file)
                eta = (time.time() - t0) / batch_count * (len(remaining) - batch_count)
                print(f"  -- Checkpoint. ETA: {eta/3600:.1f}h --", flush=True)

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
        print(f"  Rate:    {n_success/total_time:.2f} models/s")
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
