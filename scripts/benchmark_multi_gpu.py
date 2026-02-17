#!/usr/bin/env python3
"""Benchmark: run N LUMINA models in parallel on N GPUs to test contention."""
import sys, time, json, os, threading
import numpy as np
sys.path.insert(0, '.')
from lumina_ml.data_utils import LuminaRunner, Stage2Params
from lumina_ml import config as cfg

binary = str(cfg.LUMINA_CUDA)

with open('data/oide_base_values.json') as f:
    base = json.load(f)
arr = np.array([base[name]['median'] for name in cfg.STAGE2_PARAM_NAMES])

n_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f'Multi-GPU benchmark: {n_gpus} GPUs, 200K pkts, 10 iters, NLTE start=5')
print(f'Binary: {binary}')

results = {}
lock = threading.Lock()

def run_on_gpu(gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    runner = LuminaRunner(binary, nlte=True, nlte_start_iter=5)
    p = Stage2Params.from_array(arr)
    t0 = time.time()
    result = runner.run_model(p, n_packets=200000, n_iters=10,
                              tag=f'bench_gpu{gpu_id}', timeout=1800, env=env)
    elapsed = time.time() - t0
    with lock:
        results[gpu_id] = (elapsed, result is not None)
        status = 'OK' if result is not None else 'FAIL'
        print(f'  GPU{gpu_id}: {status} ({elapsed:.1f}s)', flush=True)

# Launch all GPUs in parallel
threads = []
t_start = time.time()
for gpu_id in range(n_gpus):
    t = threading.Thread(target=run_on_gpu, args=(gpu_id,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

total = time.time() - t_start
times = [results[i][0] for i in range(n_gpus)]
print(f'\nSummary ({n_gpus} GPUs):')
print(f'  Min: {min(times):.1f}s, Max: {max(times):.1f}s, Avg: {np.mean(times):.1f}s')
print(f'  Wall clock: {total:.1f}s')
print(f'  Throughput: {n_gpus/total*3600:.1f} models/hr')
