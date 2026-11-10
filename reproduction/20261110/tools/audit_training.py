#!/usr/bin/env python3
"""External input, RNG and full precision train_step audit for fixed sources."""
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import shlex
import subprocess
import sys

SCRIPT = Path(__file__).resolve()


def driver():
    original = subprocess.call

    def dispatch(command, *args, **kwargs):
        command = list(command)
        for i in range(len(command) - 1):
            if command[i:i + 2] != ['-m', 'jiuge_training.train']:
                continue
            config_path = Path(command[command.index('--jiuge-config') + 1]).resolve()
            if not config_path.is_relative_to(Path('/private/yp')):
                raise ValueError('The experiment must remain under /private/yp')
            config = json.loads(config_path.read_text())
            assert config['task']['recompute'] is False
            actual = command[:i] + [str(SCRIPT), '--audit-worker'] + command[i + 2:]
            (config_path.parent / 'audit-command.txt').write_text(shlex.join(actual) + '\n')
            (config_path.parent / 'audit-launcher.json').write_text(json.dumps({
                'script': str(SCRIPT), 'sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                'records': ['actual batch hashes', 'RNG before/after', 'full precision loss', 'gradient norm', 'skipped iteration'],
                'profiling': False, 'timing_scope': 'Audit introduces synchronization; not a performance measurement',
                'empty_cache_before_step': os.environ.get('AUDIT_EMPTY_CACHE_BEFORE_STEP') == '1',
                'preserve_restored_cpu_rng_during_iterator_rebuild': os.environ.get('AUDIT_PRESERVE_RESUME_DATALOADER_RNG') == '1',
                'trusted_checkpoint_load_override': os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD') == '1',
            }, indent=2) + '\n')
            return original(actual, *args, **kwargs)
        return original(command, *args, **kwargs)

    subprocess.call = dispatch
    try:
        runpy.run_module('jiuge_training.cli', run_name='__main__')
    finally:
        subprocess.call = original


def worker():
    sys.argv.remove('--audit-worker')
    baseline = importlib.import_module('jiuge_training.train')
    original_cache = baseline.configure_worker_caches
    original_cache()
    baseline.configure_worker_caches = lambda: None
    import numpy as np
    import torch
    from megatron.core import tensor_parallel
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    mcore = importlib.import_module('megatron.training.training')
    config_path = Path(sys.argv[sys.argv.index('--jiuge-config') + 1]).resolve()
    output = config_path.parent / f'step-audit-rank{os.environ["RANK"]}.jsonl'
    original_step, original_broadcast = mcore.train_step, tensor_parallel.broadcast_data
    original_iterators = mcore.build_train_valid_test_data_iterators
    config = json.loads(config_path.read_text())
    current = None

    def tensor_hash(value):
        value = value.detach().contiguous().cpu()
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}

    def rng():
        return {
            'python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
            'torch_cpu': tensor_hash(torch.get_rng_state()),
            'torch_cuda': tensor_hash(torch.cuda.get_rng_state()),
            'mcore_tracker': {k: tensor_hash(v) for k, v in sorted(get_cuda_rng_tracker().get_states().items())},
        }

    def rebuild_iterators(*args, **kwargs):
        enabled = os.environ.get('AUDIT_PRESERVE_RESUME_DATALOADER_RNG') == '1'
        if not enabled or not config['task'].get('load'):
            return original_iterators(*args, **kwargs)
        settings = mcore.get_args()
        if settings.num_workers != 0 or settings.dataloader_type != 'single' or settings.eval_iters != 0:
            raise RuntimeError('The RNG repair is limited to the audited deterministic num_workers=0 single iterator')
        state = torch.get_rng_state()
        try:
            return original_iterators(*args, **kwargs)
        finally:
            after = torch.get_rng_state()
            torch.set_rng_state(state)
            receipt = {'rank': int(os.environ['RANK']), 'before': tensor_hash(state),
                       'after_iterator_construction': tensor_hash(after), 'restored': tensor_hash(torch.get_rng_state()),
                       'scope': 'Native resume only; workerless deterministic samples; preserve loaded CPU RNG across rebuilding iterators.'}
            assert receipt['before'] == receipt['restored']
            (config_path.parent / f'resume-iterator-rng-rank{os.environ["RANK"]}.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def broadcast(*args, **kwargs):
        result = original_broadcast(*args, **kwargs)
        if current is not None:
            current['batch_broadcasts'].append({key: tensor_hash(value) for key, value in sorted(result.items())})
        return result

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        nonlocal current
        iteration = kwargs.get('iteration')
        if iteration is None:
            raise RuntimeError('This pinned train_step must provide iteration')
        if os.environ.get('AUDIT_EMPTY_CACHE_BEFORE_STEP') == '1':
            torch.cuda.empty_cache()
        current = {'rank': int(os.environ['RANK']), 'iteration_zero_based': iteration,
                   'consumed_train_samples_before': mcore.get_args().consumed_train_samples,
                   'rng_before': rng(), 'batch_broadcasts': []}
        try:
            result = original_step(*args, **kwargs)
            current.update(losses={k: float(v.detach().item()) for k, v in result[0].items()},
                           skipped_iter=int(result[1]), grad_norm=None if result[5] is None else float(result[5]),
                           rng_after=rng())
            with output.open('a') as stream:
                stream.write(json.dumps(current, sort_keys=True) + '\n')
            return result
        finally:
            current = None

    tensor_parallel.broadcast_data, mcore.train_step = broadcast, step
    mcore.build_train_valid_test_data_iterators = rebuild_iterators
    try:
        baseline.main()
    finally:
        tensor_parallel.broadcast_data, mcore.train_step = original_broadcast, original_step
        mcore.build_train_valid_test_data_iterators = original_iterators
        baseline.configure_worker_caches = original_cache


if __name__ == '__main__':
    worker() if '--audit-worker' in sys.argv else driver()
