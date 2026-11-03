#!/usr/bin/env python3
"""External full-model backend numerical audit; pinned training sources unchanged."""
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
BASE = Path('/private/yp/jiuge-training')


def driver():
    original = subprocess.call

    def dispatch(command, *args, **kwargs):
        command = list(command)
        for i in range(len(command) - 1):
            if command[i:i + 2] == ['-m', 'jiuge_training.train']:
                config_path = Path(command[command.index('--jiuge-config') + 1]).resolve()
                assert config_path.is_relative_to(BASE)
                actual = command[:i] + [str(SCRIPT), '--numeric-worker'] + command[i + 2:]
                (config_path.parent / 'numeric-command.txt').write_text(shlex.join(actual) + '\n')
                (config_path.parent / 'numeric-launcher.json').write_text(json.dumps({
                    'script': str(SCRIPT), 'sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                    'source_modifications': False, 'performance_measurement': False,
                    'candidate': 'split-v4 mixed: native Torch statistics and row VJP reductions; Triton forward/pointwise split dx/dw; separate x and x.float autograd edges',
                    'patch_sha256': hashlib.sha256((SCRIPT.parent / 'split_graph_patch.py').read_bytes()).hexdigest(),
                    'oracle_sha256': hashlib.sha256((SCRIPT.parent / 'oracle_rmsnorm_forward.py').read_bytes()).hexdigest(),
                    'captures': ['all initial model/master hashes', 'actual inputs and RNG every step',
                                 'full preclip master gradients steps 1/10', 'full FP32 master final state',
                                 'native final checkpoint'],
                }, indent=2) + '\n')
                return original(actual, *args, **kwargs)
        return original(command, *args, **kwargs)

    subprocess.call = dispatch
    try:
        runpy.run_module('jiuge_training.cli', run_name='__main__')
    finally:
        subprocess.call = original


def worker():
    sys.argv.remove('--numeric-worker')
    baseline = importlib.import_module('jiuge_training.train')
    original_cache = baseline.configure_worker_caches
    original_cache()
    baseline.configure_worker_caches = lambda: None
    import numpy as np
    import torch
    from megatron.core import tensor_parallel
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from jiuge_ops import ReferenceRMSNorm, TritonRMSNorm
    from split_graph_patch import install
    install()
    mcore = importlib.import_module('megatron.training.training')
    config_path = Path(sys.argv[sys.argv.index('--jiuge-config') + 1]).resolve()
    config = json.loads(config_path.read_text())
    assert config['task']['recompute'] is False
    assert config['task']['train_iters'] == 10
    rank = int(os.environ['RANK'])
    root = config_path.parent / 'numeric'
    root.mkdir(exist_ok=True)
    output = config_path.parent / f'step-audit-rank{rank}.jsonl'
    receipts = []
    current = None
    prepared_calls = 0
    norm_calls = {}

    def tensor_hash(value):
        value = value.detach().contiguous().cpu()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}

    def rng():
        return {
            'python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
            'torch_cpu': tensor_hash(torch.get_rng_state()),
            'torch_cuda': tensor_hash(torch.cuda.get_rng_state()),
            'mcore_tracker': {k: tensor_hash(v) for k, v in sorted(get_cuda_rng_tracker().get_states().items())},
        }

    def named_model(models):
        return [(f'chunk{i}.{n}', p) for i, model in enumerate(models) for n, p in model.named_parameters()]

    def main_items(optimizer, models):
        names = {id(p): name for name, p in named_model(models)}
        mapping = {}

        def visit(opt):
            if hasattr(opt, 'chained_optimizers'):
                for child in opt.chained_optimizers:
                    visit(child)
                return
            if hasattr(opt, 'model_float16_groups'):
                pairs = [(opt.model_float16_groups, opt.shard_fp32_from_float16_groups),
                         (opt.model_fp32_groups, opt.shard_fp32_groups)]
            elif hasattr(opt, 'float16_groups'):
                pairs = [(opt.float16_groups, opt.fp32_from_float16_groups)]
                for group in opt.fp32_from_fp32_groups:
                    for p in group:
                        mapping[id(p)] = names[id(p)]
            else:
                raise RuntimeError('Unreviewed optimizer type: ' + type(opt).__name__)
            for model_groups, main_groups in pairs:
                assert len(model_groups) == len(main_groups)
                for mg, pg in zip(model_groups, main_groups):
                    assert len(mg) == len(pg)
                    for model, main in zip(mg, pg):
                        assert model.numel() == main.numel(), 'Expected DP1 complete local TP shard'
                        assert id(main) not in mapping
                        mapping[id(main)] = names[id(model)]

        visit(optimizer)
        params = optimizer.get_parameters()
        assert len(params) == len(mapping) and len({id(p) for p in params}) == len(params)
        assert sum(p.numel() for p in params) == sum(p.numel() for _, p in named_model(models) if p.requires_grad)
        return [(mapping[id(p)], p) for p in params]

    def model_hash(models, stage):
        record = {name: dict(tensor_hash(p), requires_grad=p.requires_grad) for name, p in named_model(models)}
        (root / f'model-{stage}-rank{rank}.json').write_text(json.dumps(record, indent=2) + '\n')

    def snapshot(items, stage, gradient=False, save=True):
        values, schema = [], []
        for name, p in items:
            value = p.grad if gradient else p
            if value is None:
                raise RuntimeError('Missing actual master gradient for ' + name)
            cpu = value.detach().to(device='cpu', copy=True).contiguous()
            assert cpu.dtype == torch.float32, (name, cpu.dtype)
            assert bool(torch.isfinite(cpu).all()), (name, stage)
            schema.append(dict(name=name, **tensor_hash(cpu)))
            values.append(cpu)
        path = root / f'{stage}-rank{rank}.pt'
        if save:
            assert not path.exists()
            torch.save({'schema': schema, 'values': values}, path)
        receipt = {'stage': stage, 'rank': rank, 'schema': schema,
                   'elements': sum(v.numel() for v in values), 'saved': save,
                   'path': str(path) if save else None}
        receipts.append(receipt)
        (root / f'receipts-rank{rank}.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print('NUMERIC_SNAPSHOT', rank, stage, receipt['elements'], flush=True)

    original_step, original_broadcast = mcore.train_step, tensor_parallel.broadcast_data

    def broadcast(*args, **kwargs):
        result = original_broadcast(*args, **kwargs)
        if current is not None:
            current['batch_broadcasts'].append({k: tensor_hash(v) for k, v in sorted(result.items())})
        return result

    @functools.wraps(original_step)
    def step(*args, **kwargs):
        nonlocal current, prepared_calls
        bound = __import__('inspect').signature(original_step).bind(*args, **kwargs)
        iteration = bound.arguments['iteration']
        models, optimizer = bound.arguments['model'], bound.arguments['optimizer']
        items = main_items(optimizer, models)
        if iteration == 0:
            expected_class = ReferenceRMSNorm if config['backend']['rmsnorm'] == 'reference' else TritonRMSNorm
            norms = [(f'chunk{i}.{n}', m) for i, model in enumerate(models) for n, m in model.named_modules()
                     if isinstance(m, ReferenceRMSNorm)]
            assert norms and all(type(m) is expected_class for _, m in norms)
            for name, norm in norms:
                norm_calls[name] = 0

                def record(module, inputs, name=name):
                    assert list(inputs[0].shape) == [512, 1, 2560]
                    norm_calls[name] += 1

                norm.register_forward_pre_hook(record)
            model_hash(models, 'initial')
            # Initial FP32 masters are hashed as in the original audit; the fixed
            # reference snapshot already exists, so do not duplicate it here.
            snapshot(items, 'master-initial', save=False)
        before = rng()
        current = {'rank': rank, 'iteration_zero_based': iteration,
                   'consumed_train_samples_before': mcore.get_args().consumed_train_samples,
                   'rng_before': before, 'batch_broadcasts': []}
        prepare = optimizer.prepare_grads

        def prepare_with_capture(*pa, **pk):
            nonlocal prepared_calls
            result = prepare(*pa, **pk)
            prepared_calls += 1
            if iteration in (0, 9):
                snapshot(items, f'gradient-step{iteration + 1}', gradient=True)
            return result

        optimizer.prepare_grads = prepare_with_capture
        try:
            result = original_step(*args, **kwargs)
            current.update(losses={k: float(v.detach().item()) for k, v in result[0].items()},
                           skipped_iter=int(result[1]), grad_norm=None if result[5] is None else float(result[5]),
                           rng_after=rng())
            with output.open('a') as stream:
                stream.write(json.dumps(current, sort_keys=True) + '\n')
            if iteration == 9:
                snapshot(items, 'master-final')
                model_hash(models, 'final')
                assert prepared_calls == 10
                assert all(v == 160 for v in norm_calls.values()), norm_calls
                (root / f'coverage-rank{rank}.json').write_text(json.dumps({
                    'rank': rank, 'backend': config['backend']['rmsnorm'],
                    'optimizer_class': type(optimizer).__name__, 'prepared_calls': prepared_calls,
                    'master_tensors': len(items), 'master_elements': sum(p.numel() for _, p in items),
                    'rmsnorm_modules': len(norm_calls), 'rmsnorm_forward_calls': norm_calls,
                    'full_gradient_steps': [1, 10], 'gradient_location': 'after outer prepare_grads, before norm/clipping',
                }, indent=2) + '\n')
            return result
        finally:
            optimizer.prepare_grads = prepare
            current = None

    tensor_parallel.broadcast_data, mcore.train_step = broadcast, step
    try:
        baseline.main()
    finally:
        tensor_parallel.broadcast_data, mcore.train_step = original_broadcast, original_step
        baseline.configure_worker_caches = original_cache


if __name__ == '__main__':
    worker() if '--numeric-worker' in sys.argv else driver()
