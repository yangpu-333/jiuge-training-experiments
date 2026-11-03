#!/usr/bin/env python3
"""All-element CPU FP64 numerical review against an immutable preregistration."""
import argparse
import gc
import hashlib
import json
import math
import os
from compact_xor_v2 import restore
from pathlib import Path
import sys
import time

sys.modules.setdefault('tensorflow', None)
import torch

BASE = Path('/private/yp/jiuge-training')
STAGE = BASE / 'checks/20261102-adapted-local-cpt/numerical'
CHUNK = 1048576


def finish(stats):
    a2, b2, d2 = stats['sum_a2'], stats['sum_b2'], stats['sum_d2']
    a, b, d = math.sqrt(max(a2, 0)), math.sqrt(max(b2, 0)), math.sqrt(max(d2, 0))
    return dict(stats, reference_l2=a, candidate_l2=b, error_l2=d,
                relative_l2=d / a if a else (0.0 if d == 0 else None),
                cosine=max(-1.0, min(1.0, stats['sum_ab'] / (a * b))) if a and b else (1.0 if a == b == 0 else 0.0))


def empty():
    return {'sum_a2': 0.0, 'sum_b2': 0.0, 'sum_d2': 0.0, 'sum_ab': 0.0,
            'elements': 0, 'unequal_elements': 0, 'max_abs': 0.0, 'finite': True}


def merge(total, item):
    for key in ['sum_a2', 'sum_b2', 'sum_d2', 'sum_ab', 'elements', 'unequal_elements']:
        total[key] += item[key]
    total['max_abs'] = max(total['max_abs'], item['max_abs'])
    total['finite'] = total['finite'] and item['finite']


def measure(a, b, initial=None):
    assert a.shape == b.shape and a.dtype == b.dtype
    if initial is not None:
        assert initial.shape == a.shape and initial.dtype == a.dtype
    result = empty()
    x, y = a.reshape(-1), b.reshape(-1)
    z = initial.reshape(-1) if initial is not None else None
    for i in range(0, x.numel(), CHUNK):
        aa, bb = x[i:i + CHUNK].to(torch.float64), y[i:i + CHUNK].to(torch.float64)
        if z is not None:
            zz = z[i:i + CHUNK].to(torch.float64)
            aa, bb = aa - zz, bb - zz
        delta = bb - aa
        finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
        result['finite'] = result['finite'] and finite
        result['elements'] += aa.numel()
        result['unequal_elements'] += int(torch.count_nonzero(delta))
        if finite:
            result['sum_a2'] += float(torch.sum(aa * aa))
            result['sum_b2'] += float(torch.sum(bb * bb))
            result['sum_d2'] += float(torch.sum(delta * delta))
            result['sum_ab'] += float(torch.sum(aa * bb))
            if delta.numel():
                result['max_abs'] = max(result['max_abs'], float(delta.abs().max()))
    return finish(result)


def vector_gate(metrics, relative, cosine):
    return bool(metrics['finite'] and metrics['relative_l2'] is not None
                and metrics['relative_l2'] <= relative and metrics['cosine'] >= cosine)


def load_snapshot(path):
    value = restore(path)
    assert len(value['schema']) == len(value['values'])
    return value


def compare_snapshots(reference, candidate, stage, initial=False):
    rows, total = [], empty()
    for rank in range(4):
        pa = reference / f'numeric/{stage}-rank{rank}.pt'
        pb = candidate / f'numeric/{stage}-rank{rank}.pt'
        a, b = load_snapshot(pa), load_snapshot(pb)
        canonical = BASE / 'runs' / '4b-cpt-tp2-pp2-reference-no-recompute-s512-gbs16-numeric-v1'
        init = load_snapshot(canonical / f'numeric/master-initial-rank{rank}.pt') if initial else None
        assert len(a['values']) == len(b['values'])
        if init:
            assert len(init['values']) == len(a['values'])
        rank_total = empty()
        for i, (schema_a, schema_b, aa, bb) in enumerate(zip(a['schema'], b['schema'], a['values'], b['values'])):
            assert {k: v for k, v in schema_a.items() if k != 'sha256'} == {k: v for k, v in schema_b.items() if k != 'sha256'}
            if init:
                assert init['schema'][i]['name'] == schema_a['name']
            metric = measure(aa, bb, init['values'][i] if init else None)
            rows.append(dict(rank=rank, name=schema_a['name'], shape=list(aa.shape), **metric))
            merge(total, metric)
            merge(rank_total, metric)
        print('REVIEWED', stage, rank, json.dumps(finish(rank_total)), flush=True)
        del a, b, init
        gc.collect()
    return {'global': finish(total), 'tensors': rows,
            'aggregation': 'Concatenated actual rank-local optimizer tensors; includes TP replicated norms and PP tied embedding copies. Not the deduplicated MCore clipping norm.'}


def compare_model_checkpoints(reference, candidate):
    rows,total=[],empty()
    for rank in range(4):
        a=restore(reference/f'numeric/model-final-rank{rank}.pt')
        b=restore(candidate/f'numeric/model-final-rank{rank}.pt')
        assert a['metadata']['scalars']==b['metadata']['scalars']
        assert a['metadata']['scheduler']==b['metadata']['scheduler']
        assert a['metadata']['consumed_train_samples']==b['metadata']['consumed_train_samples']==160
        assert a['metadata']['iteration']==b['metadata']['iteration']==10
        assert len(a['schema'])==len(b['schema'])
        for sa,sb,aa,bb in zip(a['schema'],b['schema'],a['values'],b['values']):
            assert {k:v for k,v in sa.items() if k!='sha256'}=={k:v for k,v in sb.items() if k!='sha256'}
            item=measure(aa,bb);rows.append(dict(rank=rank,name=sa['name'],dtype=str(aa.dtype),shape=list(aa.shape),**item));merge(total,item)
        del a,b;gc.collect();print('DIRECT_MODELS_REVIEWED',rank,flush=True)
    return {'global':finish(total),'tensors':rows,'scope':'Direct full-element comparison of two separately reconstructed model exports'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['cpt', 'lora'], required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((STAGE / 'protocol.json').read_text())
    protocol_hash = hashlib.sha256((STAGE / 'protocol.json').read_bytes()).hexdigest()
    prereg = json.loads((STAGE / 'preregistration.json').read_text())
    assert protocol_hash == next(f['sha256'] for f in prereg['files'] if f['path'].endswith('/protocol.json'))
    prefix = f'4b-{options.task}-tp2-pp2-'
    suffix = '-no-recompute-s512-gbs16-numeric-v1'
    reference = BASE / 'runs' / (prefix + 'reference' + suffix.replace('numeric-v1', 'local-attention-only-v1'))
    candidate = BASE / 'runs' / (prefix + 'triton' + suffix.replace('numeric-v1', 'norm-dw-products-v1'))
    result_path = STAGE / f'{options.task}-direct-comparison-v2.json'
    assert not result_path.exists(), 'Refusing to overwrite a reviewed result'
    result = {'task': options.task, 'reference': str(reference), 'candidate': str(candidate),
              'protocol_sha256': protocol_hash, 'exact_control_failures': [], 'step_metrics': [],
              'historical_reference_failures': [], 'gradient_reviews': {}, 'passed': False}
    started = time.monotonic()
    tracked = [p for run in [reference, candidate] for p in run.rglob('*') if p.is_file()]
    stat_before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in tracked}

    def fail(kind, **details):
        result['exact_control_failures'].append(dict(kind=kind, **details))

    ca, cb = [json.loads((run / 'config.json').read_text()) for run in [reference, candidate]]
    assert ca['backend']['rmsnorm'] == 'reference' and cb['backend']['rmsnorm'] == 'triton'
    ca, cb = dict(ca), dict(cb)
    ca.pop('backend'); cb.pop('backend')
    ca['task'].pop('save'); cb['task'].pop('save')
    if ca != cb:
        fail('effective training configs differ beyond backend/save')
    old_suffix = 'control10-hybrid-v3' if options.task == 'cpt' else 'control10'
    old = BASE / 'runs' / (prefix + 'reference-no-recompute-s512-gbs16-' + old_suffix)
    for rank in range(4):
        for run in [reference, candidate]:
            assert json.loads((run / f'metrics-rank{rank}.json').read_text())['completed']
        ma, mb = [json.loads((run / f'numeric/model-initial-rank{rank}.json').read_text()) for run in [reference, candidate]]
        if ma != mb:
            fail('initial model mismatch', rank=rank)
        receipts = [json.loads((run / f'numeric/receipts-rank{rank}.json').read_text()) for run in [reference, candidate]]
        schemas = [next(r['schema'] for r in rr if r['stage'] == 'master-initial') for rr in receipts]
        if schemas[0] != schemas[1]:
            fail('initial master mismatch', rank=rank)
        for run, initial in [(reference, ma), (candidate, mb)]:
            final = json.loads((run / f'numeric/model-final-rank{rank}.json').read_text())
            assert set(initial) == set(final)
            if options.task == 'lora':
                changed = [n for n in initial if not initial[n]['requires_grad'] and initial[n] != final[n]]
                if changed:
                    fail('frozen base changed', rank=rank, run=str(run), names=changed)
                assert json.loads((run / f'training-validity-rank{rank}.json').read_text())['passed']
        aa, bb = [[json.loads(s) for s in (run / f'step-audit-rank{rank}.jsonl').read_text().splitlines()] for run in [reference, candidate]]
        historical = [json.loads(s) for s in (old / f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        canonical = [json.loads(s) for s in (BASE / 'runs' / (prefix + 'reference' + suffix) / f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        assert len(aa) == len(bb) == len(historical) == len(canonical) == 10
        for a, b, h, canonical_a in zip(aa, bb, historical, canonical):
            exact = [k for k in a if k not in ('losses', 'grad_norm')]
            changed = [k for k in exact if a[k] != b[k]]
            if changed or a['skipped_iter'] != 0 or b['skipped_iter'] != 0:
                fail('step input/state control mismatch', rank=rank, iteration=a['iteration_zero_based'], fields=changed)
            if canonical_a != h:
                result['historical_reference_failures'].append({'rank': rank, 'iteration': canonical_a['iteration_zero_based'], 'fields': [k for k in canonical_a if canonical_a[k] != h.get(k)]})
            assert set(a['losses']) == set(b['losses'])
            losses = {k: {'reference': a['losses'][k], 'candidate': b['losses'][k],
                          'abs_error': abs(a['losses'][k] - b['losses'][k]),
                          'passed': abs(a['losses'][k] - b['losses'][k]) <= protocol['loss']['atol'] + protocol['loss']['rtol'] * abs(a['losses'][k])}
                      for k in a['losses']}
            gn_a, gn_b = a['grad_norm'], b['grad_norm']
            grad_pass = math.isfinite(gn_a) and math.isfinite(gn_b) and abs(gn_a - gn_b) <= protocol['gradient_norm']['atol'] + protocol['gradient_norm']['rtol'] * abs(gn_a)
            result['step_metrics'].append({'rank': rank, 'step': a['iteration_zero_based'] + 1, 'losses': losses,
                                           'gradient_norm': {'reference': gn_a, 'candidate': gn_b, 'passed': grad_pass},
                                           'passed': grad_pass and all(x['passed'] for x in losses.values())})
    for step in [1, 10]:
        review = compare_snapshots(reference, candidate, f'gradient-step{step}')
        p = protocol['full_gradients']
        review['global_passed'] = vector_gate(review['global'], p['global_relative_l2_max'], p['global_cosine_min'])
        ref_norm = review['global']['reference_l2']
        for row in review['tensors']:
            if row['reference_l2'] >= ref_norm * p['material_reference_norm_fraction_min'] and row['reference_l2'] > 0:
                row['passed'] = vector_gate(row, p['material_tensor_relative_l2_max'], p['material_tensor_cosine_min'])
                row['gate'] = 'material tensor'
            else:
                row['passed'] = row['finite'] and row['error_l2'] <= ref_norm * p['small_tensor_error_norm_fraction_max']
                row['gate'] = 'small tensor'
        review['failed_tensors'] = [{'rank': r['rank'], 'name': r['name'], 'relative_l2': r['relative_l2'], 'cosine': r['cosine']} for r in review['tensors'] if not r['passed']]
        review['passed'] = review['global_passed'] and not review['failed_tensors']
        result['gradient_reviews'][str(step)] = review
        (STAGE / f'{options.task}-direct-comparison-v2-progress.json').write_text(json.dumps(result, indent=2) + '\n')
    result['master_updates'] = compare_snapshots(reference, candidate, 'master-final', initial=True)
    p = protocol['fp32_master_updates']
    result['master_updates']['passed'] = vector_gate(result['master_updates']['global'], p['global_relative_l2_max'], p['global_cosine_min'])
    result['final_model'] = compare_model_checkpoints(reference, candidate)
    unchanged = all(stat_before[str(p)] == [p.stat().st_size, p.stat().st_mtime_ns] for p in tracked)
    assert unchanged, 'Reviewed files changed during CPU review'
    result['files_unchanged_during_review'] = unchanged
    result['passed'] = (not result['exact_control_failures'] and not result['historical_reference_failures']
                        and all(s['passed'] for s in result['step_metrics'])
                        and all(g['passed'] for g in result['gradient_reviews'].values())
                        and result['master_updates']['passed'] and result['final_model']['global']['finite'])
    result['elapsed_cpu_review_seconds'] = time.monotonic() - started
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print('NUMERIC_REVIEW_COMPLETE', options.task, json.dumps({
        'passed': result['passed'], 'exact_control_failures': result['exact_control_failures'],
        'historical_reference_failures': result['historical_reference_failures'],
        'failed_steps': [s for s in result['step_metrics'] if not s['passed']],
        'gradients': {k: {'global': v['global'], 'failed_tensors': v['failed_tensors']} for k, v in result['gradient_reviews'].items()},
        'updates': result['master_updates']['global'], 'model': result['final_model']['global'],
    }), flush=True)


if __name__ == '__main__':
    main()

