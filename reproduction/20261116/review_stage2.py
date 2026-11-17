"""Offline analysis of persistent E2E records. No Torch, GPU or network calls."""
from pathlib import Path
import collections,datetime,hashlib,json,re,statistics,sys
import random,math
root=Path(sys.argv[1]);allow_partial='--partial' in sys.argv
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
runs=[]
for p in sorted(root.glob('performance-*/*/*/result.json')):
    folder=p.parent
    required=[folder/'stdout.log']+[folder/f'{name}-rank{rank}.json' for name in ['packing-runtime','metrics'] for rank in range(8)]
    if not all(x.exists() for x in required):
        if allow_partial:continue
        raise AssertionError(f'Incomplete download:{folder}')
    d=read(p);log=(folder/'stdout.log').read_text(encoding='utf-8')
    iteration=[float(x)/1000 for x in re.findall(r'elapsed time per iteration \(ms\):\s*([\d.]+)',log)]
    assert len(iteration)==50
    timers=re.findall(r'save-checkpoint\s*\.+:\s*\(([\d.]+),\s*([\d.]+)\)',log)
    assert len(timers)==1,(folder,timers)
    save=float(timers[0][1])/1000
    rt=[read(folder/f'packing-runtime-rank{rank}.json') for rank in range(8)]
    metrics=[read(folder/f'metrics-rank{rank}.json') for rank in range(8)]
    assert all(x['completed'] for x in metrics)
    cfg=read(folder/'config.json');assert cfg['task']['recompute'] is False
    assert cfg['task']['train_iters']==50 and cfg['task']['seq_length']==512
    assert cfg['task']['micro_batch_size']==1 and cfg['task']['global_batch_size']==16
    assert cfg['parallel']['dp']==1 and cfg['parallel']['tp']*cfg['parallel']['pp']==8
    integrity=read(folder/'checkpoint-integrity.json')
    assert sum(x['path'].endswith('model_optim_rng.pt') for x in integrity)==8
    assert sum(x['bytes'] for x in integrity if x['path'].endswith('.pt'))==sum(x['bytes'] for r in rt for x in r['writes'])
    assert all(x['fsync_rename_complete'] for r in rt for x in r['writes'])
    arm='host-only' if folder.parent.parent.name.endswith('-host-only') else 'baseline'
    get=[sum(x['seconds'] for x in r['calls']) for r in rt]
    writes=[sum(x['seconds'] for x in r['writes']) for r in rt]
    wall=d['seconds_through_exit_and_fsync'];loop=sum(iteration)
    row={'cycle':int(folder.parent.parent.name.split('-')[1]),'case':d['case'],'arm':arm,'e2e_s':wall,
        'iteration_log_sum_s':loop,'save_timer_max_s':save,'other_envelope_s':wall-loop-save,
        'pack_max_rank_s':max(get),'pack_rank_range_s':[min(get),max(get)],
        'write_fsync_max_rank_s':max(writes),'write_fsync_rank_range_s':[min(writes),max(writes)],
        'checkpoint_bytes':sum(x['bytes'] for r in rt for x in r['writes']),
        'pretrain_max_rank_s':max(m['pretrain_wall_seconds_including_load_save'] for m in metrics),
        'source':str(folder.relative_to(root))}
    markers={label:datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.datetime.fromisoformat(d['started']).tzinfo)
        for label,value in re.findall(r'\[([^\]]+)\] datetime:\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)',log)}
    start=datetime.datetime.fromisoformat(d['started'])
    row['startup_to_inprocess_s']=(markers['after in-process setup and before initialize_megatron']-start).total_seconds()
    row['distributed_init_s']=(markers['after megatron is initialized']-markers['after in-job setup and before initialize_megatron']).total_seconds()
    row['model_optimizer_build_s']=(markers['after model, optimizer, and learning rate scheduler are built']-markers['after megatron is initialized']).total_seconds()
    row['dataloader_build_s']=(markers['after dataloaders are built']-markers['after model, optimizer, and learning rate scheduler are built']).total_seconds()
    row['post_training_label_to_exit_fsync_s']=(start+datetime.timedelta(seconds=wall)-markers['after training is done']).total_seconds()
    row['training_label_to_done_s']=(markers['after training is done']-markers['before the start of training step']).total_seconds()
    assert row['post_training_label_to_exit_fsync_s']>=0
    if arm!='baseline':
        installed=[read(folder/f'stack-install-rank{rank}.json') for rank in range(8)]
        assert all(r['swiglu_fallback']==0 for r in installed)
        row['swiglu_module_counts']=[x['swiglu_modules'] for x in installed]
        row['lora_calls_per_step']=[x['lora_calls']/50 for x in installed]
        row['direct_tp_calls_per_step']=[sum(x['tp_sizes'].values())/50 for x in installed]
    runs.append(row)
assert len(runs)==48,len(runs)
original=read(Path(sys.argv[2]));output=Path(sys.argv[3])
remote='/private/yp/jiuge-training/checks/20261116-host-pool-stage2-v1/'
checked_seals={}
def local_path(p):
    assert p.startswith(remote),p
    q=root/p.removeprefix(remote);assert q.resolve().is_relative_to(root.resolve());return q

def verify_seal(manifest):
    for p,h in manifest.items():
        if p not in checked_seals:
            q=local_path(p);assert q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest()==h,p
            checked_seals[p]=h
        else:assert checked_seals[p]==h

def scalar_equal(a,b):
    assert a['iteration']==b['iteration'] and a['skipped']==b['skipped']==0
    tol=original['gradient_norm'];assert abs(a['grad_norm']-b['grad_norm'])<=tol['atol']+tol['rtol']*abs(a['grad_norm'])
    assert a['losses'].keys()==b['losses'].keys()
    for k,v in a['losses'].items():
        tol=original['loss'];assert abs(v-b['losses'][k])<=tol['atol']+tol['rtol']*abs(v)

cases=[];paired_steps=0;numeric_pairs=0;native_pairs=0
for case in ['pp2-tp4-cpt','pp2-tp4-lora','tp8-cpt','tp8-lora']:
    audit=read(root/(case+'-audit-passed.json'));assert audit['passed'];verify_seal(audit['evidence_sha256'])
    values=[]
    for i in range(1,7):
        record=read(root/(case+f'-pair-{i:02d}.json'));assert record['complete'];verify_seal(record['evidence_sha256'])
        folders={arm:local_path(p) for arm,p in record['folders'].items()}
        for arm,folder in folders.items():
            row=read(folder/'result.json');assert row['case']==case and row['mode']=='performance' and row['returncode']==0
            assert row['seconds_through_exit_and_fsync']==record['values'][arm]
            assert record['resource_window'] in folder.parent.parent.name
            cfg=read(folder/'config.json');assert cfg['task']['seed']==1234 and cfg['task']['bf16'] is True
            tp,pp=(4,2) if case.startswith('pp2') else (8,1)
            assert cfg['parallel']['tp']==tp and cfg['parallel']['pp']==pp
        for rank in range(8):
            x=read(folders['baseline']/f'performance-steps-rank{rank}.json');y=read(folders['host-only']/f'performance-steps-rank{rank}.json')
            assert len(x)==len(y)==50
            for a,b in zip(x,y):scalar_equal(a,b);paired_steps+=1
        values.append(record['values'])
    summary=read(root/(case+'-e2e-summary.json'));assert values==summary['pairs']
    means={arm:statistics.mean(v[arm] for v in values) for arm in ['baseline','host-only']}
    assert means==summary['mean_seconds']
    ratio=means['baseline']/means['host-only'];assert ratio==summary['speed_ratio']
    rng=random.Random(1234);boot=[]
    for _ in range(20000):
        sample=rng.choices(values,k=6);boot.append(sum(v['baseline'] for v in sample)/sum(v['host-only'] for v in sample))
    boot.sort();assert [boot[499],boot[19499]]==summary['paired_bootstrap95_speed_ratio']
    target=1.15 if case.endswith('cpt') else 1/.85
    assert summary['target_mean_passed']==(ratio>=target) and summary['target_ci_lower_passed']==(boot[499]>=target)
    groups={arm:[r for r in runs if r['case']==case and r['arm']==arm] for arm in ['baseline','host-only']}
    assert all(len(v)==6 for v in groups.values())
    fields=['e2e_s','iteration_log_sum_s','save_timer_max_s','other_envelope_s','pack_max_rank_s','write_fsync_max_rank_s','startup_to_inprocess_s','post_training_label_to_exit_fsync_s']
    component_means={a:{f:statistics.mean(r[f] for r in rows) for f in fields} for a,rows in groups.items()}
    cases.append({'case':case,'mean_seconds':means,'component_means':component_means,'throughput_gain_percent':100*(ratio-1),'time_reduction_percent':100*(1-1/ratio),
        'paired_bootstrap95_speed_ratio':[boot[499],boot[19499]],'mean_targets_passed':ratio>=target,'confidence_targets_passed':boot[499]>=target,
        'additional_candidate_reduction_needed_s':max(0,means['host-only']-means['baseline']/target),'faster_pairs':summary['faster_pairs']})
    numeric={}
    for p in root.glob('numeric-*/*/*/result.json'):
        d=read(p)
        if d['case']==case:
            arm='host-only' if p.parent.parent.parent.name.endswith('-host-only') else 'baseline';assert arm not in numeric;numeric[arm]=p.parent
    assert numeric.keys()=={'baseline','host-only'}
    a,b=numeric['baseline'],numeric['host-only']
    assert read(a/'numeric-review.json')['passed'] and read(b/'numeric-review.json')['passed']
    for rank in range(8):
        for when in ['initial','final']:
            for name in [f'full-{when}-rank{rank}.json',f'numeric/model-{when}-rank{rank}.json']:assert read(a/name)==read(b/name)
        assert {r['stage']:r['schema'] for r in read(a/f'numeric/receipts-rank{rank}.json')}=={r['stage']:r['schema'] for r in read(b/f'numeric/receipts-rank{rank}.json')}
        x=[json.loads(s) for s in (a/f'step-audit-rank{rank}.jsonl').read_text(encoding='utf-8').splitlines()];y=[json.loads(s) for s in (b/f'step-audit-rank{rank}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(x)==len(y)==10
        for left,right in zip(x,y):
            assert {k:v for k,v in left.items() if k not in ['losses','grad_norm']}=={k:v for k,v in right.items() if k not in ['losses','grad_norm']}
            tol=original['gradient_norm'];assert abs(left['grad_norm']-right['grad_norm'])<=tol['atol']+tol['rtol']*abs(left['grad_norm'])
            for k,v in left['losses'].items():
                tol=original['loss'];assert abs(v-right['losses'][k])<=tol['atol']+tol['rtol']*abs(v)
        if case.endswith('lora'):
            for folder in [a,b]:
                initial=read(folder/f'numeric/model-initial-rank{rank}.json');final=read(folder/f'numeric/model-final-rank{rank}.json')
                assert all(v==final[k] for k,v in initial.items() if not v['requires_grad'])
    numeric_pairs+=1
    native={}
    for p in root.glob('restore-*/*/*/result.json'):
        d=read(p)
        if d['case']==case:
            arm='host-only' if p.parent.parent.parent.name.endswith('-host-only') else 'baseline'
            native.setdefault(arm,{})['resume' if d['resume'] else 'control']=p.parent
    assert native.keys()=={'baseline','host-only'}
    for arm,folders in native.items():
        a,b=folders['control'],folders['resume'];assert read(b/'native-restore-review.json')['passed']
        for rank in range(8):
            for left,right in [('step9-state','loaded-step9'),('final-state','final-state'),('counters','counters')]:assert read(a/f'{left}-rank{rank}.json')==read(b/f'{right}-rank{rank}.json')
            x=[json.loads(s) for s in (a/f'step-audit-rank{rank}.jsonl').read_text(encoding='utf-8').splitlines()];y=[json.loads(s) for s in (b/f'step-audit-rank{rank}.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(x)==10 and len(y)==1 and x[-1]==y[0]
            if case.endswith('cpt') and arm=='host-only':assert any(c.get('native_cpu_full_byte_exact') for c in read(a/f'packing-runtime-rank{rank}.json')['calls'])
        native_pairs+=1
    for rank in range(8):assert read(native['baseline']['control']/f'final-state-rank{rank}.json')==read(native['host-only']['control']/f'final-state-rank{rank}.json')
policies=list(root.rglob('host-pool-policy-rank*.json'));assert len(policies)==576,len(policies)
for p in policies:
    d=read(p);assert d['observed_compile_threads']==(1 if d['arm']=='host-only' else 32)
    if d['arm']=='host-only':assert d['pools_before_original_shutdown']==0
    assert d['accepted_training_and_native_durable_save_unchanged'] and d['all_original_shutdown_handlers_retained']
assert len(list(root.glob('*/*/*/result.json')))==72
assert read(root/'complete.json')['complete'] and read(root/'complete.json')['valid_performance_jobs']==48
result={'passed':True,'valid_performance_jobs':48,'paired_rank_steps_original_scalar_gate':paired_steps,'full_numeric_arms':numeric_pairs*2,'native_restore_pairs':native_pairs,
    'host_pool_policy_receipts':len(policies),'sealed_files_sha_verified':len(checked_seals),'cases':cases,'runs':runs,'checkpoint_bodies_rehashed_now':False,
    'limits':'10step fullstate/gradient/update numeric audit;native save9/load9/continue10 only;50step scalar+fullcheckpoint integrity;no new50-to51 or longtraining-quality audit;timing envelopes not pure GPUcriticalpath'}
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'cases':cases,'sealed_files_sha_verified':len(checked_seals)},indent=2))
