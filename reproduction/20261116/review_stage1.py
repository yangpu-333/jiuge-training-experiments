"""Offline analysis of persistent E2E records. No Torch, GPU or network calls."""
from pathlib import Path
import collections,datetime,hashlib,json,re,statistics,sys
import random,math
root=Path(sys.argv[1]);allow_partial='--partial' in sys.argv
read=lambda p:json.loads(p.read_text())
runs=[]
for p in sorted(root.glob('performance-*/*/*/result.json')):
    folder=p.parent
    required=[folder/'stdout.log']+[folder/f'{name}-rank{rank}.json' for name in ['packing-runtime','metrics'] for rank in range(8)]
    if not all(x.exists() for x in required):
        if allow_partial:continue
        raise AssertionError(f'Incomplete download:{folder}')
    d=read(p);log=(folder/'stdout.log').read_text()
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
    arm=folder.parent.parent.name.removeprefix('performance-'+folder.parent.parent.name.split('-')[1]+'-')
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
assert len(runs)==36,len(runs)
original=read(Path(sys.argv[2]));checked_steps=0;cases=[]
for case in ['pp4-tp2-cpt','pp4-tp2-lora']:
    groups={a:[r for r in runs if r['case']==case and r['arm']==a] for a in ['baseline','current','host-only']}
    assert all(len(v)==6 for v in groups.values())
    cycles=[]
    for i in range(1,7):
        g={a:next(r for r in rows if r['cycle']==i) for a,rows in groups.items()}
        cycles.append({a:r['e2e_s'] for a,r in g.items()})
        for arm in ['current','host-only']:
            for rank in range(8):
                x=read(root/g['baseline']['source']/f'performance-steps-rank{rank}.json')
                y=read(root/g[arm]['source']/f'performance-steps-rank{rank}.json')
                assert len(x)==len(y)==50
                for a,b in zip(x,y):
                    assert a['iteration']==b['iteration'] and a['skipped']==b['skipped']==0
                    tol=original['gradient_norm'];assert abs(a['grad_norm']-b['grad_norm'])<=tol['atol']+tol['rtol']*abs(a['grad_norm'])
                    assert a['losses'].keys()==b['losses'].keys()
                    for k,v in a['losses'].items():
                        tol=original['loss'];assert abs(v-b['losses'][k])<=tol['atol']+tol['rtol']*abs(v)
                    checked_steps+=1
    summary=read(root/(case+'-e2e-summary.json'));assert cycles==summary['cycles']
    fields=['e2e_s','iteration_log_sum_s','save_timer_max_s','other_envelope_s','pack_max_rank_s','write_fsync_max_rank_s','startup_to_inprocess_s','post_training_label_to_exit_fsync_s']
    means={a:{k:statistics.mean(r[k] for r in rows) for k in fields} for a,rows in groups.items()}
    for a,b in [('baseline','host-only'),('current','host-only'),('baseline','current')]:
        ratio=means[a]['e2e_s']/means[b]['e2e_s'];stored=summary['comparisons'][a+'_vs_'+b]
        assert math.isclose(ratio,stored['speed_ratio'],rel_tol=1e-14)
        rng=random.Random(1234);boot=[]
        for _ in range(20000):
            s=rng.choices(cycles,k=6);boot.append(sum(r[a] for r in s)/sum(r[b] for r in s))
        boot.sort();assert [boot[499],boot[19499]]==stored['paired_bootstrap95_speed_ratio']
    inc=summary['comparisons']['current_vs_host-only']
    assert summary['increment_stable']==(inc['speed_ratio']>=1.005 and inc['paired_bootstrap95_speed_ratio'][0]>1 and inc['faster_cycles']>=5)
    target=1.15 if case.endswith('cpt') else 1/.85
    cases.append({'case':case,'means':means,'comparisons':summary['comparisons'],'increment_stable':summary['increment_stable'],
        'additional_reduction_needed_for_mean_targets_s':max(0,means['host-only']['e2e_s']-means['baseline']['e2e_s']/target),
        'host_only_reductions_vs_current_s':{k:means['current'][k]-means['host-only'][k] for k in fields},
        'sensitivity_exclude_cycle1_not_acceptance':{'throughput_percent':100*(sum(c['baseline'] for c in cycles[1:])/sum(c['host-only'] for c in cycles[1:])-1),'time_reduction_percent':100*(1-sum(c['host-only'] for c in cycles[1:])/sum(c['baseline'] for c in cycles[1:])),'note':'Diagnostic only; preregistered official result retains ALL6cycles'}})
    stage='packed' if case.endswith('cpt') else 'training'
    base=root/'numeric-baseline'/case/('adapted-reference-control')
    for arm in ['current','host-only']:
        numeric=root/('numeric-'+arm)/case/(stage+'-control')
        for rank in range(8):
            for when in ['initial','final']:
                for name in [f'full-{when}-rank{rank}.json',f'numeric/model-{when}-rank{rank}.json']:
                    assert read(base/name)==read(numeric/name)
            assert {r['stage']:r['schema'] for r in read(base/f'numeric/receipts-rank{rank}.json')}=={r['stage']:r['schema'] for r in read(numeric/f'numeric/receipts-rank{rank}.json')}
            x=[json.loads(s) for s in (base/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            y=[json.loads(s) for s in (numeric/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            assert len(x)==len(y)==10
            for a,b in zip(x,y):
                assert {k:v for k,v in a.items() if k not in ['losses','grad_norm']}=={k:v for k,v in b.items() if k not in ['losses','grad_norm']}
                tol=original['gradient_norm'];assert abs(a['grad_norm']-b['grad_norm'])<=tol['atol']+tol['rtol']*abs(a['grad_norm'])
                for k,v in a['losses'].items():
                    tol=original['loss'];assert abs(v-b['losses'][k])<=tol['atol']+tol['rtol']*abs(v)
        assert read(numeric/'numeric-review.json')['passed']
    controls={}
    for arm in ['current','host-only']:
        a=root/('restore-'+arm)/case/(stage+'-control');b=a.with_name(stage+'-resume');controls[arm]=a
        assert read(b/'native-restore-review.json')['passed']
        for rank in range(8):
            for left,right in [('step9-state','loaded-step9'),('final-state','final-state'),('counters','counters')]:
                assert read(a/f'{left}-rank{rank}.json')==read(b/f'{right}-rank{rank}.json')
            x=[json.loads(s) for s in (a/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            y=[json.loads(s) for s in (b/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
            assert len(x)==10 and len(y)==1 and x[-1]==y[0]
            if case.endswith('cpt'):assert any(c.get('native_cpu_full_byte_exact') for c in read(a/f'packing-runtime-rank{rank}.json')['calls'])
    for rank in range(8):assert read(controls['current']/f'final-state-rank{rank}.json')==read(controls['host-only']/f'final-state-rank{rank}.json')
policy=0
for p in root.rglob('host-pool-policy-rank*.json'):
    d=read(p);assert d['observed_compile_threads']==(1 if d['arm']=='host-only' else 32)
    if d['arm']=='host-only':assert d['pools_before_original_shutdown']==0
    assert d['accepted_training_and_native_durable_save_unchanged'] and d['all_original_shutdown_handlers_retained'];policy+=1
assert policy==400,policy
assert len(list(root.glob('*/*/*/result.json')))==50
result={'passed':True,'performance_jobs':36,'full_numeric_arms':6,'native_restore_pairs':4,'paired_rank_steps_original_scalar_gate':checked_steps,'host_pool_policy_receipts':policy,
    'checkpoint_integrity_receipts_byte_totals_verified':36,'checkpoint_bodies_rehashed_now':False,'numeric_scope':'Full10step model/master/Adam/gradient/input/RNG exact fast path; native save9/continue10 vs fresh load9/10. No new50-to51 audit.',
    'cases':cases,'runs':runs,'decomposition_limit':'Iteration/save are host log envelopes, nested rank sub-times cannot be added as device critical path.'}
output=Path(sys.argv[3]);output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'cases':cases},indent=2))
