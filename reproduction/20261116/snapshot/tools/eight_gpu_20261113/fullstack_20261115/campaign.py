"""Layered full-stack migration: numerical gates -> native restore -> E2E.

All outputs fresh and retained; completed tasks may be reused on a scheduling
continuation only after frozen sources and successful evidence are checked.
"""
from pathlib import Path
import datetime,hashlib,json,math,os,random,shlex,shutil,signal,statistics,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from corex5_numeric_campaign import review
from corex5_disk_campaign import integrity,fsync_tree
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
R=B/'checks/20261115-fullstack-v1'
T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/fullstack-20261115-v1')
CASES=['pp4-tp2-cpt','pp2-tp4-cpt','tp8-cpt','pp4-tp2-lora','pp2-tp4-lora','tp8-lora']
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def admission(mode):
    reserve={'numeric':600,'restore':600,'performance':900}[mode]
    until=datetime.datetime.fromisoformat(read(R/'admission.json')['allowed_until'])
    if datetime.datetime.now().astimezone().timestamp()+reserve>until.timestamp():
        write(R/'status.json',{'state':'needs_resource_extension','no_new_job_started':True,'allowed_until':until.isoformat()})
        raise SystemExit(75)

def run(case,stage,mode,resume=False,label=None):
    label=label or mode
    out=R/label/case/(stage+('-resume' if resume else '-control'))
    if (out/'result.json').is_file():
        row=read(out/'result.json');assert row['returncode']==0
        for rank in range(8):assert read(out/f'metrics-rank{rank}.json')['completed']
        return out,row
    assert not out.exists(),'Preserve incomplete attempt; inspect before resuming'
    admission(mode)
    assert shutil.disk_usage(B).free>=10*2**30
    assert shutil.disk_usage('/tmp').free>120*2**30
    assert 'No running processes found' in subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
    locked=read(R/'protocol.json')['sources']
    assert all(sha(Path(p))==h for p,h in locked.items()),'Frozen source changed'
    out.mkdir(parents=True)
    cfg=read(E/'prepared-v4'/case/'config.json')
    checkpoint=T/label/case/stage/'checkpoints'
    cfg['backend']['rmsnorm']='reference' if stage=='adapted-reference' else 'triton'
    cfg['task'].update(train_iters=50 if mode=='performance' else 10,
        save_interval=50 if mode=='performance' else 9,save=str(checkpoint),load=str(checkpoint) if resume else '')
    from jiuge_training.config import command,validate,prepare_checkpoint_directory,validate_resume
    from jiuge_training.data import validate_manifest
    cfg=validate(cfg);validate_manifest(cfg)
    if mode!='numeric':
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        if resume:validate_resume(cfg)
        else:prepare_checkpoint_directory(cfg)
    cp=out/'config.json';write(cp,cfg);cmd=command(cfg,cp)
    if mode=='numeric' or resume:
        i=cmd.index('--save');del cmd[i:i+2]
    i=next(i for i in range(len(cmd)-1) if cmd[i:i+2]==['-m','jiuge_training.train'])
    cmd[i:i+2]=[str(HERE/'worker.py'),'--numeric-worker' if mode=='numeric' else '--audit-worker']
    (out/'command.txt').write_text(shlex.join(cmd)+'\n')
    env=os.environ.copy();env.update(read(E/'environment.json'))
    env.update(FULLSTACK_STAGE=stage,FULLSTACK_MODE=mode,COREX5_DISK_MODE=mode,
        COREX5_AUDIT_ARM='reference' if stage=='adapted-reference' else 'fused-norm',
        AUDIT_PRESERVE_RESUME_DATALOADER_RNG='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    started=datetime.datetime.now().astimezone().isoformat()
    write(R/'status.json',{'state':'running','case':case,'stage':stage,'mode':mode,'resume':resume,'started':started})
    print('START',case,stage,mode,resume,started,flush=True);clock=time.monotonic()
    with (out/'stdout.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'process.json',{'pid':proc.pid,'process_group':proc.pid,'started':started})
        try:proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    if proc.returncode:write(out/'failed-result.json',{'returncode':proc.returncode});raise RuntimeError((case,stage,mode,proc.returncode))
    if mode!='numeric' and not resume:fsync_tree(checkpoint)
    seconds=time.monotonic()-clock
    for rank in range(8):assert read(out/f'metrics-rank{rank}.json')['completed']
    if stage!='adapted-reference':
        coverage=[read(out/f'stack-install-rank{rank}.json') for rank in range(8)]
        assert all(x['swiglu_modules']>0 and x['swiglu_fallback']==0 for x in coverage)
        assert sum(x['swiglu_modules'] for x in coverage)==62*cfg['parallel']['tp']
        if case.endswith('lora') and stage in ['linear','tp','training']:
            assert all(x['lora_calls']>0 for x in coverage)
        if case.endswith('lora') and stage in ['tp','training']:
            assert all(str(cfg['parallel']['tp']) in x['tp_sizes'] for x in coverage)
    if mode!='numeric' and not resume:write(out/'checkpoint-integrity.json',integrity(checkpoint))
    row={'case':case,'stage':stage,'mode':mode,'resume':resume,'returncode':0,'seconds_through_exit_and_fsync':seconds,'started':started,'checkpoint':str(checkpoint)}
    write(out/'result.json',row)
    print('END',case,stage,mode,resume,seconds,flush=True)
    return out,row

def exact_review(base,other,label):
    result=review(base,other,label,read(E/'numeric-v6/protocol.original.json'))
    dest=other/'numeric-review.json';write(dest,result)
    assert result['passed'],'Byte mismatch is inconclusive under original tolerances: stop for full original elementwise review, never relax gates'

def native_review(a,b):
    for rank in range(8):
        for left,right in [('step9-state','loaded-step9'),('final-state','final-state'),('counters','counters')]:
            assert read(a/f'{left}-rank{rank}.json')==read(b/f'{right}-rank{rank}.json')
        x=[json.loads(s) for s in (a/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        y=[json.loads(s) for s in (b/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        assert len(x)==10 and len(y)==1 and x[-1]==y[0]
    write(b/'native-restore-review.json',{'passed':True,'ranks':8,'full_state_inputs_rng_counters_exact':True,'fresh_process':True})

def summarize(case,pairs):
    a=statistics.mean(x['baseline'] for x in pairs);b=statistics.mean(x['candidate'] for x in pairs)
    rng=random.Random(1234);boots=[]
    for _ in range(20000):
        selected=rng.choices(pairs,k=6);boots.append(sum(x['baseline'] for x in selected)/sum(x['candidate'] for x in selected))
    boots.sort();ci=[boots[499],boots[19499]]
    threshold=1.15 if case.endswith('cpt') else 1/.85
    write(R/f'{case}-e2e-summary.json',{'pairs':pairs,'baseline_mean_s':a,'candidate_mean_s':b,
        'throughput_gain_percent':100*(a/b-1),'time_reduction_percent':100*(1-b/a),
        'paired_bootstrap95_speed_ratio':ci,'mean_target_passed':a/b>=threshold,
        'ci_lower_target_passed':ci[0]>=threshold,'baseline':'attention-only adapted local, reference RMSNorm',
        'storage':'complete synchronous native checkpoint on authorized container local disk'})

def main():
    R.mkdir(exist_ok=True);T.mkdir(parents=True,exist_ok=True)
    if not (R/'protocol.json').exists():
        dependencies=list(HERE.glob('*.py'))+list(HERE.glob('*.cu'))+list(HERE.glob('*.cpp'))
        dependencies+=list(HERE.parent.glob('corex5*.py'))
        dependencies += [B/'tools'/n for n in ['numeric_training.py','audit_training.py','fused_split_norm.py','fused_norm_dw.py','native_order_reductions.py','local_spec_patch.py']]
        write(R/'protocol.json',{'created':datetime.datetime.now().astimezone().isoformat(),'sources':{str(p):sha(p) for p in dependencies},
            'cases':CASES,'configuration':'8 BI-V150; 4B BF16 S512 MBS1 GBS16 DP1 seed1234 no recompute; TP8,TP4PP2,TP2PP4',
            'gates':'Each additive training stage exact full model/master/Adam/gradients/input/RNG 10-step fast path; then candidate CPU pack vs native full-byte; then fresh-process save9/continue10 vs load9/10; no performance before all gates',
            'numerical_mismatch':'Inconclusive under original tolerance; preserve all and stop for original full elementwise metrics, do not relax',
            'performance':'six AB/BA pairs per layout/task,50steps,process launch through complete native save/fsync/exit,CRC/SHA outside timer; autoGC retained; no profiler',
            'checkpoint_temporary_root':str(T),'all_key_evidence_persistent':str(R),
            'baseline':'four-card attention-only adapted local baseline, re-audited vs current reference per layout',
            'target':{'cpt':'throughput>=15%, time>=12% reduction','lora':'throughput>=17%, time>=15% reduction'}})
        write(R/'admission.json',{'allowed_until':'2026-11-15T01:00:00+08:00','note':'Current resource deadline. Fresh jobs require reserve; extend only with authorized coupon coverage and zero cash.'})
    assert all(sha(Path(p))==h for p,h in read(R/'protocol.json')['sources'].items())
    for case in CASES:
        reference=E/'numeric-v6/cases'/case/'reference'
        assert read(reference/'run-result.json')['returncode']==0
        for stage in (['adapted-reference','swiglu'] if case.endswith('cpt') else ['adapted-reference','swiglu','linear','tp']):
            out,_=run(case,stage,'numeric');exact_review(reference,out,stage)
    write(R/'all-training-numeric-passed.json',{'passed':True,'cases':CASES})
    for case in CASES:
        candidate='packed' if case.endswith('cpt') else 'training'
        control={}
        for stage in ['adapted-reference',candidate]:
            a,_=run(case,stage,'restore');b,_=run(case,stage,'restore',True);native_review(a,b);control[stage]=a
            if stage=='packed':
                for rank in range(8):
                    calls=read(a/f'packing-runtime-rank{rank}.json')['calls']
                    assert any(x.get('native_cpu_full_byte_exact') for x in calls)
        for rank in range(8):
            assert read(control['adapted-reference']/f'final-state-rank{rank}.json')==read(control[candidate]/f'final-state-rank{rank}.json')
    write(R/'all-native-restores-passed.json',{'passed':True,'cases':CASES,'candidate_vs_baseline_final_exact':True})
    for case in CASES:
        candidate='packed' if case.endswith('cpt') else 'training';pairs=[]
        for pair in range(6):
            values={};folders={}
            for arm in (['baseline','candidate'] if pair%2==0 else ['candidate','baseline']):
                out,row=run(case,'adapted-reference' if arm=='baseline' else candidate,'performance',label=f'performance-{pair+1:02d}')
                values[arm]=row['seconds_through_exit_and_fsync'];folders[arm]=out
            for rank in range(8):
                x=read(folders['baseline']/f'performance-steps-rank{rank}.json');y=read(folders['candidate']/f'performance-steps-rank{rank}.json')
                assert len(x)==len(y)==50
                original=read(E/'numeric-v6/protocol.original.json')
                for left,right in zip(x,y):
                    assert left['iteration']==right['iteration'] and left['skipped']==right['skipped']==0
                    for name in ['grad_norm']:
                        v=left[name];tol=original['gradient_norm'];assert abs(v-right[name])<=tol['atol']+tol['rtol']*abs(v)
                    for k,v in left['losses'].items():
                        tol=original['loss'];assert abs(v-right['losses'][k])<=tol['atol']+tol['rtol']*abs(v)
            pairs.append(values);write(R/f'{case}-pair-progress.json',pairs)
        summarize(case,pairs)
    write(R/'complete.json',{'complete':True,'training_numeric':True,'native_restores':True,'performance_cases':CASES})
    write(R/'status.json',{'state':'complete'})

if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        R.mkdir(exist_ok=True);write(R/'failure.json',{'traceback':traceback.format_exc()});raise
