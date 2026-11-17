"""Stage2: TP4PP2/TP8, fresh same-round baseline vs complete host-only stack. No GPU resource allocation in this script."""
from pathlib import Path
import datetime,hashlib,itertools,json,math,os,random,shutil,statistics,subprocess,time,traceback
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
HERE=Path(__file__).resolve().parent;OLD=B/'tools/eight_gpu_20261113/fullstack_20261115'
R=B/'checks/20261116-host-pool-stage2-v1'
T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/fullstack-20261115-v1/host-pool-stage2-20261116-v1')
read=lambda p:json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def admission(mode):
    reserve={'numeric':360,'restore':600,'performance':900}[mode]
    end=datetime.datetime.fromisoformat(read(R/'admission.json')['allowed_until']).timestamp()
    if time.time()+reserve>end:
        write(R/'status.json',{'state':'budget_admission_stop','no_new_job_started':True,'allowed_until':read(R/'admission.json')['allowed_until']});raise SystemExit(75)

source=(OLD/'campaign.py').read_text()
needle="cmd[i:i+2]=[str(HERE/'worker.py'),'--numeric-worker' if mode=='numeric' else '--audit-worker']"
assert source.count(needle)==1
source=source.replace(needle,"cmd[i:i+2]=[str(Path('/private/yp/jiuge-training/tools/host_pool_stage2_20261116/worker.py')),'--numeric-worker' if mode=='numeric' else '--audit-worker']")
needle='started=datetime.datetime.now().astimezone().isoformat()'
assert source.count(needle)==1
source=source.replace(needle,"env['HOST_POOL_ARM']=os.environ['HOST_POOL_ARM']\n    if env['HOST_POOL_ARM']=='host-only':env['TORCHINDUCTOR_COMPILE_THREADS']='1'\n    else:env.pop('TORCHINDUCTOR_COMPILE_THREADS',None)\n    "+needle)
source=source.replace('assert shutil.disk_usage(B).free>=10*2**30','assert shutil.disk_usage(B).free>=20*2**30')
assert source.count('proc.wait(timeout=1800)')==1
source=source.replace('proc.wait(timeout=1800)',"proc.wait(timeout=min(1800,max(1,int(datetime.datetime.fromisoformat(read(R/'admission.json')['hard_stop']).timestamp()-time.time()-30))))")
ns={'__name__':'reviewed_host_pool_runner','__file__':str(OLD/'campaign.py')}
exec(compile(source,str(OLD/'campaign.py'),'exec'),ns);ns.update(R=R,T=T,admission=admission)
run_original=ns['run'];exact_review=ns['exact_review'];native_review=ns['native_review']

def run(case,arm,mode,resume=False,label=None):
    os.environ['HOST_POOL_ARM']=arm
    stage='adapted-reference' if arm=='baseline' else ('packed' if case.endswith('cpt') else 'training')
    out,row=run_original(case,stage,mode,resume,label=label or mode+'-'+arm)
    for rank in range(8):
        p=read(out/f'host-pool-policy-rank{rank}.json')
        assert p['arm']==arm and p['observed_compile_threads']==(1 if arm=='host-only' else 32)
        if arm=='host-only':assert p['pools_before_original_shutdown']==0
    return out,row

def scalar_review(a,b):
    gates=read(E/'numeric-v6/protocol.original.json')
    for rank in range(8):
        x=read(a/f'performance-steps-rank{rank}.json');y=read(b/f'performance-steps-rank{rank}.json')
        assert len(x)==len(y)==50
        for left,right in zip(x,y):
            assert left['iteration']==right['iteration'] and left['skipped']==right['skipped']==0
            tol=gates['gradient_norm'];assert abs(left['grad_norm']-right['grad_norm'])<=tol['atol']+tol['rtol']*abs(left['grad_norm'])
            assert left['losses'].keys()==right['losses'].keys()
            for k,v in left['losses'].items():
                tol=gates['loss'];assert abs(v-right['losses'][k])<=tol['atol']+tol['rtol']*abs(v)

def summarize(case,pairs):
    means={a:statistics.mean(p[a] for p in pairs) for a in ['baseline','host-only']}
    rng=random.Random(1234);boot=[]
    for _ in range(20000):
        s=rng.choices(pairs,k=6);boot.append(sum(p['baseline'] for p in s)/sum(p['host-only'] for p in s))
    boot.sort();ratio=means['baseline']/means['host-only'];target=1.15 if case.endswith('cpt') else 1/.85
    result={'case':case,'pairs':pairs,'mean_seconds':means,'speed_ratio':ratio,'throughput_gain_percent':100*(ratio-1),'time_reduction_percent':100*(1-1/ratio),
        'paired_bootstrap95_speed_ratio':[boot[499],boot[19499]],'faster_pairs':sum(p['host-only']<p['baseline'] for p in pairs),
        'target_mean_passed':ratio>=target,'target_ci_lower_passed':boot[499]>=target,
        'baseline':'same-round attention-only adapted local with reference norm; not unmodified official model',
        'candidate':'full accepted training/native durable save plus compile_threads1; no batch-save or LoRA Triton epilogue',
        'attribution':'two-arm TOTAL improvement; compiler-only increment separately established in TP2PP4 first-stage experiment',
        'scope':'8BI-V150/4B BF16 S512 MBS1 GBS16 DP1 seed1234 no recompute;50steps complete local native save/fsync/process exit;CRC/SHA outside timer;no profiler'}
    write(R/(case+'-e2e-summary.json'),result);return result

def seal(folders):
    return {str(p):sha(p) for folder in folders for p in folder.rglob('*') if p.is_file()}

def checked_seal(manifest):
    assert all(Path(p).is_file() and sha(Path(p))==h for p,h in manifest.items()),'Prior evidence changed; preserve and inspect'

def main():
    budget=read(HERE/'resource-budget.json')
    assert budget['gpu_hours']==64 and budget['gpus']==8 and budget['no_automatic_extension']
    stage1=read(B/'checks/20261116-host-pool-v1/complete.json')
    assert stage1['complete'] and all(stage1['stable_increment'].values())
    previous=read(B/'checks/20261116-host-pool-v1/protocol.json')['sources']
    assert all(sha(Path(p))==h for p,h in previous.items())
    sources={**previous,**{str(HERE/n):sha(HERE/n) for n in ['campaign.py','worker.py','resource-budget.json']}}
    groups=[['pp2-tp4-cpt','pp2-tp4-lora'],['tp8-cpt','tp8-lora']]
    cases=sum(groups,[])
    if not R.exists():
        assert not T.exists(),'Preserve previous temporary attempts'
        R.mkdir(parents=True);T.mkdir(parents=True);write(R/'admission.json',budget)
        write(R/'protocol.json',{'sources':sources,'cases':cases,'budget':budget,'configuration':'8BI-V150/CoreX5/4B BF16 TP4PP2 or TP8PP1 S512 MBS1 GBS16 DP1 seed1234 no recompute',
            'gates':'PER-LAYOUT: both tasks2arms fresh10step whole-state numeric and native save9/continue10 vs fresh load9/10; exact final baseline/host states; original tolerances unchanged. Then both tasks performance, before next layout.',
            'performance':'48valid jobs,6AB/BA pairs per case,50steps complete process/native checkpoint/fsync/exit;CRC/SHA outside timer;all completed pairs retained;no profiler',
            'resume':'Verify frozen sources and sealed evidence. Reuse only complete two-arm pairs. Resource-censored incomplete pairs retained under unique attempts; rerun both arms in fresh window, never mix an unpaired time across windows.',
            'attribution':'Total result vs same-round adapted local baseline; not isolated compiler gain in these layouts'})
    else:
        assert not (R/'failure.json').exists(),'Inspect previous failure before continuation'
        assert read(R/'protocol.json')['sources']==sources
        assert all(sha(Path(p))==h for p,h in sources.items())
        T.mkdir(parents=True,exist_ok=True)
    active=read(R/'admission.json');window=active['window_id']
    (R/'resource-windows').mkdir(exist_ok=True)
    write(R/'resource-windows'/(window+'.json'),active)
    import torch,importlib.metadata
    assert torch.cuda.device_count()==8 and torch.__version__.startswith('2.10.')
    assert shutil.disk_usage(B).free>=22*2**30 and shutil.disk_usage('/tmp').free>=3000*2**30
    write(R/('environment-'+window+'.json'),{'torch':torch.__version__,'triton':importlib.metadata.version('triton'),'hostname':os.uname().nodename,'persistent_free':shutil.disk_usage(B).free,'temporary_free':shutil.disk_usage('/tmp').free})
    summaries={}
    for group in groups:
        for case in group:
            gate=R/(case+'-audit-passed.json')
            if gate.exists():checked_seal(read(gate)['evidence_sha256']);continue
            ref=E/'numeric-v6/cases'/case/'reference';folders=[]
            for arm in ['baseline','host-only']:
                out,_=run(case,arm,'numeric',label='numeric-'+window+'-'+arm);exact_review(ref,out,'host-pool-stage2-'+arm);folders.append(out)
            controls={}
            for arm in ['baseline','host-only']:
                label='restore-'+window+'-'+arm
                a,_=run(case,arm,'restore',label=label);b,_=run(case,arm,'restore',True,label=label)
                native_review(a,b);controls[arm]=a;folders.extend([a,b])
                if case.endswith('cpt') and arm=='host-only':
                    for rank in range(8):assert any(c.get('native_cpu_full_byte_exact') for c in read(a/f'packing-runtime-rank{rank}.json')['calls'])
            for rank in range(8):assert read(controls['baseline']/f'final-state-rank{rank}.json')==read(controls['host-only']/f'final-state-rank{rank}.json')
            write(gate,{'passed':True,'numeric':True,'native_restore':True,'final_state_exact':True,'window':window,'evidence_sha256':seal(folders)})
        for case in group:
            pairs=[]
            for i in range(6):
                record=R/(case+f'-pair-{i+1:02d}.json')
                if record.exists():
                    prior=read(record);assert prior['complete'];checked_seal(prior['evidence_sha256']);pairs.append(prior['values']);continue
                # Admit a whole pair with margin for its second process launch.
                hard=datetime.datetime.fromisoformat(read(R/'admission.json')['hard_stop']).timestamp()
                reserve=1800 if case.endswith('cpt') else 1500
                if time.time()+reserve>hard:
                    write(R/'status.json',{'state':'budget_admission_stop','case':case,'next_pair':i+1,'complete_pairs_this_case':len(pairs),'no_new_job_started':True,'hard_stop':read(R/'admission.json')['hard_stop']});raise SystemExit(75)
                attempt=window+'-'+str(len(list(R.glob(case+f'-pair-{i+1:02d}-attempt-*.json')))+1)
                journal=R/(case+f'-pair-{i+1:02d}-attempt-'+attempt+'.json')
                values={};folders={}
                write(journal,{'complete':False,'resource_window':window,'reason_if_incomplete':'Resource cutoff or run failure; never promote an unpaired timing','completed_arms':[]})
                for arm in (['baseline','host-only'] if i%2==0 else ['host-only','baseline']):
                    out,row=run(case,arm,'performance',label=f'performance-{i+1:02d}-'+attempt+'-'+arm)
                    values[arm]=row['seconds_through_exit_and_fsync'];folders[arm]=out
                    write(journal,{'complete':False,'resource_window':window,'completed_arms':list(values),'values':values,'folders':{a:str(p) for a,p in folders.items()}})
                scalar_review(folders['baseline'],folders['host-only'])
                result={'complete':True,'resource_window':window,'values':values,'folders':{a:str(p) for a,p in folders.items()},'evidence_sha256':seal(folders.values())}
                write(record,result);write(journal,{'complete':True,'pair_record':str(record)})
                pairs.append(values);write(R/(case+'-pair-progress.json'),pairs)
            summaries[case]=summarize(case,pairs)
    write(R/'all-audits-passed.json',{'passed':True,'cases':cases,'numeric_arms':8,'native_restore_pairs':8})
    write(R/'complete.json',{'complete':True,'valid_performance_jobs':48,'cases':cases,'mean_targets':{c:s['target_mean_passed'] for c,s in summaries.items()},'confidence_targets':{c:s['target_ci_lower_passed'] for c,s in summaries.items()}})
    write(R/'status.json',{'state':'complete'})

if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);write(R/'failure.json',{'traceback':traceback.format_exc()});raise
