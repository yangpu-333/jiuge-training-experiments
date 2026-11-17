"""TP2PP4 host-pool-only ablation; <=32GPUh, no automatic second stage."""
from pathlib import Path
import datetime,hashlib,itertools,json,math,os,random,shutil,statistics,subprocess,time,traceback
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
HERE=Path(__file__).resolve().parent;OLD=B/'tools/eight_gpu_20261113/fullstack_20261115'
R=B/'checks/20261116-host-pool-v1'
T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/fullstack-20261115-v1/host-pool-20261116-v1')
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
source=source.replace(needle,"cmd[i:i+2]=[str(Path('/private/yp/jiuge-training/tools/host_pool_20261116/worker.py')),'--numeric-worker' if mode=='numeric' else '--audit-worker']")
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

def summarize(case,cycles):
    means={a:statistics.mean(c[a] for c in cycles) for a in ['baseline','current','host-only']};comparisons={}
    target=1.15 if case.endswith('cpt') else 1/.85
    for a,b in [('current','host-only'),('baseline','host-only'),('baseline','current')]:
        rng=random.Random(1234);boot=[]
        for _ in range(20000):
            s=rng.choices(cycles,k=6);boot.append(sum(c[a] for c in s)/sum(c[b] for c in s))
        boot.sort();ratio=means[a]/means[b]
        comparisons[a+'_vs_'+b]={'speed_ratio':ratio,'throughput_gain_percent':100*(ratio-1),'time_reduction_percent':100*(1-1/ratio),
            'paired_bootstrap95_speed_ratio':[boot[499],boot[19499]],'faster_cycles':sum(c[b]<c[a] for c in cycles),
            'final_target_mean_passed':ratio>=target if a=='baseline' else None,'final_target_ci_lower_passed':boot[499]>=target if a=='baseline' else None}
    inc=comparisons['current_vs_host-only'];stable=inc['speed_ratio']>=1.005 and inc['paired_bootstrap95_speed_ratio'][0]>1 and inc['faster_cycles']>=5
    result={'case':case,'cycles':cycles,'mean_seconds':means,'comparisons':comparisons,'increment_stable':stable,
        'scope':'8BI-V150/4B BF16 TP2PP4 S512 MBS1 GBS16 DP1 seed1234 no recompute;50steps complete process including native save/fsync/exit;local checkpoint;no profiler',
        'candidate_change':'ONLY TORCHINDUCTOR_COMPILE_THREADS=1; accepted training and save exactly retained'}
    write(R/(case+'-e2e-summary.json'),result);return stable

def main():
    assert not R.exists() and not T.exists(),'Preserve previous attempts'
    budget=read(HERE/'resource-budget.json')
    assert budget['gpu_hours']==32 and budget['gpus']==8 and budget['no_automatic_extension']
    R.mkdir(parents=True);T.mkdir(parents=True);write(R/'admission.json',budget)
    old=read(B/'checks/20261115-fullstack-v1/protocol.json')['sources']
    assert all(sha(Path(p))==h for p,h in old.items())
    sources={**old,**{str(HERE/n):sha(HERE/n) for n in ['campaign.py','worker.py','resource-budget.json']}}
    write(R/'protocol.json',{'sources':sources,'created':datetime.datetime.now().astimezone().isoformat(),
        'configuration':'8BI-V150/CoreX5/4B BF16 TP2PP4 S512 MBS1 GBS16 DP1 seed1234 no recompute',
        'arms':{'baseline':'adapted local reference','current':'accepted full training+native durable save,default compiler threads','host-only':'same accepted training/save,ONLY compile_threads1'},
        'gates':'All3arms fresh full10step numeric vs original same-layout reference; current/host-only save9/continue10 vs fresh load9/10; final states exact; original tolerances unchanged',
        'performance':'6 complete permutations per case;36jobs total,50steps;all positions twice;fullcheckpoint+fsync+exit;fullCRC/SHA outside timer;no profiler',
        'stable_gate':'increment mean>=0.5%,paired bootstrap95 lower>1,>=5/6 cycles faster; original final targets separately reported',
        'budget':budget,'next_stage':'Only mark eligibility; TP4PP2/TP8 require separate resource budget, not auto-launched'})
    import torch,importlib.metadata
    assert torch.cuda.device_count()==8 and torch.__version__.startswith('2.10.')
    write(R/'environment-checked.json',{'torch':torch.__version__,'triton':importlib.metadata.version('triton'),'hostname':os.uname().nodename,
        'persistent_free':shutil.disk_usage(B).free,'temporary_free':shutil.disk_usage('/tmp').free,'cuda_devices':torch.cuda.device_count()})
    assert shutil.disk_usage(B).free>=22*2**30 and shutil.disk_usage('/tmp').free>=1500*2**30
    for case in ['pp4-tp2-cpt','pp4-tp2-lora']:
        ref=E/'numeric-v6/cases'/case/'reference'
        for arm in ['baseline','current','host-only']:
            out,_=run(case,arm,'numeric');exact_review(ref,out,'host-pool-'+arm)
        controls={}
        for arm in ['current','host-only']:
            a,_=run(case,arm,'restore');b,_=run(case,arm,'restore',True);native_review(a,b);controls[arm]=a
            if case.endswith('cpt'):
                for rank in range(8):assert any(c.get('native_cpu_full_byte_exact') for c in read(a/f'packing-runtime-rank{rank}.json')['calls'])
        for rank in range(8):assert read(controls['current']/f'final-state-rank{rank}.json')==read(controls['host-only']/f'final-state-rank{rank}.json')
        write(R/(case+'-audit-passed.json'),{'passed':True,'full10step_numeric_all3arms':True,'native_save9_load9_continue10':True,'current_host_final_state_exact':True})
    write(R/'all-audits-passed.json',{'passed':True,'cases':2,'numeric_arms':6,'native_restore_pairs':4})
    results={}
    for case in ['pp4-tp2-cpt','pp4-tp2-lora']:
        cycles=[]
        for i,order in enumerate(itertools.permutations(['baseline','current','host-only']),1):
            values={};folders={}
            for arm in order:
                out,row=run(case,arm,'performance',label=f'performance-{i:02d}-{arm}')
                values[arm]=row['seconds_through_exit_and_fsync'];folders[arm]=out
            scalar_review(folders['baseline'],folders['current']);scalar_review(folders['baseline'],folders['host-only'])
            cycles.append(values);write(R/(case+'-cycle-progress.json'),cycles)
        results[case]=summarize(case,cycles)
    write(R/'complete.json',{'complete':True,'performance_jobs':36,'stable_increment':results,'next_stage_eligible':any(results.values()),'second_stage_not_launched':True})
    write(R/'status.json',{'state':'complete','stable_increment':results})

if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        R.mkdir(parents=True,exist_ok=True);write(R/'failure.json',{'traceback':traceback.format_exc()});raise
