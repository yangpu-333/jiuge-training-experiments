"""Preregistered repeat and separate diagnostics; never modifies old evidence."""
from pathlib import Path
import datetime, hashlib, json, os, random, statistics, subprocess, sys, traceback

HERE = Path(__file__).resolve().parent
B = Path('/private/yp/jiuge-training')
E = B/'checks/20261113-eight-gpu-corex5'
R = E/'abba-diagnostics-20261114-v1'
T = Path('/tmp/yp-jiuge-corex5-e2e-20261114/abba-diagnostics-v1')

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def main():
    # Reuse the frozen run implementation, including official CLI preparation,
    # native full checkpoint, fsync, eight-rank completion and CRC/SHA checks.
    source = (HERE/'corex5_disk_campaign.py').read_text()
    source = source.replace("R=E/'disk-e2e-v1'", "R=E/'abba-diagnostics-20261114-v1'")
    source = source.replace("T=Path('/tmp/yp-jiuge-corex5-e2e-20261114')", "T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/abba-diagnostics-v1')")
    needle = "        cp=out/'config.json';write(cp,cfg);cmd=command(cfg,cp)"
    assert source.count(needle) == 1
    source = source.replace(needle, """        from jiuge_training.config import prepare_checkpoint_directory,validate
        from jiuge_training.data import validate_manifest
        cfg=validate(cfg)
        validate_manifest(cfg)
        prepare_checkpoint_directory(cfg)
"""+needle)
    start = source.index('    for name in cases:\n')
    stop = source.index("\nif __name__=='__main__':")
    source = source[:start] + '''    assert read(E/'disk-e2e-v2/complete.json')['native_restores_passed']
    write(R/'repeat-protocol.json', {
        'case':'pp4-tp2-cpt', 'pairs':6, 'orders':['AB','BA','AB','BA','AB','BA'],
        'A':'reference', 'B':'candidate fused norm', 'steps':50,
        'configuration':'8 BI-V150; 4B; DP1; TP2 PP4; BF16; S512; MBS1; GBS16; seed1234; no recomputation',
        'timer':'process launch through exit and synchronous complete checkpoint fsync',
        'integrity':'every checkpoint full ZIP CRC and SHA outside timer',
        'target':'CPT throughput>=15%, time reduction>=12%; report paired bootstrap CI, no discarded runs',
        'diagnostics':'separate from timed repeat; TP8 two full tasks and LoRA short profiler tasks',
        'baseline':'pinned Jiuge reference, not unadapted Megatron native model',
        'storage':'authorized container-local complete checkpoint; persistent code/logs/traces/integrity',
        'old_provenance':read(E/'disk-e2e-v2/protocol.json'),
    })
    pairs=[]
    for pair in range(6):
        values={}
        for arm in (['reference','candidate'] if pair%2==0 else ['candidate','reference']):
            _, row=run('pp4-tp2-cpt',arm,f'repeat-{pair+1:02d}')
            values[arm]=row['seconds_through_exit_and_fsync']
        pairs.append(values)
        write(R/'paired-progress.json',pairs)
    summarize_repeat(R,pairs)
    for case,phase in [('tp8-cpt','phase-diagnostic'),('pp4-tp2-lora','lora-profiler')]:
        for arm in ['reference','candidate']:
            os.environ['COREX5_DIAGNOSTIC_KIND']=phase
            diagnostic_run(run,case,arm,phase)
    write(R/'complete.json',{'complete':True,'repeat_pairs':6,'diagnostics_complete':True,
        'new_lora_optimization_accepted':False})
    write(R/'status.json',{'state':'complete'})
''' + source[stop:]
    # Diagnostic calls switch workers/config only in a separate interpreter
    # context; timed runs above remain byte-for-byte the successful worker.
    ns = {'__name__':'campaign_library','__file__':str(HERE/'corex5_disk_campaign.py'),
          'summarize_repeat':summarize_repeat,'diagnostic_run':diagnostic_run}
    exec(compile(source,str(HERE/'corex5_disk_campaign.py'),'exec'), ns)
    ns['main']()

def summarize_repeat(root, pairs):
    a=statistics.mean(x['reference'] for x in pairs)
    b=statistics.mean(x['candidate'] for x in pairs)
    rng=random.Random(20261114); ratios=[]
    for _ in range(20000):
        sample=rng.choices(pairs,k=len(pairs))
        ratios.append(sum(x['reference'] for x in sample)/sum(x['candidate'] for x in sample))
    ratios.sort();ci=[ratios[499],ratios[19499]]
    write(root/'repeat-summary.json', {'pairs':pairs,'reference_mean_s':a,'candidate_mean_s':b,
        'throughput_gain_percent':100*(a/b-1),'time_reduction_percent':100*(1-b/a),
        'paired_bootstrap_95_speed_ratio':ci,'all_pairs_faster':all(x['reference']>x['candidate'] for x in pairs),
        'mean_target_passed':a/b>=1.15 and 1-b/a>=.12,
        'ci_lower_target_passed':ci[0]>=1.15,'scope':'50-step fixed prepared-cache complete-task; local disk'})

def diagnostic_run(run, case, arm, phase):
    # run is a closure: temporarily override only its command builder to select
    # the diagnostic worker and optional short trace task. Restored in finally.
    cells=dict(zip(run.__code__.co_freevars,run.__closure__))
    original=cells['command'].cell_contents
    def command(cfg,path):
        if phase=='lora-profiler':
            cfg['task']['train_iters']=7
            cfg['task']['save_interval']=50
            write(path,cfg)
        result=original(cfg,path)
        # Base run subsequently replaces this module pair with disk worker.
        return result
    cells['command'].cell_contents=command
    # Replace the worker pathname in the run's code constants, leaving all
    # timing, save and integrity logic intact; LoRA trace has no checkpoint,
    # so use a dedicated launch below rather than the timed run helper.
    if phase=='lora-profiler':
        cells['command'].cell_contents=original
        return run_lora_trace(case,arm)
    oldcode=run.__code__
    constants=tuple('tools/eight_gpu_20261113/corex5_diagnostic_worker.py' if x=='tools/eight_gpu_20261113/corex5_disk_worker.py' else x for x in oldcode.co_consts)
    assert constants!=oldcode.co_consts
    cells['env'].cell_contents['COREX5_DIAGNOSTIC_KIND']=phase
    original_fsync=run.__globals__['fsync_tree']
    def measured_fsync(path):
        import time
        start=time.monotonic();original_fsync(path)
        # Stored separately, only on diagnostic tasks. Nested phase times
        # are explicitly not assumed disjoint.
        write(R/f'fsync-{case}-{arm}.json',{'seconds':time.monotonic()-start,'checkpoint':str(path)})
    run.__globals__['fsync_tree']=measured_fsync
    run.__code__=oldcode.replace(co_consts=constants)
    try:return run(case,arm,phase)
    finally:
        run.__code__=oldcode;cells['command'].cell_contents=original
        run.__globals__['fsync_tree']=original_fsync

def run_lora_trace(case, arm):
    import shutil, shlex
    from jiuge_training.config import command,validate
    from jiuge_training.data import validate_manifest
    assert shutil.disk_usage(B).free>=10*2**30
    assert 'No running processes found' in subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
    out=R/'lora-profiler'/case/arm;out.mkdir(parents=True)
    cfg=json.loads((E/'prepared-v4'/case/'config.json').read_text())
    cfg['backend']['rmsnorm']='reference' if arm=='reference' else 'triton'
    cfg['task'].update(train_iters=7,save='',load='',save_interval=50)
    cfg=validate(cfg);validate_manifest(cfg);cp=out/'config.json';write(cp,cfg)
    cmd=command(cfg,cp);i=cmd.index('-m');cmd[i:i+2]=[str(HERE/'corex5_diagnostic_worker.py'),'--audit-worker']
    (out/'command.txt').write_text(shlex.join(cmd)+'\n')
    env=os.environ.copy();env.update(json.loads((E/'environment.json').read_text()))
    env.update(COREX5_DIAGNOSTIC_KIND='lora-profiler',COREX5_DISK_MODE='lora-profiler')
    write(R/'status.json',{'state':'running','phase':'lora-profiler','case':case,'arm':arm})
    with (out/'stdout.log').open('x') as log:
        proc=subprocess.run(cmd,cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    assert proc.returncode==0
    assert len(list(out.glob('trace-rank*.json.gz')))==8
    assert all(json.loads((out/f'metrics-rank{rank}.json').read_text())['completed'] for rank in range(8))
    write(out/'result.json',{'completed':True,'instrumented':True,'performance_acceptance':False})

if __name__=='__main__':
    try:main()
    except BaseException:
        if R.is_dir():write(R/'failure.json',{'traceback':traceback.format_exc()})
        raise
