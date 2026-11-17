"""Native disk restore gates, then exploratory full-task paired measurements.

Local container disk was explicitly authorized 2026-11-14. All evidence remains
on account persistent storage. Initial single pairs are NOT final acceptance.
"""
from pathlib import Path
import datetime,hashlib,json,os,shlex,shutil,signal,subprocess,sys,time,zipfile
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
R=E/'disk-e2e-v1';T=Path('/tmp/yp-jiuge-corex5-e2e-20261114')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def integrity(path):
    rows=[]
    for p in sorted(path.rglob('*')):
        if not p.is_file():continue
        if p.suffix=='.pt':
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
        rows.append({'path':str(p.relative_to(path)),'bytes':p.stat().st_size,'sha256':sha(p)})
    assert len([x for x in rows if x['path'].endswith('model_optim_rng.pt')])==8
    return rows
def fsync_tree(path):
    for p in path.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    for p in sorted([x for x in path.rglob('*') if x.is_dir()]+[path],key=lambda x:len(x.parts),reverse=True):
        fd=os.open(p,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def main():
    assert read(E/'numeric-v6/complete.json')['passed']
    assert not R.exists() and not T.exists(),'Preserve previous attempts'
    assert shutil.disk_usage(B).free>=10*2**30
    assert shutil.disk_usage('/tmp').free>2*2**40,'Need2TiB for bounded, retained local-disk runs'
    R.mkdir();T.mkdir()
    env=os.environ.copy();env.update(read(E/'environment.json'))
    env.update(AUDIT_PRESERVE_RESUME_DATALOADER_RNG='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    sys.path.insert(0,str(B/'source/src'));from jiuge_training.config import command
    deps=list((B/'tools/eight_gpu_20261113').glob('corex5*.py'))+[B/'tools/audit_training.py']
    locked={str(p):sha(p) for p in deps}
    write(R/'protocol.json',{'authorization':'User explicitly allowed temporary checkpoint on container local disk on2026-11-14',
        'temporary_checkpoint_root':str(T),'evidence_root':str(R),'source_sha256':locked,
        'scope':'CoreX5 same-layout reference vs norm fusion; local-disk save. Not comparable to prior NAS cohort.',
        'restore_gate':'native save9/continue10 vs fresh-process load9/step10; exact state,input,RNG,counters plus full CRC/SHA',
        'performance':'50steps, complete child process from launch through save close/fsync; integrity review outside timer',
        'screen_pairs':1,'final_target_acceptance':False,'repeat_policy':'After positive screen, preregister>=6 interleaved pairs before final target claim',
        'retention':'No automatic deletions. All source/logs/reviews persistent; checkpoint bodies temporary and not guaranteed after container deletion.'})
    cases=[x['name'] for x in read(E/'prepared-v4/protocol.json')['cases']]
    cases.sort(key=lambda x:(x!='tp8-cpt',x))
    records=[]
    def run(name,arm,phase,resume=False):
        assert shutil.disk_usage(B).free>=10*2**30
        assert shutil.disk_usage(T).free>100*2**30
        assert all(sha(Path(p))==h for p,h in locked.items())
        assert 'No running processes found' in subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
        out=R/phase/name/(arm+('-resume' if resume else '-control'));out.mkdir(parents=True)
        checkpoint=T/phase/name/arm/'checkpoints';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        cfg=read(E/'prepared-v4'/name/'config.json')
        cfg['backend']['rmsnorm']='reference' if arm=='reference' else 'triton'
        cfg['task'].update(train_iters=10 if phase=='restore' else 50,save_interval=9 if phase=='restore' else 50,save=str(checkpoint),load=str(checkpoint) if resume else '')
        cp=out/'config.json';write(cp,cfg);cmd=command(cfg,cp)
        if resume:
            i=cmd.index('--save');del cmd[i:i+2]
        i=next(i for i in range(len(cmd)-1) if cmd[i:i+2]==['-m','jiuge_training.train'])
        cmd[i:i+2]=[str(B/'tools/eight_gpu_20261113/corex5_disk_worker.py'),'--audit-worker']
        (out/'command.txt').write_text(shlex.join(cmd)+'\n')
        child=env.copy();child['COREX5_DISK_MODE']=phase
        start=datetime.datetime.now().astimezone().isoformat();clock=time.monotonic()
        write(R/'status.json',{'state':'running','case':name,'arm':arm,'phase':phase,'resume':resume,'started':start})
        print('START',phase,name,arm,'resume' if resume else 'control',start,flush=True)
        with (out/'stdout.log').open('x') as log:
            proc=subprocess.Popen(cmd,env=child,cwd=B,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        assert proc.returncode==0,(name,arm,phase,resume,proc.returncode)
        if not resume:fsync_tree(checkpoint)
        seconds=time.monotonic()-clock
        row={'case':name,'arm':arm,'phase':phase,'resume':resume,'started':start,'seconds_through_exit_and_fsync':seconds,'checkpoint':str(checkpoint)}
        for rank in range(8):assert read(out/f'metrics-rank{rank}.json')['completed']
        if not resume:write(out/'checkpoint-integrity.json',integrity(checkpoint))
        write(out/'result.json',row);records.append(row);write(R/'runs.json',records)
        print('END',phase,name,arm,resume,seconds,flush=True)
        return out,row
    for name in cases:
        for arm in ['reference','candidate']:
            a,_=run(name,arm,'restore');b,_=run(name,arm,'restore',True)
            for rank in range(8):
                assert read(a/f'step9-state-rank{rank}.json')==read(b/f'loaded-step9-rank{rank}.json')
                assert read(a/f'final-state-rank{rank}.json')==read(b/f'final-state-rank{rank}.json')
                assert read(a/f'counters-rank{rank}.json')==read(b/f'counters-rank{rank}.json')
                left=[json.loads(x) for x in (a/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
                right=[json.loads(x) for x in (b/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
                assert len(left)==10 and len(right)==1 and left[-1]==right[0]
            write(R/f'{name}-{arm}-native-restore.json',{'passed':True,'ranks':8,'fresh_process':True,'storage':'container local disk'})
    write(R/'all-native-restores-passed.json',{'passed':True,'cases':cases,'arms':['reference','candidate']})
    pairs=[]
    for index,name in enumerate(cases):
        values={}
        for arm in (['reference','candidate'] if index%2==0 else ['candidate','reference']):
            _,row=run(name,arm,'performance');values[arm]=row['seconds_through_exit_and_fsync']
        ratio=values['reference']/values['candidate']
        pairs.append({'case':name,**values,'throughput_gain_percent':100*(ratio-1),'time_reduction_percent':100*(1-1/ratio),'scope':'one exploratory complete-task pair; NOT final acceptance'})
        write(R/'e2e-screen.json',pairs)
    write(R/'complete.json',{'complete':True,'native_restores_passed':True,'e2e_screen_pairs':pairs,'final_target_acceptance':False})
    write(R/'status.json',{'state':'complete'})

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if R.is_dir():write(R/'failure.json',{'traceback':traceback.format_exc()})
        raise
