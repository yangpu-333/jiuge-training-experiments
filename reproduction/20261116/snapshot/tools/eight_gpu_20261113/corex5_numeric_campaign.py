"""Six layouts/tasks: reference, RAM-loader diagnostic, fused RMSNorm audit.

No durable tensor checkpoint is written. Every attempted run and mismatch stays.
"""
from pathlib import Path
import datetime,hashlib,json,math,os,shlex,shutil,signal,subprocess,sys,time
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5';R=E/'numeric-v1'
ARMS=['reference','ram-restore','fused-norm']

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def review(base,other,arm,protocol):
    rows=[];passed=True
    for rank in range(8):
        checks={}
        for stage in ['initial','final']:
            checks['full_state_'+stage]=read(base/f'full-{stage}-rank{rank}.json')==read(other/f'full-{stage}-rank{rank}.json')
            checks['model_'+stage]=read(base/f'numeric/model-{stage}-rank{rank}.json')==read(other/f'numeric/model-{stage}-rank{rank}.json')
        reference_receipts={x['stage']:x['schema'] for x in read(base/f'numeric/receipts-rank{rank}.json')}
        candidate_receipts={x['stage']:x['schema'] for x in read(other/f'numeric/receipts-rank{rank}.json')}
        assert set(reference_receipts)=={'master-initial','gradient-step1','gradient-step10','master-final'}
        checks['full_capture_byte_parity']=reference_receipts==candidate_receipts
        left=[json.loads(x) for x in (base/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        right=[json.loads(x) for x in (other/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
        assert len(left)==len(right)==10
        checks['controls_exact']=True;checks['original_scalar_tolerances']=True
        for i,(a,b) in enumerate(zip(left,right)):
            assert a['iteration_zero_based']==b['iteration_zero_based']==i
            assert a['skipped_iter']==b['skipped_iter']==0
            assert all(math.isfinite(v) for row in [a,b] for v in row['losses'].values())
            assert math.isfinite(a['grad_norm']) and math.isfinite(b['grad_norm'])
            exact=[k for k in a if k not in ['losses','grad_norm']]
            checks['controls_exact'] &= {k:a[k] for k in exact}=={k:b[k] for k in exact}
            assert a['losses'].keys()==b['losses'].keys()
            for key,v in a['losses'].items():
                checks['original_scalar_tolerances'] &= abs(v-b['losses'][key])<=protocol['loss']['atol']+protocol['loss']['rtol']*abs(v)
            v=a['grad_norm'];checks['original_scalar_tolerances'] &= abs(v-b['grad_norm'])<=protocol['gradient_norm']['atol']+protocol['gradient_norm']['rtol']*abs(v)
        initial=read(base/f'numeric/model-initial-rank{rank}.json');final=read(base/f'numeric/model-final-rank{rank}.json')
        checks['frozen_parameters_unchanged']=all(initial[n]==final[n] for n in initial if not initial[n]['requires_grad'])
        if arm=='ram-restore':checks['ram_roundtrip']=read(other/f'ram-roundtrip-rank{rank}.json')['passed']
        rows.append({'rank':rank,'checks':checks});passed &= all(checks.values())
    return {'passed':passed,'arm':arm,'ranks':rows,'criterion':'Original scalar gates plus full-element byte parity sufficient for zero gradient/update/model error',
            'mismatch_policy':'No tolerance relaxation. Hash mismatch requires full original numerical metrics; not proof of original tolerance failure.',
            'durable_restore_tested':False,'fresh_process_resume_tested':False}

def main():
    assert not R.exists(),'Preserve every previous campaign'
    assert all(x['returncode']==0 for x in read(E/'smoke-summary.json'))
    assert len(read(E/'smoke-summary.json'))==6
    R.mkdir();sys.path.insert(0,str(B/'source/src'))
    from jiuge_training.config import command
    original_protocol=B/'checks/20261031-numeric/protocol.json'
    if not original_protocol.is_file():original_protocol=B/'checks/20261031-numeric-fix/protocol.json'
    assert original_protocol.is_file()
    protocol=read(original_protocol);write(R/'protocol.original.json',protocol)
    assert sha(B/'tools/numeric_training.py')=='a8d8d0c2ad7feea89ff628502e50039362cfcec6ac10a458fa96724483a54e76'
    dependencies=[B/'tools/numeric_training.py',B/'tools/fused_split_norm.py',B/'tools/fused_norm_dw.py',B/'tools/native_order_reductions.py',B/'tools/eight_gpu_20261113/corex5_numeric_worker.py',Path(__file__)]
    dependency_hashes={str(p):sha(p) for p in dependencies}
    write(R/'preregistration.json',{'original_protocol_sha256':sha(original_protocol),'dependency_hashes':dependency_hashes,
      'configuration':'4B/BF16/DP1/S512/MBS1/GBS16/seed1234/no recompute; TP8,TP4PP2,TP2PP4',
      'arms':ARMS,'storage':'Hash-only full-element captures; self-generated RAM serialization only',
      'scope':'Numeric exact-parity fast path plus SAME-PROCESS RAM native optimizer/model loader diagnostic; no durable or fresh-process recovery acceptance',
      'failure_policy':'Preserve logs; stop for investigation. Byte mismatch is inconclusive under original numerical tolerance, not a license to relax it.'})
    env=os.environ.copy();env.update(read(E/'environment.json'))
    cases=[]
    for item in read(E/'prepared-v4/protocol.json')['cases']:
        name=item['name'];cases.append(name)
        for arm in ARMS:
            out=R/'cases'/name/arm;out.mkdir(parents=True)
            cfg=read(E/'prepared-v4'/name/'config.json')
            cfg['task'].update(train_iters=10,save_interval=10,save=str(out/'unused-checkpoint'),load='')
            cfg['backend']['rmsnorm']='triton' if arm=='fused-norm' else 'reference'
            path=out/'config.json';write(path,cfg)
            cmd=command(cfg,path);index=cmd.index('--save');del cmd[index:index+2]
            index=next(i for i in range(len(cmd)-1) if cmd[i:i+2]==['-m','jiuge_training.train'])
            cmd[index:index+2]=[str(B/'tools/eight_gpu_20261113/corex5_numeric_worker.py'),'--numeric-worker']
            (out/'command.txt').write_text(shlex.join(cmd)+'\n')
    records=[]
    for name in cases:
        for arm in ARMS:
            assert shutil.disk_usage(B).free>=10*2**30
            assert 'No running processes found' in subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
            assert all(sha(Path(p))==value for p,value in dependency_hashes.items())
            out=R/'cases'/name/arm;cmd=shlex.split((out/'command.txt').read_text())
            child_env=env.copy();child_env['COREX5_AUDIT_ARM']=arm
            start=datetime.datetime.now().astimezone().isoformat();clock=time.monotonic()
            print('START',name,arm,start,flush=True)
            write(R/'status.json',{'case':name,'arm':arm,'state':'running','started':start,'completed_runs':len(records)})
            with (out/'stdout.log').open('x') as stream:
                proc=subprocess.Popen(cmd,env=child_env,cwd=B,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            row={'case':name,'arm':arm,'returncode':proc.returncode,'seconds':time.monotonic()-clock,'started':start}
            write(out/'run-result.json',row);records.append(row);write(R/'runs.json',records)
            print('END',name,arm,proc.returncode,row['seconds'],flush=True)
            if proc.returncode:write(R/'failure.json',row);return
            for rank in range(8):
                m=read(out/f'metrics-rank{rank}.json');assert m['completed']
                if name.endswith('lora'):assert m['adapter_validity']['passed']
                assert read(out/f'numeric/coverage-rank{rank}.json')['prepared_calls']==10
            if arm!='reference':
                result=review(R/'cases'/name/'reference',out,arm,protocol)
                write(R/f'{name}-{arm}-review.json',result)
                if not result['passed']:
                    write(R/'needs-full-numeric-review.json',{'case':name,'arm':arm,'reason':'Strict byte/control audit did not pass; inspect original gates without relaxing them'})
                    print('NEEDS_INVESTIGATION',name,arm,flush=True);return
    write(R/'complete.json',{'passed':True,'completed_runs':len(records),'configurations':cases,
      'numeric_scope':'fused RMSNorm vs same-layout reference; full-element exact parity',
      'restore_scope':'same-process RAM serialization and native optimizer/model reload only',
      'durable_checkpoint_restore_passed':False,'fresh_process_resume_passed':False})
    write(R/'status.json',{'state':'complete','completed_runs':len(records)})
    print('COREX5_NUMERIC_AND_RAM_LOADER_COMPLETE',flush=True)

if __name__=='__main__':main()
