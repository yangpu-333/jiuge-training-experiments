"""Serial reference-only full-model feasibility checks in isolated CoreX 5.0."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,shutil,signal,socket,subprocess,sys,time

B=Path('/private/yp/jiuge-training');R=B/'checks/20261113-eight-gpu-corex5'
P=R/'prepared'

def main():
    resource=json.loads((R/'resource.json').read_text())
    assert resource['container_hostname']==socket.gethostname()
    assert resource['requested_gpu_count']==8
    for rank in range(8):assert json.loads((R/f'collective-rank{rank}.json').read_text())['passed']
    env=os.environ.copy();env.update(json.loads((R/'environment.json').read_text()))
    expected={'source':'9fabd71df4897f08267df5431aa480ab9ef70a75','Megatron-LM':'5be9626709af2722333bf54797c954c09edeada3'}
    for name,commit in expected.items():
        assert subprocess.check_output(['git','-C',str(B/name),'rev-parse','HEAD'],text=True).strip()==commit
    protocol=json.loads((P/'protocol.json').read_text());records=[]
    for case in protocol['cases']:
        # Conservative local cut-off until this new container's expiry is read.
        now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        assert now.hour<19 or (now.hour==19 and now.minute<20),'No new case after 19:20 CST'
        assert shutil.disk_usage(B).free>=10*2**30,'Keep 10GiB persistent free'
        idle=subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
        assert 'No running processes found' in idle,'Other GPU process present'
        out=P/case['name'];cfgpath=out/'config.json'
        assert hashlib.sha256(cfgpath.read_bytes()).hexdigest()==case['config_sha256']
        cfg=json.loads(cfgpath.read_text())
        assert cfg['parallel']['tp']*cfg['parallel']['pp']==8 and cfg['parallel']['dp']==1
        assert cfg['task']['recompute'] is False and cfg['task']['seq_length']==512
        check='import json;from pathlib import Path;from jiuge_training.data import validate_manifest;validate_manifest(json.loads(Path('+repr(str(cfgpath))+').read_text()))'
        subprocess.run([sys.executable,'-c',check],env=env,check=True,timeout=120)
        command=shlex.split((out/'command.txt').read_text())
        assert '--save' not in command and '--load' not in command
        started=datetime.datetime.now().astimezone().isoformat();clock=time.monotonic()
        print('START',case['name'],started,flush=True)
        with (out/'stdout.log').open('x') as stream:
            proc=subprocess.Popen(command,cwd=B,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        row={'case':case['name'],'started':started,'completed':datetime.datetime.now().astimezone().isoformat(),
             'seconds':time.monotonic()-clock,'returncode':proc.returncode,
             'scope':'2-step feasibility only; no numerical acceptance, restore or E2E performance claim'}
        if proc.returncode==0:
            metrics=[json.loads((out/f'metrics-rank{rank}.json').read_text()) for rank in range(8)]
            assert all(m['completed'] for m in metrics)
            row['max_peak_allocated_bytes']=max(m['peak_allocated_bytes'] for m in metrics)
        (out/'smoke-result.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row)
        (R/'smoke-summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print('END',case['name'],row['returncode'],row['seconds'],flush=True)
        # Identical dependency errors should be fixed before spending five more launches.
        if proc.returncode!=0:
            print('STOP_AFTER_FAILURE: preserve this attempt; inspect before continuing',flush=True)
            return
    print('ALL_SIX_SMOKES_COMPLETED',flush=True)

if __name__=='__main__':main()
