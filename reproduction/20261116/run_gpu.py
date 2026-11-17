"""Fresh replay; immutable workers, new controls/references, explicit resource deadline."""
from pathlib import Path
import argparse,datetime,hashlib,itertools,json,os,runpy,shutil,subprocess
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5';H=B/'tools/eight_gpu_20261113/fullstack_20261115'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--until',required=True,help='ISO8601 resource expiry with timezone');p.add_argument('--cases',nargs='+',default=['pp4-tp2-cpt','pp4-tp2-lora','pp2-tp4-cpt','pp2-tp4-lora','tp8-cpt','tp8-lora']);a=p.parse_args()
assert a.name.replace('-','').replace('_','').isalnum()
R=B/'checks/public-replay'/a.name;T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/fullstack-20261115-v1')/('public-'+a.name)
assert not R.exists() and not T.exists();assert shutil.disk_usage(B).free>=22*2**30 and shutil.disk_usage('/tmp').free>=3000*2**30
R.mkdir(parents=True);T.mkdir(parents=True)
read=lambda f:json.loads(f.read_text());write=lambda f,d:f.write_text(json.dumps(d,indent=2)+'\n')
expiry=datetime.datetime.fromisoformat(a.until);assert expiry.tzinfo is not None
write(R/'admission.json',{'allowed_until':(expiry-datetime.timedelta(minutes=15)).isoformat()})
# Original runner retains command validation, full save/fsync, CRC/SHA, metrics and timeout.
s=(H/'campaign.py').read_text();old="cmd[i:i+2]=[str(HERE/'worker.py'),'--numeric-worker' if mode=='numeric' else '--audit-worker']"
assert s.count(old)==1;s=s.replace(old,"cmd[i:i+2]=[str(B/'tools/host_pool_20261116/worker.py'),'--numeric-worker' if mode=='numeric' else '--audit-worker']")
needle='started=datetime.datetime.now().astimezone().isoformat()';assert s.count(needle)==1
s=s.replace(needle,"env['HOST_POOL_ARM']=os.environ['HOST_POOL_ARM']\n    if env['HOST_POOL_ARM']=='host-only':env['TORCHINDUCTOR_COMPILE_THREADS']='1'\n    else:env.pop('TORCHINDUCTOR_COMPILE_THREADS',None)\n    "+needle)
s=s.replace('assert shutil.disk_usage(B).free>=10*2**30','assert shutil.disk_usage(B).free>=20*2**30')
s=s.replace('proc.wait(timeout=1800)',"proc.wait(timeout=min(1800,max(1,int(datetime.datetime.fromisoformat(read(R/'admission.json')['allowed_until']).timestamp()-time.time()))))")
ns={'__name__':'fresh_replay','__file__':str(H/'campaign.py')};exec(compile(s,str(H/'campaign.py'),'exec'),ns);ns.update(R=R,T=T)
# No historical reference or previous complete.json is read by this fresh workflow.
files=[f for f in (B/'tools').rglob('*') if f.suffix in ['.py','.cu','.cpp']];write(R/'protocol.json',{'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'adaptation':'Fresh same-layout baseline numeric reference, new R/T and explicit deadline; frozen workers unchanged','cases':a.cases})
def run(case,arm,mode,resume=False,label=None):
 os.environ['HOST_POOL_ARM']=arm
 return ns['run'](case,'adapted-reference' if arm=='baseline' else ('packed' if case.endswith('cpt') else 'training'),mode,resume,label or mode+'-'+arm)
for case in a.cases:
 arms=['baseline','current','host-only'] if case.startswith('pp4') else ['baseline','host-only'];num={}
 for arm in arms:
  out,_=run(case,arm,'numeric');num[arm]=out
  if arm!='baseline':ns['exact_review'](num['baseline'],out,arm)
  else:
   golden=read(B/'reproduction-golden-20261116'/(case+'.json'))
   for rank in range(8):
    assert read(out/f'numeric/model-initial-rank{rank}.json')==golden[rank]['initial_model'],'Original initial weights/LoRA seed differ'
    first=json.loads((out/f'step-audit-rank{rank}.jsonl').read_text().splitlines()[0])
    assert {k:v for k,v in first.items() if k not in ['losses','grad_norm']}==golden[rank]['first_step_controls'],'Original microbatches/RNG differ; check input preparation'
 controls={}
 for arm in (['current','host-only'] if case.startswith('pp4') else arms):
  x,_=run(case,arm,'restore');y,_=run(case,arm,'restore',True);ns['native_review'](x,y);controls[arm]=x
  if case.endswith('cpt') and arm!='baseline':
   for rank in range(8):assert any(c.get('native_cpu_full_byte_exact') for c in read(x/f'packing-runtime-rank{rank}.json')['calls'])
 controls=list(controls.values())
 for rank in range(8):assert read(controls[0]/f'final-state-rank{rank}.json')==read(controls[1]/f'final-state-rank{rank}.json')
 cycles=[]
 for i in range(6):
  order=list(itertools.permutations(arms))[i] if len(arms)==3 else (arms if i%2==0 else arms[::-1]);values={};folders={}
  for arm in order:
   folder,row=run(case,arm,'performance',label=f'performance-{i+1:02d}-'+arm);values[arm]=row['seconds_through_exit_and_fsync'];folders[arm]=folder
  for arm in arms[1:]:
   for rank in range(8):
    left=read(folders['baseline']/f'performance-steps-rank{rank}.json');right=read(folders[arm]/f'performance-steps-rank{rank}.json');assert len(left)==len(right)==50
    gate=read(E/'numeric-v6/protocol.original.json')
    for x,y in zip(left,right):
     assert x['iteration']==y['iteration'] and x['skipped']==y['skipped']==0
     tol=gate['gradient_norm'];assert abs(x['grad_norm']-y['grad_norm'])<=tol['atol']+tol['rtol']*abs(x['grad_norm'])
     for key,v in x['losses'].items():
      tol=gate['loss'];assert abs(v-y['losses'][key])<=tol['atol']+tol['rtol']*abs(v)
  cycles.append(values);write(R/(case+'-cycles.json'),cycles)
 # Reuse the recorded summary definitions, without executing their historical controllers.
 module=B/'tools'/('host_pool_20261116/campaign.py' if len(arms)==3 else 'host_pool_stage2_20261116/campaign.py')
 import ast
 tree=ast.parse(module.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
 import random,statistics
 scope={'R':R,'write':write,'random':random,'statistics':statistics};exec(compile(ast.Module(body=[node],type_ignores=[]),str(module),'exec'),scope);scope['summarize'](case,cycles)
write(R/'complete.json',{'complete':True,'cases':a.cases})
