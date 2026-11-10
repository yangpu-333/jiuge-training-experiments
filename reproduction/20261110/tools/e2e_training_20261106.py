"""Whole CLI task entry: retain loading, all steps, GC, and final checkpoint."""
import os,sys,json,runpy,subprocess,shlex,time,gc,functools,inspect,math
from pathlib import Path
B=Path('/private/yp/jiuge-training');SCRIPT=Path(__file__).resolve()
def driver():
 original=subprocess.call
 def dispatch(command,*a,**kw):
  cmd=list(command)
  for i in range(len(cmd)-1):
   if cmd[i:i+2]==['-m','jiuge_training.train']:
    actual=cmd[:i]+[str(SCRIPT),'--e2e-worker']+cmd[i+2:]
    index=actual.index('--save');actual[index+1]=os.environ['E2E_CHECKPOINT_SLOT']
    root=Path(actual[actual.index('--jiuge-config')+1]).parent
    (root/'actual-command.txt').write_text(shlex.join(actual)+'\n')
    return original(actual,*a,**kw)
  return original(cmd,*a,**kw)
 subprocess.call=dispatch
 try:runpy.run_module('jiuge_training.cli',run_name='__main__')
 finally:subprocess.call=original
def worker():
 sys.argv.remove('--e2e-worker')
 import jiuge_training.train as baseline
 baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
 import torch
 assert torch.cuda.device_count()==4
 cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);cfg=json.loads(cfgpath.read_text());task=cfg['task']['name'];arm=os.environ['E2E_ARM']
 assert cfg['task']['recompute'] is False and cfg['task']['seq_length']==512 and cfg['task']['global_batch_size']==16 and cfg['task']['micro_batch_size']==1
 assert cfg['parallel']['tp']==cfg['parallel']['pp']==2 and cfg['task']['train_iters']==50 and not cfg['task'].get('load')
 if arm=='baseline':
  import local_spec_ablation_patch
  os.environ['LOCAL_SPEC_ABLATION']='attention-only';local_spec_ablation_patch.install()
 else:
  assert arm=='optimized'
  import lora_linear_cpp,lora_tp_linear,fused_split_norm,fused_norm_dw
  lora_linear_cpp.install()
  if task=='lora':lora_tp_linear.install()
  (fused_norm_dw if task=='cpt' else fused_split_norm).install()
  if os.environ['E2E_USE_FAST']=='1':
   import swiglu_fast_patch_20261106
   swiglu_fast_patch_20261106.install()
 import megatron.training.training as mcore
 original=mcore.train_step;rows=[];events=[];active={};current={'iteration':None}
 def callback(phase,info):
  g=info['generation']
  if phase=='start':active[g]=(time.perf_counter(),current['iteration'])
  elif g in active:
   t,i=active.pop(g);events.append({'generation':g,'iteration':i,'seconds':time.perf_counter()-t})
 @functools.wraps(original)
 def step(*a,**kw):
  bound=inspect.signature(original).bind(*a,**kw).arguments;current['iteration']=int(bound['iteration'])
  args=mcore.get_args();assert not args.async_save and args.save==os.environ['E2E_CHECKPOINT_SLOT']
  assert args.seed==1234 and args.world_size==4 and args.tensor_model_parallel_size==args.pipeline_model_parallel_size==2
  try:
   result=original(*a,**kw)
   rows.append({'iteration':current['iteration'],'skipped':int(result[1]),'grad_norm':float(result[5]),'losses':{k:float(v.detach().item()) for k,v in result[0].items()}})
   return result
  finally:current['iteration']=None
 gc.callbacks.append(callback);mcore.train_step=step
 try:baseline.main()
 finally:gc.callbacks.remove(callback);mcore.train_step=original
 assert len(rows)==50 and all(x['skipped']==0 and math.isfinite(x['grad_norm']) and all(math.isfinite(v) for v in x['losses'].values()) for x in rows)
 settings=mcore.get_args();assert settings.consumed_train_samples==800
 (cfgpath.parent/f'e2e-rank{os.environ["RANK"]}.json').write_text(json.dumps({'completed':True,'task':task,'arm':arm,'use_fast':os.environ['E2E_USE_FAST']=='1','steps':rows,'consumed_train_samples':settings.consumed_train_samples,'processed_tokens':800*512,'gc_events':events,'automatic_gc_enabled':gc.isenabled(),'scope':'all 50 steps, loading and save retained; parent task wall time authoritative'},indent=2)+'\n')
if __name__=='__main__':worker() if '--e2e-worker' in sys.argv else driver()
