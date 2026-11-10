"""Read-only step9 restore + real step10; state assembly and per-rank save diagnosis."""
from pathlib import Path
import os,sys,json,time,functools,inspect,hashlib,shutil,gc,statistics
B=Path('/private/yp/jiuge-training');ROOT=B/'checks/20261107-save-trace'

def fingerprint(value):
 import torch
 entries=[];total=0
 def visit(x,path):
  nonlocal total
  if isinstance(x,torch.Tensor):
   assert x.device.type=='cpu' and x.is_contiguous();h=hashlib.sha256();flat=x.reshape(-1).view(torch.uint8)
   for start in range(0,flat.numel(),16*1024*1024):h.update(flat[start:start+16*1024*1024].numpy().tobytes())
   entries.append([path,'tensor',str(x.dtype),list(x.shape),h.hexdigest()]);total+=flat.numel()
  elif isinstance(x,dict):
   entries.append([path,'dict',[repr(k) for k in x]])
   for k,v in x.items():visit(v,path+'/'+repr(k))
  elif isinstance(x,(tuple,list)):
   entries.append([path,type(x).__name__,len(x)])
   for i,v in enumerate(x):visit(v,path+'/'+str(i))
  elif x is None or isinstance(x,(str,int,float,bool)):entries.append([path,type(x).__name__,x])
  else:raise TypeError(type(x))
 visit(value,'root');return {'bytes':total,'sha256':hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest(),'entries':entries}

def instrumented_native(opt,native):
 import torch,torch.distributed as dist
 rows={};changes=[]
 def wrap(owner,name,label):
  original=getattr(owner,name)
  def call(*a,**kw):
   start=time.perf_counter()
   try:return original(*a,**kw)
   finally:
    d=rows.setdefault(label,{'calls':0,'host_seconds':0});d['calls']+=1;d['host_seconds']+=time.perf_counter()-start
  setattr(owner,name,call);changes.append((owner,name,original))
 try:
  for owner,name,label in [(torch,'zeros','zeros'),(torch,'cat','cat'),(torch.Tensor,'cpu','tensor_cpu'),(torch.Tensor,'copy_','tensor_copy'),(dist,'gather','gather')]:wrap(owner,name,label)
  start=time.perf_counter();state=native(opt);rows['total_seconds']=time.perf_counter()-start
 finally:
  for owner,name,original in reversed(changes):setattr(owner,name,original)
 return state,rows

def save_profile(state,path):
 import torch,torch.serialization as ser
 records=[];base=ser._open_zipfile_writer_file
 class Proxy:
  def __init__(self,writer):self.writer=writer
  def __getattr__(self,k):return getattr(self.writer,k)
  def write_record(self,name,data,num_bytes):
   start=time.perf_counter();result=self.writer.write_record(name,data,num_bytes);records.append({'name':name,'bytes':num_bytes,'seconds':time.perf_counter()-start});return result
  def write_end_of_file(self):
   start=time.perf_counter();result=self.writer.write_end_of_file();records.append({'name':'<finalize>','bytes':0,'seconds':time.perf_counter()-start});return result
 class Open(base):
  def __init__(self,name):super().__init__(name);self.file_like=Proxy(self.file_like)
 assert not path.exists();ser._open_zipfile_writer_file=Open
 try:
  start=time.perf_counter();torch.save(state,path);elapsed=time.perf_counter()-start
 finally:ser._open_zipfile_writer_file=base
 return {'torch_save_seconds':elapsed,'write_record_seconds':sum(x['seconds'] for x in records),'records':records,'bytes':path.stat().st_size,'crc32_enabled':ser.get_crc32_options()}

def diagnose(bound):
 import torch,torch.distributed as dist
 from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
 from dp1_checkpoint_pack_20261107 import pack
 optimizer=bound['optimizer'];opts=getattr(optimizer,'chained_optimizers',[optimizer]);assert len(opts)==1 and isinstance(opts[0],DistributedOptimizer);opt=opts[0]
 assert torch.cuda.device_count()==4 and dist.get_world_size()==4 and opt.data_parallel_group.size()==1
 native=DistributedOptimizer.get_parameter_state_dp_zero;rank=dist.get_rank();root=ROOT/'save-diagnostic';root.mkdir(exist_ok=True)
 torch.cuda.synchronize();dist.barrier();state,diagnostic=instrumented_native(opt,native);reference=fingerprint(state);del state
 (root/f'native-assembly-rank{rank}.json').write_text(json.dumps({'diagnostic':diagnostic,'reference':reference},indent=2)+'\n')
 all_rows=[];pinned_supported=True
 for repeat in range(4):
  order=['native','direct','pinned'] if repeat%2==0 else ['pinned','direct','native']
  for mode in order:
   if mode=='pinned' and not pinned_supported:continue
   dist.barrier();torch.cuda.synchronize();start=time.perf_counter();error=None;state=None
   try:
    state=native(opt) if mode=='native' else pack(opt,mode)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
   except Exception as exc:error=repr(exc);elapsed=time.perf_counter()-start
   errors=[None]*4;dist.all_gather_object(errors,error)
   if any(errors):
    if state is not None:del state
    if mode!='pinned':raise RuntimeError(errors)
    pinned_supported=False;(root/f'pinned-rejection-rank{rank}.json').write_text(json.dumps(errors)+'\n');continue
   actual=fingerprint(state);assert actual==reference,(rank,repeat,mode)
   row={'repeat':repeat,'mode':mode,'seconds':elapsed,'exact_state_fingerprint_match':True,'cpu_state_bytes':actual['bytes']};gathered=[None]*4;dist.all_gather_object(gathered,row);all_rows.extend(gathered);del state
   (root/f'assembly-progress-rank{rank}.json').write_text(json.dumps(all_rows,indent=2)+'\n')
 modes=['direct']+(['pinned'] if pinned_supported else []);selection={}
 for mode in modes:
  pairs=[]
  for repeat in range(4):
   a=max(x['seconds'] for x in all_rows if x['repeat']==repeat and x['mode']=='native');b=max(x['seconds'] for x in all_rows if x['repeat']==repeat and x['mode']==mode);pairs.append({'repeat':repeat,'native_seconds':a,'candidate_seconds':b,'ratio':a/b})
  ratio=statistics.mean(x['native_seconds'] for x in pairs)/statistics.mean(x['candidate_seconds'] for x in pairs);selection[mode]={'pairs':pairs,'ratio':ratio,'selected_for_io_screen':all(x['ratio']>1 for x in pairs) and ratio>1.005}
 passing=[mode for mode,d in selection.items() if d['selected_for_io_screen']];selected=max(passing,key=lambda mode:selection[mode]['ratio']) if passing else None
 (root/f'assembly-summary-rank{rank}.json').write_text(json.dumps({'exact_state_passed':True,'reference':reference,'selection':selection,'selected':selected,'scope':'four AB/BA state-assembly phase comparisons; excludes hashing and file writes; not full checkpoint or E2E'},indent=2)+'\n')
 io_rows=[]
 # Sequential single-rank real optimizer-file writes fit the persistent budget.
 for active_rank in range(4):
  for mode in ['native']+([selected] if selected else []):
   dist.barrier()
   if rank==active_rank:
    state=native(opt) if mode=='native' else pack(opt,mode);assert fingerprint(state)==reference
    scratch=root/'scratch'/f'rank-{rank}-{mode}';scratch.mkdir(parents=True,exist_ok=False);path=scratch/'distrib_optim.pt'
    assert path.resolve().is_relative_to((root/'scratch').resolve()) and shutil.disk_usage(root).free>reference['bytes']+8*2**30
    record=save_profile(state,path);assert record['crc32_enabled'];record.update(rank=rank,mode=mode,path=str(path),file_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest())
    restored=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert fingerprint(restored)==reference;del restored,state
    record.update(full_roundtrip_exact=True,cleanup='only this newly created diagnostic scratch file removed after exact roundtrip and full SHA256');(root/f'io-rank{rank}-{mode}.json').write_text(json.dumps(record,indent=2)+'\n')
    assert path.resolve().is_relative_to((root/'scratch').resolve());path.unlink();io_rows.append(record)
   dist.barrier()
 (root/f'completed-rank{rank}.json').write_text(json.dumps({'completed':True,'selected':selected,'io_records':len(io_rows),'input_reference_unchanged':True,'scope':'optimizer state assembly + sequential per-rank optimizer-file diagnostic only; no E2E claim'},indent=2)+'\n')

def worker():
 import jiuge_training.train as baseline
 baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
 import megatron.training.training as mcore
 original=mcore.train_step;done=False
 @functools.wraps(original)
 def step(*a,**kw):
  nonlocal done
  bound=inspect.signature(original).bind(*a,**kw).arguments;result=original(*a,**kw)
  if bound['iteration']==9 and not done:diagnose(bound);done=True
  return result
 mcore.train_step=step
 import new_fast_restore_cpt_20261106 as restore
 restore.worker();assert done

if __name__=='__main__':
 if '--audit-worker' in sys.argv:worker()
 else:
  import audit_training
  audit_training.SCRIPT=Path(__file__).resolve();audit_training.driver()
