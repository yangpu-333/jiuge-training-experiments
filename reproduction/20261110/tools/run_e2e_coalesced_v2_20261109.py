"""Full-save correctness gates, then six balanced 3-arm CPT complete-task cycles."""
from pathlib import Path
import datetime,hashlib,itertools,json,os,random,shutil,signal,statistics,subprocess,sys,tarfile,time,traceback,zipfile
B=Path('/private/yp/jiuge-training');A=B/'checks/20261109-e2e-coalesced';R=A/'campaign-v2';SLOT=R/'scratch-checkpoint-slot'
if os.environ.get('ROUND_E2E_ENV')!='1':
 e=dict(os.environ,**json.loads((B/'runtime-overrides.json').read_text()));e.update(ROUND_E2E_ENV='1',CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='1',MAX_JOBS='1',E2E_USE_FAST='1',E2E_CHECKPOINT_SLOT=str(SLOT),AUDIT_PRESERVE_RESUME_DATALOADER_RNG='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
 e['PYTHONPATH']=':'.join([str(B/'tools'),str(B/'source/src'),str(B/'Megatron-LM'),e.get('PYTHONPATH','')]);os.execve(sys.executable,[sys.executable,'-u',__file__],e)
assert not R.exists();R.mkdir()
names=['checkpoint_round_common_v2_20261109.py','e2e_coalesced_v2_20261109.py','integrity_coalesced_v2_20261109.py','run_e2e_coalesced_v2_20261109.py','coalesced_state_20261109.py','dp1_checkpoint_pack_20261107.py','e2e_training_20261106.py','audit_training.py','savepack_resume_20261107.py','checkpoint_diagnostic_20261107.py','fused_norm_dw.py','swiglu_fast_patch_20261106.py','local_spec_ablation_patch.py','lora_linear_cpp.py','compact_legacy_archive_20261109.py','archive_failed_shards_20261109.py','dedup_checkpoint_twins_20261109.py']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
frozen={str(B/'tools'/name):sha(B/'tools'/name) for name in names}
orders=list(itertools.permutations(['baseline','current','coalesced']))
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config':'4 BI-V150 4B BF16 DP1 TP2 PP2 S512 MBS1 GBS16 seed1234 recompute=false',
 'arms':{'baseline':'native MCore attention + nine-grid reference RMSNorm adapted-local, native optimizer assembly','current':'accepted fused RMSNorm dw + fast SwiGLU, direct DP1 optimizer assembly','coalesced':'same current training stack, bounded GPU chunk assembly'},
 'timing':'parent before CLI creation until all child ranks exit; loading/init, 50 train steps, autoGC, final full synchronous native-format model/optimizer/RNG/scheduler save, file fsync+directory fsync+atomic rename, shutdown included; no profiler; post-exit CRC/hash/full payload decode and restoration audits excluded',
 'work':{'steps':50,'samples':800,'processed_tokens':409600},'orders':orders,
 'correctness_preflight':'for each arm fresh 11 steps: save10 only, native getter exact, CPU prewrite vs persisted optimizer bytes exact, whole ZIP CRC and SHA; native load10->11 full GPU state/scheduler/counters and actual input/RNG audit must equal in-memory11',
 'persistence':'same fresh atomic-file/fsync wrapper all arms, original torch.save ZIP64 CRC retained; one NEW scratch slot recycled only after full verification; never touch old archive paths; final valid slot retained',
 'capacity':'retain >=10GiB; budget full slot plus256MiB; user-approved exact archival of three old CRC-failed files prerequisite',
 'performance_acceptance':'six cycles, original all50 loss/grad_norm tolerances; current/coalesced complete persisted state payload fingerprints exact; final cycle native50->51 input/RNG and full current/coalesced state exact; candidate incremental all6faster and mean>=1.005; CPT final target baseline/candidate>=1.15 and lower95 bound>=1.15, time<=0.88',
 'stats':'20000 paired cycle bootstrap seed1234; retain every run including slow samples', 'deadline':'no new job after 2026-11-10 02:00 CST, each job timeout1200s; environment expires02:30','amendment':'v1 superseded before any GPU work; v2 additionally fsyncs latest-iteration tracker and contract before exit; same arms/work/numeric gates','files':frozen}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
status={};safe_to_recycle=False;estimate=60452835938+256*2**20
def save_status():
 t=R/'status.tmp';t.write_text(json.dumps(status,indent=2)+'\n');t.replace(R/'status.json')
def idle():assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
def fresh_slot():
 global safe_to_recycle
 idle()
 if SLOT.exists():
  assert safe_to_recycle and SLOT.resolve().is_relative_to(R.resolve()) and SLOT.name=='scratch-checkpoint-slot' and not SLOT.is_symlink()
  shutil.rmtree(SLOT)
 assert shutil.disk_usage(B).free>estimate+10*2**30
 SLOT.mkdir();safe_to_recycle=False
def command(entry,out,mode,steps=50,load=False):
 cmd=[sys.executable,str(B/'tools'/entry),'train','--model','4b','--task','cpt','--parallel','tp2-pp2','--backend','reference' if mode=='baseline' else 'triton','--megatron-root',str(B/'Megatron-LM'),'--run-dir',str(out)]
 values=['model.path='+json.dumps(str(B/'models/9G4B-resolved')),'task.recompute=false','task.seq_length=512','task.micro_batch_size=1','task.global_batch_size=16','task.train_iters='+str(steps),'task.save_interval='+str(steps-1 if steps!=50 else 50),'task.eval_iters=0','task.save='+json.dumps(str(SLOT)),'task.data_path='+json.dumps(str(B/'data/4b-cpt-seq512-resolved'))]
 if load:values+=['task.load='+json.dumps(str(SLOT))]
 for value in values:cmd+=['--set',value]
 return cmd
def run(label,cmd,mode,audit_target=None):
 idle();assert datetime.datetime.now()<datetime.datetime(2026,11,10,2,0),'job admission deadline'
 for p,h in frozen.items():assert sha(Path(p))==h,p
 out=R/label;assert not out.exists();env=dict(os.environ,BENCH_ARM=mode,E2E_ARM='baseline' if mode=='baseline' else 'optimized')
 if audit_target is not None:env['AUDIT_SAVE_ITER']=str(audit_target)
 print('START',label,datetime.datetime.now().isoformat(),flush=True)
 with (R/(label+'.log')).open('w') as log,(R/(label+'-telemetry.jsonl')).open('w') as telemetry:
  start=time.monotonic();p=subprocess.Popen(cmd,cwd=B/'tools',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status[label]={'pid':p.pid,'running':True,'started_monotonic':start,'started_cst':datetime.datetime.now().isoformat()};save_status()
  try:
   while p.poll() is None:
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:telemetry.write(json.dumps({'utc':time.time(),**{n:Path('/proc/pressure/'+n).read_text() for n in ['io','cpu','memory']}})+'\n');telemetry.flush()
    if time.monotonic()-start>1200:raise TimeoutError(label)
  except BaseException:
   try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
   except (ProcessLookupError,subprocess.TimeoutExpired):
    try:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
    except ProcessLookupError:pass
   raise
  elapsed=time.monotonic()-start
 status[label].update(running=False,returncode=p.returncode,seconds=elapsed,completed_cst=datetime.datetime.now().isoformat());save_status()
 print('END',label,p.returncode,elapsed,flush=True);assert p.returncode==0,label
 return elapsed,out
def read(p):return json.loads(p.read_text())
def validate_slot(out,iteration,prewrite=False):
 import torch
 from checkpoint_diagnostic_20261107 import fingerprint as cpu_fp
 from savepack_resume_20261107 import fingerprint as fp
 idle();assert (SLOT/'latest_checkpointed_iteration.txt').read_text().strip()==str(iteration)
 assert len(list(SLOT.rglob('model_optim_rng.pt')))==len(list(SLOT.rglob('distrib_optim.pt')))==4
 images={};hashes={}
 for rank,shard in enumerate(['00_000','01_000','00_001','01_001']):
  folder=SLOT/f'iter_{iteration:07d}'/f'mp_rank_{shard}'
  images[str(rank)]={}
  for filename in ['model_optim_rng.pt','distrib_optim.pt']:
   p=folder/filename
   with zipfile.ZipFile(p) as z:assert z.testzip() is None, str(p)+' ZIP CRC failed'
   hashes[str(p.relative_to(SLOT))]={'bytes':p.stat().st_size,'sha256':sha(p),'zip_crc_passed':True}
   state=torch.load(p,map_location='cpu',weights_only=filename=='distrib_optim.pt',mmap=True)
   if filename=='distrib_optim.pt':
    image=cpu_fp(state)
    if prewrite:assert image==read(out/f'prewrite-optimizer-rank{rank}.json')
    images[str(rank)]['optimizer']=image
   else:
    assert state['iteration']==iteration
    images[str(rank)]['model_and_training_state_excluding_args']=fp({k:v for k,v in state.items() if k!='args'})
   del state
 (out/'checkpoint-file-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 (out/'persisted-state-fingerprints.json').write_text(json.dumps(images,indent=2)+'\n')
 assert shutil.disk_usage(B).free>10*2**30
 return images
def scalar_pair(a,b):
 gate=read(B/'checks/20261031-fused-norm/numerical/protocol.json')
 for rank in range(4):
  left=read(a/f'e2e-rank{rank}.json');right=read(b/f'e2e-rank{rank}.json')
  assert left['completed'] and right['completed'] and left['automatic_gc_enabled'] and right['automatic_gc_enabled'] and len(left['steps'])==len(right['steps'])==50
  for x,y in zip(left['steps'],right['steps']):
   assert x['iteration']==y['iteration'] and x['skipped']==y['skipped']==0
   g,q=gate['gradient_norm'],gate['loss']
   assert abs(x['grad_norm']-y['grad_norm'])<=g['atol']+g['rtol']*abs(x['grad_norm'])
   assert x['losses'].keys()==y['losses'].keys() and all(abs(x['losses'][k]-y['losses'][k])<=q['atol']+q['rtol']*abs(x['losses'][k]) for k in x['losses'])
def bootstrap(rows,left,right):
 rng=random.Random(1234);xs=[]
 for _ in range(20000):
  sample=rng.choices(rows,k=len(rows));xs.append(sum(x[left] for x in sample)/sum(x[right] for x in sample))
 xs.sort();return [xs[500],xs[19499]]
try:
 deadline=time.monotonic()+7200
 while not (A/'legacy-corruption-archive/compact-complete.json').exists():
  assert not (A/'legacy-corruption-archive/failure.txt').exists() and not (A/'legacy-corruption-archive/compact-failure.txt').exists();assert time.monotonic()<deadline;time.sleep(10)
 assert read(A/'legacy-corruption-archive/compact-complete.json')['full_readback_exact'];idle()
 # Separate real-write/load correctness campaign; never included in performance.
 controls={}
 for mode in ['baseline','current','coalesced']:
  fresh_slot();label='preflight-control-'+mode
  _,out=run(label,command('integrity_coalesced_v2_20261109.py',R/label,mode,steps=11),mode,10)
  images=validate_slot(out,10,True)
  label='preflight-resume-'+mode
  _,resumed=run(label,command('integrity_coalesced_v2_20261109.py',R/label,mode,steps=11,load=True),mode,10)
  for rank in range(4):
   assert read(out/f'saved-state-rank{rank}.json')==read(resumed/f'loaded-state-rank{rank}.json')
   assert read(out/f'final-state-rank{rank}.json')==read(resumed/f'final-state-rank{rank}.json')
   assert read(out/f'final-counters-rank{rank}.json')==read(resumed/f'final-counters-rank{rank}.json')
   assert images[str(rank)]['optimizer']==read(resumed/f'loaded-optimizer-cpu-rank{rank}.json')
   original=[json.loads(x) for x in (out/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
   restored=[json.loads(x) for x in (resumed/f'step-audit-rank{rank}.jsonl').read_text().splitlines()]
   assert original[-1:]==restored and len(original)==11
  controls[mode]=out;safe_to_recycle=True
 for rank in range(4):assert read(controls['current']/f'final-state-rank{rank}.json')==read(controls['coalesced']/f'final-state-rank{rank}.json')
 (R/'integrity-preflight-complete.json').write_text(json.dumps({'passed':True,'all_three_arms_native10_to11_exact':True,'full_cpu_prewrite_persisted_and_reloaded_exact':True,'current_coalesced_full_gpu_final_exact':True},indent=2)+'\n')
 cycles=[];final_resume={}
 for cycle,order in enumerate(orders,1):
  row={'cycle':cycle,'order':order};outputs={};images={}
  for mode in order:
   fresh_slot();label=f'cycle{cycle}-{mode}'
   elapsed,out=run(label,command('e2e_coalesced_v2_20261109.py',R/label,mode),mode)
   row[mode]=elapsed;outputs[mode]=out;images[mode]=validate_slot(out,50)
   for rank in range(4):
    runtime=read(out/f'checkpoint-runtime-rank{rank}.json')
    assert len(runtime['writes'])==2 and all(x['fsync_and_rename_complete'] and x['crc32_enabled'] for x in runtime['writes'])
    assert len(runtime['calls'])==1 and runtime['mode']==mode
    if rank==0:assert set(runtime['metadata_synced'])=={'jiuge_contract.json','latest_checkpointed_iteration.txt'}
   if cycle==6:
    label=f'final-native50to51-{mode}'
    _,resumed=run(label,command('integrity_coalesced_v2_20261109.py',R/label,mode,steps=51,load=True),mode,50)
    for rank in range(4):assert images[mode][str(rank)]['optimizer']==read(resumed/f'loaded-optimizer-cpu-rank{rank}.json')
    final_resume[mode]=resumed
   safe_to_recycle=True
  scalar_pair(outputs['baseline'],outputs['current']);scalar_pair(outputs['baseline'],outputs['coalesced']);scalar_pair(outputs['current'],outputs['coalesced'])
  assert images['current']==images['coalesced'],'full persisted current/coalesced state differs'
  row.update(incremental_ratio=row['current']/row['coalesced'],total_ratio=row['baseline']/row['coalesced'],all_checks_passed=True)
  cycles.append(row);(R/'progress.json').write_text(json.dumps(cycles,indent=2)+'\n')
 for rank in range(4):
  for name in ['loaded-state','final-state','final-counters']:
   assert read(final_resume['current']/f'{name}-rank{rank}.json')==read(final_resume['coalesced']/f'{name}-rank{rank}.json')
  assert (final_resume['current']/f'step-audit-rank{rank}.jsonl').read_text()==(final_resume['coalesced']/f'step-audit-rank{rank}.jsonl').read_text()
 means={m:statistics.mean(x[m] for x in cycles) for m in ['baseline','current','coalesced']}
 total=means['baseline']/means['coalesced'];incremental=means['current']/means['coalesced'];ci=bootstrap(cycles,'baseline','coalesced')
 result={'completed':True,'cycles':cycles,'means_seconds':means,'incremental_speed_ratio':incremental,'incremental_time_reduction':1-means['coalesced']/means['current'],'incremental_bootstrap95':bootstrap(cycles,'current','coalesced'),'total_speed_ratio':total,'total_time_reduction':1-means['coalesced']/means['baseline'],'total_bootstrap95':ci,'processed_tokens_per_second':{k:409600/v for k,v in means.items()},'candidate_incremental_selected':all(x['incremental_ratio']>1 for x in cycles) and incremental>=1.005,'cpt_final_target_met':total>=1.15 and ci[0]>=1.15 and means['coalesced']/means['baseline']<=.88,'native_restore_and_integrity_passed':True,'scope':'fixed prepared-cache 50-step complete CPT tasks, full durable native-format checkpoint, same atomic/fsync persistence for all arms; adapted-local baseline; no LoRA claim'}
 (R/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE_TASK_E2E_CAMPAIGN_COMPLETE',flush=True)
except BaseException:
 (R/'failure.txt').write_text(traceback.format_exc());raise
finally:
 files=sorted({p for p in R.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.txt','.log','.md'} and p.stat().st_size<20*2**20 and not p.is_relative_to(SLOT)} | {Path(p) for p in frozen} | set((A/'legacy-corruption-archive').glob('*.json')) | set(A.glob('twins-dedup-*.json')) | {A/'superseded-before-gpu.json',A/'durability-capability/passed.json'})
 manifest={str(p.relative_to(B)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
 mp=R/'evidence-file-hashes.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');archive=R/'key-evidence-e2e-coalesced-20261109.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for p in files+[mp]:t.add(p,arcname=str(p.relative_to(B)),recursive=False)
 (R/'archive-complete.json').write_text(json.dumps({'completed':True,'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'large_checkpoint_and_legacy_archive_bodies_included':False},indent=2)+'\n')
