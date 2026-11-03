"""Exploratory paired LoRA timing, gated on direct numeric equivalence."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
import run_4b_smoke as helper
B=helper.BASE;ROOT=B/'checks/20261102-local-ablation/attention-only/performance';NUM=ROOT.parent/'numerical'
def read(p):return json.loads(p.read_text())
def main():
 assert read(NUM/'lora-comparison.json')['passed'] and read(NUM/'lora-direct-comparison-v2.json')['passed']
 assert not ROOT.exists();ROOT.mkdir()
 files=['adapted_local_perf.py','local_spec_ablation_patch.py','local_spec_patch.py','run_adapted_pairs.py','lora_linear_cpp.py','lora_tp_linear.py','fused_split_norm.py','phase_training.py','performance_training.py']
 protocol={'scope':'Exploratory paired native MCore local DotProductAttention with nine-grid reference RMSNorm and semantic adapters vs validated fused norm+C++ A+TP/B; LoRA only; not formal end-to-end','warmup':5,'measured':20,'order':[['native','optimized'],['optimized','native'],['native','optimized'],['optimized','native']],'gc':'both arms full GC after final warmup; auto GC stays enabled, cost recorded separately','all_steps_retained':True,'direct_numeric_gate_sha256':hashlib.sha256((NUM/'lora-direct-comparison-v2.json').read_bytes()).hexdigest(),'tools':{n:hashlib.sha256((B/'tools'/n).read_bytes()).hexdigest() for n in files}}
 (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 helper.STAGE=ROOT;helper.save_status=lambda:(ROOT/'status.json').write_text(json.dumps(helper.STATUS,indent=2)+'\n')
 for pair,order in enumerate(protocol['order'],1):
  for arm in order:
   name=f'4b-lora-tp2pp2-s512-adapted-local-pair{pair}-{arm}-20261102-v1';out=B/'runs'/name;assert not out.exists()
   assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],env=helper.ENV,text=True)
   helper.ENV.update(PERF_ARM='reference' if arm=='native' else 'triton',PERF_MODE='timing',PERF_LOCAL_ARM=arm,PHASE_OUTPUT=str(out))
   cmd=[sys.executable,str(B/'tools/adapted_local_perf.py'),'train','--model','4b','--task','lora','--parallel','tp2-pp2','--backend','reference' if arm=='native' else 'triton','--megatron-root',str(B/'Megatron-LM'),'--run-dir',str(out)]
   for v in ['model.path='+json.dumps(str(helper.MODEL)),'task.recompute=false','task.seq_length=512','task.micro_batch_size=1','task.global_batch_size=16','task.train_iters=25','task.save_interval=1000','task.eval_iters=0','task.data_path='+json.dumps(str(B/'data/4b-lora-seq512-resolved'))]:cmd+=['--set',v]
   helper.run(name,cmd,3600)
   for i in range(4):
    d=read(out/f'performance-rank{i}.json');g=read(out/f'normalized-gc-rank{i}.json');install=read(out/f'normalized-install-rank{i}.json')
    assert d['completed'] and len(d['steps'])==25 and g['setup']['auto_gc_enabled_after'] and install['arm']==arm
    assert all(x['skipped']==0 and math.isfinite(x['grad_norm']) and all(math.isfinite(v) for v in x['losses'].values()) for x in d['steps'])
   helper.STATUS[name]['validated']=True;helper.save_status()
 (ROOT/'complete.json').write_text(json.dumps({'completed':True,'pairs':4})+'\n');print('LOCAL_PAIRS_COMPLETE',flush=True)
if __name__=='__main__':main()
