import hashlib,json,math,statistics
from pathlib import Path
B=Path('/private/yp/jiuge-training');ROOT=B/'checks/20261102-adapted-local-cpt/performance';NUM=ROOT.parent/'numerical'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(ROOT/'complete.json')['completed'];proto=read(ROOT/'protocol.json');gate=read(NUM/'cpt-direct-comparison-v2.json');assert gate['passed'] and sha(NUM/'cpt-direct-comparison-v2.json')==proto['direct_numeric_gate_sha256']
 for n,h in proto['tools'].items():assert sha(B/'tools'/n)==h
 tol=read(NUM/'protocol.json')['loss'];pairs=[];hashes={}
 for pair in range(1,5):
  arms={};configs=[];scalars=[]
  for arm in ['native','optimized']:
   run=B/'runs'/f'4b-cpt-tp2pp2-s512-adapted-local-pair{pair}-{arm}-20261102-v1';cfg=read(run/'config.json');cfg.pop('backend');cfg['task'].pop('save');configs.append(cfg)
   rs=[read(run/f'performance-rank{i}.json') for i in range(4)];gc=[read(run/f'normalized-gc-rank{i}.json') for i in range(4)];inst=[read(run/f'normalized-install-rank{i}.json') for i in range(4)]
   for i,x in enumerate(inst):
    assert x['arm']==arm
    name='local_spec_ablation_patch.py' if arm=='native' else 'fused_norm_dw.py';assert x['norm_or_patch_sha256']==sha(B/'tools'/name)
    if arm=='native':
     receipt=read(run/f'local-spec-install-rank{i}.json');assert receipt['norm_count'] and receipt['attention_count'] and receipt['patch_sha256']==sha(B/'tools/local_spec_ablation_patch.py')
   values=[max(r['steps'][i]['seconds'] for r in rs) for i in range(5,25)]
   gen2=[{'rank':i,**x} for i,g in enumerate(gc) for x in g['events'] if x['generation']==2 and x['iteration'] is not None and x['iteration']>=5]
   arms[arm]={'mean_seconds':statistics.mean(values),'median_seconds':statistics.median(values),'step_seconds':values,'measured_gen2_events':gen2,'auto_gc_enabled_all':all(g['setup']['auto_gc_enabled_after'] for g in gc),'setup_gc_max_seconds':max(g['setup']['full_gc_seconds'] for g in gc),'run':str(run)}
   scalars.append([[{'iteration':x['iteration'],'losses':x['losses'],'grad_norm':x['grad_norm'],'skipped':x['skipped']} for x in r['steps']] for r in rs])
   for i in range(4):
    for n in [f'performance-rank{i}.json',f'normalized-gc-rank{i}.json',f'normalized-install-rank{i}.json']:hashes[str(run/n)]=sha(run/n)
  assert configs[0]==configs[1]
  deviations=[]
  for rank in range(4):
   for a,b in zip(scalars[0][rank],scalars[1][rank]):
    assert a['iteration']==b['iteration'] and a['skipped']==b['skipped']==0 and a['losses'].keys()==b['losses'].keys()
    for k,v in a['losses'].items():
     delta=abs(v-b['losses'][k]);bound=tol['atol']+tol['rtol']*abs(v)
     if delta>bound:deviations.append({'rank':rank,'step':a['iteration']+1,'loss':k,'absolute':delta,'bound':bound})
  pairs.append({'pair':pair,'arms':arms,'speedup':arms['native']['mean_seconds']/arms['optimized']['mean_seconds'],'loss_tolerance_failures':deviations})
 means={a:statistics.mean(p['arms'][a]['mean_seconds'] for p in pairs) for a in ['native','optimized']}
 out={'scope':proto['scope'],'pairs':pairs,'means':means,'speedup_ratio_of_means':means['native']/means['optimized'],'all_measured_gen2_absent':all(not p['arms'][a]['measured_gen2_events'] for p in pairs for a in ['native','optimized']),'all_steps_loss_within_original_tolerance':all(not p['loss_tolerance_failures'] for p in pairs),'direct_numeric_gate_passed':True,'input_hashes':hashes}
 (ROOT/'performance-summary.json').write_text(json.dumps(out,indent=2)+'\n');print('LOCAL_PERF_SUMMARY',json.dumps({'means':means,'ratio':out['speedup_ratio_of_means'],'loss_within_gate':out['all_steps_loss_within_original_tolerance'],'no_gen2':out['all_measured_gen2_absent']}),flush=True)
if __name__=='__main__':main()
