import hashlib
import json
from pathlib import Path
import statistics

BASE=Path('/private/yp/jiuge-training');ROOT=BASE/'checks/20261102-lora-tp/performance'

def main():
    assert json.loads((ROOT/'complete.json').read_text())['completed']
    out={'scope':'Exploratory LoRA A+TP C++ comparison with identical warmup-end full GC; auto GC enabled; not a GC runtime optimization or formal E2E result','tasks':{},'input_hashes':{}}
    prereg=json.loads((ROOT.parent/'numerical/preregistration.json').read_text())
    for entry in prereg['files']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    binaries=set()
    for task,pairs in [('lora',4),('cpt',1)]:
        records=[]
        for pair in range(1,pairs+1):
            arms={};oldcfg=None;oldscalar=None
            for arm in ['current','candidate']:
                run=BASE/'runs'/f'4b-{task}-tp2pp2-s512-lora-a-tp-pair{pair}-{arm}-20261102-v1'
                cfg=json.loads((run/'config.json').read_text());cfg['task'].pop('save')
                if oldcfg is None:oldcfg=cfg
                else:assert cfg==oldcfg
                rs=[json.loads((run/f'performance-rank{i}.json').read_text()) for i in range(4)]
                gc=[json.loads((run/f'normalized-gc-rank{i}.json').read_text()) for i in range(4)]
                installs=[json.loads((run/f'normalized-install-rank{i}.json').read_text()) for i in range(4)]
                for install in installs:
                    assert install['requested_lora_tp']==(arm=='candidate')
                    effective=arm=='candidate' and task=='lora'
                    assert install['lora_tp_active']==effective
                    expected=BASE/'tools'/('fused_norm_dw.py' if task=='cpt' else 'fused_split_norm.py')
                    assert install['selected_norm_sha256']==hashlib.sha256(expected.read_bytes()).hexdigest()
                    binaries.add(install['binary_sha256'])
                    assert install['tp_binary_sha256']==hashlib.sha256((BASE/'cache/torch-extensions/lora-tp-linear-v1/jiuge_lora_tp_linear_v1.so').read_bytes()).hexdigest()
                phases=[json.loads((run/f'phase-rank{i}.json').read_text())['rows'] for i in range(4)]
                scalars=[[{k:v for k,v in x.items() if k!='seconds'} for x in r['steps']] for r in rs]
                if oldscalar is None:oldscalar=scalars
                else:assert oldscalar==scalars
                values=[max(r['steps'][i]['seconds'] for r in rs) for i in range(5,25)]
                gen2=[dict(x,rank=i) for i,g in enumerate(gc) for x in g['events'] if x['generation']==2 and x['iteration'] is not None and x['iteration']>=5]
                arms[arm]={'mean_seconds':statistics.mean(values),'median_seconds':statistics.median(values),'step_seconds':values,'min_seconds':min(values),'max_seconds':max(values),'measured_gen2_events':gen2,'maxrank_setup_gc_seconds':max(g['setup']['full_gc_seconds'] for g in gc),'setup_cost_recorded':True,'auto_gc_enabled_all':all(g['setup']['auto_gc_enabled_after'] for g in gc),'process_cpu_seconds_rank0':statistics.mean(x['process_cpu_seconds'] for x in phases[0][5:]),'run':str(run)}
                for i in range(4):
                    for n in [f'performance-rank{i}.json',f'normalized-gc-rank{i}.json',f'phase-rank{i}.json']:out['input_hashes'][str(run/n)]=hashlib.sha256((run/n).read_bytes()).hexdigest()
            records.append({'pair':pair,'arms':arms,'speedup':arms['current']['mean_seconds']/arms['candidate']['mean_seconds'],'scalar_parity':True})
        means={a:statistics.mean(x['arms'][a]['mean_seconds'] for x in records) for a in ['current','candidate']}
        out['tasks'][task]={'pairs':records,'means':means,'speedup_ratio_of_means':means['current']/means['candidate'],'all_measured_gen2_absent':all(not p['arms'][a]['measured_gen2_events'] for p in records for a in ['current','candidate']),'scope':'LoRA A+TP vs accepted C++ A/B' if task=='lora' else 'Unchanged CPT negative control, no performance claim'}
    assert len(binaries)==1
    out['binaries']=list(binaries);out['preregistered_inputs_unchanged']=True
    (ROOT/'performance-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print('LORA_TP_PERF_SUMMARY_COMPLETE',json.dumps({k:{n:v[n] for n in ['means','speedup_ratio_of_means','all_measured_gen2_absent']} for k,v in out['tasks'].items()}),flush=True)

if __name__=='__main__':main()

