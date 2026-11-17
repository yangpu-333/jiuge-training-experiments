"""Native dtype-key audit repair; reuse four verified completed runs."""
from pathlib import Path
import json,hashlib,shutil
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
REUSE={('tp8-lora',a) for a in ['reference','ram-restore','fused-norm']}|{('tp8-cpt','reference')}
for arm in ['ram-restore','fused-norm']:
    assert json.loads((E/f'numeric-v4/tp8-lora-{arm}-review.json').read_text())['passed']

def reuse(out,name,arm):
    old=E/'numeric-v4/cases'/name/arm
    assert json.loads((old/'run-result.json').read_text())['returncode']==0
    for rank in range(8):
        assert json.loads((old/f'metrics-rank{rank}.json').read_text())['completed']
        assert json.loads((old/f'numeric/coverage-rank{rank}.json').read_text())['prepared_calls']==10
    assert not any(p.suffix=='.pt' for p in old.rglob('*'))
    out.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(old,out)
    manifest={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if p.is_file()}
    for n,h in manifest.items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h
    (out/'REUSED_V4.json').write_text(json.dumps({'source':str(old),'sha256':manifest,
       'reason':'Add torch.dtype metadata support only; prior tensor and numeric fingerprints unchanged'},indent=2)+'\n')

original=B/'tools/eight_gpu_20261113/corex5_numeric_campaign.py'
source=original.read_text().replace("R=E/'numeric-v1'","R=E/'numeric-v5'")
source=source.replace("corex5_numeric_worker.py'","corex5_numeric_worker_v5.py'")
needle="dependencies=[B/'tools/numeric_training.py'"
names=['corex5_norm_order.py','corex5_numeric_worker.py','corex5_numeric_worker_v2.py','corex5_numeric_worker_v3.py','corex5_numeric_worker_v4.py','corex5_numeric_campaign.py']
extra="dependencies=["+','.join("B/'tools/eight_gpu_20261113/"+n+"'" for n in names)+",B/'tools/numeric_training.py'"
assert source.count(needle)==1;source=source.replace(needle,extra)
needle="            out=R/'cases'/name/arm;out.mkdir(parents=True)"
assert source.count(needle)==1
source=source.replace(needle,"            out=R/'cases'/name/arm\n            if (name,arm) in REUSE:\n                reuse(out,name,arm);continue\n            out.mkdir(parents=True)")
needle="            assert shutil.disk_usage(B).free>=10*2**30"
assert source.count(needle)==1
source=source.replace(needle,"""            if (name,arm) in REUSE:
                records.append({'case':name,'arm':arm,'returncode':0,'reused_from':'numeric-v4'});write(R/'runs.json',records)
                if arm!='reference':
                    result=review(R/'cases'/name/'reference',R/'cases'/name/arm,arm,protocol)
                    write(R/f'{name}-{arm}-review.json',result);assert result['passed']
                print('REUSED',name,arm,flush=True);continue
"""+needle)
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve()),'reuse':reuse,'REUSE':REUSE})
