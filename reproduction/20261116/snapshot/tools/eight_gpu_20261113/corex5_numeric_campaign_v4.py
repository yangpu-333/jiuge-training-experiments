"""Port norm order to Torch2.10; reuse verified reference and RAM-loader results."""
from pathlib import Path
import json,hashlib,shutil
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
assert json.loads((E/'numeric-v3/tp8-lora-ram-restore-review.json').read_text())['passed']
probe=json.loads((E/'norm-probe-v3.json').read_text())
assert len(probe['cases'])==3
assert all(v['unequal']==0 and v['finite'] for row in probe['cases'] for k,v in row.items() if k!='scale')

def reuse(out,name,arm):
    old=E/'numeric-v3/cases'/name/arm
    assert json.loads((old/'run-result.json').read_text())['returncode']==0
    for rank in range(8):assert json.loads((old/f'metrics-rank{rank}.json').read_text())['completed']
    assert not any(p.suffix=='.pt' for p in old.rglob('*'))
    out.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(old,out)
    manifest={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old.rglob('*') if p.is_file()}
    for n,h in manifest.items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h
    (out/'REUSED_V3.json').write_text(json.dumps({'source':str(old),'sha256':manifest,
      'reason':'Norm-order change affects only fused arm; verified reference/RAM runs reused'},indent=2)+'\n')

original=B/'tools/eight_gpu_20261113/corex5_numeric_campaign.py'
source=original.read_text().replace("R=E/'numeric-v1'","R=E/'numeric-v4'")
source=source.replace("corex5_numeric_worker.py'","corex5_numeric_worker_v4.py'")
needle="dependencies=[B/'tools/numeric_training.py'"
extra="dependencies=["+','.join("B/'tools/eight_gpu_20261113/"+n+"'" for n in ['corex5_norm_order.py','corex5_numeric_worker.py','corex5_numeric_worker_v2.py','corex5_numeric_worker_v3.py','corex5_numeric_campaign.py'])+",B/'tools/numeric_training.py'"
assert source.count(needle)==1;source=source.replace(needle,extra)
needle="            out=R/'cases'/name/arm;out.mkdir(parents=True)"
assert source.count(needle)==1
source=source.replace(needle,"            out=R/'cases'/name/arm\n            if name=='tp8-lora' and arm in ['reference','ram-restore']:\n                reuse(out,name,arm);continue\n            out.mkdir(parents=True)")
needle="            assert shutil.disk_usage(B).free>=10*2**30"
assert source.count(needle)==1
source=source.replace(needle,"""            if name=='tp8-lora' and arm in ['reference','ram-restore']:
                records.append({'case':name,'arm':arm,'returncode':0,'reused_from':'numeric-v3'});write(R/'runs.json',records)
                if arm=='ram-restore':
                    result=review(R/'cases'/name/'reference',R/'cases'/name/arm,arm,protocol)
                    write(R/f'{name}-{arm}-review.json',result);assert result['passed']
                print('REUSED',name,arm,flush=True);continue
"""+needle)
namespace={'__name__':'__main__','__file__':str(Path(__file__).resolve()),'reuse':reuse}
exec(compile(source,str(original),'exec'),namespace)
