"""Versioned optimizer-loader repair; reuse only the verified completed reference."""
from pathlib import Path
import json,hashlib,shutil
B=Path('/private/yp/jiuge-training');E=B/'checks/20261113-eight-gpu-corex5'
OLD=E/'numeric-v1/cases/tp8-lora/reference'
assert json.loads((OLD/'run-result.json').read_text())['returncode']==0
for rank in range(8):
    assert json.loads((OLD/f'metrics-rank{rank}.json').read_text())['completed']
    assert json.loads((OLD/f'numeric/coverage-rank{rank}.json').read_text())['prepared_calls']==10

def reuse_reference(out):
    shutil.copytree(OLD,out)
    manifest={str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD.rglob('*') if p.is_file()}
    assert not any(p.suffix=='.pt' for p in OLD.rglob('*'))
    for name,digest in manifest.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
    (out/'REUSED_REFERENCE.json').write_text(json.dumps({'source':str(OLD),'files_sha256':manifest,
       'reason':'Only RAM-loader dispatch changed; reference math, captures and conditions unchanged'},indent=2)+'\n')

original=B/'tools/eight_gpu_20261113/corex5_numeric_campaign.py'
source=original.read_text()
assert source.count("R=E/'numeric-v1'")==1
source=source.replace("R=E/'numeric-v1'","R=E/'numeric-v2'")
source=source.replace("corex5_numeric_worker.py'","corex5_numeric_worker_v2.py'")
# Lock the shared original worker and both campaign sources too.
needle="dependencies=[B/'tools/numeric_training.py'"
assert source.count(needle)==1
source=source.replace(needle,"dependencies=[B/'tools/eight_gpu_20261113/corex5_numeric_worker.py',B/'tools/eight_gpu_20261113/corex5_numeric_campaign.py',B/'tools/numeric_training.py'")
needle="            out=R/'cases'/name/arm;out.mkdir(parents=True)"
assert source.count(needle)==1
source=source.replace(needle,"            out=R/'cases'/name/arm\n            if name=='tp8-lora' and arm=='reference':\n                out.parent.mkdir(parents=True,exist_ok=True);reuse_reference(out);continue\n            out.mkdir(parents=True)")
needle="            assert shutil.disk_usage(B).free>=10*2**30"
assert source.count(needle)==1
source=source.replace(needle,"            if name=='tp8-lora' and arm=='reference':\n                records.append({'case':name,'arm':arm,'returncode':0,'reused_from':str(OLD)});write(R/'runs.json',records);print('REUSED',name,arm,flush=True);continue\n"+needle)
namespace={'__name__':'__main__','__file__':str(Path(__file__).resolve()),'reuse_reference':reuse_reference,'OLD':OLD}
exec(compile(source,str(original),'exec'),namespace)
