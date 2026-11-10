"""Rehash four frozen checkpoint twins, then preserve every byte/path with hardlinks."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,traceback
B=Path('/private/yp/jiuge-training');R=B/'checks/20261109-e2e-coalesced'
A=B/'checks/20261106-new-account-round-v2/e2e'
C=B/'checks/20261107-save-trace/cpt-e2e-direct'
assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
groups=[]
for arm,ranks in [('baseline',['00_000','00_001','01_001']),('optimized',['00_000'])]:
 left=json.loads((A/f'cpt-pair6-{arm}/checkpoint-file-hashes.json').read_text())
 right=json.loads((C/f'pair6-{arm}/checkpoint-file-hashes.json').read_text())
 for rank in ranks:
  rel=f'iter_0000050/mp_rank_{rank}/distrib_optim.pt'
  assert left[rel]==right[rel]
  groups.append({'canonical':str(A/f'checkpoint-slots/cpt-{arm}'/rel),'duplicate':str(C/f'checkpoint-slots/{arm}'/rel),**left[rel]})
assert len(groups)==4
(R/'twins-dedup-protocol.json').write_text(json.dumps({'scope':'four SHA-identical frozen CPT optimizer shards only, byte/path preservation; no old checkpoint used for new training', 'groups':groups,'restore_independent':'copy verified bytes into fresh file then replace path; restore prior metadata from progress record','new_experiment_policy':'never write any linked archive path'},indent=2)+'\n')
records=[];before=shutil.disk_usage(B).free
def persist():
 (R/'twins-dedup-progress.json').write_text(json.dumps({'free_before':before,'free_now':shutil.disk_usage(B).free,'records':records},indent=2)+'\n')
try:
 for g in groups:
  paths=[Path(g['canonical']),Path(g['duplicate'])];metadata=[]
  for p in paths:
   assert p.resolve().is_relative_to(B.resolve()) and not p.is_symlink() and p.name=='distrib_optim.pt'
   old=p.stat();assert old.st_size==g['bytes']
   with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
   new=p.stat();assert h==g['sha256']
   assert (old.st_ino,old.st_size,old.st_mtime_ns,old.st_dev)==(new.st_ino,new.st_size,new.st_mtime_ns,new.st_dev)
   metadata.append({'path':str(p),'inode':old.st_ino,'device':old.st_dev,'mode':old.st_mode,'mtime_ns':old.st_mtime_ns,'bytes':old.st_size,'sha256':h})
   print('VERIFIED',p,flush=True)
  a,c=paths;assert a.stat().st_dev==c.stat().st_dev
  rec={'files_before':metadata,'linked':False};records.append(rec);persist()
  if a.stat().st_ino!=c.stat().st_ino:
   temporary=c.with_name(c.name+'.identical-link-20261109');assert not temporary.exists()
   os.link(a,temporary);assert temporary.stat().st_ino==a.stat().st_ino
   os.replace(temporary,c)
  assert a.stat().st_ino==c.stat().st_ino and a.stat().st_size==g['bytes']
  rec['linked']=True;persist();print('PATHS_AND_CONTENT_PRESERVED',c,flush=True)
 (R/'twins-dedup-complete.json').write_text(json.dumps({'completed':True,'all_paths_and_bytes_preserved':True,'free_before':before,'free_after':shutil.disk_usage(B).free,'records':records},indent=2)+'\n')
except BaseException:
 (R/'twins-dedup-failure.txt').write_text(traceback.format_exc());raise
