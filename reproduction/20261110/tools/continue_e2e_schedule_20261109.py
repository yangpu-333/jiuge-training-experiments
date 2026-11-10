"""Only with explicit extension permission: continue unfinished fixed-order jobs, never remeasure completed jobs."""
from pathlib import Path
import datetime,hashlib,json,os,runpy,shutil,subprocess,sys,time,traceback
B=Path('/private/yp/jiuge-training');A=B/'checks/20261109-e2e-coalesced';R=A/'campaign-v2'
approval=A/'extension-authorization.json'
assert approval.exists() and json.loads(approval.read_text())['explicit_user_approved']
if os.environ.get('E2E_EXTENSION_ENV')!='1':
 e=dict(os.environ,**json.loads((B/'runtime-overrides.json').read_text()));e.update(E2E_EXTENSION_ENV='1',ROUND_E2E_ENV='1',CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='1',MAX_JOBS='1',E2E_USE_FAST='1',E2E_CHECKPOINT_SLOT=str(R/'scratch-checkpoint-slot'),AUDIT_PRESERVE_RESUME_DATALOADER_RNG='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
 e['PYTHONPATH']=':'.join([str(B/'tools'),str(B/'source/src'),str(B/'Megatron-LM'),e.get('PYTHONPATH','')]);os.execve(sys.executable,[sys.executable,'-u',__file__],e)
while not (R/'complete.json').exists() and not ((R/'failure.txt').exists() and (R/'archive-complete.json').exists()):
 assert datetime.datetime.now()<datetime.datetime(2026,11,10,3,30),'extension admission deadline'
 time.sleep(15)
if (R/'complete.json').exists():
 (A/'extension-not-needed.json').write_text(json.dumps({'original_campaign_completed':True,'no_extra_jobs_launched':True})+'\n');sys.exit(0)
assert 'job admission deadline' in (R/'failure.txt').read_text(),'never bypass correctness/runtime failures'
assert json.loads((R/'integrity-preflight-complete.json').read_text())['passed']
while Path('/proc/316/cmdline').exists():
 try:running=Path('/proc/316/cmdline').read_bytes()
 except FileNotFoundError:break
 if b'run_e2e_coalesced_v2_20261109.py' not in running:break
 time.sleep(1)
assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
status=json.loads((R/'status.json').read_text());assert all(not r['running'] and r['returncode']==0 for r in status.values())
protocol=json.loads((R/'protocol.json').read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,h in protocol['files'].items():assert sha(Path(name))==h
evidence=json.loads((R/'evidence-file-hashes.json').read_text())
for rel,item in evidence.items():
 p=B/rel;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
C=R/'scheduling-continuation-v3';C.mkdir(exist_ok=False)
for name in ['status.json','progress.json','protocol.json','failure.txt','archive-complete.json','evidence-file-hashes.json']:
 if (R/name).exists():shutil.copy2(R/name,C/('previous-'+name))
slot=R/'scratch-checkpoint-slot'
if slot.exists() and list(slot.rglob('*.pt')):
 labels=[label for label in status if label.startswith('cycle')]
 assert labels
 manifest=json.loads((R/labels[-1]/'checkpoint-file-hashes.json').read_text())
 for rel,item in manifest.items():
  p=slot/rel;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
source=B/'tools/run_e2e_coalesced_v2_20261109.py';s=source.read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
replace("assert not R.exists();R.mkdir()", "assert R.exists() and (R/'protocol.json').exists()")
replace("(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\\n')", "saved_protocol=json.loads((R/'protocol.json').read_text());assert saved_protocol['files']==frozen;protocol=saved_protocol")
replace("status={};safe_to_recycle=False;estimate=", "status=json.loads((R/'status.json').read_text());safe_to_recycle=True;estimate=")
replace("datetime.datetime(2026,11,10,2,0)", "datetime.datetime(2026,11,10,3,30)")
start=s.index(' # Separate real-write/load correctness campaign;')
end=s.index(' cycles=[];final_resume={}',start)
s=s[:start]+" assert read(R/'integrity-preflight-complete.json')['passed']\n"+s[end:]
replace("   fresh_slot();label=f'cycle{cycle}-{mode}'\n   elapsed,out=run(label,command('e2e_coalesced_v2_20261109.py',R/label,mode),mode)\n   row[mode]=elapsed;outputs[mode]=out;images[mode]=validate_slot(out,50)",
 "   label=f'cycle{cycle}-{mode}';out=R/label\n   if label in status:\n    assert not status[label]['running'] and status[label]['returncode']==0\n    elapsed=status[label]['seconds'];assert (out/'checkpoint-file-hashes.json').is_file()\n    image=read(out/'persisted-state-fingerprints.json')\n   else:\n    fresh_slot();elapsed,out=run(label,command('e2e_coalesced_v2_20261109.py',out,mode),mode);image=validate_slot(out,50)\n   row[mode]=elapsed;outputs[mode]=out;images[mode]=image")
replace("    _,resumed=run(label,command('integrity_coalesced_v2_20261109.py',R/label,mode,steps=51,load=True),mode,50)",
 "    if label in status:\n     assert not status[label]['running'] and status[label]['returncode']==0;resumed=R/label\n    else:_,resumed=run(label,command('integrity_coalesced_v2_20261109.py',R/label,mode,steps=51,load=True),mode,50)")
replace("key-evidence-e2e-coalesced-20261109.tar.gz", "key-evidence-e2e-coalesced-20261109-continued.tar.gz")
generated=C/'continued-controller.py'
extra="\nfrozen["+repr(str(generated))+"]=sha(Path("+repr(str(generated))+"));frozen["+repr(str(Path(__file__)))+"]=sha(Path("+repr(str(Path(__file__)))+"))\n"
replace("status=json.loads((R/'status.json').read_text());safe_to_recycle=True;estimate=", extra+"status=json.loads((R/'status.json').read_text());safe_to_recycle=True;estimate=")
compile(s,str(generated),'exec');generated.write_text(s)
amendment={'created_cst':datetime.datetime.now().isoformat(),'permission':json.loads(approval.read_text()),'scope':'scheduling extension only; reuse every completed measured time and complete validation record; continue only missing original-order tasks, original six cycles, comparisons, numerical gates and bootstrap unchanged', 'original_protocol_sha256':sha(R/'protocol.json'),'generated_controller_sha256':sha(generated),'continuation_wrapper_sha256':sha(Path(__file__)),'new_last_job_admission':'2026-11-10 03:30 CST','new_environment_expiry':'2026-11-10 04:00 CST','completed_labels_reused':list(status)}
(C/'amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
for name in ['evidence-file-hashes.json','archive-complete.json']:
 (R/name).rename(C/('original-'+name))
(R/'failure.txt').rename(C/'original-deadline-failure.txt')
runpy.run_path(str(generated),run_name='__main__')
