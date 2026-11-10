"""Install an isolated, hash-checked replay of the measured CPT campaign.

Only scheduling/campaign paths change. Workers, arithmetic, tolerances, full
save/fsync/CRC/fingerprint/load checks, orders, timing and statistics stay intact.
"""
from pathlib import Path
import argparse, ast, difflib, hashlib, json, re, shutil, subprocess, sys

HERE=Path(__file__).resolve().parent
ROOT=Path('/private/yp/jiuge-training')
UPSTREAM='9fabd71df4897f08267df5431aa480ab9ef70a75'
MCORE='5be9626709af2722333bf54797c954c09edeada3'

def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def adapted_controller(name):
 original=(HERE/'tools/run_e2e_coalesced_v2_20261109.py').read_text()
 updated=original
 replacements=[
  ("R=A/'campaign-v2'", "R=A/"+repr(name)),
  ("'capacity':'retain >=10GiB; budget full slot plus256MiB; user-approved exact archival of three old CRC-failed files prerequisite'", "'capacity':'fresh project: full slot plus256MiB, retain >=10GiB; no historic archive prerequisite'"),
  ("'deadline':'no new job after 2026-11-10 02:00 CST, each job timeout1200s; environment expires02:30'", "'deadline':'no calendar deadline; same1200s per-job timeout; provision enough paid/allocated resource before launch'"),
  ("idle();assert datetime.datetime.now()<datetime.datetime(2026,11,10,2,0),'job admission deadline'", "idle()"),
  (" deadline=time.monotonic()+7200\n while not (A/'legacy-corruption-archive/compact-complete.json').exists():\n  assert not (A/'legacy-corruption-archive/failure.txt').exists() and not (A/'legacy-corruption-archive/compact-failure.txt').exists();assert time.monotonic()<deadline;time.sleep(10)\n assert read(A/'legacy-corruption-archive/compact-complete.json')['full_readback_exact'];idle()", " idle()  # Fresh workspace; no historical corruption archive to wait for."),
  (" | set((A/'legacy-corruption-archive').glob('*.json')) | set(A.glob('twins-dedup-*.json')) | {A/'superseded-before-gpu.json',A/'durability-capability/passed.json'}", ""),
  ("frozen={str(B/'tools'/name):sha(B/'tools'/name) for name in names}", "names=sorted(set(names)|{p.name for p in (B/'tools').iterdir() if p.is_file() and p.suffix in {'.py','.cpp','.cu'}})\nfrozen={str(B/'tools'/name):sha(B/'tools'/name) for name in names}")
 ]
 for before,after in replacements:
  assert updated.count(before)==1, before
  updated=updated.replace(before,after)
 ast.parse(updated)
 return original,updated

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--install',action='store_true',help='Copy verified tools into the fixed fresh Linux project; default only validates publication')
 parser.add_argument('--run-name',default='replication-20261110',help='Unique new child under checks/20261109-e2e-coalesced')
 args=parser.parse_args()
 assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_name)
 lock=json.loads((HERE/'TOOLS_SHA256.json').read_text())
 for name,item in lock.items():
  path=HERE/'tools'/name
  assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
  if path.suffix=='.py':ast.parse(path.read_text())
 original,updated=adapted_controller(args.run_name)
 print('PASS publication:',len(lock),'frozen tool files; fresh controller parses; numeric and restore gates retained')
 if not args.install:return
 assert sys.platform.startswith('linux'),'GPU replay requires Linux/CoreX'
 assert ROOT.is_dir() and not (ROOT/'tools').exists(),'Use a fresh project; existing tools are never overwritten'
 campaign=ROOT/'checks/20261109-e2e-coalesced'/args.run_name
 assert not campaign.exists(),'Never reuse or overwrite a previous measurement directory'
 for path,commit in [(ROOT/'source',UPSTREAM),(ROOT/'Megatron-LM',MCORE)]:
  assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==commit,path
  subprocess.run(['git','-C',str(path),'diff','--quiet',commit],check=True)
 manifest=json.loads((HERE/'inputs/source-manifest.original.json').read_text())
 for rel,digest in manifest['source_files_sha256'].items():assert sha(ROOT/'source'/rel)==digest,rel
 assert sha(ROOT/'source/environment.lock.json')==manifest['environment_lock_sha256']
 shutil.copytree(HERE/'tools',ROOT/'tools')
 generated=ROOT/'tools/run_reproduction_20261110.py'
 generated.write_text(updated,encoding='utf-8',newline='\n')
 (ROOT/'checks/20261109-e2e-coalesced').mkdir(parents=True,exist_ok=True)
 gate=ROOT/'checks/20261031-fused-norm/numerical/protocol.json'
 assert not gate.exists(),'Do not replace an existing numerical protocol'
 gate.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(HERE/'inputs/numerical-gate.json',gate)
 runtime=ROOT/'runtime-overrides.json'
 assert not runtime.exists(),'Do not replace existing runtime settings'
 # Versioned public settings only; no copied account credentials or private endpoints.
 overrides={'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'cache/inductor'),'TORCH_EXTENSIONS_DIR':str(ROOT/'cache/torch-extensions'),'USE_TF':'0','USE_FLAX':'0'}
 overrides['PYTHONPATH']=str(ROOT/'source/.deps')
 runtime.write_text(json.dumps(overrides,indent=2)+'\n')
 diff=''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='measured-controller',tofile='fresh-workspace-controller'))
 (ROOT/'reproduction-controller.diff').write_text(diff)
 adaptation={'upstream':UPSTREAM,'megatron':MCORE,'run_root':str(campaign),'original_controller_sha256':sha(HERE/'tools/run_e2e_coalesced_v2_20261109.py'),'generated_controller_sha256':sha(generated),'workers':'unchanged exact published bytes','changes':'unique new campaign path; remove historical disk-archive wait and calendar admission deadline; collect only new evidence; freeze transitive tool sources too','numerical_gate':json.loads(gate.read_text()),'preflight':'same6 full-save/native-restore jobs warm all3 arms before timed50-step jobs','fresh_gpu_replay_performed_during_publication':False}
 (ROOT/'reproduction-adaptation.json').write_text(json.dumps(adaptation,indent=2)+'\n')
 print('Installed. Next: prepare/check inputs, collective smoke, then python -u',generated)

if __name__=='__main__':main()
