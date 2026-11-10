"""Verify the original evidence archive, safely extract, recompute all results.

Python3.11+ standard library only. This reviews saved GPU-audit receipts; it does
not replace a new GPU execution or ship raw50GB checkpoints.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, subprocess, sys, tarfile, tempfile

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='b474c80e8d9fb9a4d85b0b73218a686e64ecfb5bef638b92bfdb23b61108afb9'

def extract(archive,destination):
 with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==EXPECTED,'Archive SHA256 mismatch'
 seen=set()
 with tarfile.open(archive,'r:gz') as pack:
  for member in pack:
   rel=PurePosixPath(member.name)
   assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name and ':' not in member.name,member.name
   assert member.name not in seen,'Duplicate archive member'
   seen.add(member.name);path=destination.joinpath(*rel.parts)
   assert path.resolve().is_relative_to(destination.resolve())
   path.parent.mkdir(parents=True,exist_ok=True)
   with pack.extractfile(member) as source,path.open('xb') as out:out.write(source.read())
 assert len(seen)==843

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--extract-to',type=Path,help='Optional NEW destination to retain extracted evidence; otherwise a temporary directory is used')
 args=parser.parse_args()
 archive=ROOT/'results/e2e-20261110/key-evidence-e2e-coalesced-20261109-continued.tar.gz'
 def verify(where):
  extract(archive,where)
  subprocess.run([sys.executable,str(ROOT/'scripts/verify_e2e_20261110.py'),str(where)],check=True)
  result=json.loads((where/'checks/20261109-e2e-coalesced/campaign-v2/complete.json').read_text())
  print('CPT throughput +%.4f%%; time -%.4f%%; target=%s' % ((result['total_speed_ratio']-1)*100,result['total_time_reduction']*100,result['cpt_final_target_met']))
 if args.extract_to:
  args.extract_to.mkdir(parents=True,exist_ok=False);verify(args.extract_to)
 else:
  with tempfile.TemporaryDirectory(prefix='.jiuge-evidence-',dir=ROOT) as tmp:
   destination=Path(tmp).resolve()
   assert destination.is_relative_to(ROOT.resolve())
   verify(destination)

if __name__=='__main__':main()
