from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[2];PACK=ROOT/'results/20261116'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'.verified-20261116');p.add_argument('--full',action='store_true');a=p.parse_args()
assert not a.output.exists(),'Use a fresh output directory'
for name,meta in json.loads((PACK/'PACKAGES.json').read_text()).items():
 f=PACK/meta['archive'];assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'];d=a.output/name;d.mkdir(parents=True)
 with tarfile.open(f) as t:
  for m in t:assert m.isfile() and (d/m.name).resolve().is_relative_to(d.resolve())
  t.extractall(d,filter='data')
 for row in json.loads((d/'PUBLICATION.json').read_text())['files']:
  f=d/row['path'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],f
 if a.full:
  subprocess.run([sys.executable,str(Path(__file__).parent/('review_'+name+'.py')),str(d/'evidence'),str(d/'original-numeric-protocol.json'),str(d/'independent-result-review.json')],check=True,stdout=subprocess.DEVNULL)
if a.full:
 subprocess.run([sys.executable,str(Path(__file__).parent/'aggregate.py'),str(a.output/'stage1'),str(a.output/'stage2')],check=True,stdout=subprocess.DEVNULL)
 assert json.loads((a.output/'stage2/all-layouts-e2e-summary.json').read_text())==json.loads((PACK/'stage2-all-layouts-e2e-summary.json').read_text())
print('PASS public packages/member hashes and full independent replay' if a.full else 'PASS package/member hashes')
