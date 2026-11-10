"""Standard-library checks for reproducibility packaging and controller changes."""
from pathlib import Path
import ast, hashlib, importlib.util, json, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reproduction/20261110'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 subprocess.run([sys.executable,str(P/'prepare.py')],check=True)
 module=load('reproduction_prepare',P/'prepare.py')
 before,after=module.adapted_controller('validation_only')
 def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
 a,b=functions(before),functions(after)
 assert a.keys()==b.keys()
 for name in a:
  if name!='run':assert a[name]==b[name],name
 assert "R=A/'validation_only'" in after
 assert 'job admission deadline' not in after
 assert 'legacy-corruption-archive/compact-complete.json' not in after
 assert 'range(20000)' in after and 'random.Random(1234)' in after
 assert (ROOT/'README.md').read_bytes()==(ROOT/'WEEKLY_REPORT_20261110.md').read_bytes()
 tools=P/'tools'
 needed=['lora_linear_cpp.cpp','lora_tp_linear.cpp','swiglu_cpp_20261105.cpp','swiglu_order_20261105.cu','native_order_reductions.py','fused_split_norm.py','lora_linear_patch.py','local_spec_patch.py','swiglu_order_patch_20261105.py']
 assert all((tools/name).is_file() for name in needed)
 for path in P.rglob('*.py'):ast.parse(path.read_text())
 gate=json.loads((P/'inputs/numerical-gate.json').read_text())
 assert gate=={'loss':{'atol':.001,'rtol':.001},'gradient_norm':{'atol':1e-12,'rtol':.01}}
 result=json.loads((ROOT/'results/e2e-20261110/complete.json').read_text())
 assert result['cpt_final_target_met'] and len(result['cycles'])==6
 print('PASS unchanged controller math/checks; native sources, report consistency and syntax')
 manifest=ROOT/'publication-manifest.json'
 if manifest.exists():
  obj=json.loads(manifest.read_text())
  for entry in obj['files']:
   path=ROOT/entry['path'];assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],entry['path']
  print('PASS publication manifest:',len(obj['files']),'files')

if __name__=='__main__':main()
