"""Execute a command with the inspected, isolated 5.0 environment."""
from pathlib import Path
import json,os,sys
B=Path('/private/yp/jiuge-training')
R=B/'checks/20261113-eight-gpu-corex5'
env=os.environ.copy();env.update(json.loads((R/'environment.json').read_text()))
assert len(sys.argv)>1
os.chdir(B)
os.execvpe(sys.argv[1],sys.argv[1:],env)
