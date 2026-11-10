"""Initialize an isolated MLP activation binding once; no per-call mutation."""
from pathlib import Path
import copy,atexit,json,os,sys
import torch.nn.functional as F
from swiglu_order_patch_20261105 import extension

def install():
 from megatron.core.transformer.mlp import MLP
 root=os.environ.get('PHASE_OUTPUT')
 if not root and '--jiuge-config' in sys.argv:root=str(Path(sys.argv[sys.argv.index('--jiuge-config')+1]).parent)
 assert root,'installation receipt requires an explicit output directory'
 receipt=Path(root)/('swiglu-fast-install-rank'+os.environ.get('RANK','unknown')+'.json')
 original=MLP.__init__;fn=extension().swiglu;counts={'eligible_modules':0,'fallback_modules':0,'binding':'one-time private MLP config, unchanged order-correct C++ kernel'}
 def initialize(self,*args,**kwargs):
  original(self,*args,**kwargs)
  c=self.config
  eligible=(not c.use_te_activation_func and not c.bias_activation_fusion and c.gated_linear_unit and c.activation_func is F.silu and c.activation_func_clamp_value is None and c.glu_linear_offset==0 and not c.add_bias_linear)
  if eligible:
   self.config=copy.copy(c);self.config.use_te_activation_func=True;self.activation_func=fn;counts['eligible_modules']+=1
  else:counts['fallback_modules']+=1
 MLP.__init__=initialize
 def report():
  receipt.write_text(json.dumps(counts,indent=2)+'\n')
 atexit.register(report)
