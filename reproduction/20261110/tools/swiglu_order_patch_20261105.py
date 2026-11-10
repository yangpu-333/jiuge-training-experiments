"""Experimental MLP patch: only for audited dense zero-bias SwiGLU."""
from pathlib import Path
import copy,atexit,json,os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as ce
B=Path('/private/yp/jiuge-training'); _mod=None
def extension():
 global _mod
 if _mod is None:
  ce.COREX_HOME='/usr/local/corex';ce.CUDA_HOME='/usr/local/corex'
  d=B/'cache/torch-extensions/swiglu-order-20261105';d.mkdir(parents=True,exist_ok=True)
  _mod=ce.load(name='jiuge_swiglu_order_20261105',sources=[str(B/'tools/swiglu_cpp_20261105.cpp'),str(B/'tools/swiglu_order_20261105.cu')],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','-ffp-contract=off'],build_directory=str(d),with_cuda=True,verbose=False)
 return _mod
def install():
 from megatron.core.transformer.mlp import MLP
 counts={'eligible':0,'fallback':0}
 def report():
  root=os.environ.get('PHASE_OUTPUT')
  if root: (Path(root)/('swiglu-install-rank'+os.environ.get('RANK','unknown')+'.json')).write_text(json.dumps(counts)+'\n')
 atexit.register(report)
 original=MLP.forward
 def forward(self,hidden_states,per_token_scale=None,**kwargs):
  c=self.config
  eligible=(not c.use_te_activation_func and not c.bias_activation_fusion and c.gated_linear_unit and c.activation_func is F.silu and c.activation_func_clamp_value is None and c.glu_linear_offset==0 and not c.add_bias_linear and per_token_scale is None)
  if not eligible:
   counts['fallback']+=1
   return original(self,hidden_states,per_token_scale=per_token_scale,**kwargs)
  counts['eligible']+=1
  # Isolate this module's configuration from other modules sharing the object.
  # Concurrent calls on the same module remain unsupported in this prototype.
  old=self.activation_func;old_config=self.config
  try:
   self.config=copy.copy(c);self.config.use_te_activation_func=True;self.activation_func=extension().swiglu
   return original(self,hidden_states,per_token_scale=per_token_scale,**kwargs)
  finally:
   self.activation_func=old;self.config=old_config
 MLP.forward=forward

