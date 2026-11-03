"""MCore local operators with explicit nine-grid model-semantic adapters."""
import functools,hashlib,inspect,json,os,sys
from pathlib import Path
import torch
from megatron.core.transformer.dot_product_attention import DotProductAttention
import jiuge_ops
import jiuge_training.model as model_module

class LocalRMSNorm(torch.nn.RMSNorm):
    """The class returned by WrappedTorchNorm is RMSNorm; subclass records identity."""
    def __init__(self,hidden_size,eps=1e-6,config=None,**kwargs):
        assert config is not None and config.normalization=='RMSNorm'
        assert not config.layernorm_zero_centered_gamma and not config.persist_layer_norm
        assert not config.sequence_parallel and not config.memory_efficient_layer_norm
        super().__init__(normalized_shape=hidden_size,eps=eps)

class NineGridLocalAttention(DotProductAttention):
    """Preserve FM9G FP32 rotary, then round Q/K as the nine-grid attention does."""
    def forward(self,query,key,value,attention_mask,*args,**kwargs):
        return super().forward(query.to(value.dtype),key.to(value.dtype),value,attention_mask,*args,**kwargs)

def install():
    # model_provider imports jiuge_ops lazily and references this attention global.
    jiuge_ops.ReferenceRMSNorm=LocalRMSNorm
    # Both audit aliases name the same native class; the CLI backend label alone
    # is not evidence of implementation and is recorded separately below.
    jiuge_ops.TritonRMSNorm=LocalRMSNorm
    model_module.HFCompatibleAttention=NineGridLocalAttention
    import megatron.training.training as mcore
    original=mcore.train_step;root=Path(sys.argv[sys.argv.index('--jiuge-config')+1]).parent
    @functools.wraps(original)
    def step(*args,**kwargs):
        bound=inspect.signature(original).bind(*args,**kwargs).arguments
        if bound['iteration']==0:
            chunks=bound['model'];norms=[m for c in chunks for m in c.modules() if type(m) is LocalRMSNorm]
            attention=[m for c in chunks for m in c.modules() if type(m) is NineGridLocalAttention]
            assert norms and attention
            receipt={'norm_count':len(norms),'attention_count':len(attention),'norm_operation':'torch.nn.RMSNorm inherited forward, as WrappedTorchNorm','attention_operation':'MCore DotProductAttention with FM9G Q/K cast after rotary','retained_model_semantics':'nine-grid residual scaling, embeddings, output and rotary hooks; config from fixed 4B model','patch_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (root/f'local-spec-install-rank{os.environ["RANK"]}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return original(*args,**kwargs)
    mcore.train_step=step
