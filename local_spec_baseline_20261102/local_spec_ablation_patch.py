"""Isolate native RMSNorm and native attention in separate fixed-model runs."""
import functools,hashlib,inspect,json,os,sys
from pathlib import Path
import jiuge_ops
import jiuge_training.model as model_module
from local_spec_patch import LocalRMSNorm,NineGridLocalAttention

def install():
    variant=os.environ['LOCAL_SPEC_ABLATION'];assert variant in ['norm-only','attention-only']
    original_norm=jiuge_ops.ReferenceRMSNorm;original_attention=model_module.HFCompatibleAttention
    if variant=='norm-only':jiuge_ops.ReferenceRMSNorm=LocalRMSNorm
    else:model_module.HFCompatibleAttention=NineGridLocalAttention
    norm_class=LocalRMSNorm if variant=='norm-only' else original_norm
    attn_class=NineGridLocalAttention if variant=='attention-only' else original_attention
    import megatron.training.training as mcore
    original=mcore.train_step;root=Path(sys.argv[sys.argv.index('--jiuge-config')+1]).parent
    @functools.wraps(original)
    def step(*args,**kwargs):
        bound=inspect.signature(original).bind(*args,**kwargs).arguments
        if bound['iteration']==0:
            chunks=bound['model'];norms=[m for c in chunks for m in c.modules() if type(m) is norm_class];attn=[m for c in chunks for m in c.modules() if type(m) is attn_class]
            assert norms and attn
            receipt={'variant':variant,'norm_class':norm_class.__name__,'attention_class':attn_class.__name__,'norm_count':len(norms),'attention_count':len(attn),'patch_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (root/f'local-spec-install-rank{os.environ["RANK"]}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return original(*args,**kwargs)
    mcore.train_step=step
