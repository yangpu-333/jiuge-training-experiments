"""Isolated CoreX5 port of the accepted four-device training components."""
from pathlib import Path
import atexit,copy,functools,hashlib,inspect,json,os,sys,time
ROOT=Path(__file__).resolve().parent
B=Path('/private/yp/jiuge-training')
CACHE=B/'checks/20261113-eight-gpu-corex5/cache/torch-extensions/fullstack-20261115-v1'
_extensions={}

def extension(name):
    if name not in _extensions:
        from torch.utils.cpp_extension import load
        import torch.utils.cpp_extension as ce
        ce.COREX_HOME=ce.CUDA_HOME='/usr/local/corex'
        build=CACHE/name;build.mkdir(parents=True,exist_ok=True)
        sources=[ROOT/(name+'.cpp')]
        if name=='swiglu':sources.append(ROOT/'swiglu.cu')
        _extensions[name]=load(name='jiuge_corex5_'+name+'_20261115_v1',sources=[str(p) for p in sources],
            extra_cflags=['-O2'],extra_cuda_cflags=['-O2','-ffp-contract=off'],
            build_directory=str(build),with_cuda=name=='swiglu',verbose=True)
    return _extensions[name]

def norm(cfg):
    import corex5_norm_order,fused_split_norm,fused_norm_dw
    corex5_norm_order.install_order()
    if cfg['task']['name']=='cpt':
        import corex5_norm_dw_native
        corex5_norm_dw_native.install();fused_norm_dw.install()
    else:fused_split_norm.install()

def install(stage,cfg,out,include_norm=False):
    import torch
    assert torch.cuda.device_count()==8 and cfg['task']['recompute'] is False
    assert cfg['parallel']['dp']==1 and cfg['parallel']['tp']*cfg['parallel']['pp']==8
    rank=os.environ['RANK'];counts={'stage':stage,'swiglu_modules':0,'swiglu_fallback':0,
        'lora_calls':0,'tp_sizes':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ['.py','.cpp','.cu']}}
    atexit.register(lambda:(out/f'stack-install-rank{rank}.json').write_text(json.dumps(counts,indent=2)+'\n'))
    if stage=='adapted-reference':
        # Same attention-only semantic adapter as the four-card accepted baseline.
        import local_spec_patch,jiuge_training.model as model
        model.HFCompatibleAttention=local_spec_patch.NineGridLocalAttention
        return
    assert stage in ['swiglu','linear','tp','packed','training']
    if include_norm:norm(cfg)
    import torch.nn.functional as F
    from megatron.core.transformer.mlp import MLP
    swiglu=extension('swiglu').swiglu;old_init=MLP.__init__
    def initialize(self,*args,**kwargs):
        old_init(self,*args,**kwargs);c=self.config
        eligible=(not c.use_te_activation_func and not c.bias_activation_fusion and c.gated_linear_unit and c.activation_func is F.silu and c.activation_func_clamp_value is None and c.glu_linear_offset==0 and not c.add_bias_linear)
        if eligible:
            self.config=copy.copy(c);self.config.use_te_activation_func=True;self.activation_func=swiglu
            counts['swiglu_modules']+=1
        else:counts['swiglu_fallback']+=1
    MLP.__init__=initialize
    if stage=='swiglu' or cfg['task']['name']!='lora':return
    from jiuge_training.lora import ParallelLoRA
    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
    import torch.distributed as dist
    linear=extension('lora_linear').linear
    tp=extension('lora_tp').linear_tp if stage in ['tp','training','packed'] else None
    def forward(self,input_,*args,**kwargs):
        assert self.dropout.p==0 and self.rank==8
        output,bias=self.base(input_,*args,**kwargs);dropped=self.dropout(input_)
        active=self.tp_group is not None and self.tp_group.size()>1
        coalescing=active and self.tp_group in dist.distributed_c10d._world.pg_coalesce_state
        if tp is not None and active and not coalescing:
            size=self.tp_group.size();assert size in [2,4,8]
            counts['tp_sizes'][str(size)]=counts['tp_sizes'].get(str(size),0)+1
            low=tp(dropped,self.lora_A,self.tp_group,self.kind=='column')
        else:
            low=linear(dropped,self.lora_A)
            if active:
                low=copy_to_tensor_model_parallel_region(low,group=self.tp_group) if self.kind=='column' else reduce_from_tensor_model_parallel_region(low,group=self.tp_group)
        counts['lora_calls']+=1
        return output+linear(low,self.lora_B)*self.scaling,bias
    ParallelLoRA.forward=forward

def install_persistence(out,slot,packed,audit):
    """Common durable native writes; optional DP1 pack with byte-exact audit."""
    import torch
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    import coalesced_state
    import corex5_numeric_worker as state
    rank=os.environ['RANK'];slot=Path(slot).resolve()
    assert slot.is_relative_to(Path('/tmp/yp-jiuge-corex5-e2e-20261114/fullstack-20261115-v1'))
    original_get=DistributedOptimizer.get_parameter_state_dp_zero;original_save=torch.save
    calls=[];writes=[]
    def fp(value):
        if isinstance(value,torch.dtype):return {'torch_dtype':str(value)}
        return state.fingerprint(value)
    # Recursive fingerprint needs dtype support too; retain it for this worker.
    oldfp=state.fingerprint
    state.fingerprint=lambda value:{'torch_dtype':str(value)} if isinstance(value,torch.dtype) else oldfp(value)
    @functools.wraps(original_get)
    def get(self,*args,**kwargs):
        bound=inspect.signature(original_get).bind(self,*args,**kwargs);bound.apply_defaults();d=bound.arguments
        active=d.get('use_gloo_comm',True) and not d.get('empty_data',False) and not d.get('return_on_all_ranks',False) and self.data_parallel_group.size()==1
        start=time.monotonic()
        value=coalesced_state.pack(self) if packed and active else original_get(self,*args,**kwargs)
        seconds=time.monotonic()-start
        row={'packed':packed,'active':bool(active),'seconds':seconds}
        if audit and packed and active:
            native=original_get(self,*args,**kwargs)
            assert fp(value)==fp(native),'Packed CPU state differs from native'
            row['native_cpu_full_byte_exact']=True;del native
        calls.append(row);return value
    @functools.wraps(original_save)
    def save(value,path,*args,**kwargs):
        if not isinstance(path,(str,os.PathLike)):return original_save(value,path,*args,**kwargs)
        path=Path(path).resolve()
        if not path.is_relative_to(slot):return original_save(value,path,*args,**kwargs)
        assert torch.serialization.get_crc32_options() and not path.exists()
        temporary=path.with_name(path.name+'.pending-rank'+rank);assert not temporary.exists()
        start=time.monotonic();original_save(value,temporary,*args,**kwargs)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        writes.append({'path':str(path),'bytes':path.stat().st_size,'seconds':time.monotonic()-start,'fsync_rename_complete':True})
    DistributedOptimizer.get_parameter_state_dp_zero=get;torch.save=save
    def finish():
        (out/f'packing-runtime-rank{rank}.json').write_text(json.dumps({'packed':packed,'audit':audit,'calls':calls,'writes':writes},indent=2)+'\n')
    atexit.register(finish)
