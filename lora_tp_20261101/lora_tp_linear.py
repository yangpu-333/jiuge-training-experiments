from pathlib import Path
import torch.distributed as dist
from torch.utils.cpp_extension import load
BASE=Path('/private/yp/jiuge-training');_extension=None
def extension():
    global _extension
    if _extension is None:
        build=BASE/'cache/torch-extensions/lora-tp-linear-v1';build.mkdir(parents=True,exist_ok=True)
        _extension=load(name='jiuge_lora_tp_linear_v1',sources=[str(BASE/'tools/lora_tp_linear.cpp')],extra_cflags=['-O2'],build_directory=str(build),with_cuda=False,verbose=True)
    return _extension
def linear_tp(x,weight,group,column):return extension().linear_tp(x,weight,group,column)
def install():
    extension()
    from jiuge_training.lora import ParallelLoRA
    from lora_linear_cpp import linear
    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
    def forward(self,input_,*args,**kwargs):
        assert self.dropout.p==0 and self.rank==8
        output,bias=self.base(input_,*args,**kwargs);dropped=self.dropout(input_)
        active=self.tp_group is not None and self.tp_group.size()>1
        # Direct PG calls must not bypass an explicitly requested coalescing manager.
        coalescing=active and self.tp_group in dist.distributed_c10d._world.pg_coalesce_state
        if active and not coalescing:
            low_rank=linear_tp(dropped,self.lora_A,self.tp_group,self.kind=='column')
        else:
            low_rank=linear(dropped,self.lora_A)
            if active:
                if self.kind=='column':low_rank=copy_to_tensor_model_parallel_region(low_rank,group=self.tp_group)
                else:low_rank=reduce_from_tensor_model_parallel_region(low_rank,group=self.tp_group)
        return output+linear(low_rank,self.lora_B)*self.scaling,bias
    ParallelLoRA.forward=forward
