"""First-order LoRA-only linear Function, retaining native BF16 GEMM layouts."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

class LoRALinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        assert x.dtype==weight.dtype==torch.bfloat16 and x.shape[-1]==weight.shape[1]
        ctx.save_for_backward(x,weight);ctx.input_shape=x.shape
        return F.linear(x,weight)
    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,w=ctx.saved_tensors
        gg=g.reshape(-1,w.shape[0]);xx=x.reshape(-1,w.shape[1])
        # Captured aten::mm native order: dW first, then dX. Preserve strides.
        dw=gg.t().mm(xx) if ctx.needs_input_grad[1] else None
        dx=gg.mm(w).reshape(ctx.input_shape) if ctx.needs_input_grad[0] else None
        return dx,dw

def linear(x,w):return LoRALinear.apply(x,w)

def install():
    from jiuge_training.lora import ParallelLoRA
    def forward(self,input_,*args,**kwargs):
        assert self.dropout.p==0 and self.rank==8
        output,bias=self.base(input_,*args,**kwargs)
        dropped=self.dropout(input_)
        low_rank=linear(dropped,self.lora_A)
        if self.tp_group is not None and self.tp_group.size()>1:
            from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
            if self.kind=='column':low_rank=copy_to_tensor_model_parallel_region(low_rank,group=self.tp_group)
            else:low_rank=reduce_from_tensor_model_parallel_region(low_rank,group=self.tp_group)
        return output+linear(low_rank,self.lora_B)*self.scaling,bias
    ParallelLoRA.forward=forward
