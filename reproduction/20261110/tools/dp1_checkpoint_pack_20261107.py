"""DP1-only legacy optimizer state assembly, preserving the native CPU schema."""
import torch

def pack(optimizer,mode='direct'):
 assert mode in ['direct','pinned']
 assert optimizer.data_parallel_group.size()==optimizer.data_parallel_group_gloo.size()==1
 keys=('param',)+optimizer.optimizer_state_keys;layouts=[];devices=set()
 # Validate all source ranges before allocation/copy. No arithmetic or state mutation.
 for gidx,maps in enumerate(optimizer.gbuf_ranges):
  assert len(maps)==1
  for dtype,buckets in maps.items():
   numel=optimizer.buffers[gidx].numel_unpadded;offset=0;pieces=[]
   for bidx,mapping in enumerate(buckets):
    bucket=optimizer.buffers[gidx].buckets[bidx];unpadded=bucket.numel_unpadded
    for param,ranges in mapping['param_map'].items():
     start=offset+ranges['gbuf_local'].start;end=offset+ranges['gbuf_local'].end
     assert offset<=start<=end<=offset+unpadded<=numel
     states=optimizer._get_main_param_and_optimizer_states(param)
     for key in keys:
      x=states[key].detach();assert x.dtype==torch.float32 and x.ndim==1 and x.numel()==end-start and x.device.type=='cuda'
      devices.add(x.device);pieces.append((key,start,end,x))
    offset+=unpadded
   assert offset==numel
   layouts.append((gidx,dtype,numel,pieces))
 assert len(devices)==1
 state={'buckets_coalesced':True}
 for gidx,dtype,numel,pieces in layouts:
  tensors={key:torch.zeros((numel,),dtype=torch.float32,device='cpu',pin_memory=mode=='pinned') for key in keys}
  tensors['numel_unpadded']=numel
  for key,start,end,x in pieces:
   if mode=='direct':tensors[key][start:end].copy_(x.cpu())
   else:tensors[key][start:end].copy_(x,non_blocking=True)
  if mode=='pinned':torch.cuda.current_stream(next(iter(devices))).synchronize()
  state[gidx]={dtype:tensors}
 return state

def install(mode='pinned'):
 from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
 original=DistributedOptimizer.get_parameter_state_dp_zero
 def get(self,use_gloo_comm=True,empty_data=False,return_on_all_ranks=False):
  if not use_gloo_comm or empty_data or return_on_all_ranks or self.data_parallel_group.size()!=1:
   return original(self,use_gloo_comm=use_gloo_comm,empty_data=empty_data,return_on_all_ranks=return_on_all_ranks)
  return pack(self,mode)
 DistributedOptimizer.get_parameter_state_dp_zero=get
 return original
