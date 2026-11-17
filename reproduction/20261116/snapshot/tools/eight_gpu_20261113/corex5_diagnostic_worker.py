"""Separate phase/LoRA profiler worker. Never used for performance acceptance."""
from pathlib import Path
import functools, gzip, inspect, json, os, shutil, sys, time

def main():
    cfgpath=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);root=cfgpath.parent
    cfg=json.loads(cfgpath.read_text());rank=int(os.environ['RANK'])
    kind=os.environ['COREX5_DIAGNOSTIC_KIND'];origin=time.monotonic()
    import torch
    import megatron.training.training as mcore
    events=[]
    def event(name, start, **extra):
        events.append({'name':name,'start_relative_s':start-origin,'duration_s':time.monotonic()-start,**extra})
        (root/f'phase-events-rank{rank}.json').write_text(json.dumps(events,indent=2)+'\n')
    def wrap(owner,name):
        if not hasattr(owner,name):return
        old=getattr(owner,name)
        @functools.wraps(old)
        def measured(*args,**kwargs):
            start=time.monotonic()
            try:return old(*args,**kwargs)
            finally:event(name,start)
        setattr(owner,name,measured)
    if kind=='phase-diagnostic':
        for name in ['setup_model_and_optimizer','build_train_valid_test_data_iterators','train_step']:
            wrap(mcore,name)
        import megatron.training.checkpointing as ck
        # Includes CPU staging and native serialization. Nested totals must
        # never be summed as disjoint phases.
        wrap(ck,'generate_state_dict')
        wrap(torch,'save')
    else:
        from jiuge_training.lora import ParallelLoRA
        import torch.nn.functional as F
        original=ParallelLoRA.forward
        active={'value':False}
        def lora(self,input_,*args,**kwargs):
            if not active['value']:return original(self,input_,*args,**kwargs)
            assert self.dropout.p==0 and self.rank==8
            scope=torch.profiler.record_function
            with scope('jiuge/lora_base'):out,bias=self.base(input_,*args,**kwargs)
            with scope('jiuge/lora_dropout'):x=self.dropout(input_)
            with scope('jiuge/lora_A'):low=F.linear(x,self.lora_A)
            with scope('jiuge/lora_TP'):
                if self.tp_group is not None and self.tp_group.size()>1:
                    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region,reduce_from_tensor_model_parallel_region
                    low=copy_to_tensor_model_parallel_region(low,group=self.tp_group) if self.kind=='column' else reduce_from_tensor_model_parallel_region(low,group=self.tp_group)
            with scope('jiuge/lora_B'):delta=F.linear(low,self.lora_B)
            with scope('jiuge/lora_scale'):delta=delta*self.scaling
            with scope('jiuge/lora_add'):out=out+delta
            return out,bias
        ParallelLoRA.forward=lora
        oldstep=mcore.train_step
        @functools.wraps(oldstep)
        def profiled(*args,**kwargs):
            bound=inspect.signature(oldstep).bind(*args,**kwargs).arguments
            if bound['iteration']!=5:return oldstep(*args,**kwargs)
            active['value']=True
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                    value=oldstep(*args,**kwargs)
                # Export uncompressed trace into anonymous RAM, then stream
                # gzip to persistent storage. No local-disk trace exception.
                fd=os.memfd_create(f'jiuge-trace-rank{rank}')
                try:
                    prof.export_chrome_trace(f'/proc/self/fd/{fd}')
                    os.lseek(fd,0,os.SEEK_SET)
                    with os.fdopen(os.dup(fd),'rb') as src,gzip.open(root/f'trace-rank{rank}.json.gz','wb',compresslevel=3) as dst:
                        while data:=src.read(1024*1024):
                            assert shutil.disk_usage(root).free>10*2**30,'Persistent reserve reached'
                            dst.write(data)
                    (root/f'profiler-table-rank{rank}.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=100))
                finally:os.close(fd)
                return value
            finally:active['value']=False
        mcore.train_step=profiled
    # Frozen disk worker installs the current accepted norm and runs the
    # official baseline; only this diagnostic process is instrumented.
    oldsave=mcore.save_checkpoint
    @functools.wraps(oldsave)
    def saved(*args,**kwargs):
        start=time.monotonic()
        try:return oldsave(*args,**kwargs)
        finally:event('native_save_checkpoint',start)
    mcore.save_checkpoint=saved
    import corex5_disk_worker
    corex5_disk_worker.main()
    event('diagnostic_worker_total',origin)

if __name__=='__main__':main()
