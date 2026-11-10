"""Common synchronous native-format persistence and DP1 packing instrumentation."""
from pathlib import Path
import functools,hashlib,json,os,time


def install(root, mode, audit=False):
    import torch
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from dp1_checkpoint_pack_20261107 import pack as direct
    from coalesced_state_20261109 import pack as coalesced
    from checkpoint_diagnostic_20261107 import fingerprint as cpu_fp
    from savepack_resume_20261107 import fingerprint as fp
    assert mode in ['baseline','current','coalesced']
    slot=Path(os.environ['E2E_CHECKPOINT_SLOT']).resolve()
    assert slot.is_relative_to(Path('/private/yp/jiuge-training/checks/20261109-e2e-coalesced'))
    original_get=DistributedOptimizer.get_parameter_state_dp_zero
    original_save=torch.save
    calls=[];writes=[]

    @functools.wraps(original_get)
    def getter(self,use_gloo_comm=True,empty_data=False,return_on_all_ranks=False):
        active=bool(use_gloo_comm and not empty_data and not return_on_all_ranks and self.data_parallel_group.size()==1)
        start=time.perf_counter()
        if active and mode!='baseline':state=coalesced(self) if mode=='coalesced' else direct(self,'direct')
        else:state=original_get(self,use_gloo_comm=use_gloo_comm,empty_data=empty_data,return_on_all_ranks=return_on_all_ranks)
        elapsed=time.perf_counter()-start
        if audit and active:
            actual=cpu_fp(state)
            if mode!='baseline':
                native=original_get(self);assert actual==cpu_fp(native);del native
        calls.append({'mode':mode,'dp1_active':active,'seconds':elapsed})
        return state

    @functools.wraps(original_save)
    def save(obj,f,*args,**kwargs):
        if not isinstance(f,(str,os.PathLike)):return original_save(obj,f,*args,**kwargs)
        path=Path(f).resolve()
        if not path.is_relative_to(slot):return original_save(obj,f,*args,**kwargs)
        assert torch.serialization.get_crc32_options(), 'CRC32 must remain enabled'
        assert not path.exists(), 'fresh slot per task; never truncate a frozen checkpoint'
        temporary=path.with_name(path.name+'.pending-rank'+os.environ['RANK'])
        assert not temporary.exists()
        if audit and path.name=='distrib_optim.pt':
            (root/f'prewrite-optimizer-rank{os.environ["RANK"]}.json').write_text(json.dumps(cpu_fp(obj),indent=2)+'\n')
        started=time.perf_counter();original_save(obj,temporary,*args,**kwargs)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        writes.append({'name':path.name,'bytes':path.stat().st_size,'seconds':time.perf_counter()-started,'fsync_and_rename_complete':True,'crc32_enabled':True})

    DistributedOptimizer.get_parameter_state_dp_zero=getter;torch.save=save
    def finish():
        metadata_synced=[]
        if writes and int(os.environ['RANK'])==0:
            for name in ['jiuge_contract.json','latest_checkpointed_iteration.txt']:
                target=slot/name;assert target.is_file()
                with target.open('rb') as stream:os.fsync(stream.fileno())
                metadata_synced.append(name)
            fd=os.open(slot,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        DistributedOptimizer.get_parameter_state_dp_zero=original_get;torch.save=original_save
        (root/f'checkpoint-runtime-rank{os.environ["RANK"]}.json').write_text(json.dumps({'mode':mode,'calls':calls,'writes':writes,'audit':audit,'metadata_synced':metadata_synced},indent=2)+'\n')
    return finish
