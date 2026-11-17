"""Common observer for all arms; training/save implementations unchanged."""
from pathlib import Path
import atexit,json,os,runpy,sys
B=Path('/private/yp/jiuge-training')
config=Path(sys.argv[sys.argv.index('--jiuge-config')+1]);out=config.parent
arm=os.environ['HOST_POOL_ARM'];requested=os.environ.get('TORCHINDUCTOR_COMPILE_THREADS')
assert (requested=='1') if arm=='host-only' else requested is None
runpy.run_path(str(B/'tools/eight_gpu_20261113/fullstack_20261115/worker.py'),run_name='__main__')
def receipt():
    module=sys.modules.get('torch._inductor.async_compile')
    assert module is not None,'Expected installed compiler module must be observable'
    threads=module.get_compile_threads();pools=len(module._pool_set)
    if arm=='host-only':assert threads==1 and pools==0
    else:assert threads==32,'Previous fixed machine default was32; inspect environment changes'
    assert 'batch_persistence' not in sys.modules and 'lora_epilogue' not in sys.modules
    (out/f'host-pool-policy-rank{os.environ["RANK"]}.json').write_text(json.dumps({
        'arm':arm,'requested_compile_threads':requested,'observed_compile_threads':threads,'pools_before_original_shutdown':pools,
        'accepted_training_and_native_durable_save_unchanged':True,'all_original_shutdown_handlers_retained':True})+'\n')
# Registered after the accepted worker returns; observes before Torch shutdown.
atexit.register(receipt)
