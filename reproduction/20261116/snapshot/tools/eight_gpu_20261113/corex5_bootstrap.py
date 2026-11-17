"""Create an independent CoreX 5.0 preflight area; never modify old runs."""
from pathlib import Path
import datetime,json,os,shutil,socket,subprocess

B=Path('/private/yp/jiuge-training')
R=B/'checks/20261113-eight-gpu-corex5'

def main():
    assert not R.exists(), 'Never replace an existing run'
    assert shutil.disk_usage(B).free >= 10*2**30
    idle=subprocess.check_output(['/usr/local/iluvatar/bin/ixsmi'],text=True)
    assert 'No running processes found' in idle
    R.mkdir()
    package=R/'overlay/megatron';package.mkdir(parents=True)
    # The vendor regular package otherwise shadows the pinned namespace package.
    (package/'__init__.py').write_text('__path__ = ["/private/yp/jiuge-training/Megatron-LM/megatron"]\n')
    env={
      'LD_LIBRARY_PATH':'/usr/local/openmpi/lib:/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/iluvatar/lib64',
      'PYTHONPATH':':'.join(str(p) for p in [R/'overlay',B/'tools',B/'source/src',B/'source/.deps',B/'Megatron-LM']),
      'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS':'1',
      'MAX_JOBS':'1','USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false',
      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
      'TRITON_CACHE_DIR':str(R/'cache/triton'),'TORCH_EXTENSIONS_DIR':str(R/'cache/torch-extensions'),
      'HF_HOME':str(R/'cache/huggingface'),'XDG_CACHE_HOME':str(R/'cache/xdg'),
    }
    (R/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    receipt={'container_hostname':socket.gethostname(),'resource_name':'PUBLIC_POD',
      'host':'PRIVATE_HOST','port':32752,'requested_gpu_count':8,
      'allocation_basis':'User provisioned eight-card environment; independently verify eight devices and idleness before probes',
      'image':'bi150-5.0.0-x86-ubuntu24.04-py3.12-llm-training-codeserver',
      'expires_cst':None,'created_cst':datetime.datetime.now().astimezone().isoformat(),
      'shm_bytes':os.statvfs('/dev/shm').f_blocks*os.statvfs('/dev/shm').f_frsize,
      'persistent_free_bytes':shutil.disk_usage(B).free,'checkpoint_writes_allowed':False}
    (R/'resource.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (R/'ixsmi-before.txt').write_text(idle)
    print(json.dumps({'prepared':str(R),'training_started':False}))

if __name__=='__main__':main()
