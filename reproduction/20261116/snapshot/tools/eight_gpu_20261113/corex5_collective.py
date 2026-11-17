"""Eight-rank correctness-only collective probe, not a performance claim."""
import datetime,json,os
from pathlib import Path
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE'])==8 and torch.cuda.device_count()==8
torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
x=torch.tensor([rank+1.0],device='cuda');dist.all_reduce(x)
assert x.item()==36
y=[torch.empty_like(x) for _ in range(8)]
dist.all_gather(y,torch.tensor([rank+1.0],device='cuda'))
assert [a.item() for a in y]==list(range(1,9))
dist.barrier();torch.cuda.synchronize()
out=Path('/private/yp/jiuge-training/checks/20261113-eight-gpu-corex5')
(out/f'collective-rank{rank}.json').write_text(json.dumps({'rank':rank,'device':local,'all_reduce':36,'all_gather':True,'passed':True})+'\n')
if rank==0:print('EIGHT_RANK_COLLECTIVES_PASSED',flush=True)
dist.destroy_process_group()
