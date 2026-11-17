"""Use the CoreX5 native row-order port; retain v3 RAM-loader diagnostics."""
from pathlib import Path
import hashlib,json,os,sys
import corex5_norm_order
if os.environ['COREX5_AUDIT_ARM']=='fused-norm':
    corex5_norm_order.install_order()
    cfg=Path(sys.argv[sys.argv.index('--jiuge-config')+1])
    patch=Path(corex5_norm_order.__file__)
    (cfg.parent/f'corex5-row-order-rank{os.environ["RANK"]}.json').write_text(json.dumps({
      'shuffle_offsets':[32,16,8,4,2,1],'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),
      'source':'Installed Torch2.10 ATen/native/cuda/Reduce.cuh:689-700',
      'two_backward_edges_preserved':True})+'\n')
original=Path(__file__).with_name('corex5_numeric_worker_v3.py')
exec(compile(original.read_text(),str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
