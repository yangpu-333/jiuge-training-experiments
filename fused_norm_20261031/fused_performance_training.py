import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import performance_training as perf

if __name__=='__main__':
    if '--performance-worker' in sys.argv:
        baseline=importlib.import_module('jiuge_training.train')
        baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
        import reduction_patch
        import fused_split_norm
        if os.environ['PERF_FUSED']=='1':reduction_patch.install=fused_split_norm.install
        root=Path(os.environ['PHASE_OUTPUT'])
        (root/f'fusion-install-rank{os.environ["RANK"]}.json').write_text(json.dumps({'fused':os.environ['PERF_FUSED']=='1','fused_sha256':hashlib.sha256(Path(fused_split_norm.__file__).read_bytes()).hexdigest(),'baseline':'reductions-full-v1','two_edges_retained':True}))
        import phase_training
        phase_training.worker()
    else:
        perf.SCRIPT=Path(__file__).resolve();perf.driver()
