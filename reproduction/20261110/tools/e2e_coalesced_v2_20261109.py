"""Unprofiled 50-step complete-task entry; common durable native checkpoint writes."""
from pathlib import Path
import os,sys,json
import e2e_training_20261106 as base
base.SCRIPT=Path(__file__).resolve()
if '--e2e-worker' in sys.argv:
    import jiuge_training.train as baseline
    baseline.configure_worker_caches();baseline.configure_worker_caches=lambda:None
    from checkpoint_round_common_v2_20261109 import install
    root=Path(sys.argv[sys.argv.index('--jiuge-config')+1]).parent
    mode=os.environ['BENCH_ARM'];os.environ['E2E_ARM']='baseline' if mode=='baseline' else 'optimized'
    finish=install(root,mode)
    try:base.worker()
    finally:finish()
else:base.driver()
