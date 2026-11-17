"""Candidate native column sum, preserving row fusion and both backward edges."""
from pathlib import Path
import os,sys,json
if os.environ['COREX5_AUDIT_ARM']=='fused-norm':
    cfg=json.loads(Path(sys.argv[sys.argv.index('--jiuge-config')+1]).read_text())
    if cfg['task']['name']=='cpt':
        import corex5_norm_dw_native
        corex5_norm_dw_native.install()
original=Path(__file__).with_name('corex5_numeric_worker_v5.py')
exec(compile(original.read_text(),str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
