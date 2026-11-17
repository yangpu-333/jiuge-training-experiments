"""Fingerprint native distributed-optimizer dtype keys without changing tensors."""
from pathlib import Path
import torch
import corex5_numeric_worker as base
original_fingerprint=base.fingerprint
def fingerprint(value):
    if isinstance(value,torch.dtype):return {'torch_dtype':str(value)}
    return original_fingerprint(value)
base.fingerprint=fingerprint
original=Path(__file__).with_name('corex5_numeric_worker_v4.py')
exec(compile(original.read_text(),str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
