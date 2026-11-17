"""CPU-FP64 diagnostic metrics; supersedes invalid GPU-FP64 probe v1."""
from pathlib import Path
original=Path(__file__).with_name('corex5_norm_probe.py')
source=original.read_text()
assert '.double().cpu()' in source
source=source.replace('.double().cpu()','.cpu().double()')
source=source.replace("norm-probe-v1.json","norm-probe-v2.json")
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
