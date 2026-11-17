"""Validate decreasing-shuffle-order port with CPU FP64 metrics."""
from pathlib import Path
import corex5_norm_order
corex5_norm_order.install_order()
original=Path(__file__).with_name('corex5_norm_probe.py')
source=original.read_text().replace('.double().cpu()','.cpu().double()')
source=source.replace('norm-probe-v1.json','norm-probe-v3.json')
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
