"""Check native column-sum candidate before its full-model audit."""
from pathlib import Path
import corex5_norm_order,corex5_norm_dw_native
corex5_norm_order.install_order()
p=Path(__file__).with_name('corex5_norm_probe.py')
s=p.read_text().replace('.double().cpu()','.cpu().double()')
s=s.replace('norm-probe-v1.json','norm-probe-v4.json')
s=s.replace('fused_split_norm.FusedSplitRMSNorm.apply','corex5_norm_dw_native.NativeColumnNorm.apply')
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p),'corex5_norm_dw_native':corex5_norm_dw_native})
