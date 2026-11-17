"""Continue five verified runs; test native weight-column sum without new gates."""
from pathlib import Path
original=Path(__file__).with_name('corex5_numeric_campaign_v5.py')
source=original.read_text()
source=source.replace("R=E/'numeric-v5'","R=E/'numeric-v6'")
source=source.replace("corex5_numeric_worker_v5.py'","corex5_numeric_worker_v6.py'")
source=source.replace("{('tp8-cpt','reference')}","{('tp8-cpt','reference'),('tp8-cpt','ram-restore')}")
source=source.replace('numeric-v4/','numeric-v5/')
source=source.replace("names=['corex5_norm_order.py'","names=['corex5_norm_dw_native.py','corex5_numeric_worker_v5.py','corex5_norm_order.py'")
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
