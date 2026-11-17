"""Retain v1/v2 failures; fresh v3 uses native scheduler fresh-counter lifecycle."""
from pathlib import Path
original=Path(__file__).with_name('corex5_numeric_campaign_v2.py')
source=original.read_text()
assert source.count("R=E/'numeric-v2'")==1
source=source.replace("R=E/'numeric-v2'","R=E/'numeric-v3'")
source=source.replace("corex5_numeric_worker_v2.py'","corex5_numeric_worker_v3.py'")
namespace={'__name__':'__main__','__file__':str(Path(__file__).resolve())}
exec(compile(source,str(original),'exec'),namespace)
