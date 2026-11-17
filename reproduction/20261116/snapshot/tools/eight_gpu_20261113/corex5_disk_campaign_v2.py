"""Use the official checkpoint preparation interface before every new save."""
from pathlib import Path
p=Path(__file__).with_name('corex5_disk_campaign.py')
s=p.read_text().replace("R=E/'disk-e2e-v1'","R=E/'disk-e2e-v2'")
s=s.replace("T=Path('/tmp/yp-jiuge-corex5-e2e-20261114')","T=Path('/tmp/yp-jiuge-corex5-e2e-20261114/campaign-v2')")
needle="        cp=out/'config.json';write(cp,cfg);cmd=command(cfg,cp)"
assert s.count(needle)==1
s=s.replace(needle,"""        from jiuge_training.config import prepare_checkpoint_directory,validate_resume,validate
        from jiuge_training.data import validate_manifest
        cfg=validate(cfg)
        validate_manifest(cfg)
        if resume:validate_resume(cfg)
        else:prepare_checkpoint_directory(cfg)
"""+needle)
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
