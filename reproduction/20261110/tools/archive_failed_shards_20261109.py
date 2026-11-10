"""User-approved lossless archive of three historical CRC-failed optimizer files."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile,time,traceback
import zstandard as zstd
B=Path('/private/yp/jiuge-training');R=B/'checks/20261109-e2e-coalesced'
old=B/'checks/20261106-new-account-round-v2/e2e/checkpoint-slots/cpt-optimized'
deadline=time.monotonic()+3600
while not (R/'twins-dedup-complete.json').exists():
 assert not (R/'twins-dedup-failure.txt').exists()
 assert time.monotonic()<deadline;time.sleep(10)
P=R/'legacy-corruption-archive';P.mkdir(exist_ok=False)
rows=[];archive=P/'old-cpt-crc-failed-three-shards.tar.zst'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
try:
 hashes=json.loads((B/'checks/20261106-new-account-round-v2/e2e/cpt-pair6-optimized/checkpoint-file-hashes.json').read_text())
 for rank in ['00_001','01_000','01_001']:
  rel=f'iter_0000050/mp_rank_{rank}/distrib_optim.pt';p=old/rel
  assert p.resolve().is_relative_to(old.resolve()) and not p.is_symlink()
  st=p.stat();assert st.st_size==hashes[rel]['bytes'];h=sha(p);assert h==hashes[rel]['sha256']
  now=p.stat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns)
  rows.append({'original_path':str(p),'member':str(p.relative_to(B)),'bytes':st.st_size,'sha256':h,'inode':st.st_ino,'mode':st.st_mode,'mtime_ns':st.st_mtime_ns})
  print('SOURCE_HASH_VERIFIED',p,flush=True)
 raw_bytes=sum(r['bytes'] for r in rows)
 assert shutil.disk_usage(B).free>raw_bytes+512*2**20+10*2**30
 manifest={'authorization':'explicit user permission 2026-11-09: losslessly archive these three known CRC-failed optimizer originals, verify every byte before removing uncompressed copies', 'archive':str(archive),'source_total_bytes':raw_bytes,'files':rows,'restore':'Use zstandard.ZstdDecompressor().stream_reader then tarfile.open(mode=r|); extract ONLY named member into a fresh temporary file under its original project directory, verify its recorded SHA256/bytes, fsync, atomically replace absent original path, restore mode/mtime; budget free space first. Never load old CRC-failed files as trusted training controls.'}
 (P/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with archive.open('xb') as f:
  with zstd.ZstdCompressor(level=3,threads=8).stream_writer(f,closefd=False) as compressed:
   with tarfile.open(fileobj=compressed,mode='w|') as tar:
    for row in rows:tar.add(row['original_path'],arcname=row['member'],recursive=False)
  f.flush();os.fsync(f.fileno())
 print('ARCHIVE_WRITTEN',archive.stat().st_size,flush=True)
 expected={r['member']:r for r in rows};seen=set()
 with archive.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as decoded:
  with tarfile.open(fileobj=decoded,mode='r|') as tar:
   for member in tar:
    assert member.isfile() and member.name in expected and member.name not in seen
    row=expected[member.name];h=hashlib.sha256();n=0
    with tar.extractfile(member) as src:
     while chunk:=src.read(16*2**20):n+=len(chunk);h.update(chunk)
    assert n==row['bytes']==member.size and h.hexdigest()==row['sha256'];seen.add(member.name)
    print('ARCHIVED_BYTES_EXACT',member.name,flush=True)
 assert seen==set(expected)
 manifest.update(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,full_readback_exact=True)
 (P/'verified-before-release.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for row in rows:
  p=Path(row['original_path']);st=p.stat()
  assert p.resolve().is_relative_to(old.resolve()) and (st.st_ino,st.st_size,st.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns'])
  p.unlink()  # ONLY explicitly approved, fully archived and verified files.
  row['uncompressed_released']=True
  (P/'release-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
 (old/'ARCHIVED_OPTIMIZER_FILES_20261109.json').write_text(json.dumps(manifest,indent=2)+'\n')
 manifest.update(completed=True,all_uncompressed_released=True,free_after_bytes=shutil.disk_usage(B).free)
 (P/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('LOSSLESS_ARCHIVE_COMPLETE',manifest['free_after_bytes'],flush=True)
except BaseException:
 (P/'failure.txt').write_text(traceback.format_exc());raise
