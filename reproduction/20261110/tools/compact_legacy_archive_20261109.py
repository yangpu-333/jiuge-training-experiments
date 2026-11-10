"""Improve approved archival compression using reversible byte shuffling, not quantization."""
from pathlib import Path
import hashlib,io,json,os,shutil,struct,tarfile,time,traceback
import numpy as np
import zstandard as zstd
B=Path('/private/yp/jiuge-training');P=B/'checks/20261109-e2e-coalesced/legacy-corruption-archive'
MAGIC=b'JIUGE-SHUFFLE4-ZSTD-V1\n';BLOCK=16*2**20
def shuffle(data):
 n=len(data)//4*4
 return np.frombuffer(data[:n],dtype=np.uint8).reshape(-1,4).T.copy().tobytes()+data[n:]
def unshuffle(data):
 n=len(data)//4*4
 return np.frombuffer(data[:n],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+data[n:]
def exact(stream,n):
 data=bytearray()
 while len(data)<n:
  block=stream.read(n-len(data))
  if not block:raise EOFError('truncated archival frame')
  data.extend(block)
 return bytes(data)
def frames(path):
 with path.open('rb') as f:
  assert f.read(len(MAGIC))==MAGIC
  with zstd.ZstdDecompressor().stream_reader(f) as stream:
   while header:=stream.read(4):
    if len(header)<4:header+=exact(stream,4-len(header))
    n=struct.unpack('<I',header)[0];assert 0<n<=BLOCK
    yield unshuffle(exact(stream,n))
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
 try:
  deadline=time.monotonic()+3600
  while not (P/'complete.json').exists():
   assert not (P/'failure.txt').exists();assert time.monotonic()<deadline;time.sleep(10)
  old=json.loads((P/'complete.json').read_text());assert old['full_readback_exact']
  source=Path(old['archive']);assert sha(source)==old['archive_sha256']
  assert shutil.disk_usage(B).free>old['source_total_bytes']+512*2**20+10*2**30
  table={r['member']:r for r in old['files']};outputs=[]
  for data in [b'',b'a',b'ab',b'abc',bytes(range(255)),bytes(range(256))*7]:assert unshuffle(shuffle(data))==data
  with source.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as decoded:
   with tarfile.open(fileobj=decoded,mode='r|') as tar:
    for member in tar:
     row=table[member.name];out=P/('rank-'+Path(row['original_path']).parent.name+'.pt.shuf4.zst');assert not out.exists()
     h=hashlib.sha256();count=0
     with tar.extractfile(member) as input,out.open('xb') as output:
      output.write(MAGIC)
      with zstd.ZstdCompressor(level=5,threads=8,write_checksum=True).stream_writer(output,closefd=False) as encoded:
       while block:=input.read(BLOCK):
        h.update(block);count+=len(block);encoded.write(struct.pack('<I',len(block)));encoded.write(shuffle(block))
      output.flush();os.fsync(output.fileno())
     assert count==row['bytes'] and h.hexdigest()==row['sha256']
     h=hashlib.sha256();count=0
     for block in frames(out):h.update(block);count+=len(block)
     assert count==row['bytes'] and h.hexdigest()==row['sha256']
     outputs.append({**row,'compact_archive':str(out),'compact_bytes':out.stat().st_size,'compact_sha256':sha(out),'full_readback_exact':True})
     (P/'compact-progress.json').write_text(json.dumps(outputs,indent=2)+'\n');print('COMPACT_ORIGINAL_BYTES_EXACT',member.name,out.stat().st_size,flush=True)
  assert len(outputs)==3
  result={'completed':True,'full_readback_exact':True,'files':outputs,'codec':'magic header then zstd stream of [uint32 little-endian original block length, reversible byte-shuffled bytes]; 16MiB blocks, transpose4byte columns; no floating-point operations or lossy conversion','restore':'import frames from compact_legacy_archive_20261109.py; concatenate yielded original byte blocks into a NEW temporary file under the approved original directory, verify original SHA256 and bytes, fsync, atomically restore original absent path; restore mode and mtime from row; budget space first','superseded_archive':str(source),'superseded_archive_sha256':old['archive_sha256'],'superseded_archive_bytes':old['archive_bytes'],'original_member_metadata':old['files']}
  (P/'compact-verified-before-release.json').write_text(json.dumps(result,indent=2)+'\n')
  assert source.resolve().is_relative_to(P.resolve()) and source.name=='old-cpt-crc-failed-three-shards.tar.zst'
  source.unlink()  # Only this run's redundant archive; all three preimages verified again.
  result['free_after_bytes']=shutil.disk_usage(B).free
  (P/'compact-complete.json').write_text(json.dumps(result,indent=2)+'\n')
  old_root=B/'checks/20261106-new-account-round-v2/e2e/checkpoint-slots/cpt-optimized'
  (old_root/'ARCHIVED_OPTIMIZER_FILES_20261109.json').write_text(json.dumps(result,indent=2)+'\n')
  print('COMPACT_ARCHIVE_COMPLETE',result['free_after_bytes'],flush=True)
 except BaseException:
  (P/'compact-failure.txt').write_text(traceback.format_exc());raise
