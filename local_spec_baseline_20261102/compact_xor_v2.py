"""Lossless candidate tensors as compressed XOR to immutable reference bytes."""
import hashlib,shutil,zlib
from pathlib import Path
import numpy as np
import torch
import compact_tensor_pack as original

FORMAT='lossless-reference-xor-zlib-tensor-v2'

def raw(t):return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()

def encode(reference,value):
 assert reference.shape==value.shape and reference.dtype==value.dtype
 a,b=raw(reference),raw(value);assert a.size==b.size
 return zlib.compress(np.bitwise_xor(a,b).tobytes(),1)

def decode(reference,payload):
 a=raw(reference);delta=np.frombuffer(zlib.decompress(payload.contiguous().numpy().tobytes()),dtype=np.uint8)
 assert delta.size==a.size
 data=np.bitwise_xor(a,delta)
 return torch.from_numpy(data).view(reference.dtype).reshape(reference.shape)

def make_pack(schema,values,reference,kind,metadata=None):
 before=original.file_stat(reference);old=original.load_reference(reference,kind)
 assert len(schema)==len(values)==len(old['schema']);changed={};raw_bytes=0
 for i,(entry,refentry,ref,value) in enumerate(zip(schema,old['schema'],old['values'],values)):
  assert {k:v for k,v in entry.items() if k!='sha256'}=={k:v for k,v in refentry.items() if k!='sha256'}
  raw_bytes+=value.numel()*value.element_size()
  if entry['sha256']!=refentry['sha256']:
   assert original.tensor_hash(ref)=={k:v for k,v in refentry.items() if k!='name'}
   assert original.tensor_hash(value)=={k:v for k,v in entry.items() if k!='name'}
   compressed=encode(ref,value)
   changed[i]=torch.from_numpy(np.frombuffer(compressed,dtype=np.uint8).copy())
 assert before==original.file_stat(reference)
 # Four ranks may write concurrently. Retain 10 GiB plus more than a full
 # four-rank uncompressed snapshot; fail before writing if headroom shrinks.
 free=shutil.disk_usage(original.BASE).free
 assert free>int(raw_bytes*4.5)+10*2**30,('insufficient disk for concurrent snapshot',free,raw_bytes)
 return {'compact_format':FORMAT,'kind':kind,'reference_path':str(reference),'reference_stat':before,'reference_schema_sha256':original.schema_hash(old['schema']),'schema':schema,'changed':changed,'metadata':metadata or {},'counts':{'tensors':len(schema),'changed_tensors':len(changed),'total_tensor_bytes':raw_bytes,'stored_tensor_bytes':sum(x.numel() for x in changed.values()),'compression':'zlib on XOR to fixed reference'}}

def restore(path):
 packed=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
 if packed.get('compact_format')!=FORMAT:return original.restore(path)
 ref=Path(packed['reference_path']);assert ref.resolve().is_relative_to(original.BASE)
 assert packed['reference_stat']==original.file_stat(ref)
 source=original.load_reference(ref,packed['kind']);assert original.schema_hash(source['schema'])==packed['reference_schema_sha256']
 values=[]
 for i,(schema,old_schema,old) in enumerate(zip(packed['schema'],source['schema'],source['values'])):
  assert original.tensor_hash(old)=={k:v for k,v in old_schema.items() if k!='name'}
  value=decode(old,packed['changed'][i]) if i in packed['changed'] else old
  assert original.tensor_hash(value)=={k:v for k,v in schema.items() if k!='name'}
  values.append(value)
 assert original.file_stat(ref)==packed['reference_stat']
 return {'schema':packed['schema'],'values':values,'metadata':packed['metadata'],'counts':packed['counts']}
