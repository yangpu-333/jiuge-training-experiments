"""Verify original FM9G-4B weights; materialize only explicit default metadata."""
from pathlib import Path
import argparse, hashlib, json, shutil

HERE=Path(__file__).resolve().parent
B=Path('/private/yp/jiuge-training')
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def model(source):
 source=source.resolve();target=B/'models/9G4B-resolved'
 assert not target.exists() and source!=target,'Keep original and normalized model paths separate; never overwrite an existing target'
 provenance=json.loads((HERE/'inputs/MODEL_SOURCE.json').read_text())
 for entry in provenance['members']:
  p=source/Path(entry['name']).name
  assert p.is_file() and p.stat().st_size==entry['size'] and sha(p)==entry['sha256'],p
 target.mkdir(parents=True)
 for entry in provenance['members']:
  p=source/Path(entry['name']).name
  if p.name!='config.json':(target/p.name).symlink_to(p,target_is_directory=p.is_dir())
 shutil.copyfile(source/'config.json',target/'config.original.json')
 shutil.copyfile(HERE/'inputs/config.resolved.json',target/'config.json')
 expected=json.loads((HERE/'inputs/data-manifest.original.json').read_text())['tokenizer_files_sha256']
 for name,digest in expected.items():assert sha(target/name)==digest,name
 print('PASS original model bytes and normalized tokenizer/config fingerprints; weights untouched')

def data():
 expected=json.loads((HERE/'inputs/data-manifest.original.json').read_text())
 path=B/'data/4b-cpt-seq512-resolved'
 observed=json.loads((path/'manifest.json').read_text())
 for key in ['source_sha256','tokenizer_files_sha256','transformers_version','task','seq_length','seed','validation_fraction','source_records','splits','format','purpose']:assert observed[key]==expected[key],key
 assert Path(observed['model_path']).resolve()==(B/'models/9G4B-resolved').resolve()
 for name,entry in expected['splits'].items():assert sha(path/(name+'.jsonl'))==entry['sha256'],name
 print('PASS exact published token splits and tokenizer fingerprints')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--model-source',type=Path,help='Extracted ORIGINAL 9G4B directory; verifies all original bytes including9GBweights')
 parser.add_argument('--check-data',action='store_true')
 args=parser.parse_args()
 if args.model_source:model(args.model_source)
 if args.check_data:data()
 if not args.model_source and not args.check_data:parser.print_help()

if __name__=='__main__':main()
