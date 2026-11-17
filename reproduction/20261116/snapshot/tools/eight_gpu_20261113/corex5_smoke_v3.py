"""Fresh attempts of the unchanged smoke protocol, preserving failed attempts."""
from pathlib import Path
import argparse
import corex5_smoke as runner

parser=argparse.ArgumentParser()
parser.add_argument('--prepared-name',required=True)
args=parser.parse_args()
assert '/' not in args.prepared_name and args.prepared_name.startswith('prepared-')
runner.P=runner.R/args.prepared_name
runner.main()
