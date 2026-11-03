"""Check local inline Markdown links without accessing external URLs."""
from pathlib import Path
from urllib.parse import urlsplit,unquote
import re,sys
ROOT=Path(__file__).resolve().parents[1]
LINK=re.compile(r'!?\[[^\]]*\]\(([^\n]+?)\)')
errors=[];checked=0;documents=0
for p in ROOT.rglob('*.md'):
 if '.git' in p.relative_to(ROOT).parts:continue
 documents+=1
 for match in LINK.finditer(p.read_text(encoding='utf-8-sig')):
  raw=match[1].strip().strip('<>');u=urlsplit(raw)
  if u.scheme or u.netloc or not u.path:continue
  target=(p.parent/unquote(u.path)).resolve();checked+=1
  if not target.is_relative_to(ROOT):errors.append(f'{p.relative_to(ROOT)}: outside repository: {raw}')
  elif not target.exists():errors.append(f'{p.relative_to(ROOT)}: missing: {raw}')
if errors:
 print('\n'.join(errors));sys.exit(1)
print(f'PASS: {documents} Markdown documents, {checked} local links, no missing targets')
