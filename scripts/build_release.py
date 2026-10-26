"""Build a source-only archive from the Git index; excludes generated/private files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',default=str(ROOT/'site/prism-research-source.zip'))
args=parser.parse_args()
paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
output=Path(args.output).resolve()
output.parent.mkdir(parents=True,exist_ok=True)
manifest={}
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(filter(None,paths)):
        path=ROOT/name
        if not path.is_file():raise ValueError('Missing staged source: '+name)
        rel=Path(name)
        if (any(p in {'.git','.venv','__pycache__','.pytest_cache','runs','smoke-demo'} for p in rel.parts)
                or (path.name.startswith('.env') and path.name!='.env.example')
                or path.suffix in {'.zip','.pyc'}):
            raise ValueError('Private/generated file in Git index: '+name)
        data=path.read_bytes()
        archive.writestr('prism-research/'+name,data)
        manifest[name]=hashlib.sha256(data).hexdigest()
output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),'utf-8')
print(f'Built {output.name}: {len(manifest)} source files')
