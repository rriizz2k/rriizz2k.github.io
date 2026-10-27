"""Package only authored public source and model assets, never credentials/runtime state."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
target=root/'public/downloads/credit-risk-source.zip'
folders=['app','lib','web','python_api']
files=['README.md','package.json','package-lock.json','vite.pages.config.ts','render.yaml',
       'scripts/export-baseline.py','scripts/check-inference.mjs','scripts/check-prediction-client.mjs','scripts/check-python-api.py',
       'scripts/parity-cases.json','public/favicon.svg','public/downloads/evaluation.json']
for folder in folders:
    for path in (root/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc']:
            files.append(str(path.relative_to(root)))
with ZipFile(target,'w',ZIP_DEFLATED) as archive:
    for name in sorted(set(files)):
        path=root/name
        if path.exists():archive.write(path,'credit-risk-portfolio/'+name)
print(target.name,target.stat().st_size,'bytes')
