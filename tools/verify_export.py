"""Offline integrity check; no dependencies, model downloads or inference."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance/export.json').read_text())
for name,expected in manifest['files'].items():
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Export changed: '+name)
    if path.suffix=='.py': ast.parse(path.read_text(), filename=name)
print('Verified '+str(len(manifest['files']))+' exported files; no model inference performed.')
