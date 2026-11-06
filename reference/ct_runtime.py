"""Paths and local-only runtime configuration for the ChunkTrust comparison."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'models/pi05_robocasa_gr1_tabletop24_step30000'
RELEASE = ROOT / 'ChunkTrust'
SCRIPTS = RELEASE / 'third_party/overlays/robotwin/eval_rebuttal/scripts'
BENCHMARK = ROOT / 'workspaces/robocasa-gr1-tabletop-tasks'
SELECTOR = ROOT / 'workspaces/starVLA/examples/Robocasa_tabletop/eval_files/horizon_selector.py'
RUN = ROOT / 'comparison-v1'
PORT = 5920


def configure():
    settings = {
        'ROBOTWIN_ROOT': str(RELEASE / 'third_party/overlays/robotwin'),
        'CHUNKTRUST_MODELS_ROOT': str(ROOT / 'models'),
        'CHUNKTRUST_WORKSPACE_ROOT': str(ROOT / 'workspaces'),
        'OPENPI_DATA_HOME': str(PACKAGE / 'openpi_cache'),
        'PYTHONPYCACHEPREFIX': str(ROOT / '.pycache'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'WANDB_MODE': 'disabled',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0',
        'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    os.environ.update(settings)
    # These also cover this interpreter, which started before configure().
    # Preserve the delivered package's cataloged bytecode files unchanged.
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(ROOT / '.pycache')
    paths = [ROOT, SCRIPTS, SELECTOR.parent, BENCHMARK,
             PACKAGE / 'openpi_runtime/src',
             PACKAGE / 'openpi_runtime/packages/openpi-client/src']
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return settings


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n')
    temp.replace(path)


def verify_freeze():
    path = RUN / 'FROZEN.json'
    if not path.exists():
        return None
    frozen = json.loads(path.read_text())
    for name, expected in frozen['sources'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen evaluation source changed: ' + name)
    return sha(path)


configure()
