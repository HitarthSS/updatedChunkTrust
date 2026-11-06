"""Independent v3 paths; previous experiments remain frozen."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ct_runtime as base

BASE = base.ROOT
PACKAGE = base.PACKAGE
SCRIPTS = base.SCRIPTS
BENCHMARK = base.BENCHMARK
SELECTOR = base.SELECTOR
RELEASE = base.RELEASE
RUN = BASE / 'comparison-v3'
PORT = 5922
sha = base.sha
atomic = base.atomic


def configure():
    settings = base.configure()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return settings


def verify_freeze():
    path = RUN / 'FROZEN.json'
    if not path.exists():
        return None
    for name, expected in json.loads(path.read_text())['sources'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen v3 source/configuration changed: ' + name)
    return sha(path)


def jsonable(value):
    import math
    import numpy as np
    if isinstance(value, dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


configure()
