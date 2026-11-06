"""Isolated four-arm experiment, importing previous controllers unchanged."""
import importlib.util
from functools import lru_cache
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
import ct_runtime as base

PACKAGE=base.PACKAGE
SCRIPTS=base.SCRIPTS
BENCHMARK=base.BENCHMARK
SELECTOR=base.SELECTOR
RUN=BASE/'comparison-v4'
PORT=5923
sha=base.sha
atomic=base.atomic


def configure():
    settings=base.configure()
    if str(HERE) in sys.path:sys.path.remove(str(HERE))
    sys.path.insert(0,str(HERE))
    return settings


def verify_sources(manifest):
    value=json.loads(Path(manifest).read_text())
    for name,expected in value['sources'].items():
        if sha(name)!=expected:raise RuntimeError('Frozen source changed: '+name)
    return sha(manifest)


def verify_freeze():
    path=RUN/'FROZEN.json'
    return verify_sources(path) if path.exists() else None


def verify_previous():
    return {version:verify_sources(BASE/('comparison-'+version)/'FROZEN.json') for version in ('v1','v2','v3')}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def originals():
    # Files are imported in place. Old source/configuration is never edited.
    for folder in ('proposal-v2','coherence-v3'):
        p=str(BASE/folder)
        if p not in sys.path:sys.path.append(p)
    v2=load('frozen_v2_evaluate',BASE/'proposal-v2/evaluate.py')
    v3=load('frozen_v3_evaluate',BASE/'coherence-v3/evaluate.py')
    configure()
    return v2,v3


def jsonable(value):
    import math
    import numpy as np
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


configure()
