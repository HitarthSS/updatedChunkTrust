"""Run a frozen historical evaluator in process-local redirected output paths.

Only its loopback endpoint/output paths are redirected; source, seeds, controller,
observations, noise and actions remain the original code. No completed result is
overwritten. These cases validate the common evaluation loop before tuning.
"""
import argparse
import json
from types import SimpleNamespace

import runtime_v4 as rt


def run(args):
    rt.verify_previous()
    old2,old3=rt.originals()
    module=old2 if args.version=='v2' else old3
    destination=rt.RUN/'compatibility'/'legacy'/args.version
    destination.mkdir(parents=True,exist_ok=True)
    proxy=SimpleNamespace(**vars(module.rt))
    proxy.RUN=destination;proxy.PORT=rt.PORT
    proxy.verify_freeze=lambda:rt.verify_sources(rt.BASE/('comparison-'+args.version)/'FROZEN.json')
    module.rt=proxy
    preflight=rt.RUN/'POLICY_PREFLIGHT.json'
    if not preflight.exists():raise RuntimeError('Common preflight must run before the legacy anchor')
    rt.atomic(destination/'POLICY_PREFLIGHT.json',json.loads(preflight.read_text()))
    seed=10010 if args.version=='v2' else 10020
    variant='authors' if args.method=='authors' else ('balanced' if args.version=='v2' else 'moderate')
    module.run(SimpleNamespace(phase='dev',method=args.method,variant=variant,task_index=0,seed=seed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['v2','v3'],required=True)
    p.add_argument('--method',choices=['authors','mine'],required=True)
    run(p.parse_args())
