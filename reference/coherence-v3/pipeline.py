"""Finite development selection, immutable freeze, then 120 fresh paired cases."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import runtime_v3 as rt
from pi05_robocasa_common import AHS_CONFIG, SAMPLING_CONFIG, task_records
from coherent_noise import PROFILES
from evaluate import result_path

DEV_TASKS = (0,1,4,7,11,17,18,22)
TEST_SEEDS = tuple(range(10,15))


def cases(phase):
    tasks = task_records()
    indices = [(i,s) for s in (10020,10021) for i in DEV_TASKS] if phase=='dev' else [
        (i,seed) for seed in TEST_SEEDS for i in range(24)]
    return [dict(pair_id=f't{i:02d}-s{seed:05d}',task_index=i,seed=seed,**tasks[i]) for i,seed in indices]


def state(name, **fields):
    payload = dict(state=name,supervisor_pid=os.getpid(),updated_unix=time.time(),
                   policy_gpu=ARGS.policy_gpu,sim_gpu=ARGS.sim_gpu,**fields)
    rt.atomic(rt.RUN/'status.json',payload)
    print(json.dumps(payload,sort_keys=True),flush=True)


def summarize(phase,variant):
    records = {method:[] for method in ('authors','mine')}
    paired = []
    for row in cases(phase):
        found = {}
        for method in ('authors','mine'):
            v = 'authors' if method=='authors' else variant
            path = result_path(phase,row['task_index'],row['seed'],method,v)
            if path.exists():
                value = json.loads(path.read_text())
                if value.get('status')!='COMPLETED':
                    raise RuntimeError('Incomplete saved result: '+str(path))
                if value['variant']!=v:
                    raise RuntimeError('Saved result variant mismatch')
                if value['environment_steps']>720 or value['policy_calls']>180:
                    raise RuntimeError('Native horizon or inference budget exceeded')
                records[method].append(value);found[method]=value
        if len(found)==2:
            if found['authors']['initial_fingerprint']!=found['mine']['initial_fingerprint']:
                raise RuntimeError('Paired initial states differ: '+row['pair_id'])
            first = []
            for method in ('authors','mine'):
                v = 'authors' if method=='authors' else variant
                p = result_path(phase,row['task_index'],row['seed'],method,v).with_suffix('.trace.jsonl')
                with p.open() as stream:
                    line=json.loads(next(stream))
                first.append((line['primary_noise_sha256'],line['primary_actions_sha256']))
            if first[0]!=first[1]:
                raise RuntimeError('Paired initial policy predictions differ: '+row['pair_id'])
            paired.append((row,found))
    arms = {}
    for method in records:
        values = records[method]
        n = len(values)
        successes = sum(r['success'] for r in values)
        arms[method] = dict(completed=n,successes=successes,
            paired_successes=sum(p[method]['success'] for _,p in paired),
            **{key:sum(r[key] for r in values) for key in ('policy_calls','extra_samples',
                'alternative_selections','noise_reused_decisions','noise_reused_positions','noise_refreshes','policy_inference_seconds','simulator_step_seconds','wall_seconds',
                'horizon_violations','inference_budget_violations')})
    per_task = {}
    for row,results in paired:
        value = per_task.setdefault(row['task_name'],dict(pairs=0,authors_successes=0,mine_successes=0))
        value['pairs']+=1
        for method in records:value[method+'_successes']+=int(results[method]['success'])
    payload = dict(phase=phase,variant=variant,total_pairs=len(cases(phase)),completed_pairs=len(paired),
        completed_episodes=sum(len(v) for v in records.values()),arms=arms,per_task=per_task,
        helped=sum(not p['authors']['success'] and p['mine']['success'] for _,p in paired),
        harmed=sum(p['authors']['success'] and not p['mine']['success'] for _,p in paired),
        initial_state_mismatches=0,updated_unix=time.time(),
        cost_interpretation='Equal 180-call ceilings; one policy call per decision, no discarded proposals; report actual totals',
        timing_caveat='Shared GPU; timing is descriptive, not a controlled speed comparison')
    name = f'dev-{variant}-summary.json' if phase=='dev' else 'test-summary.json'
    rt.atomic(rt.RUN/name,payload)
    return payload


def child_environment(gpu,policy=False):
    env = {key:value for key,value in os.environ.items()
        if not any(word in key.upper() for word in ('API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
    env.update(rt.configure())
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=':'.join(map(str,[rt.HERE,rt.BASE,rt.SCRIPTS,
        rt.PACKAGE/'openpi_runtime/src',rt.PACKAGE/'openpi_runtime/packages/openpi-client/src',
        rt.BENCHMARK,rt.SELECTOR.parent])))
    if policy:
        env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.45')
        libs=Path('/home/hitarth/rpent-reproduction/.venv/lib/python3.11/site-packages/nvidia')
        env['LD_LIBRARY_PATH']=':'.join(str(p) for p in sorted(libs.glob('*/lib')))
    else:
        env.update(JAX_PLATFORMS='cpu',MUJOCO_EGL_DEVICE_ID=str(gpu))
    return env


def start_policy():
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',rt.PORT))==0:
            raise RuntimeError('V3 policy port is occupied; refusing an unknown server')
    free=int(subprocess.check_output(['nvidia-smi','--id='+str(ARGS.policy_gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<23500:
        raise RuntimeError(f'Insufficient free memory on selected GPU {ARGS.policy_gpu}: {free} MiB')
    with (rt.RUN/'policy-server.log').open('a') as log:
        process=subprocess.Popen([str(rt.BASE/'.venv-policy/bin/python'),'-u',
            str(rt.SCRIPTS/'run_pi05_robocasa_server.py'),'--host','127.0.0.1','--port',str(rt.PORT),
            '--ready-file',str(rt.RUN/'policy-ready.json')],env=child_environment(ARGS.policy_gpu,True),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rt.atomic(rt.RUN/'policy-process.json',dict(pid=process.pid,gpu=ARGS.policy_gpu,port=rt.PORT))
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError('Policy server exited; inspect policy-server.log')
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',rt.PORT))==0:
                return process
        time.sleep(5)
    process.terminate()
    raise RuntimeError('Policy server startup deadline exceeded')


def verify_package():
    # Reuse the pinned checkpoint, independently check every published file.
    for line in (rt.PACKAGE/'PACKAGE_MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        path=rt.PACKAGE/name.lstrip('*')
        if not path.is_relative_to(rt.PACKAGE) or rt.sha(path)!=expected:
            raise RuntimeError('Package checksum mismatch: '+name)
    rt.atomic(rt.RUN/'PACKAGE_VERIFIED.json',dict(status='PASS',verified_unix=time.time(),
        model_catalog_sha256=rt.sha(rt.BASE/'modelscope-files.json')))


def episode(phase,row,method,variant):
    path=result_path(phase,row['task_index'],row['seed'],method,variant)
    if path.exists():
        if json.loads(path.read_text()).get('status')!='COMPLETED':
            raise RuntimeError('Unfinished result must be diagnosed: '+str(path))
        return
    rt.verify_freeze()
    report_variant = variant if method=='mine' else (
        json.loads((rt.RUN/'FROZEN.json').read_text())['selected_variant'] if phase=='test' else 'moderate')
    summary=summarize(phase,report_variant)
    state(phase+'_running',pair=row['pair_id'],task=row['task_name'],method=method,variant=variant,
          completed_pairs=summary['completed_pairs'],total_pairs=summary['total_pairs'])
    log=rt.RUN/'logs'/phase/variant/(row['pair_id']+'-'+method+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        process=subprocess.Popen([str(rt.BASE/'.venv-sim/bin/python'),'-u',str(rt.HERE/'evaluate.py'),
            '--phase',phase,'--method',method,'--variant',variant,'--task-index',str(row['task_index']),
            '--seed',str(row['seed'])],env=child_environment(ARGS.sim_gpu),stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state(phase+'_running',pair=row['pair_id'],task=row['task_name'],method=method,variant=variant,
              episode_pid=process.pid,completed_pairs=summary['completed_pairs'],total_pairs=summary['total_pairs'])
        try:
            code=process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.terminate()
            raise RuntimeError('Episode exceeded infrastructure deadline; preserve case: '+str(log))
    if code:
        raise RuntimeError(f'Episode failed ({code}); preserve case and inspect {log}')
    summarize(phase,report_variant)


def development():
    variants=['authors',*PROFILES]
    for index,row in enumerate(cases('dev')):
        order=variants[index%len(variants):]+variants[:index%len(variants)]
        for variant in order:
            episode('dev',row,'authors' if variant=='authors' else 'mine',variant)
        for variant in PROFILES:summarize('dev',variant)
    results={variant:summarize('dev',variant) for variant in PROFILES}
    for summary in results.values():
        if summary['completed_pairs']!=len(cases('dev')):
            raise RuntimeError('Incomplete development evaluation')
    eligible=[v for v in PROFILES if results[v]['arms']['mine']['noise_reused_decisions']>0]
    if not eligible:
        raise RuntimeError('Noise reuse was never exercised during development; diagnose before testing')
    # Fixed development-only rule. No completed test results are consulted.
    selected=max(eligible,key=lambda v:(results[v]['arms']['mine']['successes'],
        -results[v]['arms']['mine']['policy_calls'],int(v=='moderate')))
    selection=dict(selected_variant=selected,profiles=results,selected_unix=time.time(),
        rule='Maximum development successes, then fewer total policy calls, then moderate',
        test_outcomes_observed=0,accuracy_improvement_established=False)
    rt.atomic(rt.RUN/'DEV_SELECTION.json',selection)
    return selected


def freeze(selected):
    if list((rt.RUN/'test').glob('*.json')):
        raise RuntimeError('Cannot freeze after final evaluation has started')
    original=json.loads((rt.BASE/'comparison-v1/FROZEN.json').read_text())
    paths={Path(p) for p in original['sources']}
    paths.update(rt.HERE.glob('*.py'))
    paths.update(rt.RUN/name for name in ('PROTOCOL.json','dev-cases.json','test-cases.json','DEV_SELECTION.json'))
    sources={str(path):rt.sha(path) for path in sorted(paths)}
    rt.atomic(rt.RUN/'FROZEN.json',dict(frozen_unix=time.time(),sources=sources,
        selected_variant=selected,settings=asdict(PROFILES[selected]),final_pairs=120,final_episodes=240,
        test_seeds=list(TEST_SEEDS),test_outcomes_observed=0,mutation_allowed=False))
    rt.verify_freeze()


def main():
    rt.RUN.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='Scene-gated coherent sampling v3',paper='ChunkTrust, arXiv:2609.39754v1',
        source_revision='a9fd57412b3a19a0728cd726b0558dd89395fb84',
        benchmark_revision='4840e671596f93ca03651524b9f72ffb1aadfeff',
        model='NewGain/pi05_robocasa_gr1_tabletop24',optimizer_step=30000,use_qha=False,
        frozen_policy=True,authors='Unmodified released numerical AHS',
        mine='Time-aligned latent-noise reuse with public scene-change, stillness and bounded refresh',
        categories=['sampler_guidance','execution_monitoring'],profiles={k:asdict(v) for k,v in PROFILES.items()},
        development_tasks=list(DEV_TASKS),development_seeds=[10020,10021],development_episodes=48,
        test_tasks=24,test_seeds=list(TEST_SEEDS),test_episodes_per_task=5,test_episodes_per_arm=120,
        horizon=720,horizon_multiplier=1.,policy_call_ceiling=180,
        inference_cost='One policy call per decision; no resampling or discarded proposals; report actual compute',
        action_values_modified=False,maximum_proposals_per_decision=1,
        primary_noise='Authors: released independent decision noise. Mine: same fresh draws, overlap shifted by actual executed prefix and correlated; first draw identical',
        noise_formula='rho*previous_noise[executed:] + sqrt(1-rho^2)*fresh_noise[:overlap]; suffix fresh; refresh on declared public gates',
        ahs=AHS_CONFIG,sampling=SAMPLING_CONFIG,policy_gpu=ARGS.policy_gpu,simulator_gpu=ARGS.sim_gpu,
        memory='Empty episode-local posterior and rolling noise per episode; no cross-episode retrieval',
        gaps=['120 episodes/arm instead of full reported 1200',
              'Reconstructed backend/dependency environment as documented in completed v1',
              'Same trained task types with fresh initial states; no unseen-task-type claim',
              'Shared GPUs; wall time cannot establish a controlled speed comparison'],
        paid_api_calls=0,extra_models=0)
    if not (rt.RUN/'PROTOCOL.json').exists():
        rt.atomic(rt.RUN/'PROTOCOL.json',protocol)
        rt.atomic(rt.RUN/'dev-cases.json',cases('dev'))
        rt.atomic(rt.RUN/'test-cases.json',cases('test'))
    state('integrity_checking')
    rt.base.verify_freeze()
    previous=json.loads((rt.BASE/'comparison-v2/FROZEN.json').read_text())
    for name,expected in previous['sources'].items():
        if rt.sha(name)!=expected:
            raise RuntimeError('Earlier v2 frozen source changed: '+name)
    verify_package()
    for kind in ('policy','sim'):
        with (rt.RUN/f'{kind}-dependencies.txt').open('w') as stream:
            subprocess.run([str(rt.BASE/f'.venv-{kind}/bin/pip'),'freeze','--all'],stdout=stream,check=True)
    state('policy_loading')
    server=start_policy()
    try:
        if not (rt.RUN/'FROZEN.json').exists():
            selected=development()
            freeze(selected)
            state('design_frozen',variant=selected,source_files=len(json.loads((rt.RUN/'FROZEN.json').read_text())['sources']))
        selected=json.loads((rt.RUN/'FROZEN.json').read_text())['selected_variant']
        for index,row in enumerate(cases('test')):
            methods=('authors','mine') if index%2==0 else ('mine','authors')
            for method in methods:
                episode('test',row,method,'authors' if method=='authors' else selected)
            summarize('test',selected)
        result=summarize('test',selected)
        from report import report
        analysis=report()
        state('completed',summary=result,analysis_path=str(rt.RUN/'FINAL_ANALYSIS.json'))
    finally:
        server.terminate()
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:server.kill()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy-gpu',type=int,required=True)
    parser.add_argument('--sim-gpu',type=int,required=True)
    ARGS=parser.parse_args()
    try:main()
    except BaseException as exc:
        state('failed',error=str(exc),traceback=traceback.format_exc())
        raise
