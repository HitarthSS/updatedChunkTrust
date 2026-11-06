"""Finite compatibility -> development -> freeze -> 120 four-arm cases."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

import runtime_v4 as rt
import numpy as np
from pi05_robocasa_common import AHS_CONFIG,SAMPLING_CONFIG,task_records
from controller import PROFILES
from evaluate import result_path
from report import summary,report

GPU=6
DEV_TASKS=(0,1,4,7,11,17,18,22)
DEV_SEEDS=(10030,10031)
TEST_SEEDS=tuple(range(15,20))
ORDERS=(('authors','v4'),('v4','authors'))
REUSE_POLICY=False


def state(name,**fields):
    value=dict(state=name,supervisor_pid=os.getpid(),policy_gpu=GPU,sim_gpu=GPU,updated_unix=time.time(),**fields)
    rt.atomic(rt.RUN/'status.json',value);print(json.dumps(value,sort_keys=True),flush=True)


def cases(phase):
    tasks=task_records()
    if phase=='compatibility':items=[(0,10010,['authors','v2']),(0,10020,['authors','v3'])]
    elif phase=='dev':items=[(i,s,['authors','v3',*PROFILES]) for s in DEV_SEEDS for i in DEV_TASKS]
    else:
        items=[(i,s,['authors','v4']) for s in TEST_SEEDS for i in range(24)]
        # Same untouched case shuffle; user requested a two-arm comparison.
        order=np.random.default_rng(741923).permutation(len(items))
        items=[items[i] for i in order]
    return [dict(pair_id=f't{i:02d}-s{s:05d}',task_index=i,seed=s,arms=arms,**tasks[i]) for i,s,arms in items]


def child_env(policy=False):
    env={k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ('API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
    env.update(rt.configure())
    env.update(CUDA_VISIBLE_DEVICES=str(GPU),PYTHONPATH=':'.join(map(str,[rt.HERE,rt.BASE,rt.SCRIPTS,
        rt.PACKAGE/'openpi_runtime/src',rt.PACKAGE/'openpi_runtime/packages/openpi-client/src',rt.BENCHMARK,rt.SELECTOR.parent])))
    if policy:
        env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.45')
        libraries=Path('/home/hitarth/rpent-reproduction/.venv/lib/python3.11/site-packages/nvidia')
        env['LD_LIBRARY_PATH']=':'.join(str(p) for p in sorted(libraries.glob('*/lib')))
    else:env.update(JAX_PLATFORMS='cpu',MUJOCO_EGL_DEVICE_ID=str(GPU))
    return env


def gpu_snapshot():
    command=['nvidia-smi','--id='+str(GPU),'--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw','--format=csv,noheader']
    p=subprocess.run(command,capture_output=True,text=True,timeout=20)
    return dict(unix=time.time(),returncode=p.returncode,gpu_csv=p.stdout.strip())


def start_policy():
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',rt.PORT))==0:raise RuntimeError('Policy port occupied; refusing unknown endpoint')
    free=int(subprocess.check_output(['nvidia-smi','--id='+str(GPU),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<23500:raise RuntimeError('Insufficient GPU memory, not queueing silently')
    with (rt.RUN/'policy-server.log').open('a') as log:
        p=subprocess.Popen([str(rt.BASE/'.venv-policy/bin/python'),'-u',str(rt.SCRIPTS/'run_pi05_robocasa_server.py'),
            '--host','127.0.0.1','--port',str(rt.PORT),'--ready-file',str(rt.RUN/'policy-ready.json')],
            env=child_env(True),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rt.atomic(rt.RUN/'policy-process.json',dict(pid=p.pid,gpu=GPU,port=rt.PORT))
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if p.poll() is not None:raise RuntimeError('Policy process exited; inspect its log')
        if (rt.RUN/'policy-ready.json').exists():
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',rt.PORT))==0:return p
        time.sleep(5)
    p.terminate();raise RuntimeError('Policy startup deadline exceeded')


def adopt_policy():
    """Preserve the existing compiled server across the user scope amendment."""
    saved=json.loads((rt.RUN/'policy-process.json').read_text());pid=saved['pid']
    if saved['gpu']!=GPU or saved['port']!=rt.PORT:raise RuntimeError('Wrong policy process identity')
    path=Path('/proc')/str(pid)/'cmdline'
    expected=path.read_bytes()
    if str(rt.SCRIPTS/'run_pi05_robocasa_server.py').encode() not in expected or b'--port\x005923\x00' not in expected:
        raise RuntimeError('Refusing to adopt an unverified policy process')
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    old2,_=rt.originals()
    metadata=old2.verify_metadata(WebsocketClientPolicy(host='127.0.0.1',port=rt.PORT))
    if metadata!=json.loads((rt.RUN/'policy-ready.json').read_text()):raise RuntimeError('Adopted server metadata differs')

    class ExistingPolicy:
        def matches(self):
            try:return path.read_bytes()==expected
            except FileNotFoundError:return False
        def terminate(self):
            if self.matches():os.kill(pid,signal.SIGTERM)
        def kill(self):
            if self.matches():os.kill(pid,signal.SIGKILL)
        def wait(self,timeout):
            deadline=time.monotonic()+timeout
            while self.matches():
                if time.monotonic()>deadline:raise subprocess.TimeoutExpired(str(pid),timeout)
                time.sleep(.2)
            return 0

    rt.atomic(rt.RUN/'POLICY_ADOPTION.json',dict(pid=pid,gpu=GPU,port=rt.PORT,adopted_unix=time.time(),
        policy_restarted=False,metadata_verified=True,reason='User removed v2/v3 from final evaluation before any test result'))
    return ExistingPolicy()


def child(command,log,phase,**fields):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        p=subprocess.Popen([str(rt.BASE/'.venv-sim/bin/python'),'-u',*map(str,command)],env=child_env(),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state(phase+'_running',episode_pid=p.pid,log=str(log),**fields)
        deadline=time.monotonic()+3600
        with (rt.RUN/'gpu-conditions.jsonl').open('a') as monitor:
            while p.poll() is None:
                monitor.write(json.dumps(dict(phase=phase,episode_pid=p.pid,**fields,**gpu_snapshot()),sort_keys=True)+'\n');monitor.flush()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:pass
                if time.monotonic()>deadline:
                    p.terminate();raise RuntimeError('Episode infrastructure timeout; retain all partial output '+str(log))
    if p.returncode:raise RuntimeError(f'Episode failed ({p.returncode}); preserve outputs and inspect {log}')


def episode(phase,row,arm):
    output=result_path(phase,row['task_index'],row['seed'],arm)
    if output.exists():
        if json.loads(output.read_text())['status']!='COMPLETED':raise RuntimeError('Incomplete saved output')
        return
    rt.verify_freeze()
    s=summary(phase)
    child([rt.HERE/'evaluate.py','--phase',phase,'--arm',arm,'--task-index',row['task_index'],'--seed',row['seed']],
        rt.RUN/'logs'/phase/(row['pair_id']+'-'+arm+'.log'),phase,case=row['pair_id'],arm=arm,
        completed_cases=s['completed_cases'],total_cases=s['planned_cases'],task=row['task_name'])
    summary(phase)


def anchor_comparison(version,method):
    seed=10010 if version=='v2' else 10020
    arm='authors' if method=='authors' else version
    variant='authors' if method=='authors' else ('balanced' if version=='v2' else 'moderate')
    common=result_path('compatibility',0,seed,arm)
    legacy=rt.RUN/'compatibility'/'legacy'/version/'development'/variant/f't00-s{seed:05d}-{method}.json'
    a=json.loads(common.read_text());b=json.loads(legacy.read_text())
    keys=('success','first_success_step','environment_steps','policy_calls','decisions','extra_samples','alternative_selections',
          'initial_fingerprint','server_metadata','terminated','truncated')
    for k in keys:
        if a[k]!=b[k]:raise RuntimeError('Common versus frozen evaluator mismatch: '+version+'/'+method+'/'+k)
    aa=[json.loads(x) for x in common.with_suffix('.trace.jsonl').read_text().splitlines()]
    bb=[json.loads(x) for x in legacy.with_suffix('.trace.jsonl').read_text().splitlines()]
    if len(aa)!=len(bb):raise RuntimeError('Compatibility trace length differs')
    keys=('replan_index','start_step','selected_k','executed','primary_noise_sha256','primary_actions_sha256',
          'selected_actions_sha256','selected_proposal','success_observed','policy_calls_so_far')
    for i,(x,y) in enumerate(zip(aa,bb)):
        if any(x[k]!=y[k] for k in keys):raise RuntimeError(f'Compatibility trajectory mismatch at {version}/{method}/{i}')
        pairs=lambda line:[(p['noise_sha256'],p['actions_sha256']) for p in line['samples']]
        if pairs(x)!=pairs(y):raise RuntimeError('Compatibility proposal mismatch')
    historical=rt.BASE/('comparison-'+version)/'development'/variant/f't00-s{seed:05d}-{method}.json'
    old=json.loads(historical.read_text())
    old_trace=[json.loads(x) for x in historical.with_suffix('.trace.jsonl').read_text().splitlines()]
    return dict(version=version,method=method,status='PASS',exact_trajectory_match=True,decisions=len(aa),
        policy_calls=a['policy_calls'],current_success=a['success'],historical_success=old['success'],
        historical_initial_state_match=a['initial_fingerprint']==old['initial_fingerprint'],
        historical_first_action_match=aa[0]['primary_actions_sha256']==old_trace[0]['primary_actions_sha256'],
        historical_full_action_hash_match=len(aa)==len(old_trace) and all(x['selected_actions_sha256']==y['selected_actions_sha256'] for x,y in zip(aa,old_trace)))


def compatibility():
    if (rt.RUN/'COMPATIBILITY.json').exists():return
    results=[]
    for version,row in zip(('v2','v3'),cases('compatibility')):
        for method,arm in (('authors','authors'),('mine',version)):
            episode('compatibility',row,arm)
            legacy=rt.RUN/'compatibility'/'legacy'/version/'development'/('authors' if method=='authors' else ('balanced' if version=='v2' else 'moderate'))/f"t00-s{row['seed']:05d}-{method}.json"
            if not legacy.exists():
                child([rt.HERE/'legacy_anchor.py','--version',version,'--method',method],
                    rt.RUN/'logs'/'compatibility'/('legacy-'+version+'-'+method+'.log'),'compatibility',
                    case=row['pair_id'],arm='legacy-'+version+'-'+method)
            results.append(anchor_comparison(version,method))
    rt.atomic(rt.RUN/'COMPATIBILITY.json',dict(status='PASS',checks=results,physical_episodes=8,
        interpretation='Common loop exactly matches each frozen evaluator on declared development anchors; historical replay is separately descriptive',
        checked_unix=time.time()))


def development():
    arms=['authors','v3',*PROFILES]
    if not REUSE_POLICY:
        for i,row in enumerate(cases('dev')):
            order=arms[i%4:]+arms[:i%4]
            for arm in order:episode('dev',row,arm)
    s=summary('dev')
    if s['completed_cases']!=16:raise RuntimeError('Incomplete development')
    eligible=[p for p in PROFILES if s['arms'][p]['noise_reused_decisions']>0]
    if not eligible:raise RuntimeError('V4 reuse never exercised on development; diagnose')
    selected=max(eligible,key=lambda p:(s['arms'][p]['successes'],-s['arms'][p]['policy_calls'],int(p=='cautious')))
    rt.atomic(rt.RUN/'DEV_SELECTION.json',dict(selected_v4_profile=selected,summary=s,
        rule='Maximum development successes, then fewer policy calls, then cautious',test_outcomes_observed=0,
        selected_unix=time.time(),accuracy_improvement_established=False))
    return selected


def freeze(profile):
    if list((rt.RUN/'test').glob('*.json')):raise RuntimeError('Cannot freeze after test outcomes')
    sources={}
    for version in ('v1','v2','v3'):
        p=rt.BASE/('comparison-'+version)/'FROZEN.json'
        sources.update(json.loads(p.read_text())['sources']);sources[str(p)]=rt.sha(p)
    paths=list(rt.HERE.glob('*.py'))+[rt.HERE/'README.md']+[rt.RUN/n for n in (
        'PROTOCOL.json','compatibility-cases.json','dev-cases.json','test-cases.json','COMPATIBILITY.json','DEV_SELECTION.json','UNIT_TESTS.json')]
    paths.extend(rt.RUN/n for n in ('SCOPE_AMENDMENT.json','POLICY_ADOPTION.json') if (rt.RUN/n).exists())
    for p in paths:sources[str(p)]=rt.sha(p)
    rt.atomic(rt.RUN/'FROZEN.json',dict(frozen_unix=time.time(),sources=sources,selected_v4_profile=profile,
        settings=asdict(PROFILES[profile]),final_cases=120,final_physical_episodes=240,test_seeds=list(TEST_SEEDS),
        test_outcomes_observed=0,mutation_allowed=False,arms=['authors','v4'],gpu=GPU,port=rt.PORT))
    rt.verify_freeze()


def main():
    rt.RUN.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='Prediction-validated noise v4, matched authors-v4 evaluation',
        source_revision='a9fd57412b3a19a0728cd726b0558dd89395fb84',benchmark_revision='4840e671596f93ca03651524b9f72ffb1aadfeff',
        model='NewGain/pi05_robocasa_gr1_tabletop24',optimizer_step=30000,use_qha=False,frozen_policy=True,
        arms=dict(authors='Released AHS, independent noise',v4='Selected prediction-validated noise profile'),
        v4_profiles={k:asdict(v) for k,v in PROFILES.items()},categories=['sampler_guidance','execution_monitoring'],
        development_tasks=list(DEV_TASKS),development_seeds=list(DEV_SEEDS),development_physical_episodes=64,
        compatibility_physical_episodes=8,test_tasks=24,test_seeds=list(TEST_SEEDS),episodes_per_task_per_arm=5,
        test_cases=120,test_episodes_per_arm=120,test_physical_episodes=240,case_shuffle_seed=741923,
        arm_order_design='Alternating authors-v4 and v4-authors, 60 pairs each order',arm_orders=ORDERS,
        horizon=720,horizon_multiplier=1.,policy_call_ceiling=180,ahs=AHS_CONFIG,sampling=SAMPLING_CONFIG,
        policy_gpu=GPU,simulator_gpu=GPU,policy_port=rt.PORT,policy_server_count=1,serial_episodes=True,
        timing='GPU conditions every 20 seconds; one hardware placement, other users may contend; descriptive timing',
        budget='One model call per decision for both arms; all native actions count; actual totals disclosed',
        memory='Cleared each episode; no cross-episode learning or retrieval',
        test_success='Native any info[success] within horizon; no early success stopping; no recovery reset or seed replacement',
        primary_contrasts=['v4-authors'],uncertainty='Exact paired McNemar; episode/task-cluster bootstrap',
        scope_amendment='User removed v2/v3 before any test outcomes; completed compatibility/development retained; same policy process and controller',
        unseen='These 24 task types are in policy training; final states/seeds are fresh. Paper QHA held-out-head test is a different RoboTwin condition, not a base-policy unseen-task test',
        limitations=['Pilot 120/arm versus reported 1200/arm','Reconstructed historical backend/dependencies','No unseen-task-type claim'],
        paid_api_calls=0,additional_models=0,action_values_modified=False)
    if not (rt.RUN/'PROTOCOL.json').exists():
        rt.atomic(rt.RUN/'PROTOCOL.json',protocol)
        for phase in ('compatibility','dev','test'):rt.atomic(rt.RUN/(phase+'-cases.json'),cases(phase))
    state('integrity_checking')
    parents=rt.verify_previous()
    for line in (rt.PACKAGE/'PACKAGE_MANIFEST.sha256').read_text().splitlines():
        expected,name=line.split(maxsplit=1);p=rt.PACKAGE/name.lstrip('*')
        if not p.is_relative_to(rt.PACKAGE) or rt.sha(p)!=expected:raise RuntimeError('Package checksum mismatch '+name)
    rt.atomic(rt.RUN/'PACKAGE_VERIFIED.json',dict(status='PASS',previous_freezes=parents,checked_unix=time.time()))
    rt.atomic(rt.RUN/'HARDWARE.json',gpu_snapshot())
    for kind in ('policy','sim'):
        with (rt.RUN/(kind+'-dependencies.txt')).open('w') as stream:
            subprocess.run([str(rt.BASE/f'.venv-{kind}/bin/pip'),'freeze','--all'],stdout=stream,check=True)
    state('policy_adopting' if REUSE_POLICY else 'policy_loading');server=adopt_policy() if REUSE_POLICY else start_policy()
    try:
        compatibility()
        if not (rt.RUN/'FROZEN.json').exists():
            profile=development();freeze(profile);state('design_frozen',selected_v4_profile=profile)
        for i,row in enumerate(cases('test')):
            for arm in ORDERS[i%len(ORDERS)]:episode('test',row,arm)
        state('auditing',completed_cases=120,total_cases=120)
        result=report()
        state('completed',completed_cases=120,total_cases=120,summary=result,analysis_path=str(rt.RUN/'FINAL_ANALYSIS.json'))
    finally:
        server.terminate()
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:server.kill()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-policy',action='store_true',help='Adopt the verified existing server without restarting it')
    REUSE_POLICY=parser.parse_args().reuse_policy
    try:main()
    except BaseException as e:
        state('failed',error=str(e),traceback=traceback.format_exc());raise
