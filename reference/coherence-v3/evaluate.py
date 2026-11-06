"""Native GR1 evaluation: released AHS versus scene-gated coherent sampling."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import random
import time
import traceback

import runtime_v3 as rt
import numpy as np
from pi05_robocasa_common import (
    AHS_CONFIG, CHECKPOINT, NORM_ASSET_ID, NORM_STATS_SHA256, MODEL_ID,
    PACKAGE_TREE_SHA256, PROTOCOL_ID, OPTIMIZER_STEP,
    deterministic_noise, reset_with_literal_seed, instruction_text, task_records,
)
from run_pi05_robocasa_cell import (
    active_state, expand_active, active_to_action_dict, official_success,
    extract_policy_output, MODEL_IMAGE_KEY,
)
from horizon_selector import select_exec_k_horizon
from coherent_noise import CoherentNoise, PROFILES


def array_hash(value):
    value = np.ascontiguousarray(value)
    material = str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()
    return hashlib.sha256(material).hexdigest()


def result_path(phase, task_index, seed, method, variant):
    directory = rt.RUN/'development'/variant if phase=='dev' else rt.RUN/'test'
    return directory/f't{task_index:02d}-s{seed:05d}-{method}.json'


def verify_metadata(client):
    metadata = client.get_server_metadata()
    expected = dict(protocol_id=PROTOCOL_ID, model_id=MODEL_ID,
        optimizer_step=OPTIMIZER_STEP, norm_asset_id=NORM_ASSET_ID,
        norm_stats_sha256=NORM_STATS_SHA256, package_tree_sha256=PACKAGE_TREE_SHA256,
        use_qha=False, trace_num_steps_static_bound=10, trace_wire_fields=['v'], trace_wire_dtype='float32')
    for key, value in expected.items():
        if metadata.get(key)!=value:
            raise RuntimeError('Policy metadata mismatch: '+key)
    return metadata


def run(args):
    import gymnasium as gym
    from robocasa.utils.gym_utils import gymnasium_groot
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    frozen_sha = rt.verify_freeze()
    if args.phase=='test':
        if frozen_sha is None or args.seed not in range(10,15):
            raise RuntimeError('Final evaluation requires freeze and fresh test seeds 10–14')
        freeze = json.loads((rt.RUN/'FROZEN.json').read_text())
        if args.method=='mine' and args.variant!=freeze['selected_variant']:
            raise RuntimeError('Cannot change variant after evaluation freeze')
    elif args.seed not in (10020,10021):
        raise RuntimeError('Only the declared, disjoint development seed is allowed')
    if args.method=='authors' and args.variant!='authors':
        raise ValueError('Authors use the unmodified released selector')
    output = result_path(args.phase,args.task_index,args.seed,args.method,args.variant)
    if output.exists():
        raise RuntimeError('Refusing to overwrite completed result: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    row = task_records()[args.task_index]
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(row['env_name'],enable_render=True)
    record = dict(status='RUNNING',version='Scene-gated coherent sampling v3',phase=args.phase,
        method=args.method,variant=args.variant,task_index=args.task_index,seed=args.seed,**row,
        frozen_sha256=frozen_sha,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        max_environment_steps=720,horizon_multiplier=1.,policy_call_cap=180,
        started_unix=time.time(),environment_steps=0,policy_calls=0)
    live_path = output.with_suffix('.live.json')
    rt.atomic(live_path,record)
    try:
        observation,reset_info = reset_with_literal_seed(env,args.seed)
        literal = instruction_text(observation['annotation.human.coarse_action'])
        record['initial_fingerprint'] = dict(instruction=literal,
            public_state=array_hash(active_state(observation)),image=array_hash(observation[MODEL_IMAGE_KEY]),
            physics=array_hash(env.unwrapped.env.sim.get_state().flatten()))
        client = WebsocketClientPolicy(host='127.0.0.1',port=rt.PORT)
        record['server_metadata'] = verify_metadata(client)
        history = []
        posterior = {}
        rng = np.random.default_rng(AHS_CONFIG['ts_seed'])
        options = dict(AHS_CONFIG)
        options.pop('ts_seed')
        norms = json.loads((CHECKPOINT/'assets'/NORM_ASSET_ID/'norm_stats.json').read_text())['norm_stats']['actions']
        action_ranges = np.asarray(norms['q99'])-np.asarray(norms['q01'])
        sampler = CoherentNoise(action_ranges,PROFILES.get(args.variant,PROFILES['moderate']))
        calls = 0
        reused_decisions = reused_positions = refreshes = 0
        success = official_success(reset_info)
        first_success = 0 if success else None
        steps = decisions = 0
        terminated = truncated = False
        policy_seconds = server_seconds = roundtrip_seconds = sim_seconds = selection_seconds = 0.
        preflight_calls = 0
        trace_path = output.with_suffix('.trace.jsonl')
        with trace_path.open('x') as stream:
            while steps<720 and not (terminated or truncated):
                public_state = active_state(observation).copy()
                public_image = np.asarray(observation[MODEL_IMAGE_KEY]).copy()
                sampling_started = time.perf_counter()
                fresh_noise = deterministic_noise(row['task_name'],args.seed,decisions)
                if args.method=='mine':
                    used_noise,sampling_info = sampler.prepare(fresh_noise,public_state,public_image,steps)
                else:
                    used_noise = fresh_noise
                    sampling_info = dict(reused_positions=0,refresh=True,refresh_reasons=['released_independent_sampling'])
                selection_seconds += time.perf_counter()-sampling_started
                request = {'observation/image':public_image,'observation/state':expand_active(public_state),
                    'prompt':literal,'policy_noise':used_noise}
                if calls >= 180:
                    raise RuntimeError('Policy inference budget exhausted')
                calls += 1
                begin = time.perf_counter()
                response = client.infer(request)
                roundtrip = time.perf_counter()-begin
                actions,velocity,psec,ssec = extract_policy_output(response)
                if velocity.shape!=(10,16,29):
                    raise RuntimeError('Wrong fixed-trace shape')
                samples = [dict(proposal_index=0,noise_sha256=array_hash(request['policy_noise']),
                    actions_sha256=array_hash(actions),policy_seconds=psec,server_seconds=ssec,
                    roundtrip_seconds=roundtrip)]
                policy_seconds+=psec; server_seconds+=ssec; roundtrip_seconds+=roundtrip
                if not (rt.RUN/'POLICY_PREFLIGHT.json').exists():
                    if args.phase!='dev':
                        raise RuntimeError('Policy preflight must complete before final evaluation')
                    again = client.infer(deepcopy(request))
                    a2,v2,_,_ = extract_policy_output(again)
                    if not np.array_equal(a2,actions) or not np.array_equal(v2,velocity):
                        raise RuntimeError('Explicit-noise policy inference is not reproducible')
                    preflight_calls+=1
                    rt.atomic(rt.RUN/'POLICY_PREFLIGHT.json',dict(status='PASS',exact_repeat=True,
                        actions_shape=list(actions.shape),trace_shape=list(velocity.shape),
                        extra_dev_preflight_calls=1,metadata=record['server_metadata']))
                begin = time.perf_counter()
                k,author_diagnostics = select_exec_k_horizon(velocity,16,xt_step=actions,
                    history_actions=history,ts_state=posterior,ts_rng=rng,**options)
                diagnostics = dict(author_selector=author_diagnostics,sampling=sampling_info)
                selection_seconds+=time.perf_counter()-begin
                k = int(k)
                if not 4<=k<=16:
                    raise RuntimeError('Out-of-protocol execution prefix')
                actual = 0
                start_step = steps
                for action in actions[:k]:
                    before_step = time.perf_counter()
                    observation,reward,terminated,truncated,info = env.step(active_to_action_dict(action))
                    sim_seconds+=time.perf_counter()-before_step
                    steps+=1;actual+=1
                    history.append(np.asarray(action,dtype=np.float32).copy())
                    step_success = official_success(info)
                    if step_success and first_success is None:
                        first_success = steps
                    success = success or step_success
                    if steps>=720 or terminated or truncated:
                        break
                if args.method=='mine':
                    sampler.commit(used_noise,public_state,public_image,actual,sampling_info)
                    reused_decisions+=int(sampling_info['reused_positions']>0)
                    reused_positions+=sampling_info['reused_positions']
                    refreshes+=int(sampling_info['refresh'])
                line = dict(replan_index=decisions,start_step=start_step,selected_k=k,executed=actual,
                    primary_noise_sha256=samples[0]['noise_sha256'],primary_actions_sha256=samples[0]['actions_sha256'],
                    selected_actions_sha256=array_hash(actions),selected_proposal=0,samples=samples,reference_noise_sha256=array_hash(fresh_noise),
                    horizon_info=diagnostics,success_observed=bool(success),policy_calls_so_far=calls)
                if args.phase=='dev':
                    line['proposal_actions'] = [actions]
                    line['public_state'] = public_state
                stream.write(json.dumps(rt.jsonable(line),sort_keys=True,allow_nan=False)+'\n')
                stream.flush()
                decisions+=1
                rt.atomic(live_path,dict(record,environment_steps=steps,policy_calls=calls,
                    decisions=decisions,extra_samples=0,alternative_selections=0,noise_reused_decisions=reused_decisions,
                    noise_reused_positions=reused_positions,noise_refreshes=refreshes,last_update_unix=time.time()))
        record.update(status='COMPLETED',success=bool(success),first_success_step=first_success,
            environment_steps=steps,policy_calls=calls,decisions=decisions,
            extra_samples=0,alternative_selections=0,noise_reused_decisions=reused_decisions,
            noise_reused_positions=reused_positions,noise_refreshes=refreshes,policy_inference_seconds=policy_seconds,
            server_inference_seconds=server_seconds,policy_roundtrip_seconds=roundtrip_seconds,
            simulator_step_seconds=sim_seconds,selector_seconds=selection_seconds,
            wall_seconds=time.perf_counter()-started,horizon_violations=int(steps>720),
            inference_budget_violations=int(calls>180),terminated=bool(terminated),truncated=bool(truncated),
            prefix_caps=0,posterior_resets=0,paid_api_calls=0,external_vlm_calls=0,
            action_values_modified=False,extra_dev_preflight_calls=preflight_calls,
            settings=asdict(PROFILES[args.variant]) if args.method=='mine' else None)
        rt.atomic(output,record)
        rt.atomic(live_path,record)
        print(json.dumps({key:record[key] for key in ('status','task_name','seed','method','variant','success',
            'environment_steps','policy_calls','extra_samples','alternative_selections')}),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['dev','test'],required=True)
    parser.add_argument('--method',choices=['authors','mine'],required=True)
    parser.add_argument('--variant',choices=['authors',*PROFILES],required=True)
    parser.add_argument('--task-index',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        rt.atomic(rt.RUN/'LAST_ERROR.json',dict(time=time.time(),args=vars(args),error=str(exc),traceback=traceback.format_exc()))
        raise
