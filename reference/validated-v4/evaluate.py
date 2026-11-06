"""One native execution loop for authors' AHS and frozen v2/v3/new v4."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import random
import time
import traceback

import runtime_v4 as rt
import numpy as np
from pi05_robocasa_common import AHS_CONFIG,CHECKPOINT,NORM_ASSET_ID,deterministic_noise,reset_with_literal_seed,instruction_text,task_records
from run_pi05_robocasa_cell import active_state,expand_active,active_to_action_dict,official_success,extract_policy_output,MODEL_IMAGE_KEY
from horizon_selector import select_exec_k_horizon,normalize_candidate_metric
from controller import ValidatedNoise,PROFILES


def array_hash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def result_path(phase,task_index,seed,arm):
    return rt.RUN/phase/f't{task_index:02d}-s{seed:05d}-{arm}.json'


def run(args):
    import gymnasium as gym
    from robocasa.utils.gym_utils import gymnasium_groot
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    frozen=rt.verify_freeze()
    declared=json.loads((rt.RUN/(args.phase+'-cases.json')).read_text())
    matches=[r for r in declared if r['task_index']==args.task_index and r['seed']==args.seed]
    if len(matches)!=1 or args.arm not in matches[0]['arms']:
        raise RuntimeError('Undeclared phase/case/arm')
    if args.phase=='test':
        if not frozen:raise RuntimeError('Evaluation requires an immutable freeze')
        profile=json.loads((rt.RUN/'FROZEN.json').read_text())['selected_v4_profile']
    else:profile=args.arm if args.arm in PROFILES else None
    if args.arm=='v4' and profile not in PROFILES:raise RuntimeError('Missing frozen v4 profile')
    output=result_path(args.phase,args.task_index,args.seed,args.arm)
    if output.exists():raise RuntimeError('Refusing to overwrite completed result '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    old2,old3=rt.originals()
    row=task_records()[args.task_index]
    random.seed(args.seed);np.random.seed(args.seed)
    started=time.perf_counter()
    record=dict(status='RUNNING',version='Common four-arm ChunkTrust comparison v4',phase=args.phase,
        arm=args.arm,method=args.arm,variant=profile if args.arm in (*PROFILES,'v4') else args.arm,
        task_index=args.task_index,seed=args.seed,**row,frozen_sha256=frozen,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),policy_gpu=6,sim_gpu=6,
        max_environment_steps=720,horizon_multiplier=1.,policy_call_cap=180,
        started_unix=time.time(),environment_steps=0,policy_calls=0)
    live=output.with_suffix('.live.json')
    rt.atomic(live,record)
    env=gym.make(row['env_name'],enable_render=True)
    try:
        observation,reset_info=reset_with_literal_seed(env,args.seed)
        literal=instruction_text(observation['annotation.human.coarse_action'])
        record['initial_fingerprint']=dict(instruction=literal,public_state=array_hash(active_state(observation)),
            image=array_hash(observation[MODEL_IMAGE_KEY]),physics=array_hash(env.unwrapped.env.sim.get_state().flatten()))
        client=WebsocketClientPolicy(host='127.0.0.1',port=rt.PORT)
        record['server_metadata']=old2.verify_metadata(client)
        expected_metadata=json.loads((rt.RUN/'policy-ready.json').read_text())
        if record['server_metadata']!=expected_metadata:raise RuntimeError('Policy metadata changed')
        norms=json.loads((CHECKPOINT/'assets'/NORM_ASSET_ID/'norm_stats.json').read_text())['norm_stats']['actions']
        ranges=np.asarray(norms['q99'])-np.asarray(norms['q01'])
        chooser=old2.ProposalAHS(select_exec_k_horizon,normalize_candidate_metric,AHS_CONFIG,ranges,old2.PROFILES['balanced']) if args.arm=='v2' else None
        sampler=old3.CoherentNoise(ranges,old3.PROFILES['moderate']) if args.arm=='v3' else (
            ValidatedNoise(ranges,PROFILES[profile]) if args.arm in (*PROFILES,'v4') else None)
        posterior={};rng=np.random.default_rng(AHS_CONFIG['ts_seed'])
        options=dict(AHS_CONFIG);options.pop('ts_seed')
        history=[];budget=old2.CallBudget()
        steps=decisions=alternatives=reused_decisions=reused_positions=refreshes=0
        policy_seconds=server_seconds=roundtrip_seconds=sim_seconds=selector_seconds=0.
        success=official_success(reset_info);first_success=0 if success else None
        terminated=truncated=False;preflight_calls=0
        with output.with_suffix('.trace.jsonl').open('x') as trace:
            while steps<720 and not (terminated or truncated):
                before=time.perf_counter()
                state=active_state(observation).copy();image=np.asarray(observation[MODEL_IMAGE_KEY]).copy()
                fresh=deterministic_noise(row['task_name'],args.seed,decisions)
                if sampler is None:
                    used=fresh
                    sampling=dict(reused_positions=0,correlation=0.,refresh=True,refresh_reasons=['independent_sampling'])
                else:used,sampling=sampler.prepare(fresh,state,image,steps)
                request={'observation/image':image,'observation/state':expand_active(state),'prompt':literal,'policy_noise':used}
                selector_seconds+=time.perf_counter()-before
                samples=[]

                def infer(request,index):
                    nonlocal policy_seconds,server_seconds,roundtrip_seconds
                    begin=time.perf_counter();response=client.infer(request);elapsed=time.perf_counter()-begin
                    a,v,psec,ssec=extract_policy_output(response)
                    if a.shape!=(16,29) or v.shape!=(10,16,29) or not np.isfinite(a).all() or not np.isfinite(v).all():
                        raise RuntimeError('Non-finite or wrong-shaped policy output')
                    samples.append(dict(proposal_index=index,noise_sha256=array_hash(request['policy_noise']),actions_sha256=array_hash(a),
                        policy_seconds=psec,server_seconds=ssec,roundtrip_seconds=elapsed))
                    policy_seconds+=psec;server_seconds+=ssec;roundtrip_seconds+=elapsed
                    return a,v

                budget.primary();actions,velocity=infer(request,0)
                if not (rt.RUN/'POLICY_PREFLIGHT.json').exists():
                    if args.phase!='compatibility':raise RuntimeError('Missing preflight')
                    a2,v2,_,_=extract_policy_output(client.infer(deepcopy(request)))
                    if not np.array_equal(a2,actions) or not np.array_equal(v2,velocity):
                        raise RuntimeError('Repeated explicit-noise inference differs')
                    preflight_calls+=1
                    rt.atomic(rt.RUN/'POLICY_PREFLIGHT.json',dict(status='PASS',exact_repeat=True,extra_compatibility_preflight_calls=1,
                        metadata=record['server_metadata']))
                before=time.perf_counter()
                selected=0;previews=[]
                if chooser is not None:
                    first=chooser.preview(actions,velocity,history)
                    eligible,gate=chooser.request_second(first,state,image,history,steps,budget)
                    previews=[first]
                    selector_seconds+=time.perf_counter()-before
                    if eligible:
                        budget.extra(720-steps);chooser.record_extra(steps)
                        a2,v2=infer(dict(request,policy_noise=old2.extra_noise(row['task_name'],args.seed,decisions)),1)
                        before=time.perf_counter();previews.append(chooser.preview(a2,v2,history))
                    else:before=time.perf_counter()
                    selected,comparison=chooser.choose(previews,state,image,history)
                    actions=previews[selected]['actions'];k=previews[selected]['horizon']
                    alternatives+=int(selected==1)
                    diagnostics=dict(author_selector=previews[selected]['ahs'],gate=gate,comparison=comparison,sampling=sampling)
                else:
                    k,info=select_exec_k_horizon(velocity,16,xt_step=actions,history_actions=history,
                        ts_state=posterior,ts_rng=rng,**options)
                    diagnostics=dict(author_selector=info,sampling=sampling)
                selector_seconds+=time.perf_counter()-before
                k=int(k)
                if not 4<=k<=16:raise RuntimeError('Invalid AHS prefix')
                start_step=steps;actual=0
                for action in actions[:k]:
                    begin=time.perf_counter()
                    observation,reward,terminated,truncated,info=env.step(active_to_action_dict(action))
                    sim_seconds+=time.perf_counter()-begin
                    steps+=1;actual+=1;history.append(np.asarray(action,dtype=np.float32).copy())
                    ok=official_success(info)
                    if ok and first_success is None:first_success=steps
                    success=success or ok
                    if steps>=720 or terminated or truncated:break
                before=time.perf_counter()
                if chooser is not None:chooser.commit(previews[selected],state,image,actual)
                elif sampler is not None:
                    if args.arm=='v3':sampler.commit(used,state,image,actual,sampling)
                    else:sampler.commit(used,state,image,actions,actual,sampling)
                    reused_decisions+=int(sampling['reused_positions']>0)
                    reused_positions+=sampling['reused_positions'];refreshes+=int(sampling['refresh'])
                selector_seconds+=time.perf_counter()-before
                line=dict(replan_index=decisions,start_step=start_step,selected_k=k,executed=actual,
                    primary_noise_sha256=samples[0]['noise_sha256'],reference_noise_sha256=array_hash(fresh),
                    primary_actions_sha256=samples[0]['actions_sha256'],selected_actions_sha256=array_hash(actions),
                    selected_proposal=selected,samples=samples,horizon_info=diagnostics,
                    input_state_sha256=array_hash(state),input_image_sha256=array_hash(image),
                    success_observed=bool(success),policy_calls_so_far=budget.used)
                if args.phase!='test':
                    line['proposal_actions']=[p['actions'] for p in previews] if previews else [actions]
                    line['public_state']=state
                trace.write(json.dumps(rt.jsonable(line),sort_keys=True,allow_nan=False)+'\n');trace.flush()
                decisions+=1
                rt.atomic(live,dict(record,environment_steps=steps,policy_calls=budget.used,decisions=decisions,
                    extra_samples=chooser.extra_samples if chooser else 0,alternative_selections=alternatives,
                    noise_reused_decisions=reused_decisions,noise_reused_positions=reused_positions,noise_refreshes=refreshes,
                    last_update_unix=time.time()))
        record.update(status='COMPLETED',success=bool(success),first_success_step=first_success,
            environment_steps=steps,policy_calls=budget.used,decisions=decisions,
            extra_samples=chooser.extra_samples if chooser else 0,alternative_selections=alternatives,
            noise_reused_decisions=reused_decisions,noise_reused_positions=reused_positions,noise_refreshes=refreshes,
            policy_inference_seconds=policy_seconds,server_inference_seconds=server_seconds,policy_roundtrip_seconds=roundtrip_seconds,
            simulator_step_seconds=sim_seconds,selector_seconds=selector_seconds,wall_seconds=time.perf_counter()-started,
            horizon_violations=int(steps>720),inference_budget_violations=int(budget.used>180),
            terminated=bool(terminated),truncated=bool(truncated),paid_api_calls=0,external_vlm_calls=0,
            action_values_modified=False,extra_compatibility_preflight_calls=preflight_calls,
            settings=asdict(PROFILES[profile]) if args.arm in (*PROFILES,'v4') else None)
        rt.atomic(output,record);rt.atomic(live,record)
        print(json.dumps({k:record[k] for k in ('status','arm','task_name','seed','success','environment_steps','policy_calls')}),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['compatibility','dev','test'],required=True)
    p.add_argument('--arm',choices=['authors','v2','v3','v4',*PROFILES],required=True)
    p.add_argument('--task-index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        rt.atomic(rt.RUN/'LAST_ERROR.json',dict(time=time.time(),args=vars(args),error=str(e),traceback=traceback.format_exc()))
        raise
