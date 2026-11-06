"""One native GR1 episode. The authors' branch ports their released cell loop.

Differences from the release: one subprocess per episode, integrity/state
auditing, and richer timing logs. Success is evaluator-only. Controllers receive
no simulator geometry, success predicate, episode seed or task identity.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import ct_runtime as rt
import numpy as np
from pi05_robocasa_common import (
    AHS_CONFIG, CHECKPOINT, NORM_ASSET_ID, NORM_STATS_SHA256, MODEL_ID,
    PACKAGE_TREE_SHA256, PROTOCOL_ID, SAMPLING_CONFIG, OPTIMIZER_STEP,
    deterministic_noise, reset_with_literal_seed, instruction_text, task_records,
)
from run_pi05_robocasa_cell import (
    active_state, expand_active, active_to_action_dict, official_success,
    extract_policy_output, jsonable, MODEL_IMAGE_KEY,
)
from horizon_selector import select_exec_k_horizon, apply_horizon_ts_update
from my_harness import MeasuredAHS


def array_hash(value):
    value = np.ascontiguousarray(value)
    material = str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
    return hashlib.sha256(material).hexdigest()


def initial_fingerprint(env, observation):
    # Audit only; never passed to either controller.
    physics = env.unwrapped.env.sim.get_state().flatten()
    return dict(image=array_hash(observation[MODEL_IMAGE_KEY]),
                public_state=array_hash(active_state(observation)),
                physics=array_hash(physics),
                instruction=instruction_text(observation['annotation.human.coarse_action']))


def verify_metadata(client):
    metadata = client.get_server_metadata()
    expected = dict(protocol_id=PROTOCOL_ID, model_id=MODEL_ID,
                    optimizer_step=OPTIMIZER_STEP, norm_asset_id=NORM_ASSET_ID,
                    norm_stats_sha256=NORM_STATS_SHA256,
                    package_tree_sha256=PACKAGE_TREE_SHA256, use_qha=False,
                    trace_num_steps_static_bound=10,
                    trace_wire_fields=['v'], trace_wire_dtype='float32')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f'Policy metadata mismatch: {key}')
    return metadata


def main(args):
    import gymnasium as gym
    from robocasa.utils.gym_utils import gymnasium_groot  # registration
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    frozen_sha = rt.verify_freeze()
    if args.phase == 'test' and frozen_sha is None:
        raise RuntimeError('Final evaluation cannot start before design freeze')
    if args.phase == 'test' and args.seed not in range(5):
        raise RuntimeError('Test seed outside frozen range')
    if args.phase == 'dev' and args.seed not in (10000, 10001):
        raise RuntimeError('Development seed must be disjoint from test')
    tasks = task_records()
    row = tasks[args.task_index]
    result_directory = 'render-smoke' if args.render_only else args.phase
    output = rt.RUN / result_directory / f"t{args.task_index:02d}-s{args.seed:05d}-{args.method}.json"
    trace_path = output.with_suffix('.trace.jsonl')
    live_path = output.with_suffix('.live.json')
    if output.exists():
        raise RuntimeError('Refusing to overwrite an existing episode')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Constructor RNGs are fixed identically for both arms, then the exact
    # released literal-reset procedure determines the benchmark initial state.
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(row['env_name'], enable_render=True)
    record = dict(status='RUNNING', phase=args.phase, method=args.method,
                  task_index=args.task_index, seed=args.seed, **row,
                  max_environment_steps=720, horizon_multiplier=1.0,
                  frozen_sha256=frozen_sha, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  started_unix=time.time(), environment_steps=0, policy_calls=0)
    rt.atomic(live_path, record)
    try:
        observation, reset_info = reset_with_literal_seed(env, args.seed)
        fingerprint = initial_fingerprint(env, observation)
        record['initial_fingerprint'] = fingerprint
        literal = fingerprint['instruction']
        # Rendering-only smoke checks native observation/action mapping without
        # requiring loaded weights; excluded from evaluation.
        if args.render_only:
            state = active_state(observation)
            env.step(active_to_action_dict(state))
            smoke = dict(record, status='RENDER_SMOKE_PASS', environment_steps=1)
            rt.atomic(output, smoke)
            rt.atomic(live_path, smoke)
            return
        client = WebsocketClientPolicy(host='127.0.0.1', port=rt.PORT)
        record['server_metadata'] = verify_metadata(client)
        history = []
        posterior = {}
        rng = np.random.default_rng(AHS_CONFIG['ts_seed'])
        options = dict(AHS_CONFIG)
        options.pop('ts_seed')
        norms = json.loads((CHECKPOINT / 'assets' / NORM_ASSET_ID / 'norm_stats.json').read_text())['norm_stats']['actions']
        ranges = np.asarray(norms['q99']) - np.asarray(norms['q01'])
        wrapper = MeasuredAHS(select_exec_k_horizon, apply_horizon_ts_update, AHS_CONFIG, ranges)
        success = official_success(reset_info)
        first_success = 0 if success else None
        terminated = truncated = False
        steps = calls = caps = resets = 0
        policy_seconds = server_seconds = roundtrip_seconds = sim_seconds = selector_seconds = 0.
        preflight_done = (rt.RUN / 'POLICY_PREFLIGHT.json').exists()
        with trace_path.open('x') as stream:
            while steps < 720 and not (terminated or truncated):
                if calls >= 180:
                    raise RuntimeError('Exceeded common maximum policy-call budget')
                request = dict(
                    **{'observation/image': np.asarray(observation[MODEL_IMAGE_KEY]),
                       'observation/state': expand_active(active_state(observation))},
                    prompt=literal, policy_noise=deterministic_noise(row['task_name'], args.seed, calls))
                begin = time.perf_counter()
                output_policy = client.infer(request)
                elapsed = time.perf_counter() - begin
                actions, trace, psec, ssec = extract_policy_output(output_policy)
                if trace.shape != (10, 16, 29):
                    raise RuntimeError(f'Unexpected fixed trace shape: {trace.shape}')
                if not preflight_done:
                    # Fixed-noise reproducibility checked once on a DEV request.
                    # Additional inference is a preflight cost, never an action
                    # proposal or episode statistic. Both controllers use the
                    # same original request/output.
                    again = client.infer(deepcopy(request))
                    a2, v2, _, _ = extract_policy_output(again)
                    if not np.array_equal(actions, a2) or not np.array_equal(trace, v2):
                        raise RuntimeError('Repeated identical-noise inference is not deterministic')
                    rt.atomic(rt.RUN / 'POLICY_PREFLIGHT.json', dict(
                        status='PASS', actions_shape=list(actions.shape), trace_shape=list(trace.shape),
                        exact_repeat=True, external_models=0, warmup_roundtrip_seconds=elapsed,
                        metadata=record['server_metadata'], dev_request_only=True))
                    preflight_done = True
                policy_seconds += psec
                server_seconds += ssec
                roundtrip_seconds += elapsed
                begin = time.perf_counter()
                if args.method == 'authors':
                    k, info = select_exec_k_horizon(trace, 16, xt_step=actions,
                        history_actions=history, ts_state=posterior, ts_rng=rng, **options)
                else:
                    k, info = wrapper.select(trace, actions, active_state(observation), history, steps)
                    evidence = info['my_evidence']
                    caps += int(k < evidence['author_proposal_k'])
                    resets += int(evidence['memory_reset_reason'] is not None)
                selector_elapsed = time.perf_counter() - begin
                selector_seconds += selector_elapsed
                k = int(k)
                if not 4 <= k <= 16:
                    raise RuntimeError(f'Out-of-protocol prefix {k}')
                start_step = steps
                actual = 0
                for action in actions[:k]:
                    begin = time.perf_counter()
                    observation, reward, terminated, truncated, info_env = env.step(active_to_action_dict(action))
                    sim_seconds += time.perf_counter() - begin
                    steps += 1
                    actual += 1
                    history.append(np.asarray(action, dtype=np.float32).copy())
                    step_success = official_success(info_env)
                    if step_success and first_success is None:
                        first_success = steps
                    success = success or step_success
                    if terminated or truncated or steps >= 720:
                        break
                calls += 1
                trace_row = dict(replan_index=calls-1, start_step=start_step, selected_k=k,
                    executed=actual, policy_noise_sha256=array_hash(request['policy_noise']),
                    generated_actions_sha256=array_hash(actions), policy_seconds=psec,
                    server_seconds=ssec, roundtrip_seconds=elapsed,
                    selector_seconds=selector_elapsed, horizon_info=info,
                    success_observed=bool(success))
                stream.write(json.dumps(jsonable(trace_row), sort_keys=True)+'\n')
                stream.flush()
                rt.atomic(live_path, dict(record, environment_steps=steps, policy_calls=calls,
                                         last_update_unix=time.time()))
        record.update(status='COMPLETED', success=bool(success), first_success_step=first_success,
                      environment_steps=steps, policy_calls=calls,
                      policy_inference_seconds=policy_seconds, server_inference_seconds=server_seconds,
                      policy_roundtrip_seconds=roundtrip_seconds, simulator_step_seconds=sim_seconds,
                      selector_seconds=selector_seconds, wall_seconds=time.perf_counter()-started,
                      horizon_violations=int(steps > 720), terminated=bool(terminated),
                      truncated=bool(truncated), prefix_caps=caps, posterior_resets=resets,
                      external_vlm_calls=0, paid_api_calls=0,
                      interpretation='VLA inference counted in policy_calls; no separate VLM judge',
                      preflight_cost_excluded_from_episode_inference=True)
        rt.atomic(output, record)
        rt.atomic(live_path, record)
        print(json.dumps({k:record[k] for k in ('status','task_name','seed','method','success','environment_steps','policy_calls','prefix_caps')}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['dev','test'], required=True)
    parser.add_argument('--method', choices=['authors','mine'], required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        rt.atomic(rt.RUN / 'LAST_ERROR.json', dict(time=time.time(), args=vars(args),
                  error=str(exc), traceback=traceback.format_exc()))
        raise
