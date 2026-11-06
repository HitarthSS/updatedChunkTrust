"""Matched-coverage four-arm analysis and deterministic trace/budget audit."""
import json
import math

import runtime_v4 as rt
import numpy as np
from pi05_robocasa_common import deterministic_noise
from evaluate import array_hash,result_path
from analyze import wilson

ARMS=('authors','v4')
METRICS=('policy_calls','extra_samples','alternative_selections','noise_reused_decisions','noise_reused_positions',
    'noise_refreshes','policy_inference_seconds','server_inference_seconds','policy_roundtrip_seconds',
    'simulator_step_seconds','selector_seconds','wall_seconds','horizon_violations','inference_budget_violations')


def completed(phase):
    cases=json.loads((rt.RUN/(phase+'-cases.json')).read_text())
    groups=[];physical=0
    for row in cases:
        values={}
        for arm in row['arms']:
            path=result_path(phase,row['task_index'],row['seed'],arm)
            if not path.exists():continue
            v=json.loads(path.read_text())
            if v['status']!='COMPLETED':raise RuntimeError('Incomplete saved record '+str(path))
            if v['environment_steps']>720 or v['policy_calls']>180:raise RuntimeError('Budget violation')
            values[arm]=v;physical+=1
        if len(values)!=len(row['arms']):continue
        initial=[v['initial_fingerprint'] for v in values.values()]
        if any(v!=initial[0] for v in initial):raise RuntimeError('Unmatched initial state '+row['pair_id'])
        first=[];metadata=[]
        for arm in values:
            path=result_path(phase,row['task_index'],row['seed'],arm).with_suffix('.trace.jsonl')
            with path.open() as stream:line=json.loads(next(stream))
            first.append((line['primary_noise_sha256'],line['primary_actions_sha256']))
            metadata.append(values[arm]['server_metadata'])
        if any(v!=first[0] for v in first):raise RuntimeError('Initial policy output differs '+row['pair_id'])
        if any(v!=metadata[0] for v in metadata):raise RuntimeError('Different model metadata '+row['pair_id'])
        groups.append((row,values))
    return cases,groups,physical


def summary(phase):
    cases,groups,physical=completed(phase)
    if phase=='compatibility':
        out=dict(phase=phase,planned_cases=len(cases),completed_cases=len(groups),
            completed_physical_episodes=physical,planned_physical_episodes=sum(len(r['arms']) for r in cases),
            note='Noncomparative adapter checks; legacy duplicates are additional validation episodes')
        rt.atomic(rt.RUN/(phase+'-summary.json'),out)
        return out
    arms=cases[0]['arms']
    out=dict(phase=phase,planned_cases=len(cases),completed_cases=len(groups),completed_physical_episodes=physical,
        paired_physical_episodes=sum(len(v) for _,v in groups),planned_physical_episodes=sum(len(r['arms']) for r in cases),
        coverage='All displayed arms use only the same completed cases',initial_state_mismatches=0,initial_prediction_mismatches=0,
        arms={},per_task={},timing_caveat='One GPU and one policy server, serial interleaved episodes; other users may contend; descriptive timing only')
    for arm in arms:
        rows=[values[arm] for _,values in groups]
        n=len(rows);wins=sum(v['success'] for v in rows)
        out['arms'][arm]=dict(episodes=n,successes=wins,success_percent=100*wins/n if n else None,
            success_wilson_95_percent=wilson(wins,n),**{k:sum(v[k] for v in rows) for k in METRICS})
    for row,values in groups:
        task=out['per_task'].setdefault(row['task_name'],dict(cases=0,successes={a:0 for a in arms}))
        task['cases']+=1
        for arm in arms:task['successes'][arm]+=int(values[arm]['success'])
    rt.atomic(rt.RUN/(phase+'-summary.json'),out)
    return out


def contrast(groups,candidate,reference):
    d=np.array([int(v[candidate]['success'])-int(v[reference]['success']) for _,v in groups])
    helped=int(np.sum(d>0));harmed=int(np.sum(d<0));discordant=helped+harmed
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(helped,harmed)+1))/2**discordant) if discordant else 1.
    out=dict(candidate=candidate,reference=reference,episodes=len(d),helped=helped,harmed=harmed,
             difference_pp=100*float(d.mean()),mcnemar_exact_two_sided_p=p)
    rng=np.random.default_rng(643821)
    bs=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    out['paired_episode_bootstrap_95_difference_pp']=(100*np.quantile(bs,[.025,.975])).tolist()
    tasks=sorted({row['task_index'] for row,_ in groups})
    sums=np.array([sum(int(v[candidate]['success'])-int(v[reference]['success']) for row,v in groups if row['task_index']==t) for t in tasks])
    counts=np.array([sum(row['task_index']==t for row,_ in groups) for t in tasks])
    idx=rng.integers(0,len(tasks),size=(20000,len(tasks)))
    clustered=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    out['paired_task_cluster_bootstrap_95_difference_pp']=(100*np.quantile(clustered,[.025,.975])).tolist()
    return out


def audit_record(row,arm,record):
    old2,_=rt.originals()
    path=result_path('test',row['task_index'],row['seed'],arm).with_suffix('.trace.jsonl')
    lines=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    steps=calls=extras=winners=reused=positions=refreshes=0
    previous=None;previous_executed=0
    for i,line in enumerate(lines):
        if line['replan_index']!=i or line['start_step']!=steps:raise RuntimeError('Trace ordering drift')
        if not 4<=line['selected_k']<=16 or not 1<=line['executed']<=line['selected_k']:raise RuntimeError('Prefix drift')
        steps+=line['executed']
        fresh=deterministic_noise(row['task_name'],row['seed'],i)
        if line['reference_noise_sha256']!=array_hash(fresh):raise RuntimeError('Reference noise drift')
        used=fresh.copy();info=line['horizon_info']['sampling']
        if arm in ('v3','v4'):
            width=info['reused_positions'];rho=info['correlation']
            if width:
                if previous is None or width!=16-previous_executed or not 0<rho<1:raise RuntimeError('Noise alignment drift')
                used[:width]=rho*previous[previous_executed:]+math.sqrt(1-rho*rho)*fresh[:width]
            elif rho!=0:raise RuntimeError('Unexpected unused correlation')
            previous=used;previous_executed=line['executed']
            reused+=int(width>0);positions+=width;refreshes+=int(info['refresh'])
        samples=line['samples']
        if samples[0]['noise_sha256']!=array_hash(used):raise RuntimeError('Primary noise recipe mismatch')
        if line['primary_noise_sha256']!=samples[0]['noise_sha256']:raise RuntimeError('Primary input hash mismatch')
        if len(samples)>1:
            if arm!='v2' or len(samples)!=2:raise RuntimeError('Unauthorized extra inference')
            if samples[1]['noise_sha256']!=array_hash(old2.extra_noise(row['task_name'],row['seed'],i)):
                raise RuntimeError('Secondary noise mismatch')
        if line['selected_actions_sha256']!=samples[line['selected_proposal']]['actions_sha256']:
            raise RuntimeError('Generated action values changed')
        calls+=len(samples);extras+=len(samples)-1;winners+=int(line['selected_proposal']==1)
        if line['policy_calls_so_far']!=calls:raise RuntimeError('Inference accounting drift')
    expected=dict(environment_steps=steps,policy_calls=calls,decisions=len(lines),extra_samples=extras,
        alternative_selections=winners,noise_reused_decisions=reused,noise_reused_positions=positions,noise_refreshes=refreshes)
    if any(record[k]!=v for k,v in expected.items()):raise RuntimeError('Aggregate accounting mismatch')
    if steps>720 or calls>180 or record['paid_api_calls'] or record['external_vlm_calls'] or record['action_values_modified']:
        raise RuntimeError('Experiment constraint violated')


def report():
    rt.verify_freeze();rt.verify_previous()
    cases,groups,physical=completed('test')
    if len(groups)!=len(cases):raise RuntimeError('Final report requires all 120 paired cases')
    out=summary('test')
    for row,values in groups:
        for arm,record in values.items():audit_record(row,arm,record)
    contrasts=[contrast(groups,'v4','authors')]
    for item in contrasts:
        item['positive_difference_with_mcnemar_p_below_05']=bool(item['difference_pp']>0 and item['mcnemar_exact_two_sided_p']<.05)
    out['primary_contrasts']=contrasts
    out.update(trace_audit='PASS: all 240 episodes; noise recipe, action hashes, exact call accounting and native budgets',
        source_integrity='PASS: v1/v2/v3 and v4 frozen sources',selected_v4_profile=json.loads((rt.RUN/'FROZEN.json').read_text())['selected_v4_profile'],
        inference_interpretation='Same 180-call ceiling; both arms one call/decision; actual cost disclosed',
        interpretation='Five episodes/task pilot; task clustering and only five seed indices limit uncertainty estimates; no unseen-task-type claim',
        paper_target=dict(policy='pi0.5 + AHS',benchmark='RoboCasa GR1 Tabletop',reported_percent=42.5,reported_episodes=1200))
    rt.atomic(rt.RUN/'FINAL_ANALYSIS.json',out)
    return out


if __name__=='__main__':print(json.dumps(report(),indent=2))
