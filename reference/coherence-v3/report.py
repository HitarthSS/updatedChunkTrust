"""Read-only paired analysis and one-call-per-decision noise-sampling audit."""
import json

import runtime_v3 as rt
from analyze import analyze


def report():
    rt.verify_freeze()
    result=analyze(rt.RUN,'test')
    cases=json.loads((rt.RUN/'test-cases.json').read_text())
    audits={method:dict(extra_samples=0,alternative_selections=0,discarded_proposals=0,
                       inference_budget_violations=0,noise_reused_decisions=0,noise_reused_positions=0,noise_refreshes=0) for method in ('authors','mine')}
    for row in cases:
        paths={m:rt.RUN/'test'/f"{row['pair_id']}-{m}.json" for m in audits}
        if not all(p.exists() for p in paths.values()):continue
        for method,path in paths.items():
            record=json.loads(path.read_text())
            if record['status']!='COMPLETED':raise RuntimeError('Incomplete result '+str(path))
            lines=[json.loads(s) for s in path.with_suffix('.trace.jsonl').read_text().splitlines() if s.strip()]
            calls=sum(len(line['samples']) for line in lines)
            extras=sum(len(line['samples'])-1 for line in lines)
            winners=sum(line['selected_proposal']==1 for line in lines)
            steps=0
            for line in lines:
                assert line['start_step']==steps
                assert 4<=line['selected_k']<=16
                assert 1<=line['executed']<=line['selected_k']
                steps+=line['executed']
                assert steps<=720
            assert calls==len(lines)==record['policy_calls']<=180
            assert extras==winners==0
            reused=sum(line['horizon_info']['sampling']['reused_positions']>0 for line in lines) if method=='mine' else 0
            positions=sum(line['horizon_info']['sampling']['reused_positions'] for line in lines) if method=='mine' else 0
            refreshes=sum(line['horizon_info']['sampling']['refresh'] for line in lines) if method=='mine' else 0
            assert reused==record['noise_reused_decisions']
            assert positions==record['noise_reused_positions']
            assert refreshes==record['noise_refreshes']
            for key in ('noise_reused_decisions','noise_reused_positions','noise_refreshes'):
                audits[method][key]+=record[key]
            assert extras==record['extra_samples']
            assert winners==record['alternative_selections']
            assert steps==record['environment_steps']
            assert len(lines)==record['decisions']
            assert record['action_values_modified'] is False
            audits[method]['extra_samples']+=extras
            audits[method]['alternative_selections']+=winners
            audits[method]['discarded_proposals']+=extras
            audits[method]['inference_budget_violations']+=record['inference_budget_violations']
    for method in audits:result['arms'][method].update(audits[method])
    result['design']='Scene-gated coherent sampling v3'
    result['selected_variant']=json.loads((rt.RUN/'FROZEN.json').read_text())['selected_variant']
    result['sample_accounting_audit']='PASS: exactly one counted inference per decision; no discarded proposals'
    result['budget_interpretation']='Same 720-step and 180-call ceilings; actual inference computation may differ'
    result['timing_caveat']='Shared GPU; descriptive timing, no controlled speed claim'
    rt.atomic(rt.RUN/'FINAL_ANALYSIS.json',result)
    return result


if __name__=='__main__':print(json.dumps(report(),indent=2,sort_keys=True))
