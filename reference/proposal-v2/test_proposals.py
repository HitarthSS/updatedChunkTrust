"""Meaningful invariants for proposal selection, reproducibility and budgets."""
from copy import deepcopy
import unittest

import runtime_v2 as rt
import numpy as np
from pi05_robocasa_common import AHS_CONFIG
from horizon_selector import select_exec_k_horizon, normalize_candidate_metric
from proposal_selector import ProposalAHS, CallBudget, PROFILES


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.chooser = ProposalAHS(select_exec_k_horizon,normalize_candidate_metric,AHS_CONFIG,np.ones(29))
        self.actions = np.zeros((16,29),dtype=np.float32)
        self.trace = np.zeros((10,16,29),dtype=np.float32)
        self.state = np.zeros(29,dtype=np.float32)
        self.image = np.zeros((256,256,3),dtype=np.uint8)

    def test_single_proposal_matches_released_selector_across_replans(self):
        source = np.random.default_rng(91)
        state = {}; rng = np.random.default_rng(AHS_CONFIG['ts_seed'])
        options = dict(AHS_CONFIG); options.pop('ts_seed')
        history = []
        for _ in range(20):
            actions = source.normal(0,.05,(16,29)).astype(np.float32)
            trace = source.normal(0,1,(10,16,29)).astype(np.float32)
            p = self.chooser.preview(actions,trace,history)
            index,_ = self.chooser.choose([p],self.state,self.image,history)
            expected,_ = select_exec_k_horizon(trace,16,xt_step=actions,history_actions=history,
                ts_state=state,ts_rng=rng,**options)
            self.assertEqual(index,0);self.assertEqual(p['horizon'],expected)
            self.chooser.commit(p,self.state,self.image,expected)
            self.assertEqual(self.chooser.posterior,state)
            self.assertEqual(self.chooser.rng.bit_generator.state,rng.bit_generator.state)
            history.extend(actions[:expected].copy())

    def test_previewing_losing_candidates_does_not_update_memory_or_rng(self):
        rng = deepcopy(self.chooser.rng.bit_generator.state)
        first = self.chooser.preview(self.actions,self.trace,[])
        second = self.chooser.preview(self.actions+.01,self.trace+.1,[])
        self.assertEqual(self.chooser.posterior,{})
        self.assertEqual(self.chooser.rng.bit_generator.state,rng)
        self.chooser.commit(second,self.state,self.image,second['horizon'])
        self.assertEqual(self.chooser.posterior,second['posterior'])
        self.assertEqual(self.chooser.rng.bit_generator.state,second['rng_state'])

    def test_shared_scaling_distinguishes_absolute_trace_quality(self):
        chooser = ProposalAHS(select_exec_k_horizon,normalize_candidate_metric,AHS_CONFIG,np.ones(29),PROFILES['quality'])
        first = chooser.preview(self.actions,self.trace,[])
        second = chooser.preview(self.actions+.03,self.trace,[])
        first['ahs']['z_intra']=[10.,11.,12.,13.]
        second['ahs']['z_intra']=[0.,1.,2.,3.]
        first['ahs']['u_inter_raw']=second['ahs']['u_inter_raw']=[0.,0.,0.,0.]
        originals=[p['actions'].copy() for p in (first,second)]
        selected,info=chooser.choose([first,second],self.state,self.image,[])
        self.assertEqual(selected,1)
        self.assertGreater(info['shared_quality'][1],info['shared_quality'][0])
        for p,original in zip((first,second),originals):np.testing.assert_array_equal(p['actions'],original)
        self.assertFalse(info['action_values_modified'])

    def test_identical_candidates_keep_primary_without_blending(self):
        p=self.chooser.preview(self.actions,self.trace,[])
        selected,_=self.chooser.choose([p,deepcopy(p)],self.state,self.image,[])
        self.assertEqual(selected,0)

    def test_first_decision_cannot_spend_reserved_calls(self):
        b=CallBudget();b.primary()
        self.assertFalse(b.can_buy_extra(720))
        with self.assertRaises(RuntimeError):b.extra(720)
        self.assertTrue(b.can_buy_extra(704))

    def test_budget_finishes_horizon_under_adversarial_short_chunks(self):
        for seed in range(30):
            rng=np.random.default_rng(seed);b=CallBudget();steps=0;extras=0
            while steps<720:
                b.primary()
                if b.can_buy_extra(720-steps):b.extra(720-steps);extras+=1
                k=4 if rng.random()<.8 else int(rng.integers(4,17))
                steps+=min(k,720-steps)
                self.assertLessEqual(b.used+int(np.ceil((720-steps)/4)),180)
            self.assertEqual(steps,720);self.assertLessEqual(b.used,180)

    def test_hand_tracking_residual_is_not_an_arm_failure_signal(self):
        baseline=self.chooser.features(self.actions,self.state,self.image,[])
        state=self.state.copy();state[14:26]=50.
        changed=self.chooser.features(self.actions,state,self.image,[])
        self.assertEqual(baseline['boundary_score'],changed['boundary_score'])

    def test_overlapping_predictions_are_aligned_by_executed_steps(self):
        old=self.actions.copy();old[:,0]=np.arange(16)*.01
        p=self.chooser.preview(old,self.trace,[])
        self.chooser.commit(p,self.state,self.image,4)
        new=self.actions.copy();new[:12]=old[4:]
        features=self.chooser.features(new,self.state,self.image,[])
        self.assertAlmostEqual(features['overlap_rms'],0.)

    def test_preview_rejects_invalid_model_trace(self):
        trace=self.trace.copy();trace[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.chooser.preview(self.actions,trace,[])


if __name__=='__main__':unittest.main()
