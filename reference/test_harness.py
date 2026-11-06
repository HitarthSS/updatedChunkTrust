import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from my_harness import MeasuredAHS

ROOT=Path(__file__).parent
release=ROOT/('vendor' if (ROOT/'vendor').exists() else 'ChunkTrust')
path=release/'third_party/overlays/starvla/examples/Robocasa_tabletop/eval_files/horizon_selector.py'
spec=importlib.util.spec_from_file_location('authors',path);authors=importlib.util.module_from_spec(spec);spec.loader.exec_module(authors)
cfg=dict(exec_mode='expected_round',expected_temp=.8,intra_alpha=4.,intra_cut_t=.25,
    ts_epsilon=.05,ts_seed=0,ts_update_mode='kernel_forget',ts_kernel_bandwidth=10.,ts_forget_rho=.99,
    ts_update_eta=1.,z_mode='window_rms',tau_window=3,mix_mode='both',use_speed_uniformity=True,
    speed_window=20,speed_weight=.5,speed_beta=4.,speed_fallback='ehh_only')


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.wrapper=MeasuredAHS(authors.select_exec_k_horizon,authors.apply_horizon_ts_update,cfg,np.ones(29))
        self.actions=np.zeros((16,29));self.trace=np.zeros((10,16,29));self.state=np.zeros(29)

    def test_healthy_execution_matches_original_numerically(self):
        original_state={};rng=np.random.default_rng(0);options=dict(cfg);options.pop('ts_seed')
        for step in range(0,100,10):
            k,info=self.wrapper.select(self.trace,self.actions,self.state,[],step)
            expected,_=authors.select_exec_k_horizon(self.trace,16,horizon_candidates=[4,8,12,16],
                xt_step=self.actions,history_actions=[],ts_state=original_state,ts_rng=rng,**options)
            self.assertEqual(k,expected);self.assertEqual(self.wrapper.posterior,original_state)

    def test_hand_change_causes_bounded_checkpoint_without_changing_actions(self):
        actions=self.actions.copy();actions[1:,14:20]=1.;original=actions.copy()
        k,info=self.wrapper.select(self.trace,actions,self.state,[],0)
        self.assertLessEqual(k,4);np.testing.assert_array_equal(actions,original)
        self.assertEqual(info['my_evidence']['extra_policy_samples'],0)

    def test_one_tracking_error_does_not_trigger_stall(self):
        state=self.state.copy();state[:14]=.5
        _,info=self.wrapper.select(self.trace,self.actions,state,[self.state.copy()],16)
        self.assertFalse(info['my_evidence']['tracking_limited'])

    def test_persistent_arm_lag_caps_execution_and_recovers(self):
        state=self.state.copy();state[:14]=.5
        self.wrapper.select(self.trace,self.actions,state,[self.state.copy()],16)
        k,info=self.wrapper.select(self.trace,self.actions,state,[self.state.copy()],24)
        self.assertEqual(k,4);self.assertTrue(info['my_evidence']['tracking_limited'])
        _,info=self.wrapper.select(self.trace,self.actions,self.state,[self.state.copy()],28)
        self.assertFalse(info['my_evidence']['tracking_limited'])

    def test_contact_dependent_finger_residual_is_not_a_failure_signal(self):
        state=self.state.copy();state[14:26]=.5
        for step in (16,24,32):
            _,info=self.wrapper.select(self.trace,self.actions,state,[self.state.copy()],step)
            self.assertEqual(info['my_evidence']['normalized_arm_tracking_rms'],0.)
            self.assertFalse(info['my_evidence']['tracking_limited'])

    def test_forced_prefix_updates_actual_memory_center(self):
        self.actions[:,14:26]=1.
        k,info=self.wrapper.select(self.trace,self.actions,self.state,[],0)
        self.assertGreater(info['my_evidence']['author_proposal_k'],k)
        q=info['author_selector']['q_mix'][0]
        # The released kernel update spreads one unit of evidence across all
        # anchors; its peak must follow the prefix actually executed, not the
        # discarded AHS proposal.
        weights=np.exp(-((np.array([4,8,12,16])-4.)**2)/(2*10.**2))
        weights/=weights.sum()
        for horizon,weight in zip([4,8,12,16],weights):
            self.assertAlmostEqual(self.wrapper.posterior[horizon][0],.99+weight*q)
            self.assertAlmostEqual(self.wrapper.posterior[horizon][1],.99+weight*(1-q))
        self.assertGreater(self.wrapper.posterior[4][0],self.wrapper.posterior[16][0])

    def test_episode_reset_clears_all_adaptation(self):
        self.wrapper.select(self.trace,self.actions,self.state,[],0);self.wrapper.reset()
        self.assertFalse(self.wrapper.posterior);self.assertEqual(self.wrapper.lag_strikes,0)
        self.assertFalse(self.wrapper.tracking_limited)

    def test_bad_observation_fails_instead_of_inventing_geometry(self):
        bad=self.state.copy();bad[0]=np.nan
        with self.assertRaises(ValueError):self.wrapper.select(self.trace,self.actions,bad,[],0)


if __name__=='__main__':unittest.main()
