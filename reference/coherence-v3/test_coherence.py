"""Tests of temporal alignment, refresh escape routes and noise statistics."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from coherent_noise import CoherentNoise, Settings, PROFILES


class CoherenceTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(137)
        self.state=np.zeros(29,dtype=np.float32)
        self.image=np.zeros((256,256,3),dtype=np.uint8)
        self.controller=CoherentNoise(np.ones(29))

    def fresh(self):
        return self.rng.standard_normal((16,32),dtype=np.float32)

    def first(self,k=7):
        fresh=self.fresh()
        used,info=self.controller.prepare(fresh,self.state,self.image,0)
        self.controller.commit(used,self.state,self.image,k,info)
        return used

    def test_first_draw_identical_and_no_input_mutation(self):
        fresh=self.fresh(); before=fresh.copy()
        used,info=self.controller.prepare(fresh,self.state,self.image,0)
        np.testing.assert_array_equal(used,fresh)
        used[0,0]+=1
        np.testing.assert_array_equal(fresh,before)
        self.assertEqual(info['refresh_reasons'],['first_decision'])
        self.assertEqual(info['extra_policy_samples'],0)

    def test_reuse_is_aligned_to_actual_executed_prefix_and_suffix_fresh(self):
        old=self.first(k=7); fresh=self.fresh()
        used,info=self.controller.prepare(fresh,self.state,self.image,7)
        expected=.5*old[7:]+np.sqrt(.75)*fresh[:9]
        np.testing.assert_array_equal(used[:9],expected)
        np.testing.assert_array_equal(used[9:],fresh[9:])
        self.assertEqual(info['reused_positions'],9)

    def test_prepare_does_not_advance_memory(self):
        self.first(); old=deepcopy(self.controller.__dict__)
        fresh=self.fresh()
        a,info_a=self.controller.prepare(fresh,self.state,self.image,7)
        b,info_b=self.controller.prepare(fresh,self.state,self.image,7)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(info_a,info_b)
        for key in old:
            if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],getattr(self.controller,key))
            else:self.assertEqual(old[key],getattr(self.controller,key))

    def test_visual_change_and_fixed_interval_reset_without_action(self):
        self.first()
        for image,step,reason in [(np.full_like(self.image,200),7,'scene_change'),
                                  (self.image,32,'bounded_refresh_interval')]:
            fresh=self.fresh(); used,info=self.controller.prepare(fresh,self.state,image,step)
            np.testing.assert_array_equal(used,fresh)
            self.assertIn(reason,info['refresh_reasons'])
            self.assertFalse(info['action_values_modified'])

    def test_persistent_stillness_refreshes_but_finger_contact_is_not_tracking_error(self):
        self.first(k=4)
        state=self.state.copy();state[14:26]=10
        for step in (4,8,12):
            fresh=self.fresh();used,info=self.controller.prepare(fresh,state,self.image,step)
            self.controller.commit(used,state,self.image,4,info)
        self.assertIn('persistent_observed_stillness',info['refresh_reasons'])
        self.assertEqual(info['observed_arm_change'],0)
        np.testing.assert_array_equal(used,fresh)

    def test_full_prefix_or_zero_correlation_matches_independent_sampling(self):
        self.first(k=16);fresh=self.fresh()
        used,info=self.controller.prepare(fresh,self.state,self.image,16)
        np.testing.assert_array_equal(used,fresh)
        self.assertIn('no_unexecuted_noise',info['refresh_reasons'])
        self.controller=CoherentNoise(np.ones(29),replace(Settings(),correlation=0))
        self.first(k=4)
        used,info=self.controller.prepare(fresh,self.state,self.image,4)
        np.testing.assert_array_equal(used,fresh)

    def test_fixed_gate_formula_retains_standard_gaussian_scale(self):
        # This tests the algebra for an exogenous fixed reuse choice. It does
        # not claim Gaussian conditional distributions after observation gates.
        for profile in PROFILES.values():
            old=self.rng.standard_normal((200000,))
            fresh=self.rng.standard_normal((200000,))
            used=profile.correlation*old+np.sqrt(1-profile.correlation**2)*fresh
            self.assertLess(abs(float(used.mean())),.01)
            self.assertLess(abs(float(used.var())-1),.02)

    def test_invalid_noise_and_commit_fail_before_inference(self):
        bad=self.fresh();bad[0,0]=np.nan
        with self.assertRaises(ValueError):self.controller.prepare(bad,self.state,self.image,0)
        with self.assertRaises(ValueError):self.controller.commit(self.fresh(),self.state,self.image,0,{})


if __name__=='__main__':unittest.main(verbosity=2)
