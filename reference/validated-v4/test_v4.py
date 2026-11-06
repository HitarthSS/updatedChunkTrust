"""Controller behavior, frozen-controller identity and fair schedule checks."""
from copy import deepcopy
from dataclasses import replace
import inspect
import math
import unittest

import runtime_v4 as rt
import numpy as np
from controller import ValidatedNoise,Settings,PROFILES
from pi05_robocasa_common import AHS_CONFIG,deterministic_noise
from horizon_selector import select_exec_k_horizon,normalize_candidate_metric


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(593)
        self.state=np.zeros(29,dtype=np.float32)
        self.image=np.zeros((256,256,3),dtype=np.uint8)
        self.actions=np.zeros((16,29),dtype=np.float32)
        self.actions[:,:14]=np.linspace(0,.04,16)[:,None]
        self.c=ValidatedNoise(np.ones(29))

    def fresh(self):return self.rng.standard_normal((16,32),dtype=np.float32)

    def start(self,k=8):
        noise=self.fresh();used,d=self.c.prepare(noise,self.state,self.image,0)
        self.c.commit(used,self.state,self.image,self.actions,k,d)
        return used

    def test_first_draw_exact_and_observation_inputs_not_mutated(self):
        fresh=self.fresh();original=fresh.copy()
        used,d=self.c.prepare(fresh,self.state,self.image,0)
        np.testing.assert_array_equal(used,fresh)
        used[0,0]+=1
        np.testing.assert_array_equal(fresh,original)
        self.assertEqual(d['refresh_reasons'],['first_decision'])
        self.assertEqual(d['extra_policy_samples'],0)

    def test_correct_tracking_preserves_time_aligned_noise(self):
        old=self.start(k=8);fresh=self.fresh();state=self.actions[7].copy()
        used,d=self.c.prepare(fresh,state,self.image,8)
        self.assertAlmostEqual(d['correlation'],.5,places=9)
        self.assertEqual(d['reused_positions'],8)
        rho=d['correlation']
        np.testing.assert_array_equal(used[:8],rho*old[8:]+math.sqrt(1-rho*rho)*fresh[:8])
        np.testing.assert_array_equal(used[8:],fresh[8:])

    def test_motion_mismatch_releases_reuse_without_capping_actions(self):
        self.start();fresh=self.fresh()
        used,d=self.c.prepare(fresh,self.state,self.image,8)
        np.testing.assert_array_equal(used,fresh)
        self.assertEqual(d['metrics']['projected_progress'],0)
        self.assertIn('insufficient_prediction_agreement',d['refresh_reasons'])
        self.assertFalse(d['action_values_modified'])
        # A slower but correctly directed motion gives intermediate correlation.
        half=self.actions[7]*.5
        _,dh=self.c.prepare(fresh,half,self.image,8)
        self.assertGreater(dh['correlation'],0)
        self.assertLess(dh['correlation'],.5)

    def test_finger_contact_does_not_trigger_arm_failure(self):
        self.start();state=self.actions[7].copy();state[14:26]=100
        _,d=self.c.prepare(self.fresh(),state,self.image,8)
        self.assertAlmostEqual(d['correlation'],.5,places=9)
        self.assertEqual(d['metrics']['endpoint_error'],0)

    def test_idle_command_not_mislabeled_as_failed_motion(self):
        self.actions[:]=0
        self.start(k=4)
        _,d=self.c.prepare(self.fresh(),self.state,self.image,4)
        self.assertIsNone(d['metrics']['projected_progress'])
        self.assertEqual(d['next_low_progress_count'],0)
        self.assertAlmostEqual(d['correlation'],.5)

    def test_scene_change_and_interval_force_fresh_noise(self):
        self.start();state=self.actions[7]
        for image,step,reason in ((np.full_like(self.image,200),8,'large_scene_change'),
                                  (self.image,32,'bounded_refresh_interval')):
            fresh=self.fresh();used,d=self.c.prepare(fresh,state,image,step)
            np.testing.assert_array_equal(used,fresh)
            self.assertIn(reason,d['refresh_reasons'])

    def test_prepare_is_pure_and_full_prefix_retains_no_stale_slots(self):
        self.start(k=16);fresh=self.fresh();before=deepcopy(self.c.__dict__)
        a,da=self.c.prepare(fresh,self.actions[15],self.image,16)
        b,db=self.c.prepare(fresh,self.actions[15],self.image,16)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,fresh)
        self.assertEqual(da,db);self.assertIn('no_unexecuted_noise',da['refresh_reasons'])
        np.testing.assert_array_equal(before['previous_noise'],self.c.previous_noise)
        self.assertEqual(before['last_refresh_step'],self.c.last_refresh_step)

    def test_invalid_inputs_rejected(self):
        bad=self.fresh();bad[0,0]=np.nan
        with self.assertRaises(ValueError):self.c.prepare(bad,self.state,self.image,0)
        with self.assertRaises(ValueError):self.c.commit(self.fresh(),self.state,self.image,self.actions,0,{})
        names=set(inspect.signature(ValidatedNoise.prepare).parameters)
        self.assertEqual(names,{'self','fresh','observed_state','image','environment_step'})

    def test_original_controllers_and_extra_noise_loaded_unchanged(self):
        old2,old3=rt.originals()
        rt.verify_previous()
        self.assertEqual(old2.PROFILES['balanced'].maximum_extra_samples,24)
        self.assertEqual(old3.PROFILES['moderate'].correlation,.5)
        source=inspect.getfile(old2.ProposalAHS)
        self.assertEqual(source,str(rt.BASE/'proposal-v2/proposal_selector.py'))
        self.assertEqual(inspect.getfile(old3.CoherentNoise),str(rt.BASE/'coherence-v3/coherent_noise.py'))
        chooser=old2.ProposalAHS(select_exec_k_horizon,normalize_candidate_metric,AHS_CONFIG,np.ones(29),old2.PROFILES['balanced'])
        state={};rng=np.random.default_rng(AHS_CONFIG['ts_seed']);options=dict(AHS_CONFIG);options.pop('ts_seed')
        history=[]
        for i in range(12):
            actions=self.rng.normal(0,.05,(16,29)).astype(np.float32)
            v=self.rng.normal(0,1,(10,16,29)).astype(np.float32)
            p=chooser.preview(actions,v,history)
            k,_=select_exec_k_horizon(v,16,xt_step=actions,history_actions=history,ts_state=state,ts_rng=rng,**options)
            self.assertEqual(k,p['horizon'])
            chooser.commit(p,self.state,self.image,k)
            self.assertEqual(chooser.posterior,state)
            history.extend(actions[:k].copy())

    def test_case_sets_disjoint_and_crossover_positions_balanced(self):
        from pipeline import cases,ORDERS
        test=cases('test');dev=cases('dev');compat=cases('compatibility')
        self.assertEqual(len(test),120);self.assertEqual(len(dev),16)
        self.assertEqual(len({(r['task_index'],r['seed']) for r in test}),120)
        self.assertEqual({r['seed'] for r in test},set(range(15,20)))
        self.assertTrue({r['seed'] for r in test}.isdisjoint({r['seed'] for r in dev+compat}))
        for task in range(24):self.assertEqual(sum(r['task_index']==task for r in test),5)
        self.assertTrue(all(r['arms']==['authors','v4'] for r in test))
        for arm in ('authors','v4'):
            for position in range(2):self.assertEqual(sum(ORDERS[i%len(ORDERS)][position]==arm for i in range(120)),60)

    def test_common_call_budget_covers_minimum_prefix_horizon(self):
        old2,_=rt.originals()
        for seed in range(20):
            rng=np.random.default_rng(seed);b=old2.CallBudget();step=0
            while step<720:
                b.primary()
                if b.can_buy_extra(720-step) and rng.random()<.5:b.extra(720-step)
                step+=min(720-step,int(rng.integers(4,17)))
                self.assertLessEqual(b.used+math.ceil((720-step)/4),180)
            self.assertEqual(step,720)


if __name__=='__main__':unittest.main(verbosity=2)
