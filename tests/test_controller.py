"""NumPy-only contract tests for the exported noise controller."""
import unittest
import numpy as np
from my_harness import ValidatedNoise, PROFILES


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller=ValidatedNoise(np.ones(29), PROFILES['adaptive'])
        self.state=np.zeros(29,dtype=np.float32)
        self.image=np.zeros((256,256,3),dtype=np.uint8)
        self.rng=np.random.default_rng(93)

    def noise(self):return self.rng.standard_normal((16,32),dtype=np.float32)

    def test_first_noise_is_unchanged_and_copied(self):
        raw=self.noise();saved=raw.copy()
        used,info=self.controller.prepare(raw,self.state,self.image,0)
        np.testing.assert_array_equal(raw,used)
        used[0,0]+=1
        np.testing.assert_array_equal(raw,saved)
        self.assertEqual(info['extra_policy_samples'],0)

    def test_tracking_agreement_reuses_only_unexecuted_noise(self):
        old,info=self.controller.prepare(self.noise(),self.state,self.image,0)
        actions=np.zeros((16,29),dtype=np.float32)
        actions[:,:14]=np.linspace(0,.04,16)[:,None]
        self.controller.commit(old,self.state,self.image,actions,8,info)
        fresh=self.noise()
        used,next_info=self.controller.prepare(fresh,actions[7],self.image,8)
        self.assertGreater(next_info['correlation'],0)
        self.assertEqual(next_info['reused_positions'],8)
        np.testing.assert_array_equal(used[8:],fresh[8:])

    def test_large_visual_change_refreshes_all_noise(self):
        old,info=self.controller.prepare(self.noise(),self.state,self.image,0)
        actions=np.zeros((16,29),dtype=np.float32)
        self.controller.commit(old,self.state,self.image,actions,8,info)
        fresh=self.noise()
        used,next_info=self.controller.prepare(fresh,self.state,np.full_like(self.image,255),8)
        np.testing.assert_array_equal(used,fresh)
        self.assertIn('large_scene_change',next_info['refresh_reasons'])


if __name__=='__main__':unittest.main()
