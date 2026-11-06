"""Episode-local, time-aligned noise reuse for a frozen action-chunk policy.

The controller receives public observations and independent Gaussian draws;
it never receives a task label, seed, evaluator success or simulator state.
It changes sampling noise only, not policy weights or generated action values.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np


@dataclass(frozen=True)
class Settings:
    correlation: float = .5
    visual_refresh: float = .035
    maximum_refresh_interval: int = 32
    still_image: float = .001
    still_arm: float = .0015
    still_decisions: int = 3
    minimum_joint_range: float = .15


PROFILES = {
    'moderate': Settings(correlation=.5),
    'strong': Settings(correlation=.85),
}


class CoherentNoise:
    ARM_WAIST = np.r_[0:14, 26:29]

    def __init__(self, action_ranges, settings=Settings()):
        self.settings = settings
        if not 0 <= settings.correlation < 1:
            raise ValueError('Noise correlation must be in [0,1)')
        self.scale = np.maximum(np.asarray(action_ranges, dtype=float), settings.minimum_joint_range)
        if self.scale.shape != (29,) or not np.isfinite(self.scale).all():
            raise ValueError('Expected 29 finite action normalization ranges')
        self.previous_noise = self.previous_image = self.previous_state = None
        self.previous_executed = 0
        self.last_refresh_step = 0
        self.still_count = 0

    @staticmethod
    def image_summary(image):
        image = np.asarray(image)
        if image.shape != (256,256,3) or image.dtype != np.uint8:
            raise ValueError('Expected public uint8 RGB observation (256,256,3)')
        return image.astype(np.float64).reshape(16,16,16,16,3).mean(axis=(1,3))/255.

    def prepare(self, fresh, observed_state, image, environment_step):
        fresh = np.asarray(fresh)
        state = np.asarray(observed_state)
        if fresh.shape != (16,32) or fresh.dtype != np.float32 or not np.isfinite(fresh).all():
            raise ValueError('Expected finite float32 Gaussian noise (16,32)')
        if state.shape != (29,) or not np.isfinite(state).all():
            raise ValueError('Expected finite public joint positions (29,)')
        image = self.image_summary(image)
        step = int(environment_step)
        reasons = []
        visual = arm_change = None
        still = 0
        width = 0
        if self.previous_noise is None:
            reasons.append('first_decision')
        else:
            width = 16-self.previous_executed
            visual = float(np.mean(np.abs(image-self.previous_image)))
            arm = self.ARM_WAIST
            arm_change = float(np.sqrt(np.mean(((state[arm]-self.previous_state[arm])/self.scale[arm])**2)))
            still = self.still_count+1 if visual < self.settings.still_image and arm_change < self.settings.still_arm else 0
            if visual > self.settings.visual_refresh:
                reasons.append('scene_change')
            if step-self.last_refresh_step >= self.settings.maximum_refresh_interval:
                reasons.append('bounded_refresh_interval')
            if still >= self.settings.still_decisions:
                reasons.append('persistent_observed_stillness')
            if width == 0:
                reasons.append('no_unexecuted_noise')
        if self.settings.correlation == 0:
            reasons.append('independent_sampling')
        used = fresh.copy()
        reuse = width if not reasons else 0
        if reuse:
            rho = self.settings.correlation
            used[:reuse] = rho*self.previous_noise[self.previous_executed:] + math.sqrt(1-rho*rho)*fresh[:reuse]
        diagnostics = dict(reused_positions=reuse, correlation=self.settings.correlation if reuse else 0.,
            refresh=not bool(reuse), refresh_reasons=reasons, visual_change=visual,
            observed_arm_change=arm_change, next_still_count=still,
            environment_step=step, settings=asdict(self.settings),
            extra_policy_samples=0, action_values_modified=False)
        return used, diagnostics

    def commit(self, used_noise, observed_state, image, executed, diagnostics):
        if not 1 <= int(executed) <= 16:
            raise ValueError('Commit requires actually executed prefix length')
        self.previous_noise = np.asarray(used_noise).copy()
        self.previous_state = np.asarray(observed_state).copy()
        self.previous_image = self.image_summary(image)
        self.previous_executed = int(executed)
        self.still_count = int(diagnostics['next_still_count'])
        if diagnostics['refresh']:
            self.last_refresh_step = int(diagnostics['environment_step'])
            self.still_count = 0
