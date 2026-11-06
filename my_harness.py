"""Prediction-validated latent-noise reuse, without extra model queries.

Only public RGB, arm/waist positions, past generated targets and actually
executed prefix lengths are used. No task, seed, success or object-state input.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Settings:
    base_correlation: float = .5
    tracking_scale: float = .08
    image_scale: float = .06
    image_deadband: float = .01
    hard_image_refresh: float = .08
    minimum_correlation: float = .10
    maximum_refresh_interval: int = 32
    minimum_command_motion: float = .01
    low_progress_ratio: float = .15
    low_progress_strikes: int = 2
    minimum_joint_range: float = .15


PROFILES = {
    'cautious': Settings(),
    'adaptive': Settings(base_correlation=.65,tracking_scale=.12,image_scale=.10),
}


class ValidatedNoise:
    ARM_WAIST = np.r_[0:14,26:29]

    def __init__(self, action_ranges, settings=Settings()):
        self.settings=settings
        if not 0<=settings.base_correlation<1:
            raise ValueError('Invalid correlation')
        self.scale=np.maximum(np.asarray(action_ranges,dtype=float),settings.minimum_joint_range)
        if self.scale.shape!=(29,) or not np.isfinite(self.scale).all():
            raise ValueError('Expected 29 finite normalization ranges')
        self.previous_noise=self.previous_state=self.previous_image=self.expected_endpoint=None
        self.previous_executed=0
        self.last_refresh_step=0
        self.low_progress_count=0

    @staticmethod
    def summarize_image(image):
        image=np.asarray(image)
        if image.shape!=(256,256,3) or image.dtype!=np.uint8:
            raise ValueError('Expected public uint8 RGB (256,256,3)')
        return image.astype(np.float64).reshape(16,16,16,16,3).mean(axis=(1,3))/255.

    def prepare(self,fresh,observed_state,image,environment_step):
        fresh=np.asarray(fresh);state=np.asarray(observed_state,dtype=float)
        if fresh.shape!=(16,32) or fresh.dtype!=np.float32 or not np.isfinite(fresh).all():
            raise ValueError('Expected finite float32 policy noise (16,32)')
        if state.shape!=(29,) or not np.isfinite(state).all():
            raise ValueError('Expected 29 finite public joint positions')
        coarse=self.summarize_image(image);s=self.settings
        reasons=[];width=0;strikes=0;rho=0.
        metrics=dict(endpoint_error=None,command_motion=None,observed_motion=None,
                     projected_progress=None,image_change=None,tracking_confidence=None,
                     direction_confidence=None,visual_confidence=None)
        if self.previous_noise is None:
            reasons.append('first_decision')
        else:
            arm=self.ARM_WAIST;scale=self.scale[arm]
            desired=(self.expected_endpoint[arm]-self.previous_state[arm])/scale
            actual=(state[arm]-self.previous_state[arm])/scale
            error=float(np.sqrt(np.mean(((state[arm]-self.expected_endpoint[arm])/scale)**2)))
            command=float(np.sqrt(np.mean(desired**2)))
            observed=float(np.sqrt(np.mean(actual**2)))
            visual=float(np.mean(np.abs(coarse-self.previous_image)))
            tracking=math.exp(-error/s.tracking_scale)
            direction=1.
            progress=None
            if command>=s.minimum_command_motion:
                progress=float(np.dot(actual,desired)/(np.dot(desired,desired)+1e-12))
                direction=float(np.clip(progress/.5,0.,1.))
                if progress<s.low_progress_ratio and visual<s.image_deadband:
                    strikes=self.low_progress_count+1
            visual_confidence=math.exp(-max(0.,visual-s.image_deadband)/s.image_scale)
            rho=s.base_correlation*tracking*direction*visual_confidence
            width=16-self.previous_executed
            if visual>s.hard_image_refresh:reasons.append('large_scene_change')
            if strikes>=s.low_progress_strikes:reasons.append('persistent_commanded_motion_without_progress')
            if int(environment_step)-self.last_refresh_step>=s.maximum_refresh_interval:
                reasons.append('bounded_refresh_interval')
            if width==0:reasons.append('no_unexecuted_noise')
            if rho<s.minimum_correlation:reasons.append('insufficient_prediction_agreement')
            metrics=dict(endpoint_error=error,command_motion=command,observed_motion=observed,
                projected_progress=progress,image_change=visual,tracking_confidence=tracking,
                direction_confidence=direction,visual_confidence=visual_confidence)
        if reasons:rho=0.
        reuse=width if rho else 0
        used=fresh.copy()
        if reuse:
            used[:reuse]=rho*self.previous_noise[self.previous_executed:]+math.sqrt(1-rho*rho)*fresh[:reuse]
        return used,dict(reused_positions=reuse,correlation=float(rho),refresh=not bool(reuse),
            refresh_reasons=reasons,environment_step=int(environment_step),
            next_low_progress_count=strikes,metrics=metrics,settings=asdict(s),
            extra_policy_samples=0,action_values_modified=False)

    def commit(self,used_noise,observed_state,image,actions,executed,diagnostics):
        actions=np.asarray(actions)
        if actions.shape!=(16,29) or not np.isfinite(actions).all() or not 1<=int(executed)<=16:
            raise ValueError('Invalid generated actions or actually executed prefix')
        self.previous_noise=np.asarray(used_noise).copy()
        self.previous_state=np.asarray(observed_state,dtype=float).copy()
        self.previous_image=self.summarize_image(image)
        self.previous_executed=int(executed)
        self.expected_endpoint=actions[int(executed)-1].copy()
        self.low_progress_count=int(diagnostics['next_low_progress_count'])
        if diagnostics['refresh']:
            self.last_refresh_step=int(diagnostics['environment_step'])
            self.low_progress_count=0
