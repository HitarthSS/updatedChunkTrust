"""Measured-state-aware AHS: a numerical wrapper around the released selector.

Only public joint state, generated actions, executed action history and the
frozen checkpoint's normalization ranges are inputs. No task/goal predicates,
object poses, evaluator outcomes, new model weights or extra action proposals.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Settings:
    tracking_high: float = .12
    tracking_low: float = .06
    tracking_strikes: int = 2
    hand_transition: float = .20
    reset_cooldown_steps: int = 16
    minimum_joint_range: float = .15


class MeasuredAHS:
    def __init__(self, selector, update, ahs_config, action_ranges, settings=Settings()):
        self.selector=selector;self.update=update;self.config=dict(ahs_config);self.settings=settings
        self.scale=np.maximum(np.asarray(action_ranges,dtype=float),settings.minimum_joint_range)
        if self.scale.shape!=(29,) or not np.isfinite(self.scale).all():
            raise ValueError('Expected 29 finite frozen action ranges')
        self.reset()

    def reset(self):
        self.posterior={};self.rng=np.random.default_rng(int(self.config['ts_seed']))
        self.lag_strikes=0;self.tracking_limited=False;self.last_reset=-10000

    def select(self, velocity_trace, actions, observed_state, executed_history, environment_step):
        actions=np.asarray(actions,dtype=float);state=np.asarray(observed_state,dtype=float)
        if actions.shape!=(16,29) or state.shape!=(29,) or not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise ValueError('Invalid public action or state observation')
        history=[np.asarray(x,dtype=float) for x in executed_history]
        if history and (history[-1].shape!=(29,) or not np.isfinite(history[-1]).all()):
            raise ValueError('Invalid actually executed action')
        last=history[-1] if history else state
        # Hand contact can intentionally prevent a finger target from being
        # reached. Never treat that contact-dependent residual as an empty grasp.
        arm_waist=np.r_[0:14,26:29]
        residual=float(np.sqrt(np.mean(((state[arm_waist]-last[arm_waist])/self.scale[arm_waist])**2))) if history else 0.
        self.lag_strikes=self.lag_strikes+1 if residual>self.settings.tracking_high else 0
        entering_lag=not self.tracking_limited and self.lag_strikes>=self.settings.tracking_strikes
        if entering_lag:self.tracking_limited=True
        if residual<self.settings.tracking_low:self.tracking_limited=False
        hand_delta=np.max(np.abs((actions[:,14:26]-last[None,14:26])/self.scale[None,14:26]),axis=1)
        changes=np.flatnonzero(hand_delta>self.settings.hand_transition)
        transition_index=int(changes[0]) if changes.size else None
        reset_reason=None
        if entering_lag:reset_reason='persistent_measured_arm_lag'
        elif transition_index is not None and environment_step-self.last_reset>=self.settings.reset_cooldown_steps:
            reset_reason='predicted_hand_command_transition'
        if reset_reason is not None:
            self.posterior.clear();self.last_reset=int(environment_step)
        before=deepcopy(self.posterior)
        options=dict(self.config)
        options.pop('horizon_candidates',None)
        options.pop('ts_seed',None)
        k,info=self.selector(np.asarray(velocity_trace),16,horizon_candidates=[4,8,12,16],
            xt_step=actions,history_actions=history,ts_state=self.posterior,ts_rng=self.rng,**options)
        k=int(k)
        if not 4<=k<=16:raise ValueError('Released AHS proposed an out-of-protocol horizon')
        cap=16;reasons=[]
        if self.tracking_limited:cap=4;reasons.append('persistent_measured_arm_lag')
        if transition_index is not None:
            # Observe within at most four steps after the hand transition; keep
            # the released minimum execution length and never extend its choice.
            hand_cap=max(4,min(16,4*math.ceil((transition_index+1)/4)))
            cap=min(cap,hand_cap)
            if k>hand_cap:reasons.append('hand_transition_checkpoint')
        chosen=min(k,cap)
        if chosen!=k:
            # The original call has updated memory for its proposed prefix.
            # Undo that update and credit only the prefix that will be executed.
            self.posterior.clear();self.posterior.update(before)
            horizons=[4,8,12,16]
            for h in horizons:self.posterior.setdefault(h,[1.,1.])
            self.update(horizons=horizons,q_mix_vals=info['q_mix'],chosen_idx=horizons.index(chosen),
                choose_mode='measured_prefix_cap',k_exec=chosen,ts_state=self.posterior,
                ts_update_mode=self.config['ts_update_mode'],ts_kernel_bandwidth=self.config['ts_kernel_bandwidth'],
                ts_forget_rho=self.config['ts_forget_rho'],ts_update_eta=self.config['ts_update_eta'])
        evidence=dict(author_proposal_k=k,executed_prefix_k=chosen,normalized_arm_tracking_rms=residual,
            tracking_lag_strikes=self.lag_strikes,tracking_limited=self.tracking_limited,
            predicted_hand_transition_index=transition_index,memory_reset_reason=reset_reason,
            reasons=reasons,settings=asdict(self.settings),posterior_after=deepcopy(self.posterior),
            action_values_modified=False,extra_policy_samples=0)
        return chosen,dict(author_selector=info,my_evidence=evidence)
