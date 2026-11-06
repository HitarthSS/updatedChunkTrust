"""Budgeted whole-proposal selection around the unchanged, frozen-policy AHS.

No task labels, evaluator success, hidden object poses, extra model, or learned
parameters are inputs. Losing previews never update the horizon posterior/RNG.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Settings:
    quality_weight: float = .65
    overlap_weight: float = .20
    boundary_weight: float = .15
    switch_margin: float = .03
    quality_regression_allowance: float = .10
    minimum_proposal_difference: float = .003
    maximum_extra_samples: int = 24
    initial_extra_credits: int = 2
    extra_cooldown_steps: int = 8
    minimum_remaining_steps: int = 16
    uncertainty_probability: float = .45
    uncertainty_quality: float = .70
    overlap_trigger: float = .08
    boundary_trigger: float = .12
    minimum_joint_range: float = .15


PROFILES = {
    'balanced': Settings(),
    'quality': Settings(quality_weight=1., overlap_weight=0., boundary_weight=0.,
                        switch_margin=.03, quality_regression_allowance=0.),
}


class CallBudget:
    """Reserve enough primary calls to finish with four-step execution chunks."""
    def __init__(self, maximum=180):
        self.maximum = int(maximum)
        self.used = 0

    def primary(self):
        if self.used >= self.maximum:
            raise RuntimeError('Policy inference budget exhausted')
        self.used += 1

    def can_buy_extra(self, remaining_steps):
        future = math.ceil(max(0, int(remaining_steps)-4)/4)
        return self.used + 1 + future <= self.maximum

    def extra(self, remaining_steps):
        if not self.can_buy_extra(remaining_steps):
            raise RuntimeError('An extra proposal would consume reserved primary calls')
        self.used += 1


class ProposalAHS:
    ARM_WAIST = np.r_[0:14, 26:29]

    def __init__(self, selector, normalize, ahs_config, action_ranges, settings=Settings()):
        self.selector = selector
        self.normalize = normalize
        self.config = dict(ahs_config)
        self.settings = settings
        self.scale = np.maximum(np.asarray(action_ranges, dtype=float), settings.minimum_joint_range)
        if self.scale.shape != (29,) or not np.isfinite(self.scale).all():
            raise ValueError('Expected 29 finite frozen action ranges')
        if abs(settings.quality_weight + settings.overlap_weight + settings.boundary_weight-1.) > 1e-10:
            raise ValueError('Proposal score weights must sum to one')
        self.posterior = {}
        self.rng = np.random.default_rng(int(self.config['ts_seed']))
        self.previous_actions = None
        self.previous_executed = None
        self.previous_state = None
        self.previous_image = None
        self.extra_samples = 0
        self.last_extra_step = -10000

    @staticmethod
    def image_summary(image):
        image = np.asarray(image)
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError('Expected public uint8 RGB observation, 256x256x3')
        return image.astype(np.float64).reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))/255.

    def preview(self, actions, velocity_trace, history):
        actions = np.asarray(actions)
        velocity_trace = np.asarray(velocity_trace)
        if actions.shape != (16, 29) or velocity_trace.shape != (10, 16, 29):
            raise ValueError('Policy action/trace shape mismatch')
        if not np.isfinite(actions).all() or not np.isfinite(velocity_trace).all():
            raise ValueError('Non-finite policy proposal')
        state = deepcopy(self.posterior)
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        options = dict(self.config)
        options.pop('ts_seed')
        horizon, info = self.selector(velocity_trace, 16, xt_step=actions,
            history_actions=history, ts_state=state, ts_rng=rng, **options)
        horizon = int(horizon)
        if not 4 <= horizon <= 16:
            raise RuntimeError('Released AHS returned an out-of-protocol horizon')
        return dict(actions=actions, trace=velocity_trace, horizon=horizon,
                    ahs=info, posterior=state, rng_state=deepcopy(rng.bit_generator.state))

    def features(self, actions, observed_state, image, history):
        state = np.asarray(observed_state, dtype=float)
        if state.shape != (29,) or not np.isfinite(state).all():
            raise ValueError('Expected 29 finite public joint positions')
        image = self.image_summary(image)
        arm = self.ARM_WAIST
        norm = self.scale[arm]
        reference = np.asarray(history[-1], dtype=float) if len(history) else state
        old_velocity = ((np.asarray(history[-1])-history[-2])/self.scale)[arm] if len(history)>1 else np.zeros(17)
        first_velocity = (actions[0,arm]-reference[arm])/norm
        jump = float(np.sqrt(np.mean((first_velocity-old_velocity)**2)))
        positions = np.vstack([reference[arm], actions[:8,arm]])/norm
        acceleration = float(np.sqrt(np.mean(np.diff(positions,n=2,axis=0)**2)))
        boundary_score = math.exp(-4.*(.5*jump+.5*acceleration))
        context = 0.
        overlap = None
        if self.previous_actions is not None:
            unused = self.previous_actions[self.previous_executed:]
            width = min(4, len(unused))
            if width:
                difference = (actions[:width,arm]-unused[:width,arm])/norm
                overlap = float(np.sqrt(np.mean(difference**2)))
                image_change = float(np.mean(np.abs(image-self.previous_image)))
                state_change = float(np.sqrt(np.mean(((state[arm]-self.previous_state[arm])/norm)**2)))
                # Downweight a stale previous prediction after a visible/state
                # change. Agreement is a consistency signal, not task success.
                context = math.exp(-image_change/.03-state_change/.08)
        return dict(boundary_jump=jump, boundary_acceleration=acceleration,
                    boundary_score=boundary_score, overlap_rms=overlap,
                    overlap_context_weight=context,
                    overlap_score=math.exp(-4.*overlap) if overlap is not None else 0.)

    def request_second(self, primary, observed_state, image, history, environment_step, budget):
        s = self.settings
        reasons = []
        f = self.features(primary['actions'], observed_state, image, history)
        info = primary['ahs']
        quality = float(np.interp(primary['horizon'], info['horizons'], info['q_mix']))
        if max(info['expected_prob']) < s.uncertainty_probability and quality < s.uncertainty_quality:
            reasons.append('ambiguous_horizon_evidence')
        if f['overlap_rms'] is not None and f['overlap_rms']*f['overlap_context_weight'] > s.overlap_trigger:
            reasons.append('changed_overlap_prediction')
        if f['boundary_jump'] > s.boundary_trigger:
            reasons.append('large_arm_boundary_change')
        remaining = 720-int(environment_step)
        credits = min(s.maximum_extra_samples,
            s.initial_extra_credits+int(s.maximum_extra_samples*environment_step/720))
        eligible = (bool(reasons) and len(history)>=4 and remaining>=s.minimum_remaining_steps
                    and self.extra_samples < credits
                    and environment_step-self.last_extra_step >= s.extra_cooldown_steps
                    and budget.can_buy_extra(remaining))
        return bool(eligible), dict(reasons=reasons, eligible=bool(eligible),
            extra_credits_available=credits-self.extra_samples,
            budget_reservation_allows=budget.can_buy_extra(remaining), features=f)

    def shared_quality(self, previews):
        # Released scores are normalized within each chunk. Recompute only the
        # comparison scores on a shared scale across proposals and horizons.
        z = np.asarray([p['ahs']['z_intra'] for p in previews], dtype=float)
        u = np.asarray([p['ahs']['u_inter_raw'] for p in previews], dtype=float)
        zn = self.normalize(z.ravel()).reshape(z.shape)
        un = self.normalize(u.ravel()).reshape(u.shape)
        intra = np.exp(-self.config['intra_alpha']*np.maximum(0, zn))
        inter = np.exp(-self.config['speed_beta']*np.maximum(0, un))
        mixed = np.where(np.isfinite(inter), (1-self.config['speed_weight'])*intra+
                         self.config['speed_weight']*inter, intra)
        if not np.isfinite(mixed).all():
            raise ValueError('Shared proposal quality is not finite')
        common_horizon = previews[0]['horizon']
        quality = [float(np.interp(common_horizon, p['ahs']['horizons'], q))
                   for p, q in zip(previews, mixed)]
        return quality, dict(common_scoring_horizon=common_horizon,
                             raw_intra=z.tolist(), raw_inter=u.tolist(), mixed=mixed.tolist())

    def choose(self, previews, observed_state, image, history):
        if not 1 <= len(previews) <= 2:
            raise ValueError('Only one or two whole proposals are permitted')
        if len(previews) == 1:
            return 0, dict(selected_proposal=0, proposal_count=1, action_values_modified=False)
        qualities, evidence = self.shared_quality(previews)
        features = [self.features(p['actions'], observed_state, image, history) for p in previews]
        scores = []
        for quality, f in zip(qualities, features):
            context = f['overlap_context_weight']
            scores.append((self.settings.quality_weight+self.settings.overlap_weight*(1-context))*quality+
                self.settings.overlap_weight*context*f['overlap_score']+
                self.settings.boundary_weight*f['boundary_score'])
        arm = self.ARM_WAIST
        delta = (previews[1]['actions'][:8,arm]-previews[0]['actions'][:8,arm])/self.scale[arm]
        distance = float(np.sqrt(np.mean(delta**2)))
        choose_second = (scores[1] > scores[0]+self.settings.switch_margin
            and qualities[1] >= qualities[0]-self.settings.quality_regression_allowance
            and distance >= self.settings.minimum_proposal_difference)
        selected = int(choose_second)
        return selected, dict(selected_proposal=selected, proposal_count=2,
            scores=scores, shared_quality=qualities, shared_evidence=evidence,
            features=features, proposal_distance=distance, settings=asdict(self.settings),
            action_values_modified=False)

    def commit(self, preview, observed_state, image, executed):
        if not 1 <= int(executed) <= preview['horizon']:
            raise ValueError('Invalid number of actually executed actions')
        self.posterior = deepcopy(preview['posterior'])
        self.rng.bit_generator.state = deepcopy(preview['rng_state'])
        self.previous_actions = preview['actions'].copy()
        self.previous_executed = int(executed)
        self.previous_state = np.asarray(observed_state).copy()
        self.previous_image = self.image_summary(image)

    def record_extra(self, environment_step):
        self.extra_samples += 1
        self.last_extra_step = int(environment_step)
