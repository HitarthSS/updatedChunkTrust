# Integration and arm definitions

`my_harness.py` exports `ValidatedNoise`, `Settings`, and `PROFILES`. Each episode needs a fresh controller. The public inputs are uint8 RGB `(256,256,3)`, observed joint positions `(29,)`, native action ranges `(29,)`, fresh float32 Gaussian noise `(16,32)`, generated native actions `(16,29)`, the actual executed prefix length `k`, and cumulative physical step count.

```python
from my_harness import ValidatedNoise, PROFILES

controller = ValidatedNoise(action_ranges, PROFILES["adaptive"])
# At each decision, use the native RNG to obtain fresh_noise as usual.
used_noise, audit = controller.prepare(fresh_noise, public_state, rgb, step)
actions = frozen_policy.sample(observation, noise=used_noise)  # backend-specific
k = native_ahs(actions)                                    # unchanged authors' selector
# Execute exactly k actions, respecting the remaining physical horizon.
controller.commit(used_noise, public_state, rgb, actions, k, audit)
```

The policy call above is pseudocode for the backend API, not a replacement policy implementation. For the authors' harness, use the same fresh noise and native AHS without the controller. The code adds no samples or learned selector. `reference/validated-v4/` records the actual integration; earlier versions are retained for provenance and have different evaluation cohorts. Historical pipeline freeze receipts and upstream assets are required; do not bypass guards to make them launch.
