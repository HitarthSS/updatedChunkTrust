# My harness for ChunkTrust — pi0.5

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2609.39754). The evaluated behavior is in [`my_harness.py`](my_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | 42.50% AHS, 1,200 episodes |

| Our run of the authors' harness | 40/120 (33.33%) |

| **My harness** | **43/120 (35.83%)** |


Twenty-four reported RoboCasa GR1 tasks, five seeds each (15–19). One unchanged policy server; alternating arm order.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Reuse only the unexecuted, time-aligned part of the previous sampling noise.

2. Reduce correlation when observed motion disagrees with the previous prediction; refresh on large visual changes, stalls or a bounded interval.

3. Keep one policy sample per decision and the authors' AHS execution-prefix selection. No action blending or learned QHA.


Categories: `sampler_guidance`, `execution_monitoring`.

## Models and inputs

- Frozen NewGain/pi05_robocasa_gr1_tabletop24, step 30,000; ten denoising steps.

- The selected v4 profile is adaptive. It expects 29 public state coordinates and a 16x32 noise tensor.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/hf618/ChunkTrust) before installing its model stack.
2. Controller dependency: numpy==1.26.4
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 -m pip install "numpy==1.26.4"
python3 tools/verify_export.py
python -m unittest discover -s tests -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Instantiate ValidatedNoise(action_ranges, PROFILES["adaptive"]). Call prepare(fresh_noise, public_state, rgb, physical_step), pass the returned noise to the same frozen policy, let native AHS choose k, then commit(noise, public_state, rgb, native_actions, k, diagnostics).


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

720 physical actions and 180 total policy calls per episode; no multiplier.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- The exact historical simulator/backend and runtime are not fully pinned by the paper.

- Cross-server numerical drift was observed even for matching initial inputs and noise; its aggregate effect remains unresolved.

- Eight helped and five harmed; exact McNemar p=0.5811. No confirmed accuracy gain.

- Historical versions used different cohorts and must not be ranked by their absolute rates.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

3 CPU tests passed. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
