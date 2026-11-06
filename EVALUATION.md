# Evaluation: My harness for ChunkTrust — pi0.5

Twenty-four reported RoboCasa GR1 tasks, five seeds each (15–19). One unchanged policy server; alternating arm order.

## Budget

720 physical actions and 180 total policy calls per episode; no multiplier.

## Interpretation

- The exact historical simulator/backend and runtime are not fully pinned by the paper.

- Cross-server numerical drift was observed even for matching initial inputs and noise; its aggregate effect remains unresolved.

- Eight helped and five harmed; exact McNemar p=0.5811. No confirmed accuracy gain.

- Historical versions used different cohorts and must not be ranked by their absolute rates.

## Reproduction discipline

Use the reported condition only. Establish native-path parity, prepare upstream models/assets and declare case manifests before expensive runs. This export itself is a source publication, not a new evaluation. Keep all historical and new cohorts separate.
